"""A small, configurable GPT training entry point for the arithmetic dataset.

Examples (from the repository root)::

    python3 -m scripts.base_train --run-name arithmetic-small --max-iters 100 --batch-size 8
    python3 -m scripts.base_train --run-name arithmetic-config --config configs/train.json
    torchrun --standalone --nproc_per_node=4 -m scripts.base_train --run-name arithmetic-ddp --device cuda

The script intentionally keeps the data pipeline simple: every JSONL ``text``
field is encoded as one independent example and terminated with ``#``.  It
supports one process on CPU/GPU and PyTorch DistributedDataParallel (DDP) when
launched with ``torchrun``.  The master process records metrics to SwanLab.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Make ``python scripts/base_train.py`` work as well as ``python -m ...``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.gpt import GPT, GPTConfig  # noqa: E402
from model.tokenizer import ArithmeticTokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the arithmetic GPT model")
    parser.add_argument("--config", type=Path, help="JSON config file; command-line values override it")

    # Data and output
    parser.add_argument("--train-data", type=Path, default=Path("data/add-sub-int_train.json"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/add-sub-int_eval.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint path to resume")
    parser.add_argument("--save-every", type=int, default=0, help="save every N steps (0 saves only the best/final checkpoint)")

    # Model
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=32, help="per-process micro batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-iters", "--num-iterations", dest="max_iters", type=int, default=1000)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--decay-lr", action=argparse.BooleanOptionalAction, default=True)

    # Runtime and evaluation
    parser.add_argument("--device", "--device-type", dest="device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    # SwanLab uses its defaults for project/output settings; only the run name
    # is exposed because that is the experiment-specific value.
    parser.add_argument("--run-name", default=None, help="SwanLab experiment name")
    return parser


def _load_json_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a JSON object: {path}")
    # Permit either CLI spelling (``max-iters``) or Python spelling
    # (``max_iters``) in external configs.
    normalized = {key.replace("-", "_"): value for key, value in config.items()}
    aliases = {"num_iterations": "max_iters", "device_type": "device", "run": "run_name"}
    normalized = {aliases.get(key, key): value for key, value in normalized.items()}
    return normalized


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    config_probe, _ = parser.parse_known_args(argv)
    if config_probe.config is not None:
        values = _load_json_config(config_probe.config)
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(values) - known)
        if unknown:
            parser.error(f"unknown config option(s): {', '.join(unknown)}")
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    for name in ("train_data", "eval_data", "out_dir", "resume"):
        value = getattr(args, name)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    if args.config is not None and not args.config.is_file():
        parser.error(f"config file does not exist: {args.config}")
    for name in ("batch_size", "gradient_accumulation_steps", "max_iters", "block_size", "n_layer", "n_head", "n_embd", "eval_iters", "log_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_embd % args.n_head != 0:
        parser.error("--n-embd must be divisible by --n-head")
    return args


class JsonlExamples:
    """Encoded, padded examples loaded from the project's JSONL format."""

    def __init__(self, path: Path, tokenizer: ArithmeticTokenizer, block_size: int):
        self.path = path
        self.examples: list[tuple[list[int], list[int]]] = []
        eos = tokenizer.stoi["#"]
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{line_number} must contain a string field 'text'")
                try:
                    tokens = tokenizer.encode(text + "#")
                except ValueError as exc:
                    raise ValueError(f"invalid text in {path}:{line_number}: {exc}") from exc
                if len(tokens) < 2 or len(tokens) - 1 > block_size:
                    raise ValueError(f"{path}:{line_number} has {len(tokens) - 1} input tokens; block_size={block_size}")
                x = tokens[:-1]
                y = tokens[1:]
                x = x + [eos] * (block_size - len(x))
                y = y + [-1] * (block_size - len(y))
                self.examples.append((x, y))
        if not self.examples:
            raise ValueError(f"no examples found in {path}")

    def batch(self, batch_size: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self.examples), (batch_size,), generator=generator).tolist()
        x = torch.tensor([self.examples[i][0] for i in indices], dtype=torch.long, device=device)
        y = torch.tensor([self.examples[i][1] for i in indices], dtype=torch.long, device=device)
        return x, y


def create_logger(run_name: str | None, master_process: bool):
    if not master_process:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("SwanLab is required; install it with `pip install swanlab`") from exc
    return swanlab.init(experiment_name=run_name)


def _ddp_info(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if distributed and args.backend == "nccl" and not use_cuda:
        raise ValueError("NCCL requires CUDA; use --backend gloo for CPU DDP")
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if distributed:
        backend = args.backend if args.backend != "auto" else ("nccl" if use_cuda else "gloo")
        dist.init_process_group(backend=backend)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return distributed, rank, local_rank, world_size, device


def _autocast(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" or (device.type == "cpu" and dtype == torch.bfloat16)
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _lr(args: argparse.Namespace, step: int) -> float:
    if not args.decay_lr:
        return args.learning_rate
    if args.warmup_iters > 0 and step < args.warmup_iters:
        return args.learning_rate * (step + 1) / (args.warmup_iters + 1)
    if step >= args.lr_decay_iters:
        return args.min_lr
    ratio = (step - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
    ratio = min(1.0, max(0.0, ratio))
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, dataset: JsonlExamples, args: argparse.Namespace, device: torch.device, generator: torch.Generator, ddp: bool) -> float:
    model.eval()
    total = 0.0
    for _ in range(args.eval_iters):
        x, y = dataset.batch(args.batch_size, generator, device)
        with _autocast(device, _compute_dtype(args, device)):
            _, loss = model(x, y)
        total += float(loss.item())
    value = torch.tensor(total / args.eval_iters, device=device)
    if ddp:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    model.train()
    return float(value.item())


def _compute_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if args.dtype == "float32":
        return torch.float32
    if args.dtype == "bfloat16":
        return torch.bfloat16
    if args.dtype == "float16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    ddp, rank, local_rank, world_size, device = _ddp_info(args)
    master_process = rank == 0
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = ArithmeticTokenizer()
    train_data = JsonlExamples(args.train_data, tokenizer, args.block_size)
    eval_data = JsonlExamples(args.eval_data, tokenizer, args.block_size)
    train_generator = torch.Generator().manual_seed(args.seed + 1000 + rank)
    eval_generator = torch.Generator().manual_seed(args.seed + 2000 + rank)

    model_config = GPTConfig(
        block_size=args.block_size, vocab_size=tokenizer.vocab_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, bias=args.bias,
    )
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device.type)
    scaler_enabled = device.type == "cuda" and _compute_dtype(args, device) == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # compatibility with older PyTorch releases
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            checkpoint = torch.load(args.resume, map_location=device)
        state = checkpoint["model"]
        state = {key.removeprefix("_orig_mod.").removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("iter_num", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if args.compile:
        model = torch.compile(model)
    raw_model = model.module if isinstance(model, DDP) else model
    config_for_logging = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_for_logging.update(model_config=vars(model_config), vocab=tokenizer.tokens, world_size=world_size)
    logger = create_logger(args.run_name, master_process)
    if master_process:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"device={device} world_size={world_size} train={len(train_data.examples)} eval={len(eval_data.examples)}")

    last_step = start_step
    for step in range(start_step, args.max_iters):
        for group in optimizer.param_groups:
            group["lr"] = _lr(args, step)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for micro_step in range(args.gradient_accumulation_steps):
            x, y = train_data.batch(args.batch_size, train_generator, device)
            sync_context = model.no_sync() if isinstance(model, DDP) and micro_step < args.gradient_accumulation_steps - 1 else nullcontext()
            with sync_context, _autocast(device, _compute_dtype(args, device)):
                _, loss = model(x, y)
                scaled_loss = loss / args.gradient_accumulation_steps
            train_loss += float(loss.detach().item())
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        last_step = step + 1
        if master_process and (last_step % args.log_interval == 0 or last_step == 1):
            metrics = {"train/loss": train_loss / args.gradient_accumulation_steps, "train/lr": optimizer.param_groups[0]["lr"], "step_time": time.time() - start}
            print(f"step {last_step:5d}/{args.max_iters} | loss {metrics['train/loss']:.4f} | lr {metrics['train/lr']:.3g}")
            logger.log(metrics, step=last_step)
        if args.eval_interval > 0 and last_step % args.eval_interval == 0:
            val_loss = estimate_loss(model, eval_data, args, device, eval_generator, ddp)
            if master_process:
                print(f"step {last_step:5d} | val loss {val_loss:.4f}")
                logger.log({"val/loss": val_loss}, step=last_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(args.out_dir / "ckpt_best.pt", raw_model, optimizer, last_step, best_val_loss, config_for_logging)
        if args.save_every > 0 and last_step % args.save_every == 0 and master_process:
            _save_checkpoint(args.out_dir / f"ckpt_{last_step:06d}.pt", raw_model, optimizer, last_step, best_val_loss, config_for_logging)

    if master_process:
        _save_checkpoint(args.out_dir / "ckpt_last.pt", raw_model, optimizer, last_step, best_val_loss, config_for_logging)
        logger.finish()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def _save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int, best_val_loss: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter_num": step, "best_val_loss": best_val_loss, "config": config}, path)


if __name__ == "__main__":
    main()
