"""Evaluate arithmetic answers, including nested expressions, against dataset GT.

Examples (from the repository root)::

    python -m scripts.base_eval --checkpoint out/ckpt_best.pt
    python scripts/base_eval.py --checkpoint out/ckpt_last.pt \
        --eval-data data/add-sub-int_eval.json data/complex_eval.jsonl
    python -m scripts.base_eval --checkpoint out/ckpt_best.pt \
        --eval-data 'data/*_eval.json' --swanlab-mode offline --output eval.json

Dependencies: torch, swanlab. JSON and JSONL extensions both use one {"text":
"expression=answer"} object per line, matching the data generators. Ground truth
is read verbatim, never recomputed (division may have been truncated per node).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import glob
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.engine import Engine  # noqa: E402
from model.gpt import GPT, GPTConfig  # noqa: E402
from model.tokenizer import ArithmeticTokenizer  # noqa: E402

NUMBER = re.compile(r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)\Z")


@dataclass
class Example:
    line: int
    prompt: str
    answer: str
    ids: list[int]


def load_examples(path, tokenizer, block_size, max_examples=None, seed=1337):
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    raise ValueError("expected an object with a string text field")
                text = obj["text"]
                if text.endswith("#"):
                    text = text[:-1]
                if text.count("=") != 1:
                    raise ValueError("expected exactly one '='")
                expression, answer = text.split("=")
                if not expression or "#" in expression or not NUMBER.fullmatch(answer):
                    raise ValueError("expected a nonempty expression and numeric GT")
                tokenizer.encode(text)
                ids = tokenizer.encode(expression + "=")
                if len(ids) > block_size:
                    raise ValueError(f"prompt length {len(ids)} exceeds context {block_size}")
                examples.append(Example(line, expression + "=", answer, ids))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not examples:
        raise ValueError(f"no examples found in {path}")
    if max_examples is not None and len(examples) > max_examples:
        import random
        examples = random.Random(seed).sample(examples, max_examples)
        examples.sort(key=lambda ex: ex.line)
    return examples


class Metrics:
    """Accuracy denominators include every sample; error uses numeric answers.

    Unterminated outputs are always incorrect. Their numeric prefixes do not
    enter MAE/RMSE; valid_count and coverage make that denominator explicit.
    Relative error excludes zero GT and reports its own denominator.
    """
    def __init__(self):
        self.total = self.exact = self.correct = self.valid = 0
        self.invalid = self.empty = self.unterminated = self.context_limit = 0
        self.numeric_wrong = self.relative_count = 0
        self.abs_sum = self.square_sum = self.bias_sum = self.relative_sum = Decimal(0)
        self.max_error = Decimal(0)

    def add(self, prediction, target, stop_reason):
        self.total += 1
        if stop_reason != "eos":
            self.unterminated += 1
            self.context_limit += stop_reason == "context_limit"
            return {"correct": False, "exact_match": False, "absolute_error": None}
        if not NUMBER.fullmatch(prediction):
            self.invalid += 1
            self.empty += prediction == ""
            return {"correct": False, "exact_match": False, "absolute_error": None}
        self.valid += 1
        # Use Decimal throughout: long integers and generator decimal labels
        # must not become equal through binary float rounding.
        with localcontext() as ctx:
            ctx.prec = max(128, 4 * max(len(prediction), len(target), len(str(self.square_sum))))
            value, gt = Decimal(prediction), Decimal(target)
            error = value - gt
            absolute = abs(error)
            correct, exact = value == gt, prediction == target
            self.correct += correct
            self.exact += exact
            self.numeric_wrong += not correct
            self.abs_sum += absolute
            self.square_sum += error * error
            self.bias_sum += error
            self.max_error = max(self.max_error, absolute)
            if gt != 0:
                self.relative_sum += absolute / abs(gt)
                self.relative_count += 1
        return {"correct": correct, "exact_match": exact, "absolute_error": str(absolute)}

    def summary(self):
        def mean(value, count):
            return float(value / count) if count else None
        return {
            "count": self.total,
            "accuracy": self.correct / self.total,
            "exact_match": self.exact / self.total,
            "valid_count": self.valid,
            "numeric_coverage": self.valid / self.total,
            "mae": mean(self.abs_sum, self.valid),
            "rmse": float((self.square_sum / self.valid).sqrt()) if self.valid else None,
            "mean_signed_error": mean(self.bias_sum, self.valid),
            "max_absolute_error": float(self.max_error) if self.valid else None,
            "mean_relative_error": mean(self.relative_sum, self.relative_count),
            "relative_error_count": self.relative_count,
            "invalid_format_count": self.invalid,
            "empty_answer_count": self.empty,
            "unterminated_count": self.unterminated,
            "context_limit_count": self.context_limit,
            "numeric_wrong_count": self.numeric_wrong,
        }


def load_model(path, device, tokenizer):
    # base_train checkpoints contain tensors and primitive config dictionaries.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model_args = checkpoint.get("model_args") or config.get("model_config")
    if not isinstance(model_args, dict):
        raise ValueError("checkpoint needs config.model_config (base_train) or model_args")
    if config.get("vocab", tokenizer.tokens) != tokenizer.tokens:
        raise ValueError("checkpoint tokenizer vocabulary differs from ArithmeticTokenizer")
    model_config = GPTConfig(**model_args)
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary size differs from ArithmeticTokenizer")
    model = GPT(model_config)
    state = {}
    for key, value in checkpoint["model"].items():
        while key.startswith(("_orig_mod.", "module.")):
            key = key.split(".", 1)[1]
        state[key] = value
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint.get("iter_num")


def evaluate(engine, tokenizer, examples, batch_size, max_new_tokens, aggregate):
    buckets = defaultdict(list)
    for example in examples:
        buckets[len(example.ids)].append(example)
    metrics = Metrics()
    records = []
    for group in buckets.values():
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            generations = engine.generate([ex.ids for ex in batch], max_new_tokens)
            for example, generation in zip(batch, generations):
                prediction = tokenizer.decode(generation.tokens)
                score = metrics.add(prediction, example.answer, generation.stop_reason)
                aggregate.add(prediction, example.answer, generation.stop_reason)
                records.append({"line": example.line, "prompt": example.prompt,
                                "target": example.answer, "prediction": prediction,
                                "stop_reason": generation.stop_reason, **score})
            print(f"\r  evaluated {metrics.total}/{len(examples)}", end="", flush=True)
    print()
    return metrics.summary(), sorted(records, key=lambda row: row["line"])


def print_summary(name, summary):
    print(f"\n[{name}] (accuracy and exact_match require '#')")
    for key, value in summary.items():
        print(f"  {key}: {value:.6g}" if isinstance(value, float) else f"  {key}: {value}")


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-data", nargs="+", action="extend", default=None,
                        help="one or more JSONL files or quoted glob patterns; may be repeated")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32, help="answer token budget, including '#'")
    parser.add_argument("--max-examples", type=int, help="seeded subset size per dataset; default evaluates all")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--swanlab-project", default="arithmetic-eval")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "local", "disabled"), default="online")
    parser.add_argument("--output", type=Path, help="save config, metrics and every prediction as JSON")
    args = parser.parse_args(argv)
    for key in ("batch_size", "max_new_tokens", "max_examples"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    paths = []
    for pattern in args.eval_data or ["data/add-sub-int_eval.json"]:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"no datasets match: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                parser.error(f"not a file: {path}")
            if path not in paths:
                paths.append(path)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    tokenizer = ArithmeticTokenizer()
    model, step = load_model(args.checkpoint, device, tokenizer)
    model.to(dtype=getattr(torch, args.dtype))
    engine = Engine(model, tokenizer)
    datasets = [load_examples(p, tokenizer, model.config.block_size, args.max_examples, args.seed) for p in paths]
    if args.output and args.output.resolve() in [*paths, args.checkpoint.resolve()]:
        parser.error("--output must differ from the datasets and checkpoint")
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("SwanLab is required: pip install swanlab") from exc
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), model_config=asdict(model.config), vocab=tokenizer.tokens,
                  checkpoint_step=step, python_version=platform.python_version(),
                  torch_version=str(torch.__version__), swanlab_version=swanlab.__version__,
                  datasets=[{"path": str(p), "sha256": file_sha256(p),
                             "evaluated_count": len(examples)} for p, examples in zip(paths, datasets)],
                  decoding="greedy", error_denominator="numeric answers terminated by #",
                  relative_error_denominator="numeric answers terminated by # with nonzero GT")
    swanlab.init(project=args.swanlab_project, experiment_name=args.run_name, config=config, mode=args.swanlab_mode)
    report = {"config": config, "datasets": {}}
    aggregate = Metrics()
    try:
        for index, (path, examples) in enumerate(zip(paths, datasets), 1):
            name = f"{index:02d}_{path.stem}"
            print(f"Evaluating {path} ({len(examples)} examples, device={device})", flush=True)
            start = time.perf_counter()
            summary, records = evaluate(engine, tokenizer, examples, args.batch_size, args.max_new_tokens, aggregate)
            summary["elapsed_seconds"] = time.perf_counter() - start
            summary["examples_per_second"] = len(examples) / summary["elapsed_seconds"]
            print_summary(name, summary)
            swanlab.log({f"eval/{name}/{k}": v for k, v in summary.items()
                         if isinstance(v, (int, float)) and math.isfinite(v)})
            report["datasets"][name] = {"path": str(path), "metrics": summary, "predictions": records}
        report["overall"] = aggregate.summary()
        print_summary("overall (sample weighted)", report["overall"])
        swanlab.log({f"eval/overall/{k}": v for k, v in report["overall"].items()
                     if isinstance(v, (int, float)) and math.isfinite(v)})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Saved {args.output}")
    except BaseException as exc:
        swanlab.finish(state="crashed", error=str(exc))
        raise
    else:
        swanlab.finish()


if __name__ == "__main__":
    main()
