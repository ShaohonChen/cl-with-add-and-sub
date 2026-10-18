"""Small inference engine for arithmetic GPT, inspired by nanochat's engine.

Prefill each prompt once, then reuse preallocated per-layer keys and values.
Prompts in a batch must have equal length (the evaluator groups them by length).
No model parameters, training forward path, or checkpoint keys are changed.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass
class Generation:
    tokens: list[int]  # continuation only, excluding the stop token
    stop_reason: str  # eos, max_tokens, or context_limit


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.stop_id = tokenizer.encode("#")[0]

    def _forward(self, ids, cache, position, capacity):
        """Forward a prefill or one decode token, using absolute GPT positions."""
        model = self.model
        end = position + ids.size(1)
        pos = torch.arange(position, end, device=ids.device)
        x = model.transformer.drop(model.transformer.wte(ids) + model.transformer.wpe(pos))
        for index, block in enumerate(model.transformer.h):
            attn = block.attn
            batch, length, width = x.shape
            q, k, v = attn.c_attn(block.ln_1(x)).split(width, dim=2)
            q, k, v = [t.view(batch, length, attn.n_head, width // attn.n_head).transpose(1, 2)
                       for t in (q, k, v)]
            if cache[index] is None:
                shape = (batch, attn.n_head, capacity, width // attn.n_head)
                cache[index] = (k.new_empty(shape), v.new_empty(shape))
            keys, values = cache[index]
            keys[:, :, position:end] = k
            values[:, :, position:end] = v
            k, v = keys[:, :, :end], values[:, :, :end]
            # During decode the single query can see every cached position.
            # is_causal=True on a non-square SDPA matrix would hide the past.
            if hasattr(F, "scaled_dot_product_attention"):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=position == 0)
            else:
                scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
                if position == 0:
                    mask = torch.ones(length, end, dtype=torch.bool, device=x.device).tril()
                    scores = scores.masked_fill(~mask, float("-inf"))
                y = scores.softmax(dim=-1) @ v
            y = y.transpose(1, 2).contiguous().view(batch, length, width)
            x = x + attn.resid_dropout(attn.c_proj(y))
            x = x + block.mlp(block.ln_2(x))
        return model.lm_head(model.transformer.ln_f(x[:, -1:]))[:, -1]

    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], max_new_tokens: int = 32) -> list[Generation]:
        """Greedy batch generation, stopping each row independently on ``#``.

        Never slide/reset learned position embeddings when context is exhausted.
        Unfinished rows are returned explicitly as context_limit/max_tokens.
        """
        if not prompts or not prompts[0] or len({len(p) for p in prompts}) != 1:
            raise ValueError("provide nonempty prompts of equal length")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        length = len(prompts[0])
        block_size = self.model.config.block_size
        if length > block_size:
            raise ValueError(f"prompt length {length} exceeds context {block_size}")
        steps = min(max_new_tokens, block_size - length + 1)
        device = next(self.model.parameters()).device
        ids = torch.tensor(prompts, dtype=torch.long, device=device)
        cache = [None] * self.model.config.n_layer
        results = [Generation([], "max_tokens" if steps == max_new_tokens else "context_limit")
                   for _ in prompts]
        finished = [False] * len(prompts)
        position = 0
        for _ in range(steps):
            logits = self._forward(ids, cache, position, length + steps - 1)
            position += ids.size(1)
            next_ids = logits.argmax(dim=-1)
            for row, token in enumerate(next_ids.tolist()):
                if finished[row]:
                    continue
                if token == self.stop_id:
                    results[row].stop_reason = "eos"
                    finished[row] = True
                else:
                    results[row].tokens.append(token)
            if all(finished):
                break
            ids = next_ids[:, None]
        return results
