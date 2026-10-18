"""Run with: python -m unittest discover -s tests -v (requires torch)."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.engine import Engine
from model.gpt import GPT, GPTConfig
from model.tokenizer import ArithmeticTokenizer
from scripts.base_eval import Metrics, evaluate, load_examples, load_model


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = ArithmeticTokenizer()
        self.model = GPT(GPTConfig(vocab_size=self.tokenizer.vocab_size, block_size=24,
                                   n_layer=2, n_head=2, n_embd=16, dropout=0.2)).eval()
        self.engine = Engine(self.model, self.tokenizer)

    @torch.inference_mode()
    def test_cached_logits_match_full_forward(self):
        # Test bias, multiple layers, batching, prefill, and several decode steps.
        ids = torch.tensor([self.tokenizer.encode("12+3="), self.tokenizer.encode("45-6=")])
        cache = [None] * self.model.config.n_layer
        actual = self.engine._forward(ids, cache, 0, 24)
        for _ in range(5):
            expected, _ = self.model(ids)
            torch.testing.assert_close(actual, expected[:, -1], atol=1e-6, rtol=1e-5)
            token = expected[:, -1].argmax(-1)[:, None]
            position = ids.size(1)
            ids = torch.cat((ids, token), dim=1)
            actual = self.engine._forward(token, cache, position, 24)

    @torch.inference_mode()
    def test_attention_fallback(self):
        ids = torch.tensor([self.tokenizer.encode("1+2=")])
        expected = self.engine._forward(ids, [None] * 2, 0, 24)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        try:
            del torch.nn.functional.scaled_dot_product_attention
            cache = [None] * 2
            actual = self.engine._forward(ids, cache, 0, 24)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            token = actual.argmax(-1)[:, None]
            actual = self.engine._forward(token, cache, ids.size(1), 24)
        finally:
            torch.nn.functional.scaled_dot_product_attention = sdpa
        full, _ = self.model(torch.cat((ids, token), dim=1))
        torch.testing.assert_close(actual, full[:, -1], atol=1e-6, rtol=1e-5)

    def test_independent_stop_and_context_limit(self):
        stop = self.tokenizer.encode("#")[0]
        choices = [[stop, 3], [7, stop]]

        def forward(ids, cache, position, capacity):
            logits = torch.full((2, self.tokenizer.vocab_size), -100.0)
            for row, token in enumerate(choices.pop(0)):
                logits[row, token] = 100
            return logits

        with patch.object(self.engine, "_forward", side_effect=forward):
            result = self.engine.generate([[1], [2]], 4)
        self.assertEqual([(r.tokens, r.stop_reason) for r in result], [([], "eos"), ([3], "eos")])
        logits = torch.zeros(1, self.tokenizer.vocab_size)
        with patch.object(self.engine, "_forward", return_value=logits):
            result = self.engine.generate([[1] * 24], 5)[0]
            self.assertEqual((result.tokens, result.stop_reason), ([0], "context_limit"))
            self.assertEqual(self.engine.generate([[1]], 1)[0].stop_reason, "max_tokens")
        with self.assertRaises(ValueError):
            self.engine.generate([[1] * 25])

    def test_metrics_and_decimal_precision(self):
        metrics = Metrics()
        for prediction, target, reason in [("1.0", "1", "eos"), ("-3", "-1", "eos"),
                                            ("0", "0", "eos"), ("", "1", "eos"),
                                            ("1+2", "3", "eos"), ("1", "1", "max_tokens")]:
            metrics.add(prediction, target, reason)
        result = metrics.summary()
        self.assertEqual(result["accuracy"], 2 / 6)
        self.assertEqual(result["exact_match"], 1 / 6)
        self.assertEqual(result["mae"], 2 / 3)
        self.assertEqual(result["relative_error_count"], 2)
        self.assertEqual(result["mean_relative_error"], 1)
        self.assertEqual(result["invalid_format_count"], 2)
        self.assertEqual(result["unterminated_count"], 1)
        score = metrics.add("9007199254740993", "9007199254740992", "eos")
        self.assertFalse(score["correct"])
        self.assertEqual(score["absolute_error"], "1")

    def test_loading_checkpoint_and_nested_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text('\n'.join(json.dumps({"text": text}) for text in
                                      ["1+2=3", "(2+3)*4=20#", "1/3=0.333"]), encoding="utf-8")
            examples = load_examples(path, self.tokenizer, 24)
            self.assertEqual(examples[1].prompt, "(2+3)*4=")
            self.assertEqual(examples[2].answer, "0.333")
            self.assertTrue(all("#" not in ex.prompt for ex in examples))
            self.assertEqual(load_examples(path, self.tokenizer, 24, 2, 42),
                             load_examples(path, self.tokenizer, 24, 2, 42))
            ckpt = Path(tmp) / "model.pt"
            torch.save({"model": {"module._orig_mod." + k: v for k, v in self.model.state_dict().items()},
                        "config": {"model_config": asdict(self.model.config), "vocab": self.tokenizer.tokens},
                        "iter_num": 7}, ckpt)
            loaded, step = load_model(ckpt, "cpu", self.tokenizer)
            self.assertEqual(step, 7)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            metrics, records = evaluate(self.engine, self.tokenizer, examples, 2, 3, Metrics())
            self.assertEqual(metrics["count"], 3)
            self.assertEqual([row["line"] for row in records], [1, 2, 3])
            path.write_text('{"text":"1+2=nan"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "eval.json:1"):
                load_examples(path, self.tokenizer, 24)


if __name__ == "__main__":
    unittest.main()
