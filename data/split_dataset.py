#!/usr/bin/env python3
"""将 JSONL 数据集确定性地切分为 train/eval 或 train/eval/test。"""

import argparse
import random
from pathlib import Path


def output_path(input_path, split_name):
    suffix = input_path.suffix or ".jsonl"
    stem = input_path.stem if input_path.suffix else input_path.name
    return input_path.with_name(f"{stem}_{split_name}{suffix}")


def split_count(count, percent, total, name, parser):
    if percent is not None:
        if not 0 <= percent <= 100:
            parser.error(f"--{name}-percent 必须在 0 到 100 之间")
        return int(total * percent / 100)
    if not 0 <= count <= total:
        parser.error(f"--{name}-count 必须在 0 到 {total} 之间")
    return count


def main():
    parser = argparse.ArgumentParser(description="切分 JSONL 为 train/eval 或 train/eval/test")
    parser.add_argument("input", type=Path, help="输入 JSONL 文件")
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--eval-percent", type=float, help="验证集百分比，例如 20 表示 20%%")
    eval_group.add_argument("--eval-count", type=int, help="验证集条数")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--test-percent", type=float, help="测试集百分比，例如 10 表示 10%%")
    test_group.add_argument("--test-count", type=int, help="测试集条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"输入文件不存在：{args.input}")
    lines = args.input.read_text(encoding="utf-8").splitlines()
    eval_count = split_count(args.eval_count, args.eval_percent, len(lines), "eval", parser)
    has_test = args.test_percent is not None or args.test_count is not None
    test_count = split_count(args.test_count, args.test_percent, len(lines), "test", parser) if has_test else 0
    if eval_count + test_count > len(lines):
        parser.error("eval 和 test 的总条数不能超过输入数据总数")

    random.Random(args.seed).shuffle(lines)
    eval_lines = lines[:eval_count]
    test_lines = lines[eval_count:eval_count + test_count]
    train_lines = lines[eval_count + test_count:]
    for split_name, split_lines in (("train", train_lines), ("eval", eval_lines)):
        path = output_path(args.input, split_name)
        path.write_text("\n".join(split_lines) + ("\n" if split_lines else ""), encoding="utf-8")
        print(f"{split_name}：{path}（{len(split_lines)} 条）")
    if has_test:
        path = output_path(args.input, "test")
        path.write_text("\n".join(test_lines) + ("\n" if test_lines else ""), encoding="utf-8")
        print(f"test：{path}（{len(test_lines)} 条）")


if __name__ == "__main__":
    main()
