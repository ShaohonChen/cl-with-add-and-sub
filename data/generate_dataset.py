#!/usr/bin/env python3
"""生成形如 {\"text\": \"1+2=3\"} 的四则运算 JSONL 训练集。"""

import argparse
import json
import random
import warnings
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, getcontext


getcontext().prec = 50


def sample_number(low, high, places):
    """在 [low, high] 内随机取一个最多/恰好 places 位小数的数。"""
    scale = 10 ** places
    start = int((low * scale).to_integral_value(rounding=ROUND_CEILING))
    end = int((high * scale).to_integral_value(rounding=ROUND_FLOOR))
    if start > end:
        raise ValueError("范围内没有符合指定小数位数的数")
    return Decimal(random.randint(start, end)) / scale


def show(number, places):
    # 固定小数位使数据精度清晰；places 为 0 时不显示小数点。
    return f"{number:.{places}f}"


def main():
    parser = argparse.ArgumentParser(description="生成四则运算 JSONL 数据集")
    parser.add_argument("--count", type=int, default=20, help="生成条数")
    parser.add_argument("--output", default="train.jsonl", help="输出文件路径")
    parser.add_argument("--operators", default="+-*/", help="使用的运算符，例如 '+-*'")
    parser.add_argument("--a-min", type=Decimal, default=Decimal("-100"))
    parser.add_argument("--a-max", type=Decimal, default=Decimal("100"))
    parser.add_argument("--b-min", type=Decimal, default=Decimal("-100"))
    parser.add_argument("--b-max", type=Decimal, default=Decimal("100"))
    parser.add_argument("--a-places", type=int, default=0, help="a 的小数位数")
    parser.add_argument("--b-places", type=int, default=0, help="b 的小数位数")
    parser.add_argument("--c-places", type=int, default=0, help="答案的小数位数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    if (args.count < 1 or args.a_min > args.a_max or args.b_min > args.b_max
            or min(args.a_places, args.b_places, args.c_places) < 0):
        parser.error("数量、范围或小数位数不合法")
    if not args.operators or any(op not in "+-*/" for op in args.operators):
        parser.error("--operators 只能包含 + - * /")
    random.seed(args.seed)

    # 加、减、乘只保留结果可被 c-places 精确表示的题目，避免显示出错误等式。
    # 除法按需求直接截断（朝 0 截断），且最多保留三位小数。
    answer_unit = Decimal(1).scaleb(-args.c_places)
    made = attempts = 0
    seen = set()
    max_attempts = args.count * 10000
    with open(args.output, "w", encoding="utf-8") as file:
        while made < args.count and attempts < max_attempts:
            attempts += 1
            a = sample_number(args.a_min, args.a_max, args.a_places)
            b = sample_number(args.b_min, args.b_max, args.b_places)
            op = random.choice(args.operators)

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            else:
                if b == 0:
                    continue
                places = min(args.c_places, 3)
                result = (a / b).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)
                line = f"{show(a, args.a_places)}/{show(b, args.b_places)}={show(result, places)}"
                if line in seen:
                    continue
                seen.add(line)
                file.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
                made += 1
                continue

            if result != result.quantize(answer_unit):
                continue
            line = f"{show(a, args.a_places)}{op}{show(b, args.b_places)}={show(result, args.c_places)}"
            if line in seen:
                continue
            seen.add(line)
            file.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            made += 1

    if made < args.count:
        warnings.warn(
            f"只生成了 {made}/{args.count} 条唯一数据。当前范围、精度或公式规则下的唯一题目可能不足；"
            "已保留所有已生成的唯一数据。",
            RuntimeWarning,
        )


if __name__ == "__main__":
    main()
