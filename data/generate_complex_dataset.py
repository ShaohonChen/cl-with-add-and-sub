#!/usr/bin/env python3
"""生成带括号的组合四则运算 JSONL 数据集。"""

import argparse
import json
import random
import warnings
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, getcontext


getcontext().prec = 50


def sample_number(low, high, places):
    scale = 10 ** places
    start = int((low * scale).to_integral_value(rounding=ROUND_CEILING))
    end = int((high * scale).to_integral_value(rounding=ROUND_FLOOR))
    if start > end:
        raise ValueError("范围内没有符合指定小数位数的数")
    return Decimal(random.randint(start, end)) / scale


def show(number, places):
    return f"{number:.{places}f}"


def calculate(left, operator, right, division_places):
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        return None
    # 每一步除法都向 0 截断，且不超过三位小数。
    unit = Decimal(1).scaleb(-division_places)
    return (left / right).quantize(unit, rounding=ROUND_DOWN)


def precedence(operator):
    return 1 if operator in "+-" else 2


def render(node, paren_probability, is_right_child=False, parent_operator=None):
    """渲染树，并只在必要时强制添加括号以保持计算结果不变。"""
    if node[0] == "number":
        return node[1]

    _, operator, left, right, _ = node
    left_text = render(left, paren_probability, False, operator)
    right_text = render(right, paren_probability, True, operator)

    for child, text, is_right in ((left, left_text, False), (right, right_text, True)):
        if child[0] == "number":
            continue
        child_precedence = precedence(child[1])
        must_wrap = child_precedence < precedence(operator)
        # 同优先级的右子树若不加括号，计算顺序可能改变。
        must_wrap = must_wrap or (is_right and child_precedence == precedence(operator))
        optional_wrap = not must_wrap and random.random() < paren_probability
        if must_wrap or optional_wrap:
            if is_right:
                right_text = f"({text})"
            else:
                left_text = f"({text})"
    return f"{left_text}{operator}{right_text}"


def has_division(node):
    return node[0] == "operator" and (
        node[1] == "/" or has_division(node[2]) or has_division(node[3])
    )


def make_expression(args, operations):
    """随机生成一棵表达式树，返回 (文本, Decimal 结果)。"""
    count = random.randint(2, args.max_numbers)
    first = sample_number(args.number_min, args.number_max, args.number_places)
    nodes = [("number", show(first, args.number_places), first)]
    for _ in range(count - 1):
        value = sample_number(args.number_min, args.number_max, args.number_places)
        nodes.append(("number", show(value, args.number_places), value))

    division_places = min(args.answer_places, 3)
    while len(nodes) > 1:
        index = random.randrange(len(nodes) - 1)
        left = nodes.pop(index)
        right = nodes.pop(index)
        operator = random.choice(operations)
        result = calculate(left[-1], operator, right[-1], division_places)
        if result is None:
            return None
        nodes.insert(index, ("operator", operator, left, right, result))
    root = nodes[0]
    return render(root, args.paren_probability), root[-1], has_division(root)


def main():
    parser = argparse.ArgumentParser(description="生成带括号的组合四则运算 JSONL 数据集")
    parser.add_argument("--count", type=int, required=True, help="生成条数")
    parser.add_argument("--output", default="complex_train.jsonl", help="输出文件路径")
    parser.add_argument("--operators", default="+-*/", help="使用的运算符，例如 '+-*'")
    parser.add_argument("--number-min", type=Decimal, default=Decimal("-9"))
    parser.add_argument("--number-max", type=Decimal, default=Decimal("9"))
    parser.add_argument("--number-places", type=int, default=0, help="每个操作数的小数位数")
    parser.add_argument("--answer-places", type=int, default=0, help="答案的小数位数")
    parser.add_argument("--max-numbers", type=int, default=4, help="单个公式最多操作数个数")
    parser.add_argument("--max-length", type=int, default=20, help="公式加答案的最大字符数")
    parser.add_argument("--paren-probability", type=float, default=0.3, help="每个子表达式加括号的概率，默认 0.3")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()

    if (args.count < 1 or args.number_min > args.number_max or args.number_places < 0
            or args.answer_places < 0 or args.max_numbers < 2 or args.max_length < 3
            or not 0 <= args.paren_probability <= 1):
        parser.error("参数范围不合法")
    if not args.operators or any(op not in "+-*/" for op in args.operators):
        parser.error("--operators 只能包含 + - * /")
    random.seed(args.seed)

    attempts = made = 0
    seen = set()
    with open(args.output, "w", encoding="utf-8") as file:
        while made < args.count and attempts < args.count * 10000:
            attempts += 1
            expression = make_expression(args, args.operators)
            if expression is None:
                continue
            text, result, includes_division = expression
            display_places = min(args.answer_places, 3) if includes_division else args.answer_places
            answer_unit = Decimal(1).scaleb(-display_places)
            # 最终结果必须可精确按显示精度表示，避免生成错误等式。
            if result != result.quantize(answer_unit):
                continue
            line = f"{text}={show(result, display_places)}"
            if len(line) > args.max_length:
                continue
            if line in seen:
                continue
            seen.add(line)
            file.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            made += 1

    if made < args.count:
        warnings.warn(
            f"只生成了 {made}/{args.count} 条唯一数据。当前范围、长度或公式规则下的唯一题目可能不足；"
            "已保留所有已生成的唯一数据。",
            RuntimeWarning,
        )


if __name__ == "__main__":
    main()
