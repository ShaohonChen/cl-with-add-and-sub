# 算术能力评估

依赖 Python 3.11+、PyTorch 2.0+ 和 SwanLab：

```bash
pip install torch swanlab
```

在仓库根目录执行，checkpoint 使用 `scripts/base_train.py` 保存的格式：

```bash
python -m scripts.base_eval \
  --checkpoint out/ckpt_best.pt \
  --eval-data data/add-sub-int_eval.json \
  --run-name arithmetic-eval
```

一次评估多个数据集，可传入多个路径、重复 `--eval-data`，或使用带引号的通配符：

```bash
python -m scripts.base_eval \
  --checkpoint out/ckpt_best.pt \
  --eval-data 'data/*_eval.json' 'data/*_eval.jsonl' \
  --batch-size 64 --max-new-tokens 32 \
  --swanlab-mode offline --output results/eval.json
```

每个通配符必须至少匹配一个文件；重复路径只评估一次。省略 `--eval-data` 时使用
`data/add-sub-int_eval.json`。默认评估所有样本；`--max-examples 100` 按 `--seed`
为每个数据集独立采样。输入遵循逐行 `{"text": "表达式=GT"}` 格式，支持括号、
四则运算和小数，GT 可带末尾 `#`。

模型只接收第一个 `=` 及之前的字符，例如 `(2+3)*4=`，使用贪心解码，预测到
tokenizer 的 `#` 时停止。`--max-new-tokens` 包含结束符的预算。推理引擎在
`model/engine.py`，参考 nanochat 实现 prompt prefill 和每层预分配的 KV cache，
按相同输入长度组成 batch。超出上下文时停止并标记，不滑动位置嵌入。

## 指标

每个数据集以及按样本数加权的 overall 都打印并记录到 SwanLab：

| 指标 | 定义 |
| --- | --- |
| `accuracy` | 数值等于 GT 的样本数 / 总样本数，例如 `1.0` 和 `1` 数值相同 |
| `exact_match` | 完整答案字符串等于 GT 的样本数 / 总样本数 |
| `mae` / `rmse` | 与 GT 的平均绝对误差 / 均方根误差 |
| `mean_signed_error` | 平均 `预测值 - GT` |
| `max_absolute_error` | 最大绝对误差 |
| `mean_relative_error` | 平均 `abs(预测值 - GT) / abs(GT)`，排除 GT 为零的样本 |
| `valid_count` / `numeric_coverage` | 正确结束且格式合法的数值答案数量 / 占总样本比例 |
| `relative_error_count` | 参与相对误差计算的样本数 |
| `numeric_wrong_count` | 正确结束、格式合法但数值错误的答案数 |
| `invalid_format_count` | 正确结束但答案为空或格式非法的样本数 |
| `empty_answer_count` | 直接生成 `#` 的样本数，属于格式错误 |
| `unterminated_count` | 达到生成预算或上下文上限，仍未生成 `#` 的样本数 |
| `context_limit_count` | 因上下文上限停止的样本数，属于未结束输出 |

准确率要求输出以 `#` 正确结束。未结束或格式非法的答案都算错误；误差指标仅在
正确结束且格式合法的数值答案上计算，没有有效答案时显示 `None`，SwanLab 不上传
该空指标。相对误差是比值，不是百分数。

使用 Decimal 比较与计算误差，直接读取 GT，保持数据生成器对每次除法截断的语义。
未改变数据格式、词表、模型结构或上下文长度。JSON 输出包含逐条预测、停止原因、
绝对误差，以及配置、依赖版本、数据文件 SHA256 和 checkpoint step。

SwanLab 默认 `online`，可先执行 `swanlab login`。无网络时用 `--swanlab-mode offline`，
之后通过 `swanlab sync <日志目录>` 上传。`disabled` 可关闭记录。

验证命令：

```bash
python -m scripts.base_eval --help
python -m unittest discover -s tests -v
python -m py_compile data/*.py model/*.py scripts/*.py tests/*.py
```
