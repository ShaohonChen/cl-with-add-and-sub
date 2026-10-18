# AGENT.md

## 项目定位

这是一个用于验证 GPT-2 结构在四则运算任务上表现的预训练/后训练原型项目。当前仓库只包含：

- 算术数据集生成与切分脚本；
- 固定字符表的字符级 tokenizer；
- 一个 GPT-2 风格的 PyTorch 模型实现。

当前没有训练入口、后训练入口、评估脚本、依赖清单或 checkpoint 管理代码。因此不要把 `runs/prepare_data.sh` 误认为完整训练流程；它只准备数据。

除非任务明确要求，否则不要读取或修改 `reference/`、`.venv/`，也不要把生成的数据集、缓存或模型权重提交到仓库。`data/.gitignore` 已忽略 `.json` 和 `.jsonl` 文件。

## 目录与关键文件

```text
data/
  generate_dataset.py          # 单步 a op b = c 数据
  generate_complex_dataset.py  # 随机组合表达式/括号数据
  split_dataset.py             # 确定性 train/eval/test 切分
model/
  tokenizer.py                 # ArithmeticTokenizer，19 个固定字符
  gpt.py                       # GPTConfig、GPT 及训练/生成辅助方法
runs/
  prepare_data.sh              # 当前默认的加减整数数据准备命令
```

## 数据约定

### 文件格式

生成器每行写入一个 JSON 对象，字段固定为 `text`，例如：

```json
{"text": "12-3=9"}
```

这实际上是 JSONL/NDJSON，而不是一个 JSON 数组。读取时应逐行 `json.loads`，并校验对象含有字符串类型的 `text`。生成器会覆盖同名输出文件，但不会自动创建父目录。

`runs/prepare_data.sh` 使用的文件名是 `data/add-sub-int.json`，虽然扩展名为 `.json`，内容仍是 JSONL；新代码优先使用 `.jsonl` 扩展名，若要兼容现有脚本，不要根据扩展名判断内容格式。

### 简单数据生成

`data/generate_dataset.py` 从两个 Decimal 范围内采样，支持 `+-*/`。加、减、乘只保留能按 `--c-places` 精确表示的结果；除法跳过除数 0，结果按最多三位小数、`ROUND_DOWN`（向 0 截断）处理。所有样本去重，最多尝试 `count * 10000` 次，数量不足时只警告并保留已生成数据。

示例：

```bash
python3 data/generate_dataset.py \
  --count 10000 --a-min 0 --a-max 1000 --b-min 0 --b-max 1000 \
  --operators '+-' --a-places 0 --b-places 0 --c-places 0 \
  --seed 42 --output data/add-sub-int.jsonl
```

### 组合表达式生成

`data/generate_complex_dataset.py` 先构造表达式树，再渲染为带必要/随机括号的表达式。每次除法都会先按 `min(answer_places, 3)` 位小数向 0 截断，之后的运算使用这个截断值；这与“最后一步才统一舍入”的数学计算不同，评估器必须遵循同一规则。除法参与的样本答案也最多显示三位小数。结果必须能按显示精度精确表示，且整行长度不能超过 `--max-length`。

### 数据切分

`data/split_dataset.py` 使用 `random.Random(seed)` 打乱，不改变输入文件；随后依次取 eval、test，剩余为 train。百分比使用 `int(total * percent / 100)` 向下取整。输出名为 `<stem>_train.<suffix>`、`<stem>_eval.<suffix>`，有 test 时再写 `<stem>_test.<suffix>`。切分前后都应记录 seed、样本数和输入文件，避免实验不可复现或数据泄漏。

## Tokenizer 约定

`model/tokenizer.py` 是无训练状态的字符级 tokenizer。字符表及 ID 顺序固定为：

```text
0123456789+-*/().=#
```

因此 `vocab_size == 19`，ID 从左到右为 0 到 18；`#` 是约定的结束标记。数据生成器不会自动追加 `#`，训练/推理数据管道需要明确决定是否在每条例子后追加它，并在训练和评估中保持一致。tokenizer 不接受空格、换行、负号以外的 Unicode 符号或其他字符，未知字符/ID 会抛出 `ValueError`。

不要在不做词表/权重适配的情况下，把该 tokenizer 直接用于 `GPT.from_pretrained()`：预训练导入路径固定使用 GPT-2 的 50257 词表和 1024 上下文长度，而本项目 tokenizer 只有 19 个字符。

## GPT 模型约定

`model/gpt.py` 基于 nanoGPT/minGPT 风格实现：

- `GPTConfig` 默认是 GPT-2 base 配置；算术实验通常应显式设置较小的 `vocab_size=ArithmeticTokenizer().vocab_size`、`block_size`、层数、头数和 hidden size。
- `CausalSelfAttention` 优先调用 PyTorch 2.0+ 的 scaled dot-product/flash attention；不可用时使用显式 causal mask。
- 每个 block 是 pre-LN attention + GELU MLP + 残差；token embedding 与 `lm_head` 权重共享。
- `forward(idx, targets=None)` 在提供 targets 时返回完整序列 logits 和交叉熵（`ignore_index=-1`）；不提供 targets 时只计算最后一个位置的 logits，适合逐 token 生成。
- `crop_block_size()` 只能缩短上下文，不能扩展；调用后应同步确认位置 embedding、attention mask 和数据最大长度。
- `configure_optimizers()` 将二维参数做 weight decay，将 bias/LayerNorm 等一维参数不做 decay；CUDA 且 PyTorch 支持时使用 fused AdamW。
- `from_pretrained()` 仅支持 `gpt2`、`gpt2-medium`、`gpt2-large`、`gpt2-xl`，依赖 `transformers` 和可用的 Hugging Face 权重缓存/网络，并处理了 HF Conv1D 权重转置。它不是算术 tokenizer 的直接初始化入口。
- `generate()` 使用 multinomial 采样；调用方应保证 `temperature > 0`，并在需要确定性指标时设置随机种子或使用贪心解码。

新增训练代码时，应明确记录模型配置、tokenizer 词表版本、数据生成参数、切分 seed、随机种子、设备、精度和 checkpoint。不要依赖模型构造函数打印的参数量作为实验元数据的唯一来源。

## 当前缺失的训练/评估层

如果新增训练或后训练脚本，建议遵循以下边界：

1. 数据加载器逐行解析 JSONL，拒绝损坏行和未知字符，并在进入 batch 前完成 encode、EOS（如使用 `#`）和长度检查。
2. 自回归训练使用右移的输入/targets；不参与 loss 的 padding 位置填 `-1`，以匹配 `GPT.forward` 的 `ignore_index`。
3. 算术能力评估不能只看 token-level loss。至少同时报告完整答案 exact match，并区分格式错误、提前结束、非法字符和数值答案错误。
4. train/eval/test 必须使用互斥文件；不要在 eval/test 上继续更新参数或调参后再声称是独立测试结果。
5. 预训练、继续预训练、监督后训练等阶段应保存独立 checkpoint 和配置，避免覆盖基线结果。

## 验证命令

在不需要第三方依赖时可执行：

```bash
python3 -m py_compile data/*.py model/*.py
python3 data/generate_dataset.py --count 5 --a-min 0 --a-max 9 --b-min 0 --b-max 9 --operators '+-' --output /tmp/arithmetic.jsonl
python3 data/split_dataset.py /tmp/arithmetic.jsonl --eval-count 1 --test-count 1
```

安装 PyTorch 后，再增加 tokenizer round-trip、模型小配置前向（含 targets/不含 targets）、生成和 `crop_block_size` 的 smoke test。若测试涉及 `from_pretrained`，应将网络/权重缓存作为前置条件，不要把下载权重写入仓库。

当前仓库没有测试框架；修改数据规则或模型核心逻辑时，至少补充一个可重复的命令行冒烟测试，并检查 `git diff --check`。

## 编码与协作规范

- 使用 Python 3，标准库脚本保持可直接执行；新增依赖应在提交说明或依赖文件中明确记录。
- 保持数据生成的 Decimal 计算，不要改回 binary float；任何舍入规则变化都必须同步更新数据生成、标签计算和评估器。
- 保持 tokenizer 字符串顺序稳定；若必须改词表，应视为模型兼容性变更，并迁移 checkpoint/实验配置。
- 文件和 CLI 参数命名沿用现有的 snake_case 变量、kebab-case 参数风格；错误参数使用 `argparse` 报错。
- 不要提交 `data/` 下生成的数据、`.venv/`、`reference/`、`__pycache__/`、checkpoint 或日志。
- 提交前查看 `git status`，确认只包含任务相关改动；不要重置或覆盖用户已有的未提交修改。

## 完成检查清单

- [ ] 说明是否改变了数据语义、舍入规则、词表或上下文长度。
- [ ] 为新增 CLI 运行 `--help` 和最小规模 smoke test。
- [ ] 训练/评估实验记录了 seed、数据切分、模型配置和依赖版本。
- [ ] 没有把 `reference/`、`.venv/` 或生成产物纳入改动。
- [ ] `python3 -m py_compile data/*.py model/*.py` 与 `git diff --check` 通过。
