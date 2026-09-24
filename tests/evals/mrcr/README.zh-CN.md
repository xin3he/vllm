# MRCR 长上下文准确率评
c
MRCR（Multi-round Conversation Retrieval）是一个用于测试模型长上下文检索能力的评测。它来自 OpenAI 公开的 [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) 数据集，特别适合检查模型能否在很长的多轮对话中定位并复现较早出现的信息。

## MRCR 测试什么

MRCR 不只是测试模型能否在一大段文本中找到关键词，而是构造了更容易暴露长上下文问题的对话：

1. 对话中包含多个内容相近的历史 assistant 回复，也就是多个近似的 needle。
2. 当前用户问题要求模型复现其中一个特定的历史回复。
3. 目标回复前会加入随机字符串 `random_string_to_prepend`，降低模型通过固定模式或猜测获得高分的可能性。
4. 模型必须从长对话中找出正确的目标轮次，并生成对应内容。

因此，MRCR 同时考察以下能力：

- 长上下文中的信息检索；
- 多个相似候选之间的区分能力；
- 多轮对话历史的保留能力；
- 对上下文中较早 token 的持续注意能力；
- 在较长 prompt 下生成正确答案的能力。

## 为什么适合测试 KV Cache

生成阶段的每个新 token 都需要访问此前 prompt 和已生成内容对应的 Key/Value。KV Cache dtype 改变后，主要可能影响：

- 历史 Key/Value 的数值精度；
- 相似 needle 之间的注意力分数差异；
- 长上下文中远距离 token 的可检索性；
- 不同位置 token 的量化误差累积；
- 最终答案的前缀命中率和文本相似度。

短文本任务通常无法明显体现这些差异。MRCR 的 prompt 较长，并且包含多个近似答案，因此 `bfloat16`、`fp8`、`int8_per_token_head` 和 `int4_per_token_head` 之间的准确率差异更容易暴露。

建议比较 dtype 时保持以下条件一致：

- 相同模型和 tokenizer；
- 相同 MRCR 样本；
- 相同 `needles` 配置；
- 相同 `max_prompt_tokens` 和 `max_tokens`；
- 相同采样参数、随机种子和并发度；
- 仅切换 `--kv-cache-dtype`。

## 数据组织方式

数据按 needle 数量划分为三个 shard：

- `2needle/2needle_0.parquet`
- `4needle/4needle_0.parquet`
- `8needle/8needle_0.parquet`

其中：

- `2 needles`：候选较少，适合快速检查基本长上下文检索；
- `4 needles`：候选增多，可以更好地观察量化误差；
- `8 needles`：候选最多，通常最容易暴露相似信息之间的混淆。

评测默认使用 `[2, 4, 8]` 三个分桶。脚本只流式读取需要的 parquet row group，不会下载完整的约 1.4 GB 数据集。

## 评分方式

MRCR 使用两个主要指标：

### `prefix_hit_rate`

模型回复必须以样本中的 `random_string_to_prepend` 开头。否则该样本视为未命中，得分为 0。

该指标反映模型是否找到了正确的目标回复以及是否遵守了输出格式。

### `match_ratio`

当回复命中随机前缀后，评测器会去除此前缀，并使用 Python 的 `SequenceMatcher.ratio()` 计算模型输出与参考答案之间的相似度。

总体指标为所有样本的平均值，同时会分别报告：

- `match_ratio_n2`
- `match_ratio_n4`
- `match_ratio_n8`

分桶指标很重要，因为总体平均值可能掩盖某个 needle 数量下的回归。例如某个 dtype 在 2-needle 样本上正常，但在 8-needle 样本上明显退化。

## 快速运行

### 使用 Pytest

Pytest 会自动启动 server，并按照配置文件执行评测：

```bash
pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file=configs/models-small.txt
```

### 独立运行

先启动 vLLM server：

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --port 8000
```

然后运行客户端评测：

```bash
python tests/evals/mrcr/mrcr_eval.py --port 8000
```

评测器会自动从 `/v1/models` 发现模型，并从 server 配置中读取最大上下文长度。

## 批量比较 KV Cache dtype

批量脚本会针对每个 dtype 启动一个独立 server，运行同一组 MRCR 样本，然后关闭 server，再继续测试下一个 dtype：

```bash
./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
    --model Qwen/Qwen3-0.6B \
    --dtypes "bfloat16 fp8" \
    --output-dir kv-cache-results/mrcr-qwen3-kv \
    --server-arg "--max-model-len" \
    --server-arg "32768" \
    --server-arg "--reasoning-parser" \
    --server-arg qwen3
```

如果可见 GPU 数量较多，可以传入逗号分隔的设备列表。脚本会为每张设备同时启动一个 dtype 评测，当前批次结束后再启动下一批。例如 `--devices "0,1,2"` 测试四个 dtype 时，会先并行运行设备 `0,1,2`，再运行下一批的设备 `0`。如果不传 `--devices`，脚本会读取 `CUDA_VISIBLE_DEVICES`。每个并行评测使用独立端口。GPU 7 为保留设备，脚本会拒绝使用它。

如果指定端口已被占用，脚本会自动从下一个端口继续探测，并确保同一批并行评测使用互不冲突的端口。

每个 dtype 都会使用相同的 `--seed`（默认 `42`）。在数据集版本、模型、tokenizer 和上下文长度限制不变时，这会确保 streaming shuffle 的 sample 顺序和生成随机种子一致。可以通过 `--seed N` 选择并记录另一组共享 seed。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
    --model Qwen/Qwen3-0.6B \
    --dtypes "bfloat16 fp8 int8_per_token_head" \
    --devices "0,1,2"
```

脚本会生成：

```text
results/mrcr-qwen3-kv/
├── bfloat16.json
├── bfloat16.server.log
├── fp8.json
├── fp8.server.log
├── run_config.txt
└── summary.tsv
```

更激进的比较示例：

```bash
./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
    --model <model> \
    --dtypes "bfloat16 fp8 int8_per_token_head int4_per_token_head" \
    --num-samples 40 \
    --max-prompt-tokens 32768
```

脚本默认使用 `.venv/bin/python`。如果 Python 虚拟环境位于其他位置，可以设置：

```bash
VLLM_PYTHON=/path/to/python \
./tests/evals/mrcr/compare_kv_cache_dtypes.sh --model <model>
```

## 重要参数

| 参数                    | 含义                                          | 默认值                   |
| ----------------------- | --------------------------------------------- | ------------------------ |
| `--model`             | Hugging Face 模型名或本地模型路径             | 必填                     |
| `--dtypes`            | 要依次比较的 KV Cache dtype，空格分隔         | `bfloat16 fp8`         |
| `--devices`           | 轮转分配给各 dtype 的 GPU，逗号分隔           | `CUDA_VISIBLE_DEVICES` |
| `--output-dir`        | 结果、日志和汇总文件目录                      | `kv-cache-results/mrcr-<时间戳>` |
| `--port`              | 第一个 server 使用的端口；后续 dtype 递增端口 | `8000`                 |
| `--num-samples`       | 每个 dtype 使用的样本数                       | `40`                   |
| `--needles`           | 测试的 needle 分桶                            | `2 4 8`                |
| `--max-prompt-tokens` | prompt 最大 token 数                          | 根据 server 自动计算     |
| `--max-tokens`        | 最大输出 token 数                             | `2048`                 |
| `--concurrency`       | MRCR 请求并发数                               | `8`                    |
| `--seed`              | 所有 dtype 共享的 sample 和生成随机种子       | `42`                   |
| `--server-arg`        | 传给`vllm serve` 的额外参数，可重复         | 无                       |

要遍历所选 needle 桶的全部数据，传入 `--num-samples -1`。数据集目前有 2、4、8 needle 各 800 条，共 2400 条（每桶两个 parquet 文件）。实际评测条数仍受模型的 prompt token 上限筛选；运行日志中的 `Loaded N samples` 是最终条数。默认的正数采样模式保持原有取样范围。

```bash
CUDA_VISIBLE_DEVICES=0 ./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
    --model Qwen/Qwen3-0.6B \
    --devices "0" \
    --num-samples -1
```

## 配置示例

```yaml
model_name: "Qwen/Qwen3-0.6B"
# 分桶阈值可以捕获总体平均值隐藏的回归。
match_ratio_threshold:
  2: 0.30
  4: 0.15
  8: 0.10
num_samples: 30
needles: [2, 4, 8]
# max_prompt_tokens: 32768
max_tokens: 2048
concurrency: 8
server_args: "--max-model-len 32768 --reasoning-parser qwen3"
```

## 上下文长度和样本过滤

如果没有显式指定 `max_prompt_tokens`，评测器使用：

```text
max_prompt_tokens = max_model_len - max_tokens - 256
```

其中 `256` 是为输出和请求安全预留的 buffer。可以通过以下方式控制上下文长度：

- 使用 server 参数 `--max-model-len` 设置上限；
- 使用客户端参数 `--max-prompt-tokens` 进一步限制实际 prompt 长度。

样本首先使用 `n_chars × 4 ≤ max_prompt_tokens` 做粗略过滤，然后通过 server 的 `/tokenize` 接口和实际 chat template 重新确认 token 数量。这样可以避免字符数估算与真实 tokenizer 结果不一致。

## 推理模型注意事项

对于 Qwen3、DeepSeek-R1 等推理模型，建议启动 server 时指定 reasoning parser：

```bash
vllm serve <model> \
    --reasoning-parser qwen3 \
    --port 8000
```

否则 `<think>` 内容可能混入普通 assistant 回复，导致 MRCR 的答案评分受到影响。正确配置 parser 后，推理内容会进入 `message.reasoning_content`，最终答案保留在 `message.content` 中。

## 如何解读 KV Cache 影响

推荐重点查看 `summary.tsv` 中的以下列：

| 现象                        | 可能含义                                       |
| --------------------------- | ---------------------------------------------- |
| `match_ratio` 明显下降    | KV Cache 量化误差影响了答案内容或长距离检索    |
| `prefix_hit_rate` 下降    | 模型更频繁地找错目标，或输出格式受到影响       |
| 只有`match_ratio_n8` 下降 | 更多相似 needle 导致量化后的注意力区分能力不足 |
| 只有长 prompt 下下降        | 误差可能与上下文长度或远距离 token 访问有关    |
| 准确率接近但 tokens/s 提升  | dtype 主要带来性能或显存收益，精度损失较小     |
| 所有指标都不变              | 当前模型、上下文长度或样本量可能不足以暴露差异 |

为了提高结论可信度，建议先用较小样本数快速筛选，再对候选 dtype 使用更大的 `--num-samples` 和多个随机种子复测。
