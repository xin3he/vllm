# 量化 KV Cache

## FP8 KV Cache 概览

对于大型语言模型而言，高效使用内存非常重要。将 KV（Key 和 Value）Cache 量化为 FP8 可以显著降低内存占用，从而缓存更多 token，提高吞吐量并支持更长的上下文。

> **注意：** 使用 Flash Attention 3 后端和 FP8 KV Cache 时，Attention 运算也会在量化后的 FP8 域中执行。在此配置下，除了 Key 和 Value，Query 也会被量化为 FP8。

## KV Cache dtype 参考

下表列出了 vLLM Cache 配置当前接受的 dtype 字符串。实际可用性取决于模型、Attention 后端和硬件。标记为**专用**的 dtype 不能假定适用于所有模型。

| `kv_cache_dtype` | Cache 表示形式 | 典型使用场景 | 可用性与说明 |
| --- | --- | --- | --- |
| `auto` | 模型 dtype | 默认配置 | 使用模型自身的 dtype，兼容性最广。 |
| `float16` | FP16 | FP16 部署中的原生精度 Cache | 适用于希望使用 FP16 的通用非量化 Cache。 |
| `bfloat16` | BF16 | BF16 部署中的原生精度 Cache | 通用非量化 Cache。某些默认使用量化 Cache 的模型需要显式指定 BF16。 |
| `fp8` | FP8 E4M3，逐张量 | 通用 FP8 KV Cache 量化 | `fp8_e4m3` 的别名；CUDA 11.8+ 和 ROCm 支持，但仍要求所选 Attention 后端实现该模式。 |
| `fp8_e4m3` | FP8 E4M3，逐张量 | 使用 E4M3 格式的 FP8 KV Cache 量化 | CUDA 11.8+ 和 ROCm 支持；仍要求后端支持。 |
| `fp8_e5m2` | FP8 E5M2，逐张量 | 需要 E5M2 格式时的 FP8 KV Cache 量化 | 支持 CUDA 11.8+；通常不适用于 ROCm。 |
| `fp8_inc` | Gaudi FP8 E4M3 表示形式 | Intel Gaudi/HPU 上的 FP8 KV Cache | HPU 专用表示形式，不是通用的 CUDA/ROCm 选项。 |
| `fp8_per_token_head` | FP8，使用逐 token、逐 head scale | 对精度敏感的部署中的校准 FP8 Cache | 专用路径，当前主要与 Flash Attention 和校准流程配合使用。 |
| `int4_per_token_head` | Packed INT4，使用逐 token、逐 head scale | 高压缩率 KV Cache | 专用的逐 head 量化路径，需要后端实现该模式。 |
| `int8_per_token_head` | INT8，使用逐 token、逐 head scale | 低损耗的逐 head KV Cache 压缩 | 专用的逐 head 量化路径，需要后端实现该模式。 |
| `fp8_ds_mla` | DeepSeek MLA packed、block-scaled FP8 | DeepSeek V3.2/V4 系列的 MLA 压缩 Cache | 专用 MLA layout，仅适用于兼容的 DeepSeek/MLA 实现。 |
| `nvfp4_ds_mla` | DeepSeek MLA packed NVFP4 | DeepSeek V4.1 MLA 压缩 Cache | 专用 MLA layout，需要兼容的 FlashMLA/MLA 实现和硬件支持。 |
| `nvfp4` | Packed NVFP4 | NVFP4 KV Cache 量化 | 需要专用 NVFP4 kernel 和硬件支持。 |
| `nvfp4_4over6` | 使用 4-over-6 scale selection 的 packed NVFP4 | 通过重构误差选择 scale 的 NVFP4 Cache | 专用 NVFP4 layout，不是普通 Attention 后端的通用 fallback。 |
| `turboquant_k8v4` | TurboQuant K8V4 packed format | TurboQuant KV Cache 压缩 | 需要专用 TurboQuant 实现。 |
| `turboquant_4bit_nc` | TurboQuant 4-bit non-contiguous format | 4-bit TurboQuant KV Cache 压缩 | 需要专用 TurboQuant 实现。 |
| `turboquant_k3v4_nc` | TurboQuant K3V4 non-contiguous format | K3V4 TurboQuant KV Cache 压缩 | 需要专用 TurboQuant 实现。 |
| `turboquant_3bit_nc` | TurboQuant 3-bit non-contiguous format | 3-bit TurboQuant KV Cache 压缩 | 需要专用 TurboQuant 实现。 |

`fp8`、`fp8_e4m3` 和 `fp8_e5m2` 是逐张量 FP8 模式。`*_per_token_head` 模式使用不同的 scale 方案，因此需要后端专门支持。`*_ds_mla`、`nvfp4*` 和 `turboquant_*` 表示 packed layout，而不仅仅是 PyTorch 标量 dtype；它们与特定的 Attention 或量化 kernel 绑定。

对于混合模型，可以使用 `--kv-cache-dtype-skip-layers`，让指定层保留模型原生 dtype，同时量化其余层的 KV Cache。

### 支持的 FP8 KV Cache 量化方案

vLLM 支持两种主要的 FP8 KV Cache 量化策略：

- **逐张量量化：**
  每个 Q、K、V tensor 分别使用一个 scale（`q/k/v_scale = [1]`）。
- **逐 Attention head 量化：**
  每个 scale 对应一个 Attention head：`q_scale = [num_heads]`，`k/v_scale = [num_kv_heads]`。

> **注意：**
> 逐 Attention head 量化目前仅适用于 Flash Attention 后端，并且需要 `llm-compressor` 提供校准流程。

### Scale 校准方式

可以使用以下三种方式配置量化 scale 的计算方法：

1. **不校准（默认 scale）：**
   所有量化 scale 都设置为 `1.0`。

   配置方式：
   ```python
   kv_cache_dtype="fp8"
   ```

2. **推荐：使用数据集校准（通过 `llm-compressor`）：**
   使用经过选择的校准数据集估计 scale，以获得更好的精度。这需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor)。

3. **已保存的量化模型：**
   直接加载已经包含量化参数和 scale 的模型。

### 跳过指定层的 KV Cache 量化

某些 Attention 层类型（例如 sliding-window）对 KV Cache 量化更加敏感。`--kv-cache-dtype-skip-layers` 可以让指定层保留模型原生 dtype，同时让其余层使用所选量化 dtype。该参数接受层索引或层类型名称：

```bash
# 跳过所有 sliding-window attention 层。
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

# 跳过指定层索引。
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

Python API 用法：

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

---

## 示例

### 1. 不进行校准（`kv_cache_dtype="fp8"`）

所有量化 scale 都设置为 `1.0`。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. **推荐：使用数据集校准（通过 `llm-compressor`）**

为了获得更高的量化质量，建议使用数据集通过 `llm-compressor` 进行校准。这也支持逐 Attention head 量化等高级策略。

#### 安装依赖

```bash
pip install llmcompressor
```

#### 示例：将 Llama Attention 和 KV Cache 量化为 FP8

以下示例使用 `llm-compressor` 的 one-shot 校准，将 Llama Attention 和 KV Cache 量化为 FP8。`STRATEGY` 可以选择 `tensor` 或 `attn_head`。

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # 或 "attn_head"
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048


def process_and_tokenize(example: dict, tokenizer: AutoTokenizer):
    """将对话消息转换为 token。"""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )


def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,
    )


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
```

更多详细且最新的示例，请参阅 [`llm-compressor` 官方示例](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache)。
