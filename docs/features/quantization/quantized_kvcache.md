# Quantized KV Cache

## FP8 KV Cache Overview

Efficient memory usage is crucial for working with large language models. Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows.

> **Note:** When using the Flash Attention 3 backend with FP8 KV cache, attention operations are also performed in the quantized (FP8) domain. In this configuration, queries are quantized to FP8 in addition to keys and values.

## KV Cache Dtype Reference

The following table lists the dtype strings currently accepted by vLLM's cache
configuration. Availability still depends on the model, attention backend, and
hardware. The entries marked **specialized** should not be assumed to work as a
drop-in dtype for every model.

| `kv_cache_dtype` | Cache representation | Typical use | Availability and notes |
| --- | --- | --- | --- |
| `auto` | Model dtype | Default configuration | Uses the model's dtype. This is the most broadly compatible option. |
| `float16` | FP16 | Native-precision cache on FP16 deployments | General-purpose unquantized cache where FP16 is desired. |
| `bfloat16` | BF16 | Native-precision cache on BF16 deployments | General-purpose unquantized cache. Some models that default to a quantized cache require explicitly selecting BF16. |
| `fp8` | FP8 E4M3, per tensor | General FP8 KV-cache quantization | Alias for `fp8_e4m3`; supported on CUDA 11.8+ and ROCm where the selected attention backend supports it. |
| `fp8_e4m3` | FP8 E4M3, per tensor | FP8 KV-cache quantization with the E4M3 format | CUDA 11.8+ and ROCm support is documented; backend support is still required. |
| `fp8_e5m2` | FP8 E5M2, per tensor | FP8 KV-cache quantization when E5M2 is preferred | CUDA 11.8+; not generally available on ROCm. |
| `fp8_inc` | Gaudi FP8 E4M3 representation | FP8 KV cache on Intel Gaudi/HPU | HPU-specific representation; not a general CUDA/ROCm choice. |
| `fp8_per_token_head` | FP8 with per-token, per-head scales | Calibrated FP8 cache for accuracy-sensitive deployments | Specialized path; currently associated with Flash Attention and calibration support. |
| `int4_per_token_head` | Packed INT4 with per-token, per-head scales | Aggressive KV-cache compression | Specialized per-head quantization path; requires a backend that implements this mode. |
| `int8_per_token_head` | INT8 with per-token, per-head scales | Lower-loss per-head KV-cache compression | Specialized per-head quantization path; requires a backend that implements this mode. |
| `fp8_ds_mla` | DeepSeek MLA packed, block-scaled FP8 | DeepSeek V3.2/V4-family MLA compressed cache | Specialized MLA layout; use only with a compatible DeepSeek/MLA implementation. |
| `nvfp4_ds_mla` | DeepSeek MLA packed NVFP4 | DeepSeek V4.1 MLA compressed cache | Specialized MLA layout; requires the compatible FlashMLA/MLA implementation and hardware support. |
| `nvfp4` | Packed NVFP4 | NVFP4 KV-cache quantization | Specialized NVFP4 kernels and hardware support are required. |
| `nvfp4_4over6` | Packed NVFP4 with 4-over-6 scale selection | NVFP4 cache with reconstruction-error-aware scale selection | Specialized NVFP4 layout; not a universal fallback for ordinary attention backends. |
| `mxfp4_qdq` | Model dtype with simulated MXFP4 values | FP4 accuracy testing without packed storage | No memory savings or FP4 attention hardware needed. |
| `nvfp4_qdq` | Model dtype with simulated NVFP4 values | FP4 accuracy testing without packed storage | No memory savings or FP4 attention hardware needed. |
| `nvfp4_4over6_qdq` | Model dtype with simulated NVFP4 4-over-6 values | FP4 accuracy testing with scale search | No memory savings or FP4 attention hardware needed. |
| `turboquant_k8v4` | TurboQuant K8V4 packed format | TurboQuant KV-cache compression | Specialized TurboQuant implementation. |
| `turboquant_4bit_nc` | TurboQuant 4-bit non-contiguous format | 4-bit TurboQuant KV-cache compression | Specialized TurboQuant implementation. |
| `turboquant_k3v4_nc` | TurboQuant K3V4 non-contiguous format | K3V4 TurboQuant KV-cache compression | Specialized TurboQuant implementation. |
| `turboquant_3bit_nc` | TurboQuant 3-bit non-contiguous format | 3-bit TurboQuant KV-cache compression | Specialized TurboQuant implementation. |

`fp8`, `fp8_e4m3`, and `fp8_e5m2` are per-tensor FP8 modes. The
`*_per_token_head` modes use a different scaling scheme and therefore require
backend-specific support. The `*_ds_mla`, packed `nvfp4` modes, and `turboquant_*` values
describe packed layouts rather than only a PyTorch scalar dtype; they are tied
to particular attention or quantization kernels.

For hybrid models, `--kv-cache-dtype-skip-layers` can leave selected layers at
the model's native dtype while quantizing the remaining KV cache layers.

### FP4 QDQ Simulation

Use `--kv-cache-dtype mxfp4_qdq`, `--kv-cache-dtype nvfp4_qdq`, or
`--kv-cache-dtype nvfp4_4over6_qdq` to simulate FP4 rounding of keys and values
before storing them in a floating-point KV cache. MXFP4 uses 32-value E8M0
(power-of-two) blocks; NVFP4 uses 16-value FP8 E4M3 scales. Both round values
to the E2M1 codebook. The `nvfp4_4over6` mode evaluates scales based on
`max/6` and `max/4` for each block and selects the one with lower squared
reconstruction error (choosing `max/6` on a tie). For example:

```bash
vllm serve <model> --kv-cache-dtype mxfp4_qdq
```

This is an accuracy experiment, not packed FP4 storage: it does **not** save
cache memory or require FP4 attention hardware. Only standard attention layers
are simulated; MLA and model-specific caches are not affected. Layers selected with
`--kv-cache-dtype-skip-layers` also skip QDQ.

### Supported FP8 KV-Cache Quantization Schemes

vLLM supports two main quantization strategies for the FP8 KV-cache:

- **Per-tensor quantization:**  
  A single scale is applied for each Q, K, and V tensor individually. (`q/k/v_scale = [1]`)
- **Per-attention-head quantization:**  
  Each scale corresponds to an attention head: `q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`.

> **Note:**  
> Per-attention-head quantization is currently available **only with the Flash Attention backend** and requires the calibration pathway provided by **llm-compressor**.

### Scale Calibration Approaches

You can configure how the quantization scales are computed in vLLM using three different approaches:

1. **No calibration (default scales):**  
   All quantization scales are set to `1.0`.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   ```

2. **[Recommended] Calibration with a dataset (via llm-compressor):**  
   Scales are estimated using a curated calibration dataset for maximum accuracy.  
   This requires the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.  
   _See example below!_

#### Additional `kv_cache_dtype` Options

- `kv_cache_dtype="auto"`: Use the model's default data type
- `kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)
- `kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+

### Skipping Specific Layers from KV-Cache Quantization

Some attention layer types (e.g. sliding-window) are more sensitive to KV-cache quantization. The `--kv-cache-dtype-skip-layers` flag leaves the specified layers at the model's native dtype while keeping the rest of the layers under the chosen quantized dtype. The flag accepts either layer indices or layer-type names:

```bash
# Skip every sliding-window attention layer.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

# Skip specific layer indices.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

Programmatic usage:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

---

## Examples

### 1. No Calibration (`kv_cache_dtype="fp8"`)

All quantization scales are set to 1.0.

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

### 2. **[Recommended] Calibration Using a Dataset (with `llm-compressor`)**

For the highest-quality quantization, we recommend calibrating against a dataset using `llm-compressor`. This enables advanced strategies such as per-attention-head quantization.

#### Install the required package

```bash
pip install llmcompressor
```

#### Example: Quantize Llama Attention & KV Cache to FP8

```python
"""
Quantize Llama attention + KV cache to FP8 (choose either 'tensor' or 'attn_head' strategy)
using llm-compressor one-shot calibration.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # or "attn_head"
NUM_CALIB_SAMPLES = 512   # Good starting value
MAX_SEQ_LEN = 2048

# -----------------------------
# Helpers
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """Convert chat messages to tokens."""
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
                targets=["LlamaAttention"],  # Quantize queries: q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # Quantize KV cache: k/v_scale
    )

# -----------------------------
# Main
# -----------------------------
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

For more detailed and up-to-date examples, see the [`llm-compressor` official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).
