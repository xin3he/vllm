# MRCR Long-Context Accuracy Evaluation

Smoke test for long-context behavior using OpenAI's public [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) dataset. The model sees a long chat with several near-duplicate "needles" and must reproduce a specific earlier assistant turn verbatim, prepended with a random anti-guessing string.

**Scoring:** if the response doesn't start with `random_string_to_prepend`, score is 0; otherwise the prefix is stripped and the mean `SequenceMatcher.ratio()` against the reference answer is reported.

## What MRCR Tests

MRCR (Multi-round Conversation Retrieval) evaluates whether a model can retrieve
the correct earlier assistant turn from a long multi-turn conversation. Each
sample contains several near-duplicate historical answers, asks for one specific
answer, and prepends the target with a random anti-guessing string. The task
therefore tests long-context retrieval, discrimination between similar candidates,
conversation-history retention, and generation after a long prompt.

The dataset is split into three buckets:

- `2 needles`: a quick check of basic long-context retrieval;
- `4 needles`: a harder test with more similar candidates;
- `8 needles`: the most difficult bucket and usually the most sensitive to retrieval errors.

## Why It Is Useful for KV Cache Evaluation

During decoding, every generated token attends to the keys and values stored for
the prompt and previous generated tokens. Changing the KV cache dtype can alter
the numerical precision of those states and make similar needles harder to
distinguish. This effect is often small on short prompts but is easier to expose
with MRCR's long prompts and near-duplicate answers.

When comparing KV cache dtypes, keep the model, sample set, needle buckets,
context limits, sampling settings, seed, and concurrency fixed. Change only
`--kv-cache-dtype`.

## Interpreting Results

In addition to the aggregate `match_ratio`, inspect `match_ratio_n2`,
`match_ratio_n4`, and `match_ratio_n8`. A regression limited to `match_ratio_n8`
can indicate that quantization affects discrimination among more similar
candidates even when the aggregate score looks stable. A lower `prefix_hit_rate`
usually means that the model retrieved the wrong target or violated the expected
answer format.

The batch comparison also reports `tokens_per_second`. Similar accuracy with
higher throughput indicates a useful trade-off; unchanged accuracy across all
settings may simply mean that the prompt length or sample count is not sufficient
to expose the difference.

## Usage

```bash
# Pytest (spawns the server)
pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file=configs/models-small.txt

# Standalone (server already running; model and context auto-discovered)
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --port 8000
python tests/evals/mrcr/mrcr_eval.py --port 8000
```

### Compare KV Cache Dtypes

Use the batch script to start a fresh server for each dtype, run the same MRCR
sample set, and write one JSON result plus one server log per dtype:

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

When multiple visible GPUs are available, pass a comma-separated device list.
The script runs up to one dtype evaluation per device in parallel, then starts
the next batch when the current batch finishes. Thus four dtypes with
`--devices "0,1,2"` run as batches `0,1,2` and then `0`. `CUDA_VISIBLE_DEVICES`
is used when `--devices` is omitted. GPU 7 is reserved and rejected by the
script. Each parallel run uses its own port.
If a requested port is already in use, the script automatically probes the next
available port and assigns distinct ports to parallel runs.

Every dtype run receives the same `--seed` value (default: `42`), so it uses
the same deterministic streaming-shuffle order and generation seed, provided
the dataset revision, model, tokenizer, and context limits are unchanged. Use
`--seed N` to select and record another shared seed.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
  --model Qwen/Qwen3-0.6B \
  --dtypes "bfloat16 fp8 int8_per_token_head" \
  --devices "0,1,2"
```

The script writes `summary.tsv`, per-dtype JSON files, and server logs under
`--output-dir` (default: `kv-cache-results/mrcr-<timestamp>/`). Set
`VLLM_PYTHON` when the virtual environment is not at `.venv/bin/python`. The
default comparison is `bfloat16` versus `fp8`; for a
more aggressive comparison, for example:

```bash
./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
  --model <model> \
  --dtypes "bfloat16 fp8 int8_per_token_head int4_per_token_head" \
  --num-samples 10 \
  --max-prompt-tokens 32768
```

## Configuration

```yaml
model_name: "Qwen/Qwen3-0.6B"
# Per-needle thresholds catch bucket-specific regressions (sliding window,
# chunked prefill, prefix cache) that an aggregate can hide. A scalar
# (e.g. `match_ratio_threshold: 0.20`) is also accepted and checked against
# the mean match ratio.
match_ratio_threshold:
  2: 0.30
  4: 0.15
  8: 0.10
num_samples: 30
needles: [2, 4, 8]
# max_prompt_tokens: 32768       # Optional; defaults to server max_model_len - max_tokens - 256
max_tokens: 2048
concurrency: 8
server_args: "--max-model-len 32768 --reasoning-parser qwen3"
```

## Notes

- Samples stream from three parquet shards (`{N}needle/{N}needle_0.parquet`); only the first few row groups are fetched, not the full 1.4 GB repo.
- `max_prompt_tokens` defaults to `max_model_len - max_tokens - 256`, i.e. fills whatever context the server advertises. Set `--max-model-len` on the server to control the smoke-test context length; override `--max-prompt-tokens` on the client to cap below that.
- Sample length is pre-filtered by `n_chars × 4 ≤ max_prompt_tokens`, then verified via the server's `/tokenize` endpoint under the actual chat template.
- Reasoning models: start the server with `--reasoning-parser <name>` (e.g. `qwen3`, `deepseek_r1`) so `<think>` goes to `message.reasoning_content` and doesn't contaminate the scored answer.
