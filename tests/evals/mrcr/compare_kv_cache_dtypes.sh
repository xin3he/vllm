#!/usr/bin/env bash
set -euo pipefail

source /data/xinhe/.venv/bin/activate
export no_proxy='localhost,127.0.0.1,.example.com'
export NO_PROXY="$no_proxy"
export http_proxy="http://127.0.0.1:2080"
export https_proxy="http://127.0.0.1:2080"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export HF_HOME=/data/xinhe/.cache/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  compare_kv_cache_dtypes.sh --model MODEL [options]

Required:
  --model MODEL              Hugging Face model or local model path

Options:
  --dtypes "DTYPE ..."       KV cache dtypes (default: "bfloat16 fp8")
    --devices "ID,..."         Devices to rotate across (default: CUDA_VISIBLE_DEVICES)
    --output-dir DIR            Result directory (default: kv-cache-results/mrcr-<timestamp>)
  --port PORT                 First server port (default: 8000)
  --num-samples N             MRCR samples per dtype (default: 40)
  --needles "N ..."           Needle buckets (default: "2 4 8")
  --max-prompt-tokens N       Prompt token limit (default: server-derived)
  --max-tokens N              Maximum generated tokens (default: 2048)
  --concurrency N             MRCR request concurrency (default: 8)
    --seed N                    Sampling and generation seed (default: 42)
  --server-arg ARG            Extra vLLM server argument; repeatable
  --help                      Show this help

Examples:
  ./tests/evals/mrcr/compare_kv_cache_dtypes.sh \
    --model Qwen/Qwen3-0.6B \
    --dtypes "bfloat16 fp8" \
    --devices "0,1" \
    --server-arg "--max-model-len" \
    --server-arg "32768" \
    --server-arg "--reasoning-parser" \
    --server-arg qwen3
EOF
}

MODEL=""
DTYPES=(bfloat16 fp8)
OUTPUT_DIR="kv-cache-results/mrcr-$(date +%Y%m%d-%H%M%S)"
PORT=9000
NUM_SAMPLES=40
NEEDLES=(2 4 8)
MAX_PROMPT_TOKENS=""
MAX_TOKENS=2048
CONCURRENCY=8
SEED=42
SERVER_ARGS=()
CHILD_PIDS=()
PYTHON_BIN="${VLLM_PYTHON:-$(command -v python || true)}"
DEVICES="${CUDA_VISIBLE_DEVICES:-}"

require_value() {
    if (($# < 2)); then
        echo "error: $1 requires a value" >&2
        exit 2
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: $name must be a positive integer: $value" >&2
        exit 2
    fi
}

while (($# > 0)); do
    case "$1" in
        --model)
            require_value "$@"
            MODEL="$2"
            shift 2
            ;;
        --dtypes)
            require_value "$@"
            read -r -a DTYPES <<< "$2"
            shift 2
            ;;
        --devices)
            require_value "$@"
            DEVICES="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$@"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            require_value "$@"
            PORT="$2"
            shift 2
            ;;
        --num-samples)
            require_value "$@"
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --needles)
            require_value "$@"
            read -r -a NEEDLES <<< "$2"
            shift 2
            ;;
        --max-prompt-tokens)
            require_value "$@"
            MAX_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --max-tokens)
            require_value "$@"
            MAX_TOKENS="$2"
            shift 2
            ;;
        --concurrency)
            require_value "$@"
            CONCURRENCY="$2"
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED="$2"
            shift 2
            ;;
        --server-arg)
            require_value "$@"
            SERVER_ARGS+=("$2")
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "error: --model is required" >&2
    usage >&2
    exit 2
fi
require_positive_integer --port "$PORT"
if ((PORT > 65535)); then
    echo "error: --port must not exceed 65535: $PORT" >&2
    exit 2
fi
require_positive_integer --num-samples "$NUM_SAMPLES"
require_positive_integer --max-tokens "$MAX_TOKENS"
require_positive_integer --concurrency "$CONCURRENCY"
if [[ -n "$MAX_PROMPT_TOKENS" ]]; then
    require_positive_integer --max-prompt-tokens "$MAX_PROMPT_TOKENS"
fi
if [[ ! "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "error: --seed must be an integer: $SEED" >&2
    exit 2
fi
for needle in "${NEEDLES[@]}"; do
    if [[ "$needle" != "2" && "$needle" != "4" && "$needle" != "8" ]]; then
        echo "error: unsupported needle count: $needle (expected 2, 4, or 8)" >&2
        exit 2
    fi
done
if ((${#DTYPES[@]} == 0)); then
    echo "error: --dtypes must contain at least one dtype" >&2
    exit 2
fi
declare -A SEEN_DTYPES=()
for dtype in "${DTYPES[@]}"; do
    if [[ -n "${SEEN_DTYPES[$dtype]:-}" ]]; then
        echo "error: duplicate dtype would overwrite results: $dtype" >&2
        exit 2
    fi
    SEEN_DTYPES["$dtype"]=1
done
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "error: Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Set VLLM_PYTHON to a suitable interpreter if needed." >&2
    exit 2
fi
if [[ -z "$DEVICES" ]]; then
    echo "error: no devices found; set CUDA_VISIBLE_DEVICES or pass --devices" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if ((${#DEVICE_LIST[@]} == 0)); then
    echo "error: --devices must contain at least one device" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' \
    "model: $MODEL" \
    "dtypes: ${DTYPES[*]}" \
    "devices: ${DEVICE_LIST[*]}" \
    "seed: $SEED" \
    "port: $PORT" > "$OUTPUT_DIR/run_config.txt"

cleanup() {
    ((${#CHILD_PIDS[@]} == 0)) || kill "${CHILD_PIDS[@]}" 2>/dev/null || true
    ((${#CHILD_PIDS[@]} == 0)) || wait "${CHILD_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_dtype() {
    local dtype="$1"
    local device="$2"
    local port="$3"
    local safe_dtype="${dtype//[^a-zA-Z0-9_.-]/_}"
    local result_file="$OUTPUT_DIR/${safe_dtype}.json"
    local log_file="$OUTPUT_DIR/${safe_dtype}.server.log"

    echo "Testing kv_cache_dtype=$dtype on CUDA device=$device (port=$port)"

    CUDA_VISIBLE_DEVICES="$device" setsid vllm serve "$MODEL" \
        --kv-cache-dtype "$dtype" \
        --port "$port" \
        --disable-uvicorn-access-log \
        "${SERVER_ARGS[@]}" >"$log_file" 2>&1 &
    local server_pid=$!
    cleanup_server() {
        kill -- -"$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if ! wait_for_server "$port" "$server_pid" "$log_file"; then
        echo "error: failed to start server for kv_cache_dtype=$dtype on port=$port; stopping it" >&2
        cleanup_server
        trap - EXIT INT TERM
        return 1
    fi

    local -a eval_args=(
        --model "$MODEL"
        --port "$port"
        --num-samples "$NUM_SAMPLES"
        --max-tokens "$MAX_TOKENS"
        --concurrency "$CONCURRENCY"
        --seed "$SEED"
        --save-results "$result_file"
    )
    if ((${#NEEDLES[@]} > 0)); then
        eval_args+=(--needles "${NEEDLES[@]}")
    fi
    if [[ -n "$MAX_PROMPT_TOKENS" ]]; then
        eval_args+=(--max-prompt-tokens "$MAX_PROMPT_TOKENS")
    fi

    CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" \
        tests/evals/mrcr/mrcr_eval.py "${eval_args[@]}"

    cleanup_server
    trap - EXIT INT TERM
}

wait_for_server() {
    local port="$1"
    local server_pid="$2"
    local log_file="$3"
    local url="http://127.0.0.1:$port/v1/models"
    local attempts=0
    until curl --silent --fail "$url" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "error: server exited before becoming ready; log: $log_file" >&2
            tail -n 20 "$log_file" >&2 || true
            return 1
        fi
        ((attempts += 1))
        if ((attempts >= 600)); then
            echo "error: server did not become ready: $url" >&2
            return 1
        fi
        sleep 1
    done
}

port_is_available() {
    ! (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1
}

find_available_port() {
    local port="$1"
    while ! port_is_available "$port"; do
        if ((port >= 65535)); then
            echo "error: no available port at or above $1" >&2
            return 1
        fi
        echo "Port $port is occupied; trying $((port + 1))" >&2
        ((port += 1))
    done
    printf '%s' "$port"
}

for ((batch_start = 0; batch_start < ${#DTYPES[@]}; batch_start += ${#DEVICE_LIST[@]})); do
    batch_pids=()
    batch_ports=()
    batch_end=$((batch_start + ${#DEVICE_LIST[@]}))
    if ((batch_end > ${#DTYPES[@]})); then
        batch_end=${#DTYPES[@]}
    fi

    echo "============================================================"
    echo "Starting dtype batch: indexes $batch_start-$((batch_end - 1))"
    for ((dtype_index = batch_start; dtype_index < batch_end; dtype_index++)); do
        device_index=$((dtype_index - batch_start))
        device="${DEVICE_LIST[$device_index]}"
        port_candidate="$PORT"
        if ((${#batch_ports[@]} > 0)); then
            port_candidate=$((batch_ports[-1] + 1))
        fi
        port="$(find_available_port "$port_candidate")"
        batch_ports+=("$port")
        (
            run_dtype "${DTYPES[$dtype_index]}" "$device" "$port"
        ) &
        batch_pids+=("$!")
        CHILD_PIDS+=("$!")
    done

    for pid in "${batch_pids[@]}"; do
        wait "$pid"
    done
    CHILD_PIDS=()
    PORT=$((batch_ports[-1] + 1))
done

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

try:
    from prettytable import PrettyTable
except ImportError as exc:
    raise SystemExit(
        "prettytable is required to render the summary; install it with "
        "'uv pip install prettytable'"
    ) from exc

result_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(result_dir.glob("*.json")):
    result = json.loads(path.read_text())
    row = {
        "dtype": path.stem,
        "match_ratio": result["match_ratio"],
        "prefix_hit_rate": result["prefix_hit_rate"],
        "tokens_per_second": result["tokens_per_second"],
    }
    row.update(result.get("per_needle", {}))
    rows.append(row)

if not rows:
    raise SystemExit("No result JSON files found")

columns = ["dtype", "match_ratio", "prefix_hit_rate", "tokens_per_second"]
for row in rows:
    for key in row:
        if key not in columns:
            columns.append(key)

summary = result_dir / "summary.tsv"
with summary.open("w") as f:
    f.write("\t".join(columns) + "\n")
    for row in rows:
        f.write("\t".join(str(row.get(key, "")) for key in columns) + "\n")

print("\nSummary")
table = PrettyTable(columns)
table.align["dtype"] = "l"
for row in rows:
    table.add_row(
        [
            f"{value:.4f}" if isinstance(value, float) else (value or "")
            for key in columns
            for value in [row.get(key, "")]
        ]
    )
print(table)
print(f"\nSaved summary: {summary}")
PY
