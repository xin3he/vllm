#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  compare_kv_cache_dtypes.sh --model MODEL [options]

Required:
  --model MODEL              Hugging Face model or local model path

Options:
  --dtypes "DTYPE ..."       KV cache dtypes (default: "bfloat16 fp8")
    --devices "ID,..."         Devices to rotate across (default: CUDA_VISIBLE_DEVICES)
  --output-dir DIR            Result directory (default: mrcr-kv-cache-results-<timestamp>)
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
OUTPUT_DIR="mrcr-kv-cache-results-$(date +%Y%m%d-%H%M%S)"
PORT=8000
NUM_SAMPLES=40
NEEDLES=(2 4 8)
MAX_PROMPT_TOKENS=""
MAX_TOKENS=2048
CONCURRENCY=8
SEED=42
SERVER_ARGS=()
SERVER_PID=""
PYTHON_BIN="/home/xinhe/auto-round/.venv/bin/python"
DEVICES="${CUDA_VISIBLE_DEVICES:-}"

while (($# > 0)); do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dtypes)
            read -r -a DTYPES <<< "$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --needles)
            read -r -a NEEDLES <<< "$2"
            shift 2
            ;;
        --max-prompt-tokens)
            MAX_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --server-arg)
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
if ((${#DTYPES[@]} == 0)); then
    echo "error: --dtypes must contain at least one dtype" >&2
    exit 2
fi
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
for device in "${DEVICE_LIST[@]}"; do
    if [[ "$device" == "7" ]]; then
        echo "error: GPU 7 is reserved and cannot be used" >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"
printf '%s\n' \
    "model: $MODEL" \
    "dtypes: ${DTYPES[*]}" \
    "devices: ${DEVICE_LIST[*]}" \
    "seed: $SEED" \
    "port: $PORT" > "$OUTPUT_DIR/run_config.txt"

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

run_dtype() {
    local dtype="$1"
    local device="$2"
    local port="$3"
    local safe_dtype="${dtype//[^a-zA-Z0-9_.-]/_}"
    local result_file="$OUTPUT_DIR/${safe_dtype}.json"
    local log_file="$OUTPUT_DIR/${safe_dtype}.server.log"

    echo "Testing kv_cache_dtype=$dtype on CUDA device=$device (port=$port)"

    CUDA_VISIBLE_DEVICES="$device" vllm serve "$MODEL" \
        --kv-cache-dtype "$dtype" \
        --port "$port" \
        --disable-uvicorn-access-log \
        --reasoning-parser qwen3 \
        "${SERVER_ARGS[@]}" >"$log_file" 2>&1 &
    local server_pid=$!
    cleanup_server() {
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server EXIT INT TERM

    wait_for_server "$port"

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
    local url="http://127.0.0.1:$port/v1/models"
    local attempts=0
    until curl --silent --fail "$url" >/dev/null 2>&1; do
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
    done

    for pid in "${batch_pids[@]}"; do
        wait "$pid"
    done
    PORT=$((batch_ports[-1] + 1))
done

"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

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
print("\t".join(columns))
for row in rows:
    print("\t".join(str(row.get(key, "")) for key in columns))
print(f"\nSaved summary: {summary}")
PY
