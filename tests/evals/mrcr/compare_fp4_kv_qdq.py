# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare FP4 KV QDQ against GPU quantization and packed-cache kernels.

Run from the vLLM root with
    .venv/bin/python -m tests.evals.mrcr.compare_fp4_kv_qdq
MXFP4 uses the GPU Triton quantizer, not a native MXFP4 KV-cache writer.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch

from vllm.model_executor.layers.attention.kv_cache_qdq import fp4_kv_cache_qdq

logger = logging.getLogger(__name__)
FORMATS = ("mxfp4", "nvfp4", "nvfp4_4over6")
HEAD_SIZE = 64
BLOCK_SIZE = 16


def make_inputs(device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(20260923)
    random_k = (torch.randn(8, 2, HEAD_SIZE, generator=generator) * 2).to(
        device=device, dtype=torch.bfloat16
    )
    random_v = (torch.randn(8, 2, HEAD_SIZE, generator=generator) * 0.4).to(
        device=device, dtype=torch.bfloat16
    )
    # Both /4 and /6 candidates occur; include zeros, signs and FP4 ties.
    pattern = torch.tensor(
        [6.0] + [4.5] * 15 + [4.0] * 8 + [2.0] * 8
        + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] + [0.0] * 8
        + [-6.0] + [-4.5] * 15,
        dtype=torch.bfloat16,
        device=device,
    ).reshape(1, 1, HEAD_SIZE)
    boundary_k = pattern.expand(4, 2, -1).contiguous()
    boundary_v = -boundary_k
    return {"random": (random_k, random_v), "boundary": (boundary_k, boundary_v)}


def mxfp4_gpu_qdq(source: torch.Tensor) -> torch.Tensor:
    from tests.quantization.reference_mxfp4 import dq_mxfp4_torch
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        downcast_to_mxfp,
    )

    rows = source.reshape(-1, HEAD_SIZE).contiguous()
    packed, scales, _ = downcast_to_mxfp(rows, axis=-1)
    return dq_mxfp4_torch(packed, scales, source.dtype).reshape(source.shape)


def nvfp4_cache_qdq(
    key: torch.Tensor, value: torch.Tensor, cache_dtype: str, capability: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import nvfp4_split_data_scale

    tokens, heads, head_size = key.shape
    full_dim = head_size // 2 + head_size // 16
    # HND physical layout, with [K_data | K_scale | V_data | V_scale] per page.
    cache = torch.zeros(
        1, 2, heads, BLOCK_SIZE, full_dim, dtype=torch.uint8, device=key.device
    ).permute(0, 1, 3, 2, 4)
    key_cache, value_cache = cache[:, 0], cache[:, 1]
    slots = torch.arange(tokens, dtype=torch.int64, device=key.device)
    scale = torch.ones(1, device=key.device, dtype=torch.float32)
    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, cache_dtype, scale, scale
    )

    def read(side: torch.Tensor, swizzled: bool) -> torch.Tensor:
        data, block_scales = nvfp4_split_data_scale(side)
        return dequant_nvfp4_kv_cache(
            data.permute(0, 2, 1, 3),
            block_scales.permute(0, 2, 1, 3),
            scale.item(),
            head_size,
            BLOCK_SIZE,
            swizzled_scales=swizzled,
        )[0, :, :tokens].permute(1, 0, 2).to(key.dtype)

    return read(key_cache, False), read(value_cache, capability != 120)


def compare(
    format: str, case: str, side: str, source: torch.Tensor, actual: torch.Tensor
) -> tuple[str, str, str, str, float, float, int, int]:
    expected = fp4_kv_cache_qdq(source, format)
    difference = (actual.float() - expected.float()).abs()
    mismatches = (actual != expected).nonzero()
    max_diff = difference.max().item()
    mean_diff = difference.mean().item()
    status = "PASS" if len(mismatches) == 0 else "FAIL"
    logger.info(
        "%s %-14s %-8s %s: shape=%s dtype=%s max_abs=%.8g mean_abs=%.8g "
        "mismatches=%d/%d",
        format, case, side, status, tuple(source.shape), source.dtype,
        max_diff, mean_diff, len(mismatches), source.numel(),
    )
    for index in mismatches[:8].tolist():
        position = tuple(index)
        logger.info(
            "  index=%s input=%g qdq=%g gpu=%g abs_diff=%g",
            position, source[position].item(), expected[position].item(),
            actual[position].item(), difference[position].item(),
        )
    return (format, case, side, status, max_diff, mean_diff,
            len(mismatches), source.numel())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--output-dir",
        default=f"kv-cache-results/fp4-qdq-{datetime.now():%Y%m%d-%H%M%S}",
        help="Directory for detailed log and summary (default: timestamped under kv-cache-results)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "compare.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logger.info("Saving comparison results under %s", output_dir)
    results = []
    capability = 0
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        major, minor = torch.cuda.get_device_capability(device)
        capability = major * 10 + minor
        logger.info("GPU: %s (SM%d), torch %s", torch.cuda.get_device_name(device),
                    capability, torch.__version__)
    else:
        logger.info("No CUDA device: GPU comparisons will be skipped")

    cases = make_inputs(device)
    for format in FORMATS:
        if device.type != "cuda" or (format != "mxfp4" and capability < 100):
            logger.info("SKIP %s: %s", format, "CUDA required" if device.type != "cuda"
                        else "native NVFP4 KV cache requires SM100+")
            results.append((format, "-", "-", "SKIP", 0.0, 0.0, 0, 0))
            continue
        if format != "mxfp4":
            try:
                from vllm import _custom_ops as ops  # noqa: F401
            except ImportError:
                logger.info("SKIP %s: native KV cache extension unavailable", format)
                results.append((format, "-", "-", "SKIP", 0.0, 0.0, 0, 0))
                continue
            if not hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash"):
                logger.info("SKIP %s: native KV cache op unavailable", format)
                results.append((format, "-", "-", "SKIP", 0.0, 0.0, 0, 0))
                continue
        logger.info("Comparing %s using %s", format,
                    "GPU Triton MXFP4 quantizer" if format == "mxfp4"
                    else "native packed NVFP4 KV cache writer")
        for case, (key, value) in cases.items():
            try:
                if format == "mxfp4":
                    actual_key, actual_value = mxfp4_gpu_qdq(key), mxfp4_gpu_qdq(value)
                else:
                    actual_key, actual_value = nvfp4_cache_qdq(
                        key, value, format, capability
                    )
                results.append(compare(format, case, "K", key, actual_key))
                results.append(compare(format, case, "V", value, actual_value))
            except (ImportError, RuntimeError, AssertionError):
                logger.exception("GPU comparison failed for %s/%s", format, case)
                results.append((format, case, "K/V", "ERROR", 0.0, 0.0, 0, 0))

    table = [
        "| Format | Case | Side | Status | Max abs | Mean abs | Mismatches |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for format, case, side, status, maximum, mean, count, total in results:
        table.append(
            f"| {format} | {case} | {side} | {status} | {maximum:.6g} | "
            f"{mean:.6g} | {count}/{total} |"
        )
    summary = "\n".join(table) + "\n"
    print(f"\n{summary}", end="")
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("Detailed log: %s", log_file)
    logger.info("Summary: %s", output_dir / "summary.md")
    return int(any(row[3] in ("FAIL", "ERROR") for row in results))


if __name__ == "__main__":
    raise SystemExit(main())