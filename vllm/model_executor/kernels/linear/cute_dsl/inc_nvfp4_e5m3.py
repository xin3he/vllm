# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

_compiled_cache: dict[tuple[torch.dtype, int], Any] = {}


def is_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _stream():
    from cuda.bindings.driver import CUstream

    from vllm.utils.torch_utils import current_stream

    return CUstream(current_stream().cuda_stream)


def _compile(dtype: torch.dtype, k: int):
    import cutlass.cute as cute
    from cutlass import BFloat16, Float16, Float32, Uint8
    from quack.compile_utils import make_fake_tensor

    from ._inc_nvfp4_e5m3 import INCNvFp4E5M3Gemm

    input_dtype = BFloat16 if dtype == torch.bfloat16 else Float16
    m = cute.sym_int()
    n = cute.sym_int()
    x = make_fake_tensor(input_dtype, (m, k), divisibility=8)
    weight = make_fake_tensor(Uint8, (n, k // 2), divisibility=8)
    weight_scale = make_fake_tensor(Float32, (n, k // 16), divisibility=1)
    output = make_fake_tensor(Float32, (m, n), divisibility=1)
    kernel = cute.compile(
        INCNvFp4E5M3Gemm(k),
        x,
        weight,
        weight_scale,
        output,
        1,
        1,
        _stream(),
        options="--enable-tvm-ffi",
    )
    _compiled_cache[(dtype, k)] = kernel
    return kernel


@torch.library.custom_op(
    "vllm::inc_nvfp4_e5m3_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _inc_nvfp4_e5m3_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    cache_key = (x.dtype, k)
    kernel = _compiled_cache.get(cache_key) or _compile(*cache_key)
    output = torch.empty((m, n), dtype=torch.float32, device=x.device)
    kernel(x, weight, weight_scale, output, m, n, _stream())
    return output


@torch.library.register_fake("vllm::inc_nvfp4_e5m3_gemm")
def _inc_nvfp4_e5m3_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )


def inc_nvfp4_e5m3_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("INC NVFP4 E5M3 GEMM expects 2D tensors.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("INC NVFP4 E5M3 GEMM expects FP16 or BF16 input.")
    return _inc_nvfp4_e5m3_gemm(x, weight, weight_scale)