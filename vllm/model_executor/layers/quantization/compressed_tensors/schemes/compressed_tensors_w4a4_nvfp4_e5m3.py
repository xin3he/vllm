# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 variant with unsigned E5M3 block scales ("nvfp4+"/nvfp4-e5m3), as
produced by AutoRound's ``nvfp4-e5m3-pack-quantized`` compressed-tensors
export format.

Unlike standard NVFP4 (``CompressedTensorsW4A4Fp4``), this format:
  * stores per-group weight scales as raw ``uint8`` values encoded in
    unsigned E5M3 (5-bit exponent, 3-bit mantissa), not ``float8_e4m3fn``;
  * has no per-tensor ``weight_global_scale``/``input_global_scale``
    (i.e. no double quantization).

There is no fused CUDA kernel for this scale encoding yet, so this scheme
performs a reference (eager PyTorch) dequant + fake-quant matmul, mirroring
AutoRound's own ``NVFP4E5M3QuantLinear`` reference implementation.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4E5M3"]

_FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# Per-device cache of the FP4 magnitude lookup table. `_unpack_fp4` runs
# inside the model's forward pass, which may be traced under CUDA graph
# capture; moving a CPU-resident tensor to a CUDA device there raises
# "Cannot copy between CPU and CUDA tensors during CUDA graph capture".
# Caching the device copy (populated on the first, non-captured call)
# avoids doing that host->device copy again during capture/replay.
_FP4_MAGNITUDE_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_magnitudes(device: torch.device) -> torch.Tensor:
    cached = _FP4_MAGNITUDE_CACHE.get(device)
    if cached is None:
        cached = _FP4_MAGNITUDES.to(device)
        _FP4_MAGNITUDE_CACHE[device] = cached
    return cached


def _decode_e5m3(values: torch.Tensor) -> torch.Tensor:
    """Decode unsigned E5M3 (5-bit exponent, 3-bit mantissa) uint8 -> float32."""
    exponent = (values >> 3).to(torch.int32)
    mantissa = (values & 0x07).to(torch.float32)
    normal = (1.0 + mantissa / 8.0) * torch.exp2((exponent - 15).float())
    subnormal = mantissa * (2.0**-17)
    return torch.where(exponent == 0, subnormal, normal)


def _encode_e5m3(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(min=0.0)
    mantissa, exponent = torch.frexp(values)
    exponent_bits = (exponent + 14).clamp(0, 31)
    mantissa_bits = ((mantissa - 0.5) * 16.0).round().clamp(0, 7)
    encoded = (exponent_bits.to(torch.uint8) << 3) | mantissa_bits.to(torch.uint8)
    subnormal = (values / (2.0**-14) * 8.0).round().clamp(1, 7).to(torch.uint8)
    return torch.where(
        values == 0,
        torch.zeros_like(encoded),
        torch.where(values < 2**-14, subnormal, encoded),
    )


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(-1)
    magnitude_indices = (nibbles & 0x07).long()
    magnitude = _fp4_magnitudes(packed.device)[magnitude_indices]
    return torch.where(nibbles & 0x08 != 0, -magnitude, magnitude)


def _qdq_fp4_v2(x: torch.Tensor, group_size: int) -> torch.Tensor:
    original_shape = x.shape
    grouped = x.float().reshape(-1, group_size)
    scale = grouped.abs().amax(dim=1, keepdim=True) / 6.0
    scale = _decode_e5m3(_encode_e5m3(scale))
    scaled = torch.where(scale == 0, torch.zeros_like(grouped), grouped / scale)
    magnitude = scaled.abs()
    quantized = torch.where(
        magnitude < 2.0,
        torch.round(magnitude * 2.0) / 2.0,
        torch.where(
            magnitude < 4.0,
            torch.round(magnitude),
            2.0 * torch.round(magnitude / 2.0),
        ),
    ).clamp(max=6.0)
    return (
        (torch.copysign(quantized, scaled) * scale).reshape(original_shape).to(x.dtype)
    )


class CompressedTensorsW4A4Fp4E5M3(CompressedTensorsScheme):
    """Reference (unfused) implementation for AutoRound nvfp4-e5m3 ("nvfp4+")
    compressed-tensors checkpoints."""

    group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if input_size_per_partition % self.group_size:
            raise ValueError(
                f"NVFP4 E5M3 requires input features divisible by "
                f"{self.group_size}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        def load_e5m3_scale(param: Parameter, loaded_weight: torch.Tensor, *args):
            # The AutoRound exporter saves weight_scale as a flat column
            # vector; reshape it back to (out_features, in_features / group)
            # before applying the usual (possibly sharded) weight loader.
            weight_loader(param, loaded_weight.reshape(-1, param.data.shape[-1]), *args)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=load_e5m3_scale,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = Parameter(
            _decode_e5m3(layer.weight_scale), requires_grad=False
        )
        # Warm the per-device magnitude table cache now (outside any CUDA
        # graph capture) so `_unpack_fp4` never needs a CPU->CUDA copy later.
        _fp4_magnitudes(layer.weight_packed.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = _unpack_fp4(layer.weight_packed).reshape(
            layer.output_size_per_partition, layer.input_size_per_partition
        )
        weight = weight.reshape(-1, self.group_size)
        weight = weight * layer.weight_scale.reshape(-1, 1)
        weight = weight.reshape(
            layer.output_size_per_partition, layer.input_size_per_partition
        )
        return F.linear(_qdq_fp4_v2(x, self.group_size), weight.to(x.dtype), bias)
