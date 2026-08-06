# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCLinearScheme, INCScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig


class INCNvFp4E5M3LinearScheme(INCLinearScheme):
    """Reference implementation for AutoRound FP4-v2 E5M3 checkpoints."""

    group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition % self.group_size:
            raise ValueError("FP4-v2 requires input features divisible by 16.")

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        def load_e5m3_scale(
            param: Parameter, loaded_weight: torch.Tensor, *args
        ) -> None:
            weight_loader(
                param,
                loaded_weight.reshape(-1, param.data.shape[1]),
                *args,
            )

        layer.register_parameter(
            "weight_packed",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // self.group_size,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=load_e5m3_scale,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = Parameter(
            _decode_e5m3(layer.weight_scale), requires_grad=False
        )

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


class INCNvFp4E5M3Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_nvfp4_e5m3

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: torch.nn.Module,
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> INCLinearMethod:
        del config, layer, prefix, layer_config
        return INCLinearMethod(INCNvFp4E5M3LinearScheme())


def _decode_e5m3(values: torch.Tensor) -> torch.Tensor:
    exponent = (values >> 3).to(torch.int32)
    mantissa = (values & 0x07).to(torch.float32)
    normal = (1.0 + mantissa / 8.0) * torch.pow(
        torch.tensor(2.0, device=values.device), exponent - 15
    )
    subnormal = mantissa * (2.0**-17)
    return torch.where(exponent == 0, subnormal, normal)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(-1)
    magnitude_indices = (nibbles & 0x07).long()
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )[magnitude_indices]
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
        (torch.copysign(quantized, scaled) * scale)
        .reshape(original_shape)
        .to(x.dtype)
    )


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