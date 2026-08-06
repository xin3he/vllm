# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.linear.cute_dsl.inc_nvfp4_e5m3 import (
    inc_nvfp4_e5m3_gemm,
    is_available,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_nvfp4_e5m3_scheme import _decode_e5m3, _qdq_fp4_v2
from .inc_scheme import INCLinearScheme, INCScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig


class INCCuteNvFp4E5M3LinearScheme(INCLinearScheme):
    """CuteDSL implementation for AutoRound FP4-v2 E5M3 checkpoints."""

    group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

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
            _decode_e5m3(layer.weight_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        quantized_x = _qdq_fp4_v2(x, self.group_size).reshape(
            -1, layer.input_size_per_partition
        )
        out = inc_nvfp4_e5m3_gemm(
            quantized_x,
            layer.weight_packed,
            layer.weight_scale,
        ).to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class INCCuteNvFp4E5M3Scheme(INCScheme):
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
        if not current_platform.has_device_capability(80) or not is_available():
            raise NotImplementedError(
                "INC NVFP4 E5M3 CuteDSL requires an sm_80+ GPU and CUTLASS Python."
            )
        return INCLinearMethod(INCCuteNvFp4E5M3LinearScheme())