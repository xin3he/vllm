# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.linear.cute_dsl.inc_nvfp4_e5m3 import is_available
from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.schemes.factory import resolve_scheme
from vllm.model_executor.layers.quantization.inc.schemes.inc_cute_nvfp4_e5m3_scheme import (  # noqa: E501
    INCCuteNvFp4E5M3LinearScheme,
    INCCuteNvFp4E5M3Scheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_nvfp4_e5m3_scheme import (
    INCNvFp4E5M3ReferenceLinearScheme,
    _encode_e5m3,
)
from vllm.platforms import current_platform


def test_nvfp4_e5m3_defaults_to_cutedsl() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=16,
        sym=True,
        packing_format="auto_round:llm_compressor_nvfp4_e5m3",
        backend="auto",
        data_type="fp4_v2",
        quantized=True,
    )

    assert isinstance(resolve_scheme(layer_config), INCCuteNvFp4E5M3Scheme)


def _make_layer(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = weight_packed.shape[1] * 2
    layer.output_size_per_partition = weight_packed.shape[0]
    layer.register_parameter(
        "weight_packed", Parameter(weight_packed.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", Parameter(weight_scale.clone(), requires_grad=False)
    )
    return layer


@pytest.mark.skipif(
    not current_platform.has_device_capability(80) or not is_available(),
    reason="INC NVFP4 E5M3 CuteDSL requires sm_80+ and CUTLASS Python.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 64, 64), (2, 70, 160), (2048, 70, 160)])
@torch.inference_mode()
def test_cutedsl_matches_reference_and_cuda_graph(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, output_size, input_size = shape
    weight_packed = torch.randint(
        0,
        256,
        (output_size, input_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    weight_scale = _encode_e5m3(
        torch.rand(
            (output_size, input_size // 16),
            dtype=torch.float32,
            device=device,
        )
        / 8
    )
    reference_layer = _make_layer(weight_packed, weight_scale)
    cute_layer = _make_layer(weight_packed, weight_scale)
    reference = INCNvFp4E5M3ReferenceLinearScheme()
    cute = INCCuteNvFp4E5M3LinearScheme()
    reference.process_weights_after_loading(reference_layer)
    cute.process_weights_after_loading(cute_layer)

    x = torch.randn((batch_size, input_size), dtype=dtype, device=device) / 2
    bias = torch.randn((output_size,), dtype=dtype, device=device) / 10
    expected = reference.apply_weights(reference_layer, x, bias)
    actual = cute.apply_weights(cute_layer, x, bias)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    compiled = torch.compile(
        lambda value: cute.apply_weights(cute_layer, value, bias), fullgraph=True
    )
    compiled_reference = torch.compile(
        lambda value: reference.apply_weights(reference_layer, value, bias),
        fullgraph=True,
    )
    compiled_output = compiled(x)
    compiled_expected = compiled_reference(x)
    torch.testing.assert_close(
        compiled_output, compiled_expected, rtol=5e-2, atol=5e-2
    )

    static_x = x.clone()
    cute.apply_weights(cute_layer, static_x, bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = cute.apply_weights(cute_layer, static_x, bias)

    static_x.copy_(x * 0.5)
    graph.replay()
    graph_expected = reference.apply_weights(reference_layer, static_x, bias)
    torch.testing.assert_close(graph_output, graph_expected, rtol=5e-2, atol=5e-2)