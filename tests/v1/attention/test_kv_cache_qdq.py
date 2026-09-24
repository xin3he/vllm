# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.kv_cache_qdq import fp4_kv_cache_qdq
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder


@pytest.mark.parametrize("format,group_size", [("mxfp4", 32), ("nvfp4", 16)])
def test_fp4_kv_cache_qdq_group_scaling(format: str, group_size: int):
    tensor = torch.zeros(2, 1, group_size + 1, dtype=torch.bfloat16)
    tensor[..., 0] = 6
    tensor[..., 1] = 0.6
    tensor[..., group_size] = 0.6

    restored = fp4_kv_cache_qdq(tensor, format)

    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert restored[..., 0].eq(6).all()
    assert restored[..., 1].eq(0.5).all()
    expected = 0.5 if format == "mxfp4" else 0.609375
    assert restored[..., group_size].eq(expected).all()
    assert tensor[..., 1].eq(torch.tensor(0.6, dtype=tensor.dtype)).all()


def test_fp4_kv_cache_qdq_formats_have_distinct_group_sizes():
    tensor = torch.zeros(1, 1, 32)
    tensor[..., 0] = 6
    tensor[..., 16] = 0.6

    assert fp4_kv_cache_qdq(tensor, "mxfp4")[0, 0, 16] == 0.5
    assert fp4_kv_cache_qdq(tensor, "nvfp4")[0, 0, 16] > 0.5


def test_mxfp4_scale_uses_rounded_block_maximum():
    tensor = torch.zeros(32, dtype=torch.bfloat16)
    tensor[0] = 3.25
    tensor[1] = 0.5859375

    assert fp4_kv_cache_qdq(tensor, "mxfp4")[1] == 0.5


def test_nvfp4_midpoint_uses_reciprocal_scale():
    tensor = torch.zeros(16, dtype=torch.bfloat16)
    tensor[0] = 5.625
    tensor[1] = 1.171875
    tensor[2] = -1.171875

    restored = fp4_kv_cache_qdq(tensor, "nvfp4")

    assert restored[1] == 1.40625
    assert restored[2] == -1.40625


def test_nvfp4_4over6_selects_lower_error_for_each_group():
    tensor = torch.tensor([4.0] * 8 + [2.0] * 8 + [6.0] * 16)
    restored = fp4_kv_cache_qdq(tensor, "nvfp4_4over6")

    torch.testing.assert_close(restored[:16], tensor[:16])
    torch.testing.assert_close(restored[16:], fp4_kv_cache_qdq(tensor[16:], "nvfp4"))
    assert (restored - tensor).square().sum() < (
        fp4_kv_cache_qdq(tensor, "nvfp4") - tensor
    ).square().sum()


def test_nvfp4_4over6_ties_choose_max_over_six():
    tensor = torch.zeros(16)
    torch.testing.assert_close(
        fp4_kv_cache_qdq(tensor, "nvfp4_4over6"),
        fp4_kv_cache_qdq(tensor, "nvfp4"),
    )


def test_nvfp4_e2m1_midpoints_round_to_even():
    tensor = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] + [0.0] * 8)
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0] + [0.0] * 8)
    torch.testing.assert_close(fp4_kv_cache_qdq(tensor, "nvfp4"), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_qdq_compiles_midpoint_indices():
    tensor = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] + [0.0] * 8,
        device="cuda",
        dtype=torch.bfloat16,
    ).repeat(4, 2, 1)
    compiled_qdq = torch.compile(fp4_kv_cache_qdq, backend="inductor")

    torch.testing.assert_close(
        compiled_qdq(tensor, "nvfp4"),
        fp4_kv_cache_qdq(tensor, "nvfp4"),
    )


def test_fp4_kv_cache_qdq_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported KV cache QDQ format"):
        fp4_kv_cache_qdq(torch.zeros(16), "invalid")


@pytest.mark.parametrize("format", ["nvfp4", "nvfp4_4over6"])
def test_flashinfer_qdq_nvfp4_keeps_model_dtype_query(format: str, monkeypatch):
    from vllm.v1.attention.backends import flashinfer

    monkeypatch.setattr(
        flashinfer.current_platform,
        "is_device_capability_family",
        lambda family: family == 100,
    )
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(
                disable_flashinfer_q_quantization=False
            )
        ),
        cache_config=SimpleNamespace(kv_cache_qdq=format),
        cache_dtype="auto",
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        kv_cache_spec=SimpleNamespace(dtype=torch.bfloat16),
    )

    assert FlashInferMetadataBuilder.get_q_data_type(builder, is_prefill=True) == (
        torch.bfloat16
    )
    assert FlashInferMetadataBuilder.get_q_data_type(builder, is_prefill=False) == (
        torch.bfloat16
    )