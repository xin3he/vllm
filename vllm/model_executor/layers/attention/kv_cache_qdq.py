# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F


def fp4_kv_cache_qdq(tensor: torch.Tensor, format: str) -> torch.Tensor:
    """Simulate block-scaled E2M1 KV storage in the input's floating dtype."""
    if format not in ("mxfp4", "nvfp4", "nvfp4_4over6"):
        raise ValueError(f"Unsupported KV cache QDQ format: {format}")

    group_size = 32 if format == "mxfp4" else 16
    head_dim = tensor.shape[-1]
    padded_dim = (head_dim + group_size - 1) // group_size * group_size
    values = F.pad(tensor.float(), (0, padded_dim - head_dim)).reshape(
        *tensor.shape[:-1], -1, group_size
    )
    absmax = values.abs().amax(dim=-1, keepdim=True)
    levels = values.new_tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6))
    boundaries = values.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5))

    def restore(scale: torch.Tensor) -> torch.Tensor:
        scaled = (
            values / scale if format == "mxfp4" else values * scale.reciprocal()
        ).abs().contiguous()
        indices = torch.bucketize(scaled, boundaries)
        indices = indices + (
            (scaled == 0.75) | (scaled == 1.75) | (scaled == 3.5)
        ).long()
        return levels[indices] * values.sign() * scale

    if format == "mxfp4":
        rounded_max = (
            (absmax.contiguous().view(torch.int32) + 0x200000) & 0x7F800000
        ).view(torch.float32)
        exponent = (torch.log2(rounded_max.clamp(min=2**-126)) - 2).clamp(
            -127, 127
        )
        scale = torch.exp2(exponent)
    else:
        scale = (absmax / 6).clamp(min=2**-9, max=448)
        scale = scale.to(torch.float8_e4m3fn).float()
    scale = torch.where(absmax == 0, 1.0, scale)

    restored = restore(scale)
    if format == "nvfp4_4over6":
        scale4 = (absmax / 4).clamp(min=2**-9, max=448)
        scale4 = scale4.to(torch.float8_e4m3fn).float()
        scale4 = torch.where(absmax == 0, 1.0, scale4)
        restored4 = restore(scale4)
        error6 = (restored - values).square().sum(dim=-1, keepdim=True)
        error4 = (restored4 - values).square().sum(dim=-1, keepdim=True)
        restored = torch.where(error4 < error6, restored4, restored)

    return restored.reshape(*tensor.shape[:-1], padded_dim)[..., :head_dim].to(
        tensor.dtype
    )