# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream


class INCNvFp4E5M3Gemm:
    def __init__(
        self, k: int, block_size: int = 128, token_tile: int = 16
    ) -> None:
        self.k = k
        self.block_size = block_size
        self.token_tile = token_tile
        self.num_warps = block_size // cute.arch.WARP_SIZE
        self.k_rounds = (k + block_size - 1) // block_size

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        weight_scale: cute.Tensor,
        output: cute.Tensor,
        m: cutlass.Int32,
        n: cutlass.Int32,
        stream: CUstream,
    ) -> None:
        self.kernel(x, weight, weight_scale, output, m).launch(
            grid=[n, (m + self.token_tile - 1) // self.token_tile, 1],
            block=[self.block_size, 1, 1],
            smem=self.num_warps * self.token_tile * 4,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        weight_scale: cute.Tensor,
        output: cute.Tensor,
        m: cutlass.Int32,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        output_idx, token_block, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        token_start = token_block * self.token_tile
        acc = cute.make_rmem_tensor((self.token_tile,), cutlass.Float32)
        acc.fill(0.0)

        for k_round in cutlass.range_constexpr(self.k_rounds):
            k_idx = thread_idx + k_round * self.block_size
            if k_idx < self.k:
                packed = weight[output_idx, k_idx // 2]
                nibble = cutlass.Uint8(0)
                if k_idx % 2 == 0:
                    nibble = packed & cutlass.Uint8(0x0F)
                else:
                    nibble = packed >> cutlass.Uint8(4)
                magnitude_idx = nibble & cutlass.Uint8(0x07)
                magnitude = cutlass.Float32(0.0)
                if magnitude_idx == 1:
                    magnitude = 0.5
                elif magnitude_idx == 2:
                    magnitude = 1.0
                elif magnitude_idx == 3:
                    magnitude = 1.5
                elif magnitude_idx == 4:
                    magnitude = 2.0
                elif magnitude_idx == 5:
                    magnitude = 3.0
                elif magnitude_idx == 6:
                    magnitude = 4.0
                elif magnitude_idx == 7:
                    magnitude = 6.0
                if nibble & cutlass.Uint8(0x08) != 0:
                    magnitude = -magnitude
                scaled_weight = (
                    magnitude * weight_scale[output_idx, k_idx // 16]
                )
                for token_offset in cutlass.range_constexpr(self.token_tile):
                    token_idx = token_start + token_offset
                    if token_idx < m:
                        acc[token_offset] += (
                            x[token_idx, k_idx].to(cutlass.Float32) * scaled_weight
                        )

        for token_offset in cutlass.range_constexpr(self.token_tile):
            acc[token_offset] = cute.arch.warp_reduction_sum(acc[token_offset])

        smem_layout = cute.make_layout((self.num_warps, self.token_tile))
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32, smem_layout, byte_alignment=16
        )
        if cute.arch.lane_idx() == 0:
            for token_offset in cutlass.range_constexpr(self.token_tile):
                partials[warp_idx, token_offset] = acc[token_offset]

        cute.arch.sync_threads()
        if thread_idx < self.token_tile:
            token_idx = token_start + thread_idx
            if token_idx < m:
                total = cutlass.Float32(0.0)
                for partial_idx in cutlass.range_constexpr(self.num_warps):
                    total += partials[partial_idx, thread_idx]
                output[token_idx, output_idx] = total