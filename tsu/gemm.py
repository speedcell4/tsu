import triton
import triton.language as tl

from torch import Tensor


@triton.jit
def gemm_fwd_kernel(
        x_ptr: tl.tensor, x_s0, x_s1,
        y_ptr: tl.tensor, y_s0, y_s1,
        o_ptr: tl.tensor, o_s0, o_s1,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        N: tl.constexpr, BLOCK_N: tl.constexpr,
        K: tl.constexpr, BLOCK_K: tl.constexpr):
    x_bp = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(x_s0, x_s1),
        offsets=(tl.program_id(0) * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    y_bp = tl.make_block_ptr(
        base=y_ptr,
        shape=(K, N),
        strides=(y_s0, y_s1),
        offsets=(0, tl.program_id(1) * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K):
        x = tl.load(x_bp, boundary_check=(0, 1), padding_option='zero').to(dtype=tl.float32)
        y = tl.load(y_bp, boundary_check=(0, 1), padding_option='zero').to(dtype=tl.float32)
        z += tl.dot(x, y, allow_tf32=False)

        x_bp = tl.advance(x_bp, offsets=(0, BLOCK_K))
        y_bp = tl.advance(y_bp, offsets=(BLOCK_K, 0))

    o_bp = tl.make_block_ptr(
        base=o_ptr,
        shape=(M, N),
        strides=(o_s0, o_s1),
        offsets=(tl.program_id(0) * BLOCK_M, tl.program_id(1) * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    tl.store(o_bp, z.to(dtype=o_ptr.dtype.element_ty), boundary_check=(0, 1))


def gemm_fwd(x: Tensor, y: Tensor, BLOCK: int = 32):
    o = x.new_zeros((x.size(0), y.size(1)))

    gemm_fwd_kernel[(x.size(0) + BLOCK - 1) // BLOCK, (y.size(1) + BLOCK - 1) // BLOCK,](
        x, x.stride(0), x.stride(1),
        y, y.stride(0), y.stride(1),
        o, o.stride(0), o.stride(1),
        x.size(0), BLOCK,
        y.size(1), BLOCK,
        x.size(1), BLOCK,
    )

    return o
