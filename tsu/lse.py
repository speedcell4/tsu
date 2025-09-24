import triton
import triton.language as tl

from torch import Tensor


@triton.jit
def load_block(block_ptr, boundary_check, other):
    x = tl.load(block_ptr, boundary_check=boundary_check, padding_option='nan')
    return tl.where(x != x, other, x)


@triton.jit
def init_se(x: tl.tensor, BLOCK_N: tl.constexpr):
    m = tl.full((BLOCK_N,), dtype=x.dtype.element_ty, value=-float('inf'))
    se = tl.zeros((BLOCK_N,), dtype=x.dtype.element_ty)

    return se, m


@triton.jit
def update_se(x: tl.tensor, se: tl.tensor, m: tl.tensor):
    m, mp = tl.maximum(m, tl.max(x, axis=1)), m
    se = se * tl.exp(mp - m) + tl.sum(tl.exp(x - m[:, None]), axis=1)

    return se, m


@triton.jit
def lse_fwd_kernel(
        x_ptr: tl.tensor, x_s0: int, x_s1: int,
        o_ptr: tl.tensor, o_s0: int,
        N: int, BLOCK_N: tl.constexpr,
        D: int, BLOCK_D: tl.constexpr):
    x_block = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, D),
        strides=(x_s0, x_s1),
        offsets=(BLOCK_N * tl.program_id(0), 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    se, x_m = init_se(x_ptr, BLOCK_N)

    for _ in tl.range(0, D, BLOCK_D):
        xs = load_block(x_block, boundary_check=(0, 1), other=-float('inf'))
        se, x_m = update_se(xs, se, x_m)

        x_block = tl.advance(x_block, offsets=(0, BLOCK_D))

    o_block = tl.make_block_ptr(
        base=o_ptr,
        shape=(N,),
        strides=(o_s0,),
        offsets=(BLOCK_N * tl.program_id(0),),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    tl.store(o_block, (tl.log(se) + x_m).to(dtype=o_ptr.dtype.element_ty), boundary_check=(0,))


def lse_fwd(x: Tensor, BLOCK: int = 128):
    N, D = x.size()

    o = x.new_zeros((N,))
    lse_fwd_kernel[(N + BLOCK - 1) // BLOCK,](
        x, x.stride(0), x.stride(1),
        o, o.stride(0),
        N, BLOCK, D, BLOCK,
    )
    return o
