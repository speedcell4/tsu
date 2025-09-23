import torch
import triton
import triton.language as tl

from torch import Tensor


@triton.jit
def lse_fwd_kernel(
        x: tl.tensor, x_s0, x_s1,
        o: tl.tensor, o_s0,
        N: tl.constexpr, BLOCK_N: tl.constexpr,
        D: tl.constexpr, BLOCK_D: tl.constexpr):
    x_bp = tl.make_block_ptr(
        base=x,
        shape=(N, D),
        strides=(x_s0, x_s1),
        offsets=(BLOCK_N * tl.program_id(0), 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    m = tl.full((BLOCK_N,), dtype=x.dtype.element_ty, value=-float('inf'))
    lse = tl.zeros((BLOCK_N,), dtype=x.dtype.element_ty)

    for _ in tl.range(0, D, BLOCK_D):
        xs = tl.load(x_bp, boundary_check=(0, 1), padding_option='nan')
        xs = tl.where(xs != xs, -float('inf'), xs)

        m, mp = tl.maximum(m, tl.max(xs, axis=1)), m
        lse = lse * tl.exp(mp - m) + tl.sum(tl.exp(xs - m[:, None]), axis=1)

        x_bp = tl.advance(x_bp, offsets=(0, BLOCK_D))

    o_bp = tl.make_block_ptr(
        base=o,
        shape=(BLOCK_N,),
        strides=(o_s0,),
        offsets=(BLOCK_N * tl.program_id(0),),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    tl.store(o_bp, (tl.log(lse) + m).to(dtype=o.dtype.element_ty), boundary_check=(0,))


def lse_fwd(x: Tensor, BLOCK: int = 128):
    N, D = x.size()

    o = x.new_zeros((N,))
    lse_fwd_kernel[(N + BLOCK - 1) // BLOCK,](
        x, x.stride(0), x.stride(1),
        o, o.stride(0),
        N, BLOCK, D, BLOCK,
    )
    return o


def lse(n: int = 3, d: int = 1000):
    x = torch.randn((n, d)).cuda()
    actual = lse_fwd(x)
    expected = x.logsumexp(dim=-1)
    print(actual)
    print(expected)

    torch.testing.assert_close(actual, expected)
