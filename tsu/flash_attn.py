import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_qkv_forward_kernel(
        q_ptr: tl.tensor, q_s0: int, q_s1: int, q_s2: int,
        k_ptr: tl.tensor, k_s0: int, k_s1: int, k_s2: int,
        v_ptr: tl.tensor, v_s0: int, v_s1: int, v_s2: int,

        o_ptr: tl.tensor, o_s0: int, o_s1: int, o_s2: int,
        l_ptr: tl.tensor, l_s0: int, l_s1: int,

        BH: int, Q: int, KV: int, D: int, scale: float,
        BLOCK_BH: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    q_block = tl.make_block_ptr(
        base=q_ptr,
        shape=(BH, Q, D),
        strides=(q_s0, q_s1, q_s2),
        offsets=(BLOCK_BH * tl.program_id(0), BLOCK_Q * tl.program_id(1), 0),
        block_shape=(BLOCK_BH, BLOCK_Q, BLOCK_D),
        order=(2, 1, 0),
    )
    k_block = tl.make_block_ptr(
        base=k_ptr,
        shape=(BH, D, KV),
        strides=(k_s0, k_s2, k_s1),
        offsets=(BLOCK_BH * tl.program_id(0), 0, 0),
        block_shape=(BLOCK_BH, BLOCK_D, BLOCK_KV),
        order=(2, 0, 1),
    )
    v_block = tl.make_block_ptr(
        base=v_ptr,
        shape=(BH, KV, D),
        strides=(v_s0, v_s1, v_s2),
        offsets=(BLOCK_BH * tl.program_id(0), 0, 0),
        block_shape=(BLOCK_BH, BLOCK_KV, BLOCK_D),
        order=(2, 1, 0),
    )
    kv_index = tl.arange(0, BLOCK_KV)  # [kv]

    q = tl.load(q_block, boundary_check=(0, 1, 2), padding_option='zero')  # [bh, q, d]

    m = tl.zeros((BLOCK_BH, BLOCK_Q), dtype=tl.float32) - float('inf')
    l = tl.zeros((BLOCK_BH, BLOCK_Q), dtype=tl.float32)
    o = tl.zeros((BLOCK_BH, BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for _ in tl.range(0, KV, BLOCK_KV):
        k = tl.load(k_block, boundary_check=(0, 1, 2), padding_option='zero')  # [bh, d, kv]
        v = tl.load(v_block, boundary_check=(0, 1, 2), padding_option='zero')  # [bh, kv, d]

        a = tl.dot(q, k, input_precision='ieee') * scale  # [bh, q, kv]
        a = tl.where(kv_index[None, None, :] < KV, a, -float('inf'))

        m, r = tl.maximum(tl.max(a, axis=2), m), m
        ratio1 = tl.exp(r - m)  # [bh, q]
        ratio2 = tl.exp(a - m[:, :, None])  # [bh, q, kv]

        l = l * ratio1 + tl.sum(ratio2, axis=2)
        o = o * ratio1[:, :, None] + tl.dot(ratio2, v, input_precision='ieee')

        k_block = tl.advance(k_block, offsets=(0, 0, BLOCK_KV))
        v_block = tl.advance(v_block, offsets=(0, BLOCK_KV, 0))
        kv_index += BLOCK_KV

    l_block = tl.make_block_ptr(
        base=l_ptr,
        shape=(BH, Q),
        strides=(l_s0, l_s1),
        offsets=(BLOCK_BH * tl.program_id(0), BLOCK_Q * tl.program_id(1)),
        block_shape=(BLOCK_BH, BLOCK_Q),
        order=(1, 0),
    )
    o_block = tl.make_block_ptr(
        base=o_ptr,
        shape=(BH, Q, D),
        strides=(o_s0, o_s1, o_s2),
        offsets=(BLOCK_BH * tl.program_id(0), BLOCK_Q * tl.program_id(1), 0),
        block_shape=(BLOCK_BH, BLOCK_Q, BLOCK_D),
        order=(2, 1, 0),
    )

    tl.store(l_block, (tl.log(l) + m).to(dtype=l_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(o_block, (o / l[:, :, None]).to(dtype=o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def flash_attn(q, k, v):
    o = torch.randn((q.size(0), q.size(1), v.size(2))).cuda()
    l = torch.randn((q.size(0), q.size(1))).cuda()

    BLOCK_Q = 16
    BLOCK_KV = 16
    BLOCK_D = max(16, triton.next_power_of_2(q.size(-1)))
    BLOCK_BH = 512 // BLOCK_D

    flash_attn_qkv_forward_kernel[(q.size(0) + BLOCK_BH - 1) // BLOCK_BH, (q.size(1) + BLOCK_Q - 1) // BLOCK_Q](
        q, q.stride(0), q.stride(1), q.stride(2),
        k, k.stride(0), k.stride(1), k.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        o, o.stride(0), o.stride(1), o.stride(2),
        l, l.stride(0), l.stride(1),
        BH=q.size(0), Q=q.size(1), KV=k.size(1), D=q.size(2), scale=q.size(-1) ** -0.5,
        BLOCK_BH=BLOCK_BH,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )

    return o, l
