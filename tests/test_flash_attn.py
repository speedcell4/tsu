import torch

from tsu.flash_attn import flash_attn


def test_flash_attn():
    q = torch.randn((20 * 30, 100, 111)).cuda()
    k = torch.randn((20 * 30, 200, 111)).cuda()
    v = torch.randn((20 * 30, 200, 111)).cuda()

    o, l = flash_attn(q, k, v)

    expected_o = (q @ k.transpose(-1, -2)).mul(q.size(-1) ** -0.5).softmax(dim=-1) @ v
    expected_l = (q @ k.transpose(-1, -2)).mul(q.size(-1) ** -0.5).logsumexp(dim=-1)

    print(torch.testing.assert_close(actual=o, expected=expected_o))
    print(torch.testing.assert_close(actual=l, expected=expected_l))
