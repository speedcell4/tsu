import torch

from tsu.lse import lse_fwd


def test_lse(batch=10000, dim=10000):
    x = torch.randn((batch, dim)).cuda()
    actual = lse_fwd(x)
    expected = x.logsumexp(dim=-1)

    torch.testing.assert_close(actual, expected)
