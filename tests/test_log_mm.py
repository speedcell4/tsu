import torch
import triton

from tsu.log_mm import log_mm_fwd


def test_log_mm():
    x = torch.randn((1000, 2000)).cuda()
    y = torch.randn((2000, 3000)).cuda()

    actual = log_mm_fwd(x, y)
    expected = (x[:, None, :] + y[None, :, :]).logsumexp(dim=1)

    triton.testing.assert_close(actual, expected)
