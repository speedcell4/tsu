import torch
import triton

from tsu.gemm import gemm_fwd


def test_gemm():
    x = torch.randn((1000, 2000)).cuda()
    y = torch.randn((2000, 3000)).cuda()

    actual = gemm_fwd(x, y)
    expected = x @ y

    triton.testing.assert_close(actual, expected)
