import unittest
import torch
from nsf_hifigan.nvSTFT import dynamic_range_compression_torch
from reflow.reflow import RectifiedFlow
from reflow.lynxnet2adaln import LYNXNet2AdaLN


class NumericalStabilityTest(unittest.TestCase):
    def test_fp16_dynamic_range_compression_backward_safe(self):
        # Input in float16 with near-zero energy
        x = torch.zeros((2, 80, 20), dtype=torch.float16, requires_grad=True)
        compressed = dynamic_range_compression_torch(x, clip_val=1e-5)
        loss = compressed.sum()
        loss.backward()

        self.assertFalse(torch.isnan(x.grad).any(), "FP16 gradient produced NaN")
        self.assertFalse(torch.isinf(x.grad).any(), "FP16 gradient produced Inf")
        # Gradient should not exceed max representable in float16
        self.assertLessEqual(x.grad.abs().max().item(), 65504.0)

    def test_norm_spec_clamping(self):
        backbone = LYNXNet2AdaLN(in_dims=4, dim_cond=4, dim_global_cond=4, n_layers=2, n_chans=8)
        flow = RectifiedFlow(backbone, out_dims=4, spec_min=-12.0, spec_max=2.0)

        # Mel values outside the range [-12, 2]
        outlier_mels = torch.tensor([[-20.0, -12.0, 0.0, 2.0, 10.0]])
        normed = flow.norm_spec(outlier_mels)

        # Normalized values must be clamped strictly to [-1, 1]
        self.assertTrue((normed >= -1.0).all())
        self.assertTrue((normed <= 1.0).all())

        # Denorm of clamped values must stay within [-12, 2]
        outlier_norm = torch.tensor([[-5.0, -1.0, 0.0, 1.0, 5.0]])
        denormed = flow.denorm_spec(outlier_norm)
        self.assertTrue((denormed >= -12.0).all())
        self.assertTrue((denormed <= 2.0).all())


if __name__ == '__main__':
    unittest.main()
