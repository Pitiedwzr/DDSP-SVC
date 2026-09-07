import unittest
import torch
from reflow.lynxnet2adaln import LYNXNet2AdaLN
from reflow.reflow import RectifiedFlow


class PerformanceOptimizationsTest(unittest.TestCase):
    def test_adaln_broadcast_equivalence(self):
        backbone = LYNXNet2AdaLN(
            in_dims=8, dim_cond=8, dim_global_cond=4, n_layers=2, n_chans=16
        )
        spec = torch.randn(2, 1, 8, 12)
        cond = torch.randn(2, 8, 12)
        spk = torch.randn(2, 4)

        t_scalar = torch.tensor([0.3, 0.7])
        t_expanded = t_scalar[:, None].expand(-1, 12)

        out_scalar = backbone(spec, 1000 * t_scalar, cond, spk)
        out_expanded = backbone(spec, 1000 * t_expanded, cond, spk)

        torch.testing.assert_close(out_scalar, out_expanded, rtol=1e-4, atol=1e-4)

    def test_batched_cfg_matches_sequential_velocity(self):
        backbone = LYNXNet2AdaLN(
            in_dims=8, dim_cond=8, dim_global_cond=4, n_layers=2, n_chans=16
        )
        flow = RectifiedFlow(backbone, out_dims=8)

        x = torch.randn(2, 1, 8, 10)
        t = torch.tensor([0.2, 0.5])
        cond = torch.randn(2, 8, 10)
        global_cond = torch.randn(2, 4)
        null_cond = torch.zeros(2, 4)
        cfg_scale = 2.5

        # Batched CFG via _get_velocity
        v_batched = flow._get_velocity(x, t, cond, global_cond, cfg_scale=cfg_scale, null_global_cond=null_cond)

        # Sequential reference
        v_cond = backbone(x, 1000 * t, cond, global_cond)
        v_uncond = backbone(x, 1000 * t, cond, null_cond)
        v_seq = v_uncond + cfg_scale * (v_cond - v_uncond)

        torch.testing.assert_close(v_batched, v_seq, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
