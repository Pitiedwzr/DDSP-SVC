import unittest
import torch
from reflow.lynxnet2adaln import LYNXNet2AdaLN, LYNXNet2AdaLNBlock, DecoupledLYNXNet2AdaLNBlock

class TestBackboneImprovements(unittest.TestCase):
    def test_default_fused_atan_compatibility(self):
        model = LYNXNet2AdaLN(
            in_dims=32,
            dim_cond=32,
            dim_global_cond=64,
            n_layers=2,
            n_chans=64,
            block_type='fused',
            gating_act='atan'
        )
        x = torch.randn(2, 1, 32, 20)
        t = torch.tensor([100.0, 200.0])
        cond = torch.randn(2, 32, 20)
        global_cond = torch.randn(2, 64)
        out = model(x, t, cond, global_cond)
        self.assertEqual(out.shape, (2, 1, 32, 20))

    def test_gating_activations(self):
        for act in ['atan', 'silu', 'glu']:
            model = LYNXNet2AdaLN(
                in_dims=16,
                dim_cond=16,
                dim_global_cond=32,
                n_layers=2,
                n_chans=32,
                block_type='fused',
                gating_act=act
            )
            x = torch.randn(2, 1, 16, 15)
            t = torch.tensor([50.0, 150.0])
            cond = torch.randn(2, 16, 15)
            global_cond = torch.randn(2, 32)
            out = model(x, t, cond, global_cond)
            self.assertEqual(out.shape, (2, 1, 16, 15))
            self.assertFalse(torch.isnan(out).any())

    def test_decoupled_block(self):
        for act in ['atan', 'silu', 'glu']:
            model = LYNXNet2AdaLN(
                in_dims=16,
                dim_cond=16,
                dim_global_cond=32,
                n_layers=2,
                n_chans=32,
                block_type='decoupled',
                gating_act=act
            )
            self.assertIsInstance(model.residual_layers[0], DecoupledLYNXNet2AdaLNBlock)
            x = torch.randn(2, 1, 16, 15)
            t = torch.tensor([50.0, 150.0])
            cond = torch.randn(2, 16, 15)
            global_cond = torch.randn(2, 32)
            out = model(x, t, cond, global_cond)
            self.assertEqual(out.shape, (2, 1, 16, 15))
            self.assertFalse(torch.isnan(out).any())

    def test_decoupled_gradient_flow(self):
        model = LYNXNet2AdaLN(
            in_dims=16,
            dim_cond=16,
            dim_global_cond=32,
            n_layers=2,
            n_chans=32,
            block_type='decoupled',
            gating_act='silu'
        )
        # Avoid zero-init alpha trapping in unit test of gradient flow
        for layer in model.residual_layers:
            nn = torch.nn
            nn.init.normal_(layer.film_proj1.weight, std=0.02)
            nn.init.normal_(layer.film_proj2.weight, std=0.02)
        nn.init.normal_(model.output_projection.weight, std=0.02)

        x = torch.randn(2, 1, 16, 15, requires_grad=True)
        t = torch.tensor([50.0, 150.0])
        cond = torch.randn(2, 16, 15, requires_grad=True)
        global_cond = torch.randn(2, 32, requires_grad=True)
        out = model(x, t, cond, global_cond)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(cond.grad)
        self.assertIsNotNone(global_cond.grad)
        self.assertFalse(torch.isnan(x.grad).any())

if __name__ == '__main__':
    unittest.main()
