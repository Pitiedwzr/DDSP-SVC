import unittest
import torch
import torch.nn as nn
from optimizer.muon import get_params_for_muon, Muon, Muon_AdamW
from optimizer.aurora import get_params_for_aurora, Aurora, Aurora_AdamW, aurora_update


class OptimizerCorrectnessTest(unittest.TestCase):
    def test_muon_excludes_depthwise_conv_and_embeddings(self):
        class ModelWithConvs(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 64)
                self.depthwise = nn.Conv1d(64, 64, kernel_size=31, groups=64)
                self.pointwise = nn.Conv1d(64, 128, kernel_size=1)
                self.linear = nn.Linear(128, 64)

        model = ModelWithConvs()
        muon_params = get_params_for_muon(model)
        muon_param_ids = {id(p) for p in muon_params}

        # Depthwise conv weight and embedding must NOT be in muon_params
        self.assertNotIn(id(model.depthwise.weight), muon_param_ids)
        self.assertNotIn(id(model.emb.weight), muon_param_ids)

        # Standard dense 2D weights must be in muon_params
        self.assertIn(id(model.pointwise.weight), muon_param_ids)
        self.assertIn(id(model.linear.weight), muon_param_ids)

    def test_aurora_dimension_scaling(self):
        W = torch.randn(64, 64)
        G = torch.randn(64, 64)
        momentum = torch.zeros(64, 64)

        # Should execute without error using float32 / bf16 fallback
        updated_W = aurora_update(W.clone(), G.clone(), momentum.clone(), eta=0.01, use_bf16=False)
        self.assertEqual(updated_W.shape, W.shape)
        self.assertFalse(torch.isnan(updated_W).any())

    def test_muon_adamw_step_executes_cleanly(self):
        layer = nn.Linear(32, 16)
        opt = Muon_AdamW(layer, lr=0.01)
        x = torch.randn(4, 32)
        y = layer(x).sum()
        y.backward()
        opt.step()
        self.assertFalse(torch.isnan(layer.weight).any())


if __name__ == '__main__':
    unittest.main()
