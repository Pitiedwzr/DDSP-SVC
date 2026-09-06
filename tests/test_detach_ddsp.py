import unittest
import torch
from reflow.vocoder import Unit2Wav


class DetachDdspConditionTest(unittest.TestCase):
    def test_detach_ddsp_cond_controls_gradient_flow(self):
        # Setup small Unit2Wav instances
        kwargs = dict(
            sampling_rate=44100,
            block_size=512,
            win_length=2048,
            n_unit=16,
            n_spk=1,
            out_dims=16,
            n_aux_layers=1,
            n_aux_chans=16,
            n_layers=1,
            n_chans=16,
            use_aux_f0=False,
            use_self_flow=False,
        )
        model_detached = Unit2Wav(**kwargs, detach_ddsp_cond=True)
        model_attached = Unit2Wav(**kwargs, detach_ddsp_cond=False)
        with torch.no_grad():
            model_detached.reflow_model.velocity_fn.output_projection.weight.fill_(0.1)
            model_attached.reflow_model.velocity_fn.output_projection.weight.fill_(0.1)

        # Mock vocoder that returns differentiable mel from audio
        class MockVocoder:
            def extract(self, audio):
                # Simple differentiable projection to [B, T, out_dims]
                B, L = audio.shape
                frames = L // 512
                # average pool into frames then expand to out_dims
                pooled = audio[:, :frames * 512].reshape(B, frames, 512).mean(-1, keepdim=True)
                return pooled.expand(-1, -1, 16)

        mock_vocoder = MockVocoder()
        units = torch.randn(1, 4, 16)
        f0 = torch.full((1, 4, 1), 220.0)
        volume = torch.full((1, 4, 1), 0.5)
        gt_spec = torch.randn(1, 4, 16)

        # 1. Detached: reflow loss should NOT propagate gradients to ddsp_model
        model_detached.zero_grad()
        ddsp_loss_d, reflow_loss_d, _ = model_detached(
            units, f0, volume, vocoder=mock_vocoder, gt_spec=gt_spec, infer=False
        )
        reflow_loss_d.backward()
        ddsp_has_grad_d = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_detached.ddsp_model.parameters())
        self.assertFalse(ddsp_has_grad_d, "Detached model should not have DDSP gradients from reflow_loss")

        # 2. Attached: reflow loss SHOULD propagate gradients to ddsp_model
        model_attached.zero_grad()
        ddsp_loss_a, reflow_loss_a, _ = model_attached(
            units, f0, volume, vocoder=mock_vocoder, gt_spec=gt_spec, infer=False
        )
        reflow_loss_a.backward()
        ddsp_has_grad_a = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_attached.ddsp_model.parameters())
        self.assertTrue(ddsp_has_grad_a, "Attached model should have DDSP gradients from reflow_loss")


if __name__ == '__main__':
    unittest.main()
