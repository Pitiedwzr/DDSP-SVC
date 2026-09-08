import unittest

import torch

from reflow.lynxnet2adaln import LYNXNet2AdaLN
from reflow.reflow import RectifiedFlow


class TestSelfFlowSpanMasking(unittest.TestCase):
    def test_span_masking_shape_and_length(self):
        model = LYNXNet2AdaLN(
            in_dims=32,
            n_layers=2,
            n_chans=64,
            dim_cond=32,
            dim_global_cond=64,
            use_self_flow=True,
            self_flow_projector_dim=64,
        )
        rf = RectifiedFlow(
            velocity_fn=model,
            out_dims=32,
            use_self_flow=True,
            self_flow_mask_ratio=0.5,
            self_flow_span_length=4,
        )
        batch_size = 8
        seq_len = 50
        mask = rf._sample_self_flow_mask(batch_size, seq_len, torch.device("cpu"))
        self.assertEqual(mask.shape, (batch_size, seq_len))
        self.assertEqual(mask.dtype, torch.bool)

    def test_span_masking_contiguity(self):
        model = LYNXNet2AdaLN(
            in_dims=32,
            n_layers=2,
            n_chans=64,
            dim_cond=32,
            dim_global_cond=64,
            use_self_flow=True,
            self_flow_projector_dim=64,
        )
        rf = RectifiedFlow(
            velocity_fn=model,
            out_dims=32,
            use_self_flow=True,
            self_flow_mask_ratio=0.5,
            self_flow_span_length=6,
        )
        mask = rf._sample_self_flow_mask(1, 100, torch.device("cpu"))[0]
        self.assertTrue(mask.any().item())

    def test_selective_loss_computation(self):
        model = LYNXNet2AdaLN(
            in_dims=32,
            n_layers=2,
            n_chans=64,
            dim_cond=32,
            dim_global_cond=64,
            use_self_flow=True,
            self_flow_projector_dim=64,
        )
        rf = RectifiedFlow(
            velocity_fn=model,
            out_dims=32,
            use_self_flow=True,
            self_flow_student_layer=1,
            self_flow_teacher_layer=2,
            self_flow_mask_ratio=0.5,
            self_flow_span_length=3,
            self_flow_loss_on_masked_only=True,
        )
        # x_1: [B, 1, M, T]
        x_1 = torch.randn(2, 1, 32, 20)
        t = torch.tensor([0.2, 0.4])
        cond = torch.randn(2, 32, 20)
        global_cond = torch.randn(2, 64)

        _v_loss, sf_loss = rf.reflow_loss(
            x_1,
            t,
            cond,
            global_cond=global_cond,
            loss_type="l2",
            teacher_velocity_fn=model,
            return_self_flow_loss=True,
        )
        self.assertIsNotNone(sf_loss)
        self.assertTrue(sf_loss >= 0)
        self.assertFalse(torch.isnan(sf_loss))

    def test_projection_and_cosine_run_in_float32_under_autocast(self):
        model = LYNXNet2AdaLN(
            in_dims=32,
            n_layers=2,
            n_chans=64,
            dim_cond=32,
            dim_global_cond=64,
            use_self_flow=True,
            self_flow_projector_dim=64,
        )
        rf = RectifiedFlow(
            velocity_fn=model,
            out_dims=32,
            use_self_flow=True,
            self_flow_student_layer=1,
            self_flow_teacher_layer=2,
        )
        projector_input_dtypes = []
        hook = model.self_flow_projector[0].register_forward_pre_hook(
            lambda _module, inputs: projector_input_dtypes.append(inputs[0].dtype)
        )
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, sf_loss = rf.reflow_loss(
                    torch.randn(2, 1, 32, 20),
                    torch.tensor([0.2, 0.4]),
                    torch.randn(2, 32, 20),
                    global_cond=torch.randn(2, 64),
                    teacher_velocity_fn=model,
                    return_self_flow_loss=True,
                )
        finally:
            hook.remove()

        self.assertEqual(projector_input_dtypes, [torch.float32])
        self.assertEqual(sf_loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(sf_loss))


if __name__ == "__main__":
    unittest.main()
