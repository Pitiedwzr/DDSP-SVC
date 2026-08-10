import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from logger import utils
from reflow.lynxnet2adaln import LYNXNet2AdaLN
from reflow.reflow import RectifiedFlow


class SelfFlowTest(unittest.TestCase):
    def make_backbone(self, use_self_flow=True):
        return LYNXNet2AdaLN(
            in_dims=8,
            dim_cond=8,
            dim_global_cond=4,
            n_layers=4,
            n_chans=16,
            use_self_flow=use_self_flow,
            self_flow_projector_dim=12,
        )

    def test_scalar_and_token_timesteps_have_matching_shapes(self):
        model = self.make_backbone(use_self_flow=False)
        spec = torch.randn(2, 1, 8, 7)
        cond = torch.randn(2, 8, 7)
        speaker = torch.randn(2, 4)

        scalar_output = model(spec, torch.rand(2), cond, speaker)
        token_output = model(spec, torch.rand(2, 7), cond, speaker)

        self.assertEqual(scalar_output.shape, spec.shape)
        self.assertEqual(token_output.shape, spec.shape)

    def test_self_flow_loss_backpropagates_to_student_only_parameters(self):
        student = self.make_backbone(use_self_flow=True)
        teacher = self.make_backbone(use_self_flow=True)
        teacher.load_state_dict(student.state_dict())
        flow = RectifiedFlow(
            student,
            out_dims=8,
            use_self_flow=True,
            self_flow_student_layer=2,
            self_flow_teacher_layer=3,
            self_flow_mask_ratio=0.5,
            self_flow_condition_mask_ratio=1.0,
        )

        condition = torch.randn(2, 7, 8)
        target = torch.randn(2, 7, 8)
        speaker = torch.randn(2, 4)
        flow_loss, representation_loss = flow(
            condition,
            gt_spec=target,
            global_cond=speaker,
            infer=False,
            teacher_velocity_fn=teacher,
            return_self_flow_loss=True,
        )
        (flow_loss + 0.8 * representation_loss).backward()

        self.assertEqual(flow_loss.ndim, 0)
        self.assertEqual(representation_loss.ndim, 0)
        self.assertIsNotNone(flow.condition_mask_token.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in student.self_flow_projector.parameters()
        ))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


class EmaResumeTest(unittest.TestCase):
    def test_checkpoint_restores_ema_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
            with torch.no_grad():
                model.weight.fill_(2.0)
            ema.update_parameters(model)

            checkpoint_path = Path(temp_dir) / "model_1.pt"
            torch.save({
                "global_step": 1,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, checkpoint_path)

            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
            restored_ema = AveragedModel(
                restored_model, multi_avg_fn=get_ema_multi_avg_fn(0.9))
            step, _, _ = utils.load_model(
                temp_dir,
                restored_model,
                restored_optimizer,
                ema_model=restored_ema,
            )

            self.assertEqual(step, 1)
            for expected, actual in zip(ema.parameters(), restored_ema.parameters()):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
