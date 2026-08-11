import unittest

from ddsp.unit2control import Unit2Control as RuntimeUnit2Control
from export_onnx import Unit2Control as OnnxUnit2Control


class OnnxArchitectureTest(unittest.TestCase):
    def test_control_network_accepts_runtime_state_dict(self):
        splits = {
            'harmonic_magnitude': 9,
            'harmonic_phase': 9,
            'noise_magnitude': 9,
            'noise_phase': 9,
        }
        kwargs = dict(
            input_channel=4,
            block_size=4,
            n_spk=2,
            output_splits=splits,
            num_layers=1,
            dim_model=16,
            use_norm=True,
            use_attention=False,
            use_pitch_aug=True,
            use_f0_conditioning=True,
        )
        runtime = RuntimeUnit2Control(**kwargs)
        onnx_model = OnnxUnit2Control(**kwargs)

        self.assertEqual(
            set(runtime.state_dict()),
            set(onnx_model.state_dict()))
        onnx_model.load_state_dict(runtime.state_dict())


if __name__ == '__main__':
    unittest.main()
