import unittest

import torch

from ddsp.alignment import align_units, pad_short_audio


class AlignmentTest(unittest.TestCase):
    def test_short_audio_padding_uses_resampled_tensor(self):
        audio_res = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        padded = pad_short_audio(audio_res, min_samples=16)

        self.assertEqual(padded.shape, (1, 16))
        torch.testing.assert_close(padded[:, :10], audio_res)
        torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 6))

    def test_unit_alignment_preserves_batch_dimension(self):
        units = torch.tensor([
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ])

        aligned = align_units(units, n_frames=5, ratio=0.5)

        self.assertEqual(aligned.shape, (2, 5, 1))
        torch.testing.assert_close(
            aligned[:, :, 0],
            torch.tensor([
                [1.0, 1.0, 2.0, 3.0, 3.0],
                [10.0, 10.0, 20.0, 30.0, 30.0],
            ]))


if __name__ == '__main__':
    unittest.main()
