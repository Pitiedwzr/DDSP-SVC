import unittest

import numpy as np

from ddsp.pitch import interpolate_unvoiced_f0


class PitchInterpolationTest(unittest.TestCase):
    def test_interpolation_preserves_voicing_mask(self):
        f0, voiced = interpolate_unvoiced_f0(
            np.array([100.0, 0.0, 200.0], dtype=np.float32),
            f0_min=65.0)

        np.testing.assert_allclose(f0, [100.0, 150.0, 200.0])
        np.testing.assert_array_equal(voiced, [1.0, 0.0, 1.0])

    def test_fully_unvoiced_contour_stays_zero(self):
        f0, voiced = interpolate_unvoiced_f0(
            np.zeros(3, dtype=np.float32),
            f0_min=65.0)

        np.testing.assert_array_equal(f0, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(voiced, np.zeros(3, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
