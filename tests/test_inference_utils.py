import unittest

from reflow.inference_utils import parse_speaker_mix, validate_infer_step


class InferenceUtilsTest(unittest.TestCase):
    def test_infer_step_must_be_positive(self):
        self.assertEqual(validate_infer_step(10), 10)
        with self.assertRaises(ValueError):
            validate_infer_step(0)
        with self.assertRaises(ValueError):
            validate_infer_step(-1)

    def test_speaker_mix_parser_is_safe_and_normalized(self):
        self.assertEqual(
            parse_speaker_mix('1:0.25，2：0.75'),
            {1: 0.25, 2: 0.75})
        self.assertIsNone(parse_speaker_mix('None'))
        with self.assertRaises((ValueError, SyntaxError)):
            parse_speaker_mix("__import__('os').system('echo unsafe')")


if __name__ == '__main__':
    unittest.main()
