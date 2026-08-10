import unittest

import torch
import yaml
from torch.optim.swa_utils import AveragedModel

from logger import utils


class CheckpointIoTest(unittest.TestCase):
    def test_runtime_config_is_safe_yaml(self):
        config = {
            'device': torch.device('cpu'),
            'data': {'extensions': ('wav', 'flac')},
        }

        serialized = utils.make_config_serializable(config)
        dumped = yaml.safe_dump(serialized)
        restored = yaml.safe_load(dumped)

        self.assertEqual(restored['device'], 'cpu')
        self.assertEqual(restored['data']['extensions'], ['wav', 'flac'])

    def test_ema_state_loads_into_base_model(self):
        model = torch.nn.Linear(3, 2)
        ema = AveragedModel(model)

        base_state = utils.unwrap_ema_state_dict(ema.state_dict())
        restored = torch.nn.Linear(3, 2)
        restored.load_state_dict(base_state)

        self.assertNotIn('n_averaged', base_state)
        torch.testing.assert_close(restored.weight, ema.module.weight)
        torch.testing.assert_close(restored.bias, ema.module.bias)


if __name__ == '__main__':
    unittest.main()
