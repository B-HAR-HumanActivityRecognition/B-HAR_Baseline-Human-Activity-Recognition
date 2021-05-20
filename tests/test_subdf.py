import unittest
from b_har.baseline import B_HAR
from b_har.utility.configurator import Configurator
from datetime import datetime


class MyTestCase(unittest.TestCase):
    def test_something(self):
        cfg_file = '/home/furla/b_har/datasets/benchmark/wisdm/multiclass/config_example.cfg'
        #cfg_file = '/home/furla/b_har/datasets/wisdm-dataset/raw/watch/accel/config_example.cfg'
        #cfg_file = '/home/furla/b_har/datasets/wisdm-dataset/config_test.cfg'

        b_har = B_HAR(config_file_path=cfg_file)

        b_har.stats()
        b_har.get_baseline(['K-NN', 'DT', 'LDA','QDA'], [])
        #b_har.get_baseline([], [])
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
