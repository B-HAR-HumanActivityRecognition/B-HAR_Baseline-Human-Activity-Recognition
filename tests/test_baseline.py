import unittest
from har_baseline.baseline import BHar
from har_baseline.utility.configurator import Configurator


class StatsTestCase(unittest.TestCase):
    p = '/Users/cristianturetta/PycharmProjects/har_baseline/config.cfg'
    bhar = BHar(p)

    def test_print_stats(self):
        self.bhar.stats()
        self.assertEqual(True, True)

    def test_evaluation(self):
        self.bhar.get_baseline(['RF', 'DT', 'LDA', 'QDA', 'K-NN'], ['m1_acc'])
        self.assertEqual(True, True)

    def test_evaluation_dl_near_miss(self):
        # Configurator().set('settings', 'resampling', 'over')
        # Configurator().set('settings', 'resampling_technique', 'smote')
        self.bhar.get_baseline([], ['m1_acc'], [0])
        self.assertEqual(True, True)

    def test_features_extraction(self):
        Configurator(self.p).set('settings', 'features_extraction', 'True')
        Configurator(self.p).set('settings', 'features_selection', 'True')
        Configurator(self.p).set('settings', 'selection_technique', 'l1')

        self.bhar.get_baseline(['LDA', 'QDA'], ['m1_acc'])
        self.assertEqual(True, True)

    def test_configurator(self):
        print('get:', Configurator(self.p).get('settings', 'log_dir'))


if __name__ == '__main__':
    unittest.main()
