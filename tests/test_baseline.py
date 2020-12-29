import unittest
from har_baseline.baseline import BHar
from har_baseline.utility.configurator import Configurator
from datetime import datetime


class StatsTestCase(unittest.TestCase):
    p = '/Users/cristianturetta/PycharmProjects/har_baseline/config.cfg'
    ds_path = '/Users/cristianturetta/HAR Datasets/B-HAR Compatible/'

    def test_daphnet(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('daphnet', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'daphnet'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '4')
        Configurator(self.p).set('settings', 'sampling_frequency', '64')
        Configurator(self.p).set('settings', 'overlap', '0.5')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_hhar_phone(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('hhar_phone', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'hhar_phone'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '2')
        Configurator(self.p).set('settings', 'sampling_frequency', '200')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_hhar_watch(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('hhar_watch', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'hhar_watch'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '2')
        Configurator(self.p).set('settings', 'sampling_frequency', '200')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_mhealth(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('mhealth', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'mhealth'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'dcp')
        Configurator(self.p).set('settings', 'time', '5')
        Configurator(self.p).set('settings', 'sampling_frequency', '50')
        Configurator(self.p).set('settings', 'overlap', '2.5')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_papam(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('papam', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'papam'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '5')
        Configurator(self.p).set('settings', 'sampling_frequency', '100')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_wisdm_v1(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('wisdm_v1', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'wisdm_v1'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '10')
        Configurator(self.p).set('settings', 'sampling_frequency', '20')
        Configurator(self.p).set('settings', 'overlap', '0')
        Configurator(self.p).set('cleaning', 'low_cut', '9')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    def test_wisdm_v2(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('wisdm_v2', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'wisdm_v2'))
        Configurator(self.p).set('settings', 'data_treatment', 'segmentation')
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '10')
        Configurator(self.p).set('settings', 'sampling_frequency', '20')
        Configurator(self.p).set('settings', 'overlap', '0')
        Configurator(self.p).set('cleaning', 'low_cut', '6')
        bhar.get_baseline(['LDA', 'QDA'], [])
        self.assertEqual(True, True)

    # -----------------------------

    def test_daphnet_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('daphnet', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'daphnet'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '4')
        Configurator(self.p).set('settings', 'sampling_frequency', '64')
        Configurator(self.p).set('settings', 'overlap', '0.5')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_hhar_phone_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('hhar_phone', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'hhar_phone'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '2')
        Configurator(self.p).set('settings', 'sampling_frequency', '200')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_hhar_watch_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('hhar_watch', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'hhar_watch'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '2')
        Configurator(self.p).set('settings', 'sampling_frequency', '200')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_mhealth_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('mhealth', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'mhealth'))
        Configurator(self.p).set('dataset', 'header_type', 'dcp')
        Configurator(self.p).set('settings', 'time', '5')
        Configurator(self.p).set('settings', 'sampling_frequency', '50')
        Configurator(self.p).set('settings', 'overlap', '2.5')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_papam_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('papam', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'papam'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '5')
        Configurator(self.p).set('settings', 'sampling_frequency', '100')
        Configurator(self.p).set('settings', 'overlap', '1')
        Configurator(self.p).set('cleaning', 'low_cut', '20')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_wisdm_v1_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('wisdm_v1', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'wisdm_v1'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '10')
        Configurator(self.p).set('settings', 'sampling_frequency', '20')
        Configurator(self.p).set('settings', 'overlap', '0')
        Configurator(self.p).set('cleaning', 'low_cut', '9')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)

    def test_wisdm_v2_features(self):
        Configurator(self.p).set('settings', 'log_dir', 'logs')
        bhar = BHar(self.p)
        print('%s -> %s' % ('wisdm_v2', str(datetime.fromtimestamp(datetime.timestamp(datetime.now()))).split('.')[0]))
        Configurator(self.p).set('dataset', 'path', '%s%s' % (self.ds_path, 'wisdm_v2'))
        Configurator(self.p).set('dataset', 'header_type', 'tdcp')
        Configurator(self.p).set('settings', 'time', '10')
        Configurator(self.p).set('settings', 'sampling_frequency', '20')
        Configurator(self.p).set('settings', 'overlap', '0')
        Configurator(self.p).set('cleaning', 'low_cut', '6')
        bhar.get_baseline([], ['m1_acc'])
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
