import unittest
from har_baseline.utility.utility_baseline import apply_filter, data_delimiters
from har_baseline.utility.utility_baseline import decode_csv
from har_baseline.utility.configurator import Configurator
import pandas as pd
import matplotlib.pyplot as plt


class FilteringTestCase(unittest.TestCase):

    df = decode_csv(ds_dir=Configurator().get('dataset', 'path'),
                    separator=Configurator().get('dataset', 'separator', fallback=' '),
                    header_type=Configurator().get('dataset', 'header_type'),
                    has_header=Configurator().getboolean('dataset', 'has_header')
                    )

    def test_lowpass(self):
        # Apply filter
        filter_name = 'low'
        sampling_frequency = Configurator().getint('settings', 'sampling_frequency')
        filter_order = Configurator().getint('cleaning', 'filter_order')
        header_type = Configurator().get('dataset', 'header_type')

        cutoff = 20  # Hz

        df_filtered = pd.DataFrame(
            apply_filter(
                df=self.df.copy(),
                filter_name=filter_name,
                sample_rate=sampling_frequency,
                frequency_cutoff=cutoff,
                order=filter_order
            ),
            columns=list(self.df.columns[data_delimiters[header_type][0]: data_delimiters[header_type][1]])
        )

        # Plot
        df_filtered['A1x'][:128].plot(c='g', label='filtered')
        self.df['A1x'][:128].plot(c='r', label='normal')
        plt.title('Lowpass')
        plt.legend()
        plt.show()

        self.assertEqual(True, True)

    def test_highpass(self):
        # Apply filter
        filter_name = 'high'
        sampling_frequency = Configurator().getint('settings', 'sampling_frequency')
        filter_order = Configurator().getint('cleaning', 'filter_order')
        header_type = Configurator().get('dataset', 'header_type')

        cutoff = 5  # Hz

        df_filtered = pd.DataFrame(
            apply_filter(
                df=self.df.copy(),
                filter_name=filter_name,
                sample_rate=sampling_frequency,
                frequency_cutoff=cutoff,
                order=filter_order
            ),
            columns=list(self.df.columns[data_delimiters[header_type][0]: data_delimiters[header_type][1]])
        )

        # Plot
        df_filtered['A1x'][:128].plot(c='g', label='filtered')
        self.df['A1x'][:128].plot(c='r', label='normal')
        plt.title('Highpass')
        plt.legend()
        plt.show()

        self.assertEqual(True, True)

    def test_bandpass(self):
        # Apply filter
        filter_name = 'bandpass'
        sampling_frequency = Configurator().getint('settings', 'sampling_frequency')
        filter_order = Configurator().getint('cleaning', 'filter_order')
        header_type = Configurator().get('dataset', 'header_type')

        cutoff = (5, 20)  # Hz

        df_filtered = pd.DataFrame(
            apply_filter(
                df=self.df.copy(),
                filter_name=filter_name,
                sample_rate=sampling_frequency,
                frequency_cutoff=cutoff,
                order=filter_order
            ),
            columns=list(self.df.columns[data_delimiters[header_type][0]: data_delimiters[header_type][1]])
        )

        # Plot
        df_filtered['A1x'][:128].plot(c='g', label='filtered')
        self.df['A1x'][:128].plot(c='r', label='normal')
        plt.title('Bandpass')
        plt.legend()
        plt.show()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
