"""
MIT License

Copyright (c) 2024 Josefine Petrick, Institute for Electrical Engineering in Medicine - Universität zu Lübeck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from progress.bar import Bar

from code.find_better_ecg import BetterEcgDetector
from code.heartbeat_calculating import HeartRateCalculator
from code.online_envelope import EnvelopeCalculator
from code.online_qrs_detection import QrsDetector
from code.online_semg_ecg_removal_multi_channel import SwtEmgDenoise


class SEMGOnlineFilter:
    """
    Interface for using the multichannel online sEMG filter for your purposes.
    In multichannel measurements a 5-second initializing phase (duration can be passed as parameter) for identifying the
    best ECG for QRS detection happens before actual processing.
    Just call filter_sEMG_online(measured_values) for every new received samples.
    """

    def __init__(self, num_channels: int, delay: int, fs: int, envelope_window=256, initialization_time=5.0):
        """
        :param num_channels: the number of channels of the measured sEMG signal
        :param delay: the delay of the peak detection in samples (300 are recommended)
        :param fs: the sampling frequency of the sEMG signal
        :param initialization_time: the time in seconds the initialization for finding the best signal for QRS detection
         should take. 0 if no initialization wanted -> first channel will be used for ECG detection
        """
        self.num_channels = num_channels
        self.fs = fs
        self.num_initialization_samples = initialization_time * fs
        self.swt_denoising = SwtEmgDenoise(fs, delay, num_channels)
        self.qrs_detector = QrsDetector(delay)
        self.heart_rate_calculator = HeartRateCalculator(delay)
        self.envelope_calculators = [EnvelopeCalculator(False, envelope_window) for _ in range(num_channels)]
        self.received_measurement_values_cnt = 0
        self.best_ecg_signal = 0
        self.progress_bar = Bar('Identifying best ECG', max=self.num_initialization_samples, fill='#')

        if self.num_channels > 1:
            self.better_ecg_detector = BetterEcgDetector(num_channels, initialization_time, fs)

        if num_channels > 1 and initialization_time > 0:
            print("Identifying the best signal for QRS detection.")

    def filter_sEMG_online(self, measured_values: int or list) -> (list, list) or (int, int):
        """
        Does all the steps to remove the ECG from a respiratory sEMG measurement.
        In multichannel cases, the best signal for detecting the QRS regions is identified first.
        :param measured_values: the new packet of measured values (one sample of each channel) or the new measured value
        :return: for multichannel two lists: the denoised values and the envelopes; else: the two values of the channel.
         During initialization the function returns zero for all channels.
        """

        if self.num_channels > 1 and self.received_measurement_values_cnt < self.num_initialization_samples:
            self._get_best_ecg_signal(measured_values)
            self.received_measurement_values_cnt += 1
            return [0 for _ in range(self.num_channels)], [0 for _ in range(self.num_channels)]

        else:
            if self.num_channels > 1 and self.received_measurement_values_cnt == self.num_initialization_samples:
                self.progress_bar.finish()
                print("Initialization completed successfully.")
                self.received_measurement_values_cnt += 1

            if isinstance(measured_values, float):
                peak = self.qrs_detector.qrs_detection(measured_values)
                heart_rate = self.heart_rate_calculator.get_heartrate(peak)
                denoised_value = self.swt_denoising.swt_emg_denoising([measured_values], peak, heart_rate)[0]
                envelope_value = self.envelope_calculators[0].calculate_envelope(denoised_value)

                return denoised_value, envelope_value

            else:
                peak = self.qrs_detector.qrs_detection(measured_values[self.best_ecg_signal])
                heart_rate = self.heart_rate_calculator.get_heartrate(peak)
                denoised_values = self.swt_denoising.swt_emg_denoising(measured_values, peak, heart_rate)
                envelope_values = [self.envelope_calculators[i].calculate_envelope(denoised_values[i])
                                   for i in range(self.num_channels)]

                return denoised_values, envelope_values

    def _get_best_ecg_signal(self, measured_values: list):
        """
        Identifies and sets the best signal out of the given ones for QRS detection and updates the progress bar.
        :param measured_values: the new measured values
        """
        result_of_detection = self.better_ecg_detector.find_better_ecg(measured_values)
        if result_of_detection != -1:
            self.best_ecg_signal = result_of_detection
        self.progress_bar.next()
