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

from collections import Counter
from statistics import mean

import numpy as np
from scipy import signal

from code import heartbeat_calculating
from code import online_filter
from code import online_qrs_detection


class BetterEcgDetector:
    """
    This class finds the best ECG for QRS-detection out of several ECGs by looking at the height of the peaks and,
    whether mainly positive or negative peaks are detected
    """

    def __init__(self, num_ECGs: int, duration: float, fs: int):
        """
        Initializes all objects needed in order to detect peaks online in a signal, a baseline filter and arrays to
        store important information
        :param num_ECGs: the numer of ECGs, from which the best shall be chosen
        :param duration: how long should be searched for the best ECG in seconds -> 5 seconds are recommended
        :param fs: the sampling frequency of the signals
        """
        assert num_ECGs > 1, "More than one ECG is needed for a comparison"
        assert duration > 2.5, "The duration should be at least 2.5 seconds."
        # at least two heart beats are needed in order to have a good comparison

        self.duration = duration * fs  # calculates the number of samples for the given duration
        self.num_ecgs = num_ECGs
        self.ecg_signals = [np.array([]) for _ in range(num_ECGs)]  # saves all ECG signals

        self.ecg_peak_directions = [Counter({1: 0, -1: 0}) for _ in range(num_ECGs)]
        # for each ECG counts numbers of positive and negative peaks: 1 -> positive peak, -1 -> negative peak

        self.ecg_peak_heights = [np.array([]) for _ in range(num_ECGs)]
        # saves the heights of the detected peaks for each signal

        self.peak_detectors = [online_qrs_detection.QrsDetector(300) for _ in range(num_ECGs)]
        self.heart_rate_calculator = heartbeat_calculating.HeartRateCalculator(300)

        self.maxima = np.array([0 for _ in range(num_ECGs)])
        order = 2
        baseline_filter_threshold = 1
        b_baseline, a_baseline = signal.butter(order, baseline_filter_threshold, 'hp', analog=False, fs=1024,
                                               output='ba')
        self.baseline_filter = online_filter.OnlineFilter(1, b_baseline, a_baseline)

    def find_better_ecg(self, new_ecg_values: list):
        """
        The method to find the best ECG within the given duration
        :param new_ecg_values: An array or a list containing the new ECG values ordered by channel
        :return: -1 if the detection is not finished yet, otherwise the index of the best ECG
        """
        for i in range(self.num_ecgs):
            peak_detector = self.peak_detectors[i]
            value = self.baseline_filter.filter(new_ecg_values[i])
            self.ecg_signals[i] = np.append(self.ecg_signals[i], value)
            peak = peak_detector.qrs_detection(value)
            if peak == 1:
                if len(self.ecg_signals[i]) > 300:  # don't look at the first 300 values because there might be noise
                    self.ecg_peak_heights[i] = np.append(self.ecg_peak_heights[i], abs(value))
                    if value > 0:
                        self.ecg_peak_directions[i][1] += 1
                    else:
                        self.ecg_peak_directions[i][-1] += 1

        if len(self.ecg_signals[0]) < self.duration:
            return -1  # detection is not finished

        else:  # last iteration
            means = [mean(self.ecg_peak_heights[i]) for i in range(self.num_ecgs)]
            mean_first, mean_second = self.get_first_and_second_max_ecg(means)

            max_counts = [self.ecg_peak_directions[i].most_common(1)[0][1] for i in range(self.num_ecgs)]
            count_first, count_second = self.get_first_and_second_max_ecg(max_counts)

            good_count = Counter()
            for i in range(self.num_ecgs):
                good_count[i] = 0

            # evaluation
            good_count[mean_first] += 2
            good_count[mean_second] += 1
            good_count[count_first] += 2.1  # regularity is evaluated higher than the height of the peaks in a draw
            good_count[count_second] += 1

            return good_count.most_common(1)[0][0]

    def get_first_and_second_max_ecg(self, values: list):
        """
        Finds the indices of the highest and the second-highest values in an array. This method does not consider
        equal values because this case is assumed to be very unlikely
        :param values: An array with the values to compare.
        :return: The indices of the highest and the second-highest values in an array.
        """
        first = np.argmax(values)

        if first == self.num_ecgs - 1:
            second = np.argmax(values[:-1])
        elif first == 0:
            second = np.argmax(values[1:]) + 1
        else:
            position = np.argmax(max(values[:first]), max(values[first + 1:]))
            if position == 0:
                second = np.argmax(values[:first])
            else:
                second = first + 1 + np.argmax(values[first + 1:])

        return first, second
