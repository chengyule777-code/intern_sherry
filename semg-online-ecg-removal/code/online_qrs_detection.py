"""
Original work:
BSD 3-Clause License

Copyright (c) 2021, Reza Sameni
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Modified work:
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

Algorithm using the Pan-Tomkins qrs-detection-algorithm for detecting qrs-complexes of cardiac artifacts in sEMG-
measurements:
J. Pan and W. J. Tompkins (1985), A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering,
BME-32(3), 230-236.

Modification from Eike Petersen:
Petersen, E., Sauer, J., Graßhoff, J., and Rostalski, P.(2022). Removing Cardiac Artifacts From Single-Channel
    Respiratory Electromyograms. IEEE Access, 8, 30905-30917.
"""
import numpy as np
from scipy import signal

from code import online_filter


class QrsDetector:
    """
    Test, whether the value at index 'look_at_value' is a QRS peak
    return 1, if there is a peak, otherwise 0
    """

    def __init__(self, delay: int):
        assert isinstance(delay, int)
        assert 280 <= delay <= 400  # tested to work between these numbers of samples, recommended are 300 samples
        order = 2
        baseline_filter_threshold = 1
        b_baseline, a_baseline = signal.butter(order, baseline_filter_threshold, 'hp', analog=False, fs=1024,
                                               output='ba')
        self.baseline_filter = online_filter.OnlineFilter(1, b_baseline, a_baseline)
        self.window_len = 0.15  # 150 ms
        self.window_length = 800  # samples
        self.signal_length = delay
        self.look_at_value = delay - 1  # index
        self.fs = 1024
        self.thr = 0.3
        self.fp1 = 8.0
        self.fp2 = 20.0  # cut-off frequency for first low pass
        self.buffered_signal = np.zeros(self.signal_length)
        self.filtered_signal = np.zeros(0)
        self.squared_signal = np.zeros(self.signal_length)
        self.mean_signal = np.zeros(0)
        self.sum_for_mean = 0
        b, a = signal.butter(4, [self.fp1, self.fp2], 'bp', analog=False, fs=1024, output='ba')
        self.bandpass = online_filter.OnlineFilter(1, b, a)
        self.box_last = 0
        self.up_idx = 0
        self.down_idx = 0
        self.found_peaks = np.zeros(0)
        self.peak_max_values = [0, 0]
        self.peak_min_values = [0, 0]
        self.peak_values = [0, 0]
        self.last_max = 0
        self.pre_last_max = 0
        self.pre_pre_last_max = 0
        self.pre_pre_pre_last_max = 0
        self.max_cnt = 0
        self.win_len = 153  # ~ 0.15 * 1024

    def qrs_detection(self, x: float):
        """
        :param x: measured value from an EMG signal with ecg artifacts
        :return: 1, if the idx look_at_value is a peak, otherwise 0
        """
        assert isinstance(x, float), "the input values of the sEMG signal must be floats"

        x = self.baseline_filter.filter(x)

        self.found_peaks += 1

        self.buffered_signal = np.append(x, self.buffered_signal[:-1])

        # bandpass from Pan-Tompkins:
        # "to reduce muscle noise, 60Hz interference, baseline wander and T-Wave interference" (p.232)
        # smoothing not used (yet), so no delay
        self.filtered_signal = np.append(self.bandpass.filter(self.buffered_signal[0]), self.filtered_signal[:])
        if len(self.filtered_signal) > 2:
            self.filtered_signal = self.filtered_signal[:-1]

        # adjust shape by differentiation, squaring and smoothing with moving average (Pan-Tomkins algorithm)
        if len(self.filtered_signal) > 1:
            value = (self.filtered_signal[1] - self.filtered_signal[0]) ** 2
            self.squared_signal = np.append([value, value], self.squared_signal[1:self.win_len - 1])
            # first item is nan after differentiating -> double it

        self.sum_for_mean += self.squared_signal[0]
        if len(self.buffered_signal) > self.win_len:
            self.sum_for_mean -= self.squared_signal[-1]
        mean_value = self.sum_for_mean / len(self.squared_signal)

        self.mean_signal = np.append(
            np.ones(int(self.win_len / 2)) * mean_value,  # centred mean
            self.mean_signal[int(self.win_len / 2) - 1:])

        if len(self.mean_signal) > self.signal_length:
            self.mean_signal = self.mean_signal[:-1]

        # threshold calculation using smoothed maximum of the signal (Petersen)
        max_now = max(self.mean_signal)
        if self.pre_pre_pre_last_max > 1000 * max(self.last_max, self.pre_last_max, self.pre_pre_last_max):
            self.pre_pre_pre_last_max = self.last_max

        # it is the first max
        if self.pre_last_max == 0 and self.pre_pre_last_max == 0:
            v_max = max(self.last_max, max_now, self.pre_last_max, self.pre_pre_last_max, self.pre_pre_pre_last_max)
            if self.max_cnt >= self.signal_length:
                self.pre_pre_pre_last_max = self.pre_pre_last_max
                self.pre_pre_last_max = self.pre_last_max
                self.pre_last_max = self.last_max
                self.last_max = max_now
                self.max_cnt = 0

        # max not considered if the considered value is significantly higher than the last detected peaks -> artifact
        elif max_now > 10 * self.last_max and \
                max_now > 15 * self.pre_last_max and \
                max_now > 15 * self.pre_pre_last_max and \
                max_now > 15 * self.pre_pre_pre_last_max and \
                self.last_max > 0 and self.pre_last_max > 0 and self.pre_pre_last_max > 0:
            v_max = max(self.last_max, self.pre_last_max, self.pre_pre_last_max)

        # usual case
        else:
            v_max = max(self.last_max, max_now, self.pre_last_max, self.pre_pre_last_max, self.pre_pre_pre_last_max)
            if self.max_cnt >= self.signal_length:
                self.pre_pre_pre_last_max = self.pre_pre_last_max
                self.pre_pre_last_max = self.pre_last_max
                self.pre_last_max = self.last_max
                self.last_max = max_now
                self.max_cnt = 0
        self.max_cnt += 1

        # actual peak detection
        # (delay of 64 because of mean )
        # found a part of a box
        if self.mean_signal[0] > self.thr * v_max:
            # found the start of a box
            if self.box_last == 0:
                self.up_idx = int(self.win_len / 2)
                self.box_last = 1
                self.peak_max_values = [self.buffered_signal[int(self.win_len / 2)], 0]
                self.peak_min_values = [self.buffered_signal[int(self.win_len / 2)], 0]
            # else just a part of the box
            else:
                if self.buffered_signal[int(self.win_len / 2)] > self.peak_max_values[0]:
                    self.peak_max_values = [self.buffered_signal[int(self.win_len / 2)], int(self.win_len / 2)]
                if self.buffered_signal[int(self.win_len / 2)] < self.peak_min_values[0]:
                    self.peak_min_values = [self.buffered_signal[int(self.win_len / 2)], int(self.win_len / 2)]

        # found that here is no box
        else:
            # found the end of a box
            if self.box_last == 1:
                self.down_idx = int(self.win_len / 2)
                self.box_last = 0
            # else no box here

        if self.down_idx == int(self.win_len / 2) and 700 > self.up_idx - self.down_idx >= 130:
            # only allow physiologically possible peaks
            if self.peak_max_values[0] >= abs(self.peak_min_values[0]):
                self.peak_values = self.peak_max_values
            else:
                self.peak_values = self.peak_min_values
            if len(self.found_peaks) > 0 and self.found_peaks[0] - self.peak_values[1] > self.fs * 0.5 or len(
                    self.found_peaks) == 0:
                self.found_peaks = np.append(self.peak_values[1], self.found_peaks)
                self.up_idx = 0
                self.down_idx = 0

        self.up_idx += 1
        self.down_idx += 1
        self.peak_max_values[1] += 1
        self.peak_min_values[1] += 1

        if len(self.found_peaks) > 0 and self.found_peaks[-1] > self.look_at_value:
            self.found_peaks = self.found_peaks[:-1]

        if len(self.found_peaks) > 0 and self.found_peaks[-1] == self.look_at_value:
            self.found_peaks = self.found_peaks[:-1]
            return 1

        else:
            return 0
