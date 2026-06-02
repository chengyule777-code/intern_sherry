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

import numpy as np
import pywt

from code.online_filter import OnlineFilter


def get_sampled_coefficients(coefficients: list, num_levels: int):
    """
    returns the coefficients for the 2nd and 3rd level of the stationary wavelet transform,
    which are the coefficients of the 1st level but padded with zeros in between
    :param coefficients: the coefficients of the 1st level
    :param num_levels: the number of levels of the filter bank
    :return: the coefficients of the higher levels
    """
    all_coefficients = [np.zeros(len(coefficients) * 2 - 1)]
    for i in range(len(coefficients)):
        all_coefficients[0][i * 2] = coefficients[i]

    for i in range(num_levels - 2):
        all_coefficients.append(np.zeros(len(all_coefficients[i]) * 2 - 1))
        for j in range(len(all_coefficients[i])):
            all_coefficients[i + 1][j * 2] = all_coefficients[i][j]

    return all_coefficients


class FilterBank:
    """
    A three level filter bank for a stationary wavelet transform using ring buffers and the Daubechies2 wavelet with a
    delay of 21 samples
    """

    def __init__(self, num_levels: int, daubechies_filter_order: int):
        """
        :param num_levels: the number of levels of the filter bank
        creates all the coefficients and filters needed for the swt
        """
        self.num_levels = num_levels

        filter_delays = [(daubechies_filter_order-1) * (2 ** i) for i in range(num_levels)]

        self.delay = sum(filter_delays)

        buffer_lengths = []
        buffer_length = 0
        for i in filter_delays[::-1]:
            buffer_length += i
            buffer_lengths.append(buffer_length + 1)

        # buffers for delaying, sorted: delay for level one is at the last position
        self.buffers = [np.zeros(i) for i in buffer_lengths]

        # for ring-buffers
        self.buffer_pointers = [0 for _ in range(num_levels - 1)]

        self.ret = 0

        daubechies_filter = 'db' + str(int(daubechies_filter_order/2))
        # filter coefficients
        self.decomposition_lowpass = [pywt.Wavelet(daubechies_filter).filter_bank[0]]
        for coefficients in get_sampled_coefficients(self.decomposition_lowpass[0], num_levels):
            self.decomposition_lowpass.append(coefficients)

        self.decomposition_highpass = [pywt.Wavelet(daubechies_filter).filter_bank[1]]
        for coefficients in get_sampled_coefficients(self.decomposition_highpass[0], num_levels):
            self.decomposition_highpass.append(coefficients)

        self.recomposition_lowpass = [pywt.Wavelet(daubechies_filter).filter_bank[2]]
        for coefficients in get_sampled_coefficients(self.recomposition_lowpass[0], num_levels):
            self.recomposition_lowpass.append(coefficients)

        self.recomposition_highpass = [pywt.Wavelet(daubechies_filter).filter_bank[3]]
        for coefficients in get_sampled_coefficients(self.recomposition_highpass[0], num_levels):
            self.recomposition_highpass.append(coefficients)

        # filters
        self.decomposition_filter_low = [OnlineFilter(0, self.decomposition_lowpass[i])
                                         for i in range(len(self.decomposition_lowpass))]

        self.decomposition_filter_high = [OnlineFilter(0, self.decomposition_highpass[i])
                                          for i in range(len(self.decomposition_highpass))]

        self.recomposition_filter_low = [OnlineFilter(0, self.recomposition_lowpass[i])
                                         for i in range(len(self.recomposition_lowpass))]

        self.recomposition_filter_high = [OnlineFilter(0, self.recomposition_highpass[i])
                                          for i in range(len(self.recomposition_highpass))]

    def swt(self, input_value: float):
        """
        stationary wavelet transform of the input value
        :param input_value: the value, which should be filtered next
        :return: the filtered values for each of the layers of the filter bank (each with a different delay)
        """
        decomposed_coefficients = [(self.decomposition_filter_low[0].filter(input_value),
                                    self.decomposition_filter_high[0].filter(input_value))]

        for i in range(self.num_levels - 1):
            next_decomposed_coefficients = (self.decomposition_filter_low[i + 1].filter(decomposed_coefficients[i][0]),
                                            self.decomposition_filter_high[i + 1].filter(decomposed_coefficients[i][0]))
            decomposed_coefficients.append(next_decomposed_coefficients)

        return decomposed_coefficients[::-1]

    def iswt(self, y_lowpass: float, y_highpasses: list):
        """
        inverse stationary wavelet transform
        :param y_lowpass: the filtered value from the lowpass of the lowest level (high level count)
        :param y_highpasses: the filtered values of all highpass coefficients sorted ascending, lowest level first
        :return: the final result of the stationary wavelet transform with a delay
        """

        result_lower_level = (self.recomposition_filter_low[-1].filter(y_lowpass) + self.recomposition_filter_high[-1]
                              .filter(y_highpasses[0]))

        for i, highpass in enumerate(y_highpasses[1:]):
            r_lowpass = self.recomposition_filter_low[- (i + 2)].filter(result_lower_level)
            r_highpass = self.recomposition_filter_high[- (i + 2)].filter(highpass)
            self.buffers[i][self.buffer_pointers[i]] = r_highpass
            self.buffer_pointers[i] = (self.buffer_pointers[i] + 1) % self.buffers[i].size

            result_lower_level = (self.buffers[i][self.buffer_pointers[i]] + r_lowpass / 2)

        return result_lower_level / 2

    def get_delay(self):
        return self.delay
