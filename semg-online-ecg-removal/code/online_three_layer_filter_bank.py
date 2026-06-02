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


def get_sampled_coefficients(coefficients: list):
    """
    returns the coefficients for the 2nd and 3rd level of the stationary wavelet transform,
    which are the coefficients of the 1st level but padded with zeros in between
    :param coefficients: the coefficients of the 1st level
    :return: the coefficients of the 2nd and 3rd level
    """
    coefficients2 = np.zeros(len(coefficients) * 2 - 1)
    for i in range(0, len(coefficients)):
        coefficients2[i * 2] = coefficients[i]

    coefficients3 = np.zeros(len(coefficients2) * 2 - 1)
    for i in range(0, len(coefficients2)):
        coefficients3[i * 2] = coefficients2[i]

    return coefficients2, coefficients3


class FilterBank:
    """
    A three level filter bank for a stationary wavelet transform using ring buffers and the Daubechies2 wavelet with a
    delay of 21 samples
    """

    def __init__(self):
        """
        creates all the coefficients and filters needed for the swt
        """
        self.buffer1 = np.zeros(19)
        self.buffer2 = np.zeros(13)

        # for ring-buffers
        self.pointer_buffer1 = 0
        self.pointer_buffer2 = 0

        self.ret = 0

        # filter coefficients
        self.decomposition_lowpass_1 = pywt.Wavelet('db2').filter_bank[0]
        self.decomposition_lowpass_2, self.decomposition_lowpass_3 = get_sampled_coefficients(
            self.decomposition_lowpass_1)

        self.decomposition_highpass_1 = pywt.Wavelet('db2').filter_bank[1]
        self.decomposition_highpass_2, self.decomposition_highpass_3 = get_sampled_coefficients(
            self.decomposition_highpass_1)

        self.recomposition_lowpass_1 = pywt.Wavelet('db2').filter_bank[2]
        self.recomposition_lowpass_2, self.recomposition_lowpass_3 = get_sampled_coefficients(
            self.recomposition_lowpass_1)

        self.recomposition_highpass_1 = pywt.Wavelet('db2').filter_bank[3]
        self.recomposition_highpass_2, self.recomposition_highpass_3 = get_sampled_coefficients(
            self.recomposition_highpass_1)

        # filters
        self.decomposition_filter_low_1 = OnlineFilter(0, self.decomposition_lowpass_1)
        self.decomposition_filter_high_1 = OnlineFilter(0, self.decomposition_highpass_1)

        self.decomposition_filter_low_2 = OnlineFilter(0, self.decomposition_lowpass_2)
        self.decomposition_filter_high_2 = OnlineFilter(0, self.decomposition_highpass_2)

        self.decomposition_filter_low_3 = OnlineFilter(0, self.decomposition_lowpass_3)
        self.decomposition_filter_high_3 = OnlineFilter(0, self.decomposition_highpass_3)

        self.recomposition_filter_low_1 = OnlineFilter(0, self.recomposition_lowpass_1)
        self.recomposition_filter_high_1 = OnlineFilter(0, self.recomposition_highpass_1)

        self.recomposition_filter_low_2 = OnlineFilter(0, self.recomposition_lowpass_2)
        self.recomposition_filter_high_2 = OnlineFilter(0, self.recomposition_highpass_2)

        self.recomposition_filter_low_3 = OnlineFilter(0, self.recomposition_lowpass_3)
        self.recomposition_filter_high_3 = OnlineFilter(0, self.recomposition_highpass_3)

    def swt(self, input_value: float):
        """
        stationary wavelet transform of the input value
        :param input_value: the value, which should be filtered next
        :return: the filtered values for each of the layers of the filter bank (each with a different delay)
        """
        y_lowpass_1 = self.decomposition_filter_low_1.filter(input_value)
        y_highpass_1 = self.decomposition_filter_high_1.filter(input_value)

        y_lowpass_2 = self.decomposition_filter_low_2.filter(y_lowpass_1)
        y_highpass_2 = self.decomposition_filter_high_2.filter(y_lowpass_1)

        y_lowpass_3 = self.decomposition_filter_low_3.filter(y_lowpass_2)
        y_highpass_3 = self.decomposition_filter_high_3.filter(y_lowpass_2)

        return (y_lowpass_3, y_highpass_3), (y_lowpass_2, y_highpass_2), (y_lowpass_1, y_highpass_1)

    def iswt(self, y_lowpass_3: float, y_highpass_3: float, y_highpass_2: float, y_highpass_1: float):
        """
        inverse stationary wavelet transform
        :param y_lowpass_3: the filtered value from the lowpass of level 3
        :param y_highpass_3: the filtered value from the highpass of level 3
        :param y_highpass_2: the filtered value from the highpass of level 2
        :param y_highpass_1: the filtered value from the highpass of level 1
        :return: the final result of the stationary wavelet transform with a delay of 21 samples
        """
        r_lowpass_3 = self.recomposition_filter_low_3.filter(y_lowpass_3)
        r_highpass_3 = self.recomposition_filter_high_3.filter(y_highpass_3)

        r_level_3_added = r_lowpass_3 + r_highpass_3
        r_lowpass_2 = self.recomposition_filter_low_2.filter(r_level_3_added)
        r_highpass_2 = self.recomposition_filter_high_2.filter(y_highpass_2)

        self.buffer2[self.pointer_buffer2] = r_highpass_2
        if self.pointer_buffer2 == self.buffer2.size - 1:
            self.pointer_buffer2 = 0
        else:
            self.pointer_buffer2 += 1

        r_level_2_added = self.buffer2[self.pointer_buffer2 % self.buffer2.size] + r_lowpass_2 / 2

        r_lowpass_1 = self.recomposition_filter_low_1.filter(r_level_2_added)
        r_highpass_1 = self.recomposition_filter_high_1.filter(y_highpass_1)

        self.buffer1[self.pointer_buffer1] = r_highpass_1
        if self.pointer_buffer1 == self.buffer1.size - 1:
            self.pointer_buffer1 = 0
        else:
            self.pointer_buffer1 += 1

        return (self.buffer1[self.pointer_buffer1 % self.buffer1.size] + r_lowpass_1 / 2) / 2
