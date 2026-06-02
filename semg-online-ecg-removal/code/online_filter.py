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


class OnlineFilter:
    """
    Creates a filter and ring buffer(s) of the minimal required size to filter inputs with the given filter coefficients
    gets an input of size 1 and creates an output of size 1
    """

    def __init__(self, type_of_filter: int, coefficients_in: list, coefficients_out: list = None):
        """
        initializes a new filter with a type and the coefficients
        :param type_of_filter: 0 -> FIR, 1 -> IIR
        :param coefficients_in: nominator of an IIR filter or the coefficients of a FIR filter
        :param coefficients_out: denominator of an IIR filter (with a norming factor at index 0) or none
        """

        self.buffer_input = np.zeros(len(coefficients_in))

        self.type_of_filter = type_of_filter

        self.norm = 1

        if self.type_of_filter == 1:
            self.buffer_output = np.zeros(len(coefficients_out) - 1)
            self.coefficients_out = coefficients_out[1:len(coefficients_out)]
            self.norm = coefficients_out[0]

        self.coefficients_in = coefficients_in
        self.ret = 0

        # for ring-buffers
        self.input_pointer = 0
        self.output_pointer = 0

        self.cnt = 1

    def filter(self, input_value: float):
        """
        filters the given value as the next value of the input
        :param input_value: the next value, which should be filtered
        :return: the next normalized value (filtered value from delay samples ago, where delay = len(coefficients_in)-1)
        """
        self.ret = 0
        self.buffer_input[self.input_pointer] = input_value
        if self.type_of_filter == 0 and self.cnt < len(self.coefficients_in) - 1:  # periodic extension as in matlab swt
            n = self.input_pointer + 1
            i = 0
            while n < len(self.coefficients_in):
                self.buffer_input[n] = self.buffer_input[i]
                n += 1
                i += 1

        if self.input_pointer == len(self.coefficients_in) - 1:
            self.input_pointer = 0
        else:
            self.input_pointer += 1
        for idx, i in enumerate(self.coefficients_in):
            self.ret += i * self.buffer_input[(self.input_pointer - 1 - idx) % len(self.coefficients_in)]
        if self.type_of_filter == 1:
            for idx, i in enumerate(self.coefficients_out):
                self.ret -= i * self.buffer_output[(self.output_pointer - 1 - idx) % len(self.coefficients_out)]
                self.ret = self.ret
            self.buffer_output[self.output_pointer] = self.ret / self.norm
            if self.output_pointer == len(self.coefficients_out) - 1:
                self.output_pointer = 0
            else:
                self.output_pointer += 1
        else:
            if self.output_pointer == len(self.coefficients_in):
                self.output_pointer = 0
            else:
                self.output_pointer += 1

        self.cnt += 1
        return self.ret / self.norm
