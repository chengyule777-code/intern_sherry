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

from collections import deque


class EnvelopeCalculator:
    """
    Calculates the envelope of a signal online
    """

    def __init__(self, delayed: bool, window_len: int):
        """
        :param delayed: True: a centred mean should be calculated -> delay of window_len/2,
                        False: no delay but not centred
        :param window_len: how many samples should be considered
        """
        self.delayed = delayed
        self.buffer = deque()
        self.window_len = window_len
        self.sum = 0
        self.n = 0

    def calculate_envelope(self, value: float):
        """
        calculates the envelope (mean) of the new sample (not delayed) or of the sample at index -window_len/2
        :param value: the value of the new sample
        :return: the mean of the window
        """
        if self.delayed:
            if self.n < self.window_len:
                self.n += 1
            else:
                self.sum -= self.buffer.pop()
        else:
            if self.n < self.window_len / 2:
                self.n += 1
            else:
                self.sum -= self.buffer.pop()

        self.buffer.appendleft(abs(value))
        self.sum += abs(value)

        return self.sum / self.n
