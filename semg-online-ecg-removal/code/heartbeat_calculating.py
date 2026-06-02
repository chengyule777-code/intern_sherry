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


class HeartRateCalculator:
    """
    calculates and saves the heartrate based on the last four detected heartbeats
    """

    def __init__(self, delay: int):
        """
        initializes a new heartrate calculator with a start rate of 768, which corresponds to about the average resting
        heart rate of an adult
        :param delay: how many samples the last detection of a heartbeat is delayed
        """
        assert isinstance(delay, int)
        assert 0 <= delay <= 500  # if the delay is bigger than the heartrate, the algorithm doesn't work
        self.delay = delay
        self.last_beats = np.zeros(0)
        self.heart_rate = 768  # 60*1024 (samples/minute) / 80 (average bpm)
        self.started = True  # for removing the first peak, which is not really a peak

    def update_heart_rate(self):
        """
        calculates the approximate heart rate based on the last four detected heart beats and saves it
        :return: nothing
        """
        if len(self.last_beats) > 0 and self.started:
            self.last_beats = []
            self.started = False

        if len(self.last_beats) > 4:
            self.last_beats = self.last_beats[:4]

        if len(self.last_beats) > 1:
            sum_for_average = 0
            for i in range(len(self.last_beats) - 1):
                sum_for_average += self.last_beats[i + 1] - self.last_beats[i]

            self.heart_rate = int(sum_for_average / (len(self.last_beats) - 1))

    def get_next_beat(self, value: int):
        """
        calculates the index of the next beat
        :param value: 1, if index 0 is a beat, else 0
        :return: the approximate index of the next beat
        """
        if value == 1:
            self.last_beats = np.append(0, self.last_beats)
            self.update_heart_rate()
        elif len(self.last_beats) == 0:
            return False

        last_beat_real_idx = self.last_beats[0] + self.delay

        for i in range(len(self.last_beats)):
            self.last_beats[i] = self.last_beats[i] + 1

        return last_beat_real_idx - self.heart_rate

    def get_heartrate(self, peak: int):
        """
        :param peak: whether the value at index 0 is a peak (1 -> is peak)
        :return: the last heartrate calculated
        """
        if peak == 1:
            self.last_beats = np.append(0, self.last_beats)
            self.update_heart_rate()
        for i in range(len(self.last_beats)):
            self.last_beats[i] = self.last_beats[i] + 1

        return self.heart_rate
