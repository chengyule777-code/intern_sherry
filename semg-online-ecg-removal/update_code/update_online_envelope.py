"""
Updated online RMS envelope calculator.

The original envelope stage used a rolling average of absolute values. This
version uses root-mean-square (RMS), which better reflects signal energy while
remaining cheap enough for sample-by-sample processing.
"""

from __future__ import annotations

from collections import deque
from math import sqrt


class UpdatedEnvelopeCalculator:
    """
    Calculate a streaming RMS envelope over a fixed-duration window.

    ``window_s`` is converted to samples once because the rolling buffer needs
    an integer capacity. After that, each sample updates the running square sum
    in O(1), then returns ``sqrt(mean(square))`` for the current window.
    """

    def __init__(self, fs: int, window_s: float = 0.25):
        if not isinstance(fs, int):
            raise TypeError("fs must be an integer sampling rate in Hz.")
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if window_s <= 0:
            raise ValueError("window_s must be positive.")

        self.fs = fs
        self.window_s = float(window_s)
        self.window_samples = max(1, int(round(window_s * fs)))
        self.square_buffer = deque()
        self.square_sum = 0.0

    def calculate_envelope(self, value: float) -> float:
        """
        Process one sample and return the RMS envelope for the current window.
        """
        squared_value = float(value) ** 2

        if len(self.square_buffer) == self.window_samples:
            self.square_sum -= self.square_buffer.pop()

        self.square_buffer.appendleft(squared_value)
        self.square_sum += squared_value

        return sqrt(self.square_sum / len(self.square_buffer))
