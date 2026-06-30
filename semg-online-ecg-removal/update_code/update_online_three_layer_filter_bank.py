"""
Updated three-level online stationary wavelet filter bank.

The filter bank uses the Daubechies 2 wavelet and processes one sample at a
time. It is intentionally small and explicit because the denoising stage needs
to understand the three detail bands separately before reconstructing one EMG
sample.
"""

from __future__ import annotations

import numpy as np
import pywt

from update_code.update_online_filter import FIR_FILTER, OnlineFilter


def upsample_coefficients(coefficients, level: int) -> np.ndarray:
    """
    Insert zeros between wavelet coefficients for SWT level ``level``.

    Level 1 uses the original coefficients. Level 2 inserts one zero between
    taps, and level 3 inserts three zeros between taps.
    """
    if level < 1:
        raise ValueError("level must be at least 1.")

    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("coefficients must be a non-empty 1D sequence.")
    if level == 1:
        return values.copy()

    step = 2 ** (level - 1)
    sampled = np.zeros((len(values) - 1) * step + 1, dtype=float)
    sampled[::step] = values
    return sampled


class UpdatedFilterBank:
    """
    Three-level SWT/ISWT filter bank for streaming single-sample processing.

    ``swt`` decomposes the current sample into low/detail coefficients. ``iswt``
    reconstructs one delayed sample after the denoising stage edits those detail
    coefficients.
    """

    def __init__(self):
        wavelet = pywt.Wavelet("db2")
        dec_low, dec_high, rec_low, rec_high = wavelet.filter_bank

        self.decomposition_lowpass = self._build_level_filters(dec_low)
        self.decomposition_highpass = self._build_level_filters(dec_high)
        self.recomposition_lowpass = self._build_level_filters(rec_low)
        self.recomposition_highpass = self._build_level_filters(rec_high)

        # The inverse SWT branches have different filter delays. These buffers
        # align high-pass branches with the lower-frequency reconstruction path.
        self.level_1_delay = np.zeros(19, dtype=float)
        self.level_2_delay = np.zeros(13, dtype=float)
        self.level_1_delay_index = 0
        self.level_2_delay_index = 0

    def swt(self, input_value: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Return SWT coefficients ordered from low-frequency level 3 to level 1.
        """
        low_1 = self.decomposition_lowpass[0].filter(input_value)
        high_1 = self.decomposition_highpass[0].filter(input_value)

        low_2 = self.decomposition_lowpass[1].filter(low_1)
        high_2 = self.decomposition_highpass[1].filter(low_1)

        low_3 = self.decomposition_lowpass[2].filter(low_2)
        high_3 = self.decomposition_highpass[2].filter(low_2)

        return (low_3, high_3), (low_2, high_2), (low_1, high_1)

    def iswt(
        self,
        lowpass_3: float,
        highpass_3: float,
        highpass_2: float,
        highpass_1: float,
    ) -> float:
        """
        Reconstruct one sample from the low-frequency level 3 branch and details.
        """
        rec_low_3 = self.recomposition_lowpass[2].filter(lowpass_3)
        rec_high_3 = self.recomposition_highpass[2].filter(highpass_3)
        level_3_sum = rec_low_3 + rec_high_3

        rec_low_2 = self.recomposition_lowpass[1].filter(level_3_sum)
        rec_high_2 = self.recomposition_highpass[1].filter(highpass_2)
        delayed_high_2 = self._push_delay(self.level_2_delay, "level_2_delay_index", rec_high_2)
        level_2_sum = delayed_high_2 + rec_low_2 / 2

        rec_low_1 = self.recomposition_lowpass[0].filter(level_2_sum)
        rec_high_1 = self.recomposition_highpass[0].filter(highpass_1)
        delayed_high_1 = self._push_delay(self.level_1_delay, "level_1_delay_index", rec_high_1)

        return float((delayed_high_1 + rec_low_1 / 2) / 2)

    @staticmethod
    def _build_level_filters(coefficients) -> list[OnlineFilter]:
        return [
            OnlineFilter(FIR_FILTER, upsample_coefficients(coefficients, level))
            for level in (1, 2, 3)
        ]

    def _push_delay(self, buffer: np.ndarray, index_attr: str, value: float) -> float:
        index = getattr(self, index_attr)
        buffer[index] = value
        index = (index + 1) % len(buffer)
        setattr(self, index_attr, index)
        return float(buffer[index])
