"""
Updated R-R interval calculator for online ECG artifact removal.

``get_heartrate`` keeps the historical public name used by the rest of the
project, but it returns an average R-R interval in samples. The SWT denoising
stage uses that sample interval to predict the next ECG gate.
"""

from __future__ import annotations

from collections import deque


class UpdatedHeartRateCalculator:
    """
    Track recent R-peaks and return the average R-R interval in samples.

    The calculator receives one delayed peak marker per sample from the QRS
    detector. It counts samples since the previous R-peak, stores recent
    peak-to-peak intervals, then returns their average for SWT gating.
    """

    def __init__(
        self,
        delay: int,
        fs: int = 1024,
        initial_bpm: float = 80.0,
        max_peaks: int = 4,
    ):
        if not isinstance(delay, int):
            raise TypeError("delay must be an integer number of samples.")
        if delay < 0:
            raise ValueError("delay must be non-negative.")
        if not isinstance(fs, int):
            raise TypeError("fs must be an integer sampling rate in Hz.")
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if initial_bpm <= 0:
            raise ValueError("initial_bpm must be positive.")
        if max_peaks < 2:
            raise ValueError("max_peaks must be at least 2.")

        self.delay = delay
        self.fs = fs
        self.samples_since_last_peak = None
        self.recent_rr_interval_samples = deque(maxlen=max_peaks - 1)
        self.average_rr_interval_samples = int(round(60 * fs / initial_bpm))

    def get_heartrate(self, peak: int | bool) -> int:
        """
        Process one peak marker and return the average R-R interval in samples.

        :param peak: 1/True if the current sample is an R-peak marker, else 0/False.
        :return: average recent R-R interval in samples.
        """
        if self._is_peak(peak):
            # A new peak closes the previous R-R interval. The first peak only
            # starts the counter, so the initial interval remains in use until
            # the second peak provides a real measurement.
            if self.samples_since_last_peak is not None:
                self.recent_rr_interval_samples.append(self.samples_since_last_peak)
                self.average_rr_interval_samples = int(
                    round(sum(self.recent_rr_interval_samples) / len(self.recent_rr_interval_samples))
                )

            self.samples_since_last_peak = 0

        if self.samples_since_last_peak is not None:
            self.samples_since_last_peak += 1

        return self.average_rr_interval_samples

    def get_bpm(self) -> float:
        """Return the current interval estimate converted to beats per minute."""
        return 60 * self.fs / self.average_rr_interval_samples

    @staticmethod
    def _is_peak(peak: int | bool) -> bool:
        if isinstance(peak, bool):
            return peak
        if isinstance(peak, int) and peak in (0, 1):
            return peak == 1
        raise ValueError("peak must be 0, 1, False, or True.")
