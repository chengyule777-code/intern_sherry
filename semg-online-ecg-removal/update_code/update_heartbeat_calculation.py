"""
Updated R-R interval calculator for online ECG artifact removal.

The updated pipeline keeps physiological timing in seconds so changing the
sampling rate does not change the meaning of "time since last heartbeat".
"""

from __future__ import annotations

from collections import deque


class UpdatedHeartRateCalculator:
    """
    Track recent R-peaks and return the average R-R interval in seconds.

    The calculator receives one delayed peak marker per sample from the QRS
    detector. It measures elapsed time since the previous marker, stores recent
    peak-to-peak intervals, then returns their average for SWT gating.
    """

    def __init__(
        self,
        fs: int = 1024,
        initial_bpm: float = 80.0,
        max_peaks: int = 4,
    ):
        if not isinstance(fs, int):
            raise TypeError("fs must be an integer sampling rate in Hz.")
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if initial_bpm <= 0:
            raise ValueError("initial_bpm must be positive.")
        if max_peaks < 2:
            raise ValueError("max_peaks must be at least 2.")

        self.fs = fs
        self.time_since_last_peak_s = None
        self.recent_rr_interval_s = deque(maxlen=max_peaks - 1)
        self.average_rr_interval_s = 60.0 / initial_bpm

    def get_rr_interval_s(self, peak: int | bool, dt_s: float | None = None) -> float:
        """
        Process one peak marker and return the average R-R interval in seconds.

        :param peak: 1/True if the current sample is an R-peak marker, else 0/False.
        :param dt_s: elapsed seconds since the previous sample; defaults to 1 / fs.
        :return: average recent R-R interval in seconds.
        """
        dt_s = 1.0 / self.fs if dt_s is None else float(dt_s)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive when provided.")

        if self._is_peak(peak):
            # A new peak closes the previous R-R interval. The first peak only
            # starts the counter, so the initial interval remains in use until
            # the second peak provides a real measurement.
            if self.time_since_last_peak_s is not None:
                self.recent_rr_interval_s.append(self.time_since_last_peak_s)
                self.average_rr_interval_s = sum(self.recent_rr_interval_s) / len(self.recent_rr_interval_s)

            self.time_since_last_peak_s = 0.0

        if self.time_since_last_peak_s is not None:
            self.time_since_last_peak_s += dt_s

        return self.average_rr_interval_s

    def get_bpm(self) -> float:
        """Return the current R-R interval estimate converted to beats per minute."""
        return 60.0 / self.average_rr_interval_s

    @staticmethod
    def _is_peak(peak: int | bool) -> bool:
        if isinstance(peak, bool):
            return peak
        if isinstance(peak, int) and peak in (0, 1):
            return peak == 1
        raise ValueError("peak must be 0, 1, False, or True.")
