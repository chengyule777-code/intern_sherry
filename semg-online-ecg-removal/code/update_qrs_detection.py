"""
Efficient streaming QRS detection for future sEMG ECG-removal work.

This module keeps the same public shape as the original detector:
feed one sample into ``qrs_detection`` and receive ``1`` when a delayed
QRS peak is reported, otherwise ``0``.
"""

from collections import deque

from scipy import signal

from code import online_filter


class UpdatedQrsDetector:
    """
    Detect ECG QRS peaks from an online sEMG stream.

    The implementation follows the same broad Pan-Tompkins flow as
    ``QrsDetector`` but avoids shifting NumPy arrays on every sample.
    """

    def __init__(self, delay: int, fs: int = 1024):
        assert isinstance(delay, int)
        assert 280 <= delay <= 400

        self.delay = delay
        self.look_at_value = delay - 1
        self.fs = fs
        self.threshold_ratio = 0.3
        self.window_length = int(0.15 * fs)
        self.center_offset = self.window_length // 2
        self.min_qrs_width = 130
        self.max_qrs_width = 700
        self.refractory_samples = int(0.5 * fs)

        b_baseline, a_baseline = signal.butter(2, 1, "hp", analog=False, fs=fs, output="ba")
        self.baseline_filter = online_filter.OnlineFilter(1, b_baseline, a_baseline)

        b_bandpass, a_bandpass = signal.butter(4, [8.0, 20.0], "bp", analog=False, fs=fs, output="ba")
        self.bandpass_filter = online_filter.OnlineFilter(1, b_bandpass, a_bandpass)

        self.centered_samples = deque([0.0] * (self.center_offset + 1), maxlen=self.center_offset + 1)
        self.energy_window = deque([0.0] * self.window_length, maxlen=self.window_length)
        self.energy_sum = 0.0
        self.previous_filtered = 0.0
        self.have_previous_filtered = False

        self.current_max = 0.0
        self.recent_maxima = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)
        self.samples_since_max_update = 0

        self.in_candidate = False
        self.candidate_width = 0
        self.peak_max_value = 0.0
        self.peak_max_age = 0
        self.peak_min_value = 0.0
        self.peak_min_age = 0

        self.samples_since_last_peak = self.refractory_samples
        self.pending_peak_ages = deque()

    def qrs_detection(self, x: float) -> int:
        """
        Process one sample.

        :param x: measured value from an EMG signal with ECG artifacts
        :return: 1 if the delayed sample is a QRS peak, otherwise 0
        """
        assert isinstance(x, float), "the input values of the sEMG signal must be floats"

        centered_sample = self._filter_and_shape_sample(x)
        rolling_mean = self._update_rolling_energy()
        threshold_max = self._update_threshold_max(rolling_mean)

        # The detector is a small state machine:
        # 1) enter a candidate when the smoothed spike energy crosses the adaptive threshold,
        # 2) track the strongest positive/negative centered sample while inside the candidate,
        # 3) when the energy falls below threshold, validate width and refractory spacing,
        # 4) schedule a delayed output so callers still receive a simple 0/1 stream.
        if threshold_max > 0 and rolling_mean > self.threshold_ratio * threshold_max:
            self._update_candidate(centered_sample)
        elif self.in_candidate:
            self._finish_candidate()

        return self._advance_peak_timers()

    def _filter_and_shape_sample(self, x: float) -> float:
        baseline_corrected = self.baseline_filter.filter(x)
        self.centered_samples.append(baseline_corrected)
        centered_sample = self.centered_samples[0]

        filtered = self.bandpass_filter.filter(baseline_corrected)
        if self.have_previous_filtered:
            energy = (filtered - self.previous_filtered) ** 2
        else:
            energy = 0.0
            self.have_previous_filtered = True
        self.previous_filtered = filtered

        oldest_energy = self.energy_window[0]
        self.energy_window.append(energy)
        self.energy_sum += energy - oldest_energy

        return centered_sample

    def _update_rolling_energy(self) -> float:
        return self.energy_sum / self.window_length

    def _update_threshold_max(self, rolling_mean: float) -> float:
        self.current_max = max(self.current_max, rolling_mean)
        recent_max = max(self.recent_maxima)

        # Very large one-off bursts can dominate the adaptive threshold and hide real QRS events.
        # When all recent history is non-zero and the current block is an extreme outlier,
        # ignore it for thresholding while still allowing the detector state to continue.
        if recent_max > 0 and all(value > 0 for value in self.recent_maxima):
            if self.current_max > 10 * self.recent_maxima[-1] and self.current_max > 15 * recent_max:
                threshold_max = recent_max
            else:
                threshold_max = max(self.current_max, recent_max)
        else:
            threshold_max = max(self.current_max, recent_max)

        self.samples_since_max_update += 1
        if self.samples_since_max_update >= self.delay:
            self.recent_maxima.append(self.current_max)
            self.current_max = 0.0
            self.samples_since_max_update = 0

        return threshold_max

    def _update_candidate(self, centered_sample: float) -> None:
        if not self.in_candidate:
            self.in_candidate = True
            self.candidate_width = 0
            self.peak_max_value = centered_sample
            self.peak_max_age = self.center_offset
            self.peak_min_value = centered_sample
            self.peak_min_age = self.center_offset
        else:
            if centered_sample > self.peak_max_value:
                self.peak_max_value = centered_sample
                self.peak_max_age = self.center_offset
            if centered_sample < self.peak_min_value:
                self.peak_min_value = centered_sample
                self.peak_min_age = self.center_offset

        self.candidate_width += 1
        self.peak_max_age += 1
        self.peak_min_age += 1

    def _finish_candidate(self) -> None:
        candidate_width = self.candidate_width
        peak_age = self.peak_max_age
        if abs(self.peak_min_value) > self.peak_max_value:
            peak_age = self.peak_min_age

        self.in_candidate = False
        self.candidate_width = 0

        if not self.min_qrs_width <= candidate_width < self.max_qrs_width:
            return
        if self.samples_since_last_peak < self.refractory_samples:
            return

        # The original detector increments peak ages before returning, so schedule one sample behind
        # the measured age to keep the public 0/1 output aligned with existing call sites.
        self.pending_peak_ages.append(max(0, peak_age - 1))
        self.samples_since_last_peak = 0

    def _advance_peak_timers(self) -> int:
        self.samples_since_last_peak += 1

        for index in range(len(self.pending_peak_ages)):
            self.pending_peak_ages[index] += 1

        while self.pending_peak_ages and self.pending_peak_ages[0] > self.look_at_value:
            self.pending_peak_ages.popleft()

        if self.pending_peak_ages and self.pending_peak_ages[0] == self.look_at_value:
            self.pending_peak_ages.popleft()
            return 1

        return 0
