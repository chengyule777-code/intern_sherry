"""
Efficient streaming QRS detection for future sEMG ECG-removal work.

This module keeps the same public shape as the original detector:
feed one sample into ``qrs_detection`` and receive ``1`` when a delayed
QRS peak is reported, otherwise ``0``.
"""

from scipy import signal

from code import update_online_filter


class UpdatedQrsDetector:
    """
    Detect ECG QRS peaks from an online sEMG stream.

    The implementation follows the same broad Pan-Tompkins flow as
    ``QrsDetector`` but avoids shifting NumPy arrays on every sample.
    In simple terms, it turns sharp ECG-like changes into a smoothed
    "energy" signal, watches for that energy to rise above a moving
    threshold, then reports one delayed 0/1 peak marker.
    """

    def __init__(self, delay: int, fs: int = 1024):
        assert isinstance(delay, int)
        assert 280 <= delay <= 400

        self.fs = fs
        self.threshold_ratio = 0.3
        self.window_length = int(0.15 * fs)  # smooth spike energy over about 150 ms
        self.min_qrs_width_s = 0.13  # reject boxes too narrow to be a plausible QRS region
        self.max_qrs_width_s = 0.70  # reject boxes too wide to be a plausible QRS region
        self.refractory_s = 0.5  # reject peaks closer than this many seconds apart
        self.peak_output_delay_s = delay / fs  # seconds before returning 1 for an accepted peak

        # 2nd-order Butterworth high-pass filter at 1 Hz cutoff.
        b_baseline, a_baseline = signal.butter(2, 1, "hp", analog=False, fs=fs, output="ba")
        self.baseline_filter = update_online_filter.OnlineFilter(1, b_baseline, a_baseline)

        # 4th-order Butterworth band-pass filter at the QRS-focused 8-20 Hz range.
        b_bandpass, a_bandpass = signal.butter(4, [8.0, 20.0], "bp", analog=False, fs=fs, output="ba")
        self.bandpass_filter = update_online_filter.OnlineFilter(1, b_bandpass, a_bandpass)

        self.energy_buffer = [0.0] * self.window_length  # ring buffer for recent squared differences
        self.energy_index = 0  # next slot to replace in the ring buffer
        self.energy_sum = 0.0  # rolling sum, so the moving average is O(1)
        self.previous_filtered = 0.0  # previous band-passed sample for the derivative step
        self.have_previous_filtered = False  # first sample cannot produce a difference yet

        self.current_max = 0.0  # largest rolling energy seen in the current threshold block
        self.recent_maxima = [0.0, 0.0, 0.0, 0.0]  # last four block maxima for adaptive thresholding
        self.recent_max_index = 0  # next old block maximum to overwrite
        self.latest_recent_max = 0.0  # newest completed block maximum
        self.threshold_update_interval_s = delay / fs  # match the old delay-sized threshold block in seconds
        self.threshold_update_elapsed_s = 0.0  # elapsed time since current_max was last saved

        self.in_candidate = False  # True while the energy is above threshold
        self.candidate_width_s = 0.0  # how long the current above-threshold box lasted
        self.peak_max_value = 0.0  # strongest upward sample inside the current box
        self.peak_max_age_s = 0.0  # seconds since that upward sample occurred
        self.peak_min_value = 0.0  # strongest downward sample inside the current box
        self.peak_min_age_s = 0.0  # seconds since that downward sample occurred

        self.seconds_since_last_peak = self.refractory_s  # allow the first valid peak immediately
        self.seconds_until_peak_output = None  # seconds left before returning a delayed peak marker

    def qrs_detection(self, x: float, dt_s: float | None = None) -> int:
        """
        Process one sample.

        :param x: measured value from an EMG signal with ECG artifacts
        :param dt_s: elapsed seconds since the previous sample; defaults to 1 / fs
        :return: 1 if the delayed sample is a QRS peak, otherwise 0
        """
        x = float(x)  # accept int/numpy numeric inputs while processing everything as float
        dt_s = 1.0 / self.fs if dt_s is None else float(dt_s)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive when provided.")

        # Convert the raw sample into QRS "energy", then update the adaptive threshold.
        current_sample = self._filter_and_shape_sample(x)
        rolling_mean = self._update_rolling_energy()
        threshold_max = self._update_threshold_max(rolling_mean, dt_s)

        # The detector is a small state machine:
        # 1) enter a candidate when the smoothed spike energy crosses the adaptive threshold,
        # 2) track the strongest positive/negative current sample while inside the candidate,
        # 3) when the energy falls below threshold, validate width and refractory spacing,
        # 4) schedule a delayed output so callers still receive a simple 0/1 stream.
        if threshold_max > 0 and rolling_mean > self.threshold_ratio * threshold_max:
            # Energy is high enough: we are inside a possible QRS region.
            self._update_candidate(current_sample, dt_s)
        elif self.in_candidate:
            # Energy just dropped: the possible QRS region ended, so validate it.
            self._finish_candidate()

        return self._advance_peak_timers(dt_s)

    def _filter_and_shape_sample(self, x: float) -> float:
        # Remove slow drift first, so QRS decisions are not pulled by baseline movement.
        baseline_corrected = self.baseline_filter.filter(x)

        # Keep the frequency range where QRS spikes are most visible.
        filtered = self.bandpass_filter.filter(baseline_corrected)
        if self.have_previous_filtered:
            # Difference + square makes sharp changes large and always positive.
            energy = (filtered - self.previous_filtered) ** 2
        else:
            # On the first sample there is no previous point to compare against.
            energy = 0.0
            self.have_previous_filtered = True
        self.previous_filtered = filtered

        # Replace the oldest energy value with the newest one and adjust the rolling sum.
        oldest_energy = self.energy_buffer[self.energy_index]
        self.energy_buffer[self.energy_index] = energy
        self.energy_sum += energy - oldest_energy
        self.energy_index = (self.energy_index + 1) % self.window_length

        return baseline_corrected

    def _update_rolling_energy(self) -> float:
        # Average recent spike energy to make a wider, smoother QRS "box".
        return self.energy_sum / self.window_length

    def _update_threshold_max(self, rolling_mean: float, dt_s: float) -> float:
        # Track the strongest energy in the current block of samples.
        self.current_max = max(self.current_max, rolling_mean)
        recent_max = max(self.recent_maxima)

        # Very large one-off bursts can dominate the adaptive threshold and hide real QRS events.
        # When all recent history is non-zero and the current block is an extreme outlier,
        # ignore it for thresholding while still allowing the detector state to continue.
        if recent_max > 0 and all(value > 0 for value in self.recent_maxima):
            if self.current_max > 10 * self.latest_recent_max and self.current_max > 15 * recent_max:
                # Treat this block as an artifact for threshold purposes.
                threshold_max = recent_max
            else:
                # Normal case: threshold may follow the current block if it is the strongest.
                threshold_max = max(self.current_max, recent_max)
        else:
            # Startup case: use whatever maximum exists while history is still filling.
            threshold_max = max(self.current_max, recent_max)

        # Use elapsed seconds rather than a fixed sample count, so future streams can run at
        # 1000 Hz, 1024 Hz, or slightly irregular sample intervals without changing this logic.
        self.threshold_update_elapsed_s += dt_s
        if self.threshold_update_elapsed_s >= self.threshold_update_interval_s:
            # Every delay-equivalent time interval, save its maximum and start measuring a new block.
            self.recent_maxima[self.recent_max_index] = self.current_max
            self.recent_max_index = (self.recent_max_index + 1) % len(self.recent_maxima)
            self.latest_recent_max = self.current_max
            self.current_max = 0.0
            self.threshold_update_elapsed_s = 0.0

        return threshold_max

    def _update_candidate(self, current_sample: float, dt_s: float) -> None:
        if not self.in_candidate:
            # This is the first above-threshold sample, so start a new candidate box.
            self.in_candidate = True
            self.candidate_width_s = 0.0
            self.peak_max_value = current_sample
            self.peak_max_age_s = 0.0
            self.peak_min_value = current_sample
            self.peak_min_age_s = 0.0
        else:
            if current_sample > self.peak_max_value:
                # New strongest upward point inside this candidate.
                self.peak_max_value = current_sample
                self.peak_max_age_s = 0.0
            if current_sample < self.peak_min_value:
                # New strongest downward point inside this candidate.
                self.peak_min_value = current_sample
                self.peak_min_age_s = 0.0

        # Ages advance by real elapsed time, so irregular sample intervals are represented correctly.
        self.candidate_width_s += dt_s
        self.peak_max_age_s += dt_s
        self.peak_min_age_s += dt_s

    def _finish_candidate(self) -> None:
        candidate_width_s = self.candidate_width_s
        peak_age_s = self.peak_max_age_s
        if abs(self.peak_min_value) > self.peak_max_value:
            # Some ECG artifacts are inverted; use the downward spike if it is stronger.
            peak_age_s = self.peak_min_age_s

        # Leave candidate mode before validation so rejected boxes do not linger.
        self.in_candidate = False
        self.candidate_width_s = 0.0

        if not self.min_qrs_width_s <= candidate_width_s < self.max_qrs_width_s:
            # Too short or too long to be a believable QRS region.
            return
        if self.seconds_since_last_peak < self.refractory_s:
            # Too close to the last accepted peak to be physiologically likely.
            return

        # The peak already happened inside the finished box. Count down until that
        # peak reaches the requested delay position, then return 1 for one sample.
        seconds_until_output = self.peak_output_delay_s - peak_age_s
        self.seconds_until_peak_output = seconds_until_output if seconds_until_output >= 0 else None
        self.seconds_since_last_peak = 0.0

    def _advance_peak_timers(self, dt_s: float) -> int:
        # This counter enforces the refractory period between accepted peaks.
        self.seconds_since_last_peak += dt_s

        if self.seconds_until_peak_output is None:
            # No accepted peak is waiting to be reported.
            return 0

        self.seconds_until_peak_output -= dt_s
        if self.seconds_until_peak_output <= 0:
            # The accepted peak has reached the delayed output position.
            self.seconds_until_peak_output = None
            return 1

        return 0
