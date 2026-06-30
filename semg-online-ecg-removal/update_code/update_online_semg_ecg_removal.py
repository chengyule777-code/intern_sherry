"""
Updated single-channel online sEMG ECG artifact removal.

This version is single-channel only: pass one measured sample in and receive
one denoised EMG sample back.
"""

from __future__ import annotations

from collections import deque
from statistics import median

from update_code.update_online_three_layer_filter_bank import UpdatedFilterBank


class UpdatedSwtEmgDenoise:
    """
    Remove ECG-like artifacts from one streaming EMG signal using SWT details.

    Each input sample is split into three frequency bands. Around predicted
    ECG R-peaks, the algorithm uses a stricter threshold so sharp cardiac
    artifacts are zeroed before the sample is reconstructed.
    """

    def __init__(self, fs: int, delay_s: float):
        if not isinstance(fs, int):
            raise TypeError("fs must be an integer sampling rate in Hz.")
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if delay_s < 0:
            raise ValueError("delay must be non-negative.")

        self.filter_bank = UpdatedFilterBank()
        self.fs = fs
        self.delay_s = float(delay_s)
        self.num_levels = 3

        self.emg_thresholds = [10.0] * self.num_levels
        self.qrs_thresholds = [4.0] * self.num_levels
        self.detail_histories = [
            # The median history is a real buffer, so it still needs an integer
            # number of samples even though the target duration is 250 ms.
            deque(maxlen=max(1, int(fs / 4))) for _ in range(self.num_levels)
        ]

        self.time_since_last_peak_s = 0.0

    def swt_emg_denoising(
        self,
        sig: float,
        peak: int | bool,
        rr_interval_s: float,
        dt_s: float | None = None,
    ) -> float:
        """
        Process one measured sample and return one denoised EMG sample.

        :param sig: one raw EMG sample that may include an ECG artifact
        :param peak: 1/True when the delayed QRS detector reports an R-peak
        :param rr_interval_s: estimated R-R interval in seconds
        :param dt_s: elapsed seconds since the previous sample; defaults to 1 / fs
        """
        if rr_interval_s <= 0:
            raise ValueError("rr_interval_s must be positive.")
        dt_s = 1.0 / self.fs if dt_s is None else float(dt_s)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive when provided.")

        self._advance_peak_position(peak, dt_s)
        time_until_predicted_peak_s = self.time_since_last_peak_s - rr_interval_s
        if time_until_predicted_peak_s >= 0:
            # Once the expected next peak time has passed, start the cycle again
            # so the next gate is measured against the following R-R interval.
            self.time_since_last_peak_s = 0.0
            time_until_predicted_peak_s = -rr_interval_s

        swt_coefficients = self.filter_bank.swt(float(sig))
        denoised_coefficients = []

        # For each SWT detail band:
        # 1) measure the recent typical detail size with a rolling median,
        # 2) lower the threshold inside the ECG gate so cardiac spikes are caught,
        # 3) zero coefficients that are too large to be plausible EMG,
        # 4) keep the cleaned details for inverse SWT reconstruction.
        for idx, ((lowpass, detail), qrs_threshold, emg_threshold) in enumerate(
            zip(swt_coefficients, self.qrs_thresholds, self.emg_thresholds)
        ):
            level = self.num_levels - idx
            is_inside_qrs_gate = self._is_inside_qrs_gate(time_until_predicted_peak_s, level, dt_s)

            detail_magnitude = abs(detail)
            typical_detail = self._update_typical_detail(idx, detail_magnitude)
            threshold_multiplier = qrs_threshold if is_inside_qrs_gate else emg_threshold
            threshold = threshold_multiplier * typical_detail

            cleaned_detail = detail if detail_magnitude < threshold else 0.0
            cleaned_lowpass = 0.0 if level == 3 else lowpass
            denoised_coefficients.append((cleaned_lowpass, cleaned_detail))

        return self.filter_bank.iswt(
            denoised_coefficients[0][0],
            denoised_coefficients[0][1],
            denoised_coefficients[1][1],
            denoised_coefficients[2][1],
        )

    def _advance_peak_position(self, peak: int | bool, dt_s: float) -> None:
        if peak in (1, True):
            # The QRS detector reports peaks after a fixed delay. Store the
            # physical delay here so the gate lines up with the original ECG artifact.
            self.time_since_last_peak_s = self.delay_s
        elif peak in (0, False):
            self.time_since_last_peak_s += dt_s
        else:
            raise ValueError("peak must be 0, 1, False, or True.")

    def _is_inside_qrs_gate(self, time_until_predicted_peak_s: float, level: int, dt_s: float) -> bool:
        # Lower-frequency SWT bands represent wider ECG shapes, so they need a
        # wider gate. Higher-frequency bands use a narrower gate around the spike.
        gate_width_s = level * 0.2
        half_gate_s = gate_width_s / 2
        near_next_predicted_peak = time_until_predicted_peak_s + half_gate_s >= 0
        near_last_detected_peak = self.time_since_last_peak_s - half_gate_s + dt_s <= 0
        return near_next_predicted_peak or near_last_detected_peak

    def _update_typical_detail(self, level_index: int, detail_magnitude: float) -> float:
        history = self.detail_histories[level_index]
        history.appendleft(detail_magnitude)
        return float(median(history))
