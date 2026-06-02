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

Based upon the emg-ecg-denoising algorithm from:
Petersen, E., Sauer, J., Graßhoff, J., and Rostalski, P.(2022). Removing Cardiac Artifacts From Single-Channel
Respiratory Electromyograms. IEEE Access, 8, 30905-30917.
"""

from collections import deque
from statistics import median

import online_three_layer_filter_bank


class SwtEmgDenoise:
    """
    This class can be used for denoising an EMG signal and removing cardiac artifacts online using a stationary wavelet
    transform.
    """

    def __init__(self, fs: int, delay: int):
        """
        filter_bank: the filter bank used for SWT
        peaks: Array, signifying the positions of R-peaks
        :param fs: the sampling rate
        :param delay: the delay of the peak detection
        """
        self.filter_bank = online_filter_bank.FilterBank()
        self.fs = fs
        self.coefficients_highpass_1 = deque()
        self.coefficients_highpass_2 = deque()
        self.coefficients_highpass_3 = deque()
        self.num_level = 3
        self.emg_thresholds = [10 for _ in range(self.num_level)]
        self.qrs_thresholds = [4 for _ in range(self.num_level)]
        self.idx_last_peak = 0
        self.delay = delay

    def swt_emg_denoising(self, sig: float, peak: int, heartrate: int):
        """
        This is a variant of the commonly used wavelet denoising that is optimized with respect to ECG removal.
        The position of R-peaks is considered and used in a special thresholding method.

        :param peak: 1 if sig is a peak, else 0
        :param heartrate: updated heartrate
        :param sig: new value of the signal to be denoised.
        :return results: one value of the denoised emg signal
        """

        if peak == 1:
            self.idx_last_peak = self.delay
        else:
            self.idx_last_peak += 1

        idx_new_peak = self.idx_last_peak - heartrate

        # Perform actual stationary wavelet transform with the Daubechies 2 wavelet in 3 levels
        swt_coefficients = self.filter_bank.swt(sig)

        # Create and apply adaptive thresholds for every frequency band.
        empty_coeff = 0
        swt_coefficients_emg = [(empty_coeff, empty_coeff) for _ in range(self.num_level)]

        if idx_new_peak == 0:
            self.idx_last_peak = 0
            idx_new_peak = -heartrate

        # For each detail coefficient calculate and apply a threshold, which gates the qrs complexes
        for idx, coeff, ecg_thr, emg_thr in \
                zip(range(self.num_level), swt_coefficients, self.qrs_thresholds, self.emg_thresholds):
            # Create gates around detected R peaks. The width of each gate decreases with higher frequency bands. The
            # assumption is, that the QRS-complex is wide in low frequency bands and sharpens in higher bands.
            # Note that the swt bands are ordered from low to high frequencies.

            level = self.num_level - idx
            win_gate = int(level * 0.2 * self.fs)

            if idx_new_peak + win_gate / 2 >= 0 or self.idx_last_peak - win_gate / 2 + 1 <= 0:
                gate = True  # returns true if value in gate
            else:
                gate = False

            detail_coeff = coeff[1]  # get highpass results of level

            # First the QRS complexes are removed roughly using the previously defined gates.
            gated_detail_coeff = abs(detail_coeff)

            # Apply median
            if level == 1:
                self.coefficients_highpass_1.appendleft(gated_detail_coeff)
                median_detail_coeff = median(self.coefficients_highpass_1)
                if len(self.coefficients_highpass_1) > self.fs / 4:
                    self.coefficients_highpass_1.pop()
            elif level == 2:
                self.coefficients_highpass_2.appendleft(gated_detail_coeff)
                median_detail_coeff = median(self.coefficients_highpass_2)
                if len(self.coefficients_highpass_2) > self.fs / 4:
                    self.coefficients_highpass_2.pop()
            else:
                self.coefficients_highpass_3.appendleft(gated_detail_coeff)
                median_detail_coeff = median(self.coefficients_highpass_3)
                if len(self.coefficients_highpass_3) > self.fs / 4:
                    self.coefficients_highpass_3.pop()

            # Subsequently set smaller threshold in segmented qrs-complex areas.
            if gate:
                wt_thresh = ecg_thr * median_detail_coeff  # set gates to lower threshold
            else:
                wt_thresh = emg_thr * median_detail_coeff
            below = abs(detail_coeff) < wt_thresh  # returns a boolean array with True under threshold (no peak)

            # high frequencies
            # below threshold -> EMG
            if not below:
                detail_coeff = 0  # set all False to 0 (all peaks)

            if level == 3:
                swt_coefficients_emg[idx] = (0, detail_coeff)  # lowpass set to 0
            else:
                swt_coefficients_emg[idx] = (coeff[0], detail_coeff)  # lowpass stays the same

        emg_denoised = self.filter_bank.iswt(
            0, swt_coefficients_emg[0][1], swt_coefficients_emg[1][1], swt_coefficients_emg[2][1])

        return emg_denoised
