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

from code.online_three_layer_filter_bank import FilterBank


class SwtEmgDenoise:
    """
    This class can be used for denoising a multi-channel EMG signal and removing cardiac artifacts online using a
    stationary wavelet transform.
    """

    def __init__(self, fs, delay, channels):
        """
        filter_bank: the filter bank used for SWT
        peaks: Array, signifying the positions of R-peaks
        :param fs: the sampling rate
        :param delay: the delay of the peak detection
        :param channels: the number of recording channels
        """
        self.filter_banks = []
        for _ in range(channels):
            self.filter_banks.append(FilterBank())
        self.channels = channels
        self.fs = fs
        self.coefficients_highpass_1 = [deque() for _ in range(channels)]
        self.coefficients_highpass_2 = [deque() for _ in range(channels)]
        self.coefficients_highpass_3 = [deque() for _ in range(channels)]
        self.num_level = 3
        self.emg_thresholds = [10 for _ in range(self.num_level)]
        self.qrs_thresholds = [4 for _ in range(self.num_level)]
        self.idx_last_peak = 0
        self.delay = delay

    def swt_emg_denoising(self, sig, peak, heartrate):
        """
        This is a variant of the commonly used wavelet denoising that is optimized with respect to ECG removal.
        The position of R-peaks is considered and used in a special thresholding method.

        :param peak: 1 if sig is a peak, else 0
        :param heartrate: updated heartrate
        :param sig: an array of the new values of the signal to be denoised.
        :return results: the values of the denoised emg signals as an array
        """

        if peak == 1:
            self.idx_last_peak = self.delay
        else:
            self.idx_last_peak += 1

        idx_new_peak = self.idx_last_peak - heartrate

        # Perform actual stationary wavelet transform with the Daubechies 2 wavelet in 3 levels
        swt_coefficients = []
        for idx, filter_bank in enumerate(self.filter_banks):
            swt_coefficients.append(filter_bank.swt(sig[idx]))

        # Create and apply adaptive thresholds for every frequency band.
        empty_coeff = 0
        swt_coefficients_emg = [[(empty_coeff, empty_coeff) for _ in range(self.num_level)] for _ in
                                range(self.channels)]

        if idx_new_peak == 0:
            self.idx_last_peak = 0
            idx_new_peak = -heartrate

        # For each detail coefficient calculate and apply a threshold, which gates the qrs complexes
        for idx, ecg_thr, emg_thr in zip(range(self.num_level), self.qrs_thresholds, self.emg_thresholds):
            # Create gates around detected R peaks. The width of each gate decreases with higher frequency bands. The
            # assumption is, that the QRS-complex is wide in low frequency bands and sharpens in higher bands.
            # Note that the swt bands are ordered from low to high frequencies.

            level = self.num_level - idx
            coeffs = [i[idx] for i in swt_coefficients]
            win_gate = int(level * 0.2 * self.fs)

            if idx_new_peak + win_gate / 2 >= 0 or self.idx_last_peak - win_gate / 2 + 1 <= 0:
                gate = True  # returns true if value in gate
            else:
                gate = False

            detail_coeffs = []
            for coefficient in coeffs:
                detail_coeffs.append(coefficient[1])  # get highpass results of level

            # First the QRS complexes are removed roughly using the previously defined gates.
            gated_detail_coeffs = []
            for detail_coeff in detail_coeffs:
                gated_detail_coeffs.append(abs(detail_coeff))

            # Apply median
            median_detail_coeffs = []
            for i in range(self.channels):
                if level == 1:
                    self.coefficients_highpass_1[i].appendleft(gated_detail_coeffs[i])
                    median_detail_coeffs.append(median(self.coefficients_highpass_1[i]))
                    if len(self.coefficients_highpass_1[i]) > self.fs / 4:
                        self.coefficients_highpass_1[i].pop()
                elif level == 2:
                    self.coefficients_highpass_2[i].appendleft(gated_detail_coeffs[i])
                    median_detail_coeffs.append(median(self.coefficients_highpass_2[i]))
                    if len(self.coefficients_highpass_2[i]) > self.fs / 4:
                        self.coefficients_highpass_2[i].pop()
                else:
                    self.coefficients_highpass_3[i].appendleft(gated_detail_coeffs[i])
                    median_detail_coeffs.append(median(self.coefficients_highpass_3[i]))
                    if len(self.coefficients_highpass_3[i]) > self.fs / 4:
                        self.coefficients_highpass_3[i].pop()

            # Subsequently set smaller threshold in segmented qrs-complex areas.
            wt_threshs = []
            below = []
            for i in range(self.channels):
                if gate:
                    wt_threshs.append(ecg_thr * median_detail_coeffs[i])  # set gates to lower threshold
                else:
                    wt_threshs.append(emg_thr * median_detail_coeffs[i])
                below.append(abs(detail_coeffs[i]) < wt_threshs[
                    i])  # returns a boolean array with True under threshold (no peak)

            # high frequencies
            # below threshold -> EMG

            for i in range(self.channels):
                if not below[i]:
                    detail_coeffs[i] = 0  # set all False to 0 (all peaks)

                if level == 3:
                    swt_coefficients_emg[i][idx] = (0, detail_coeffs[i])  # lowpass set to 0
                else:
                    swt_coefficients_emg[i][idx] = (coeffs[i][0], detail_coeffs[i])  # lowpass stays the same

        emg_denoised = [self.filter_banks[i].iswt(
            0, swt_coefficients_emg[i][0][1], swt_coefficients_emg[i][1][1], swt_coefficients_emg[i][2][1]) for i in
            range(self.channels)]

        return emg_denoised
