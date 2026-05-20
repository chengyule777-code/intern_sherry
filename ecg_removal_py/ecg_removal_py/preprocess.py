from __future__ import annotations

import math
from typing import Sequence

from .config import PreprocessConfig


def preprocess_emg_for_ats(
    signal: Sequence[float],
    *,
    config: PreprocessConfig,
) -> tuple[list[float], list[float], list[int]]:
    raw = [float(v) for v in signal]
    if not raw:
        return [], [], []
    notch = apply_notch(raw, fs_hz=config.fs_hz, f0_hz=config.powerline_hz, q=config.notch_q, passes=config.notch_passes)
    detect_sig = apply_highpass(notch, fs_hz=config.fs_hz, cutoff_hz=config.hp_detect_hz, order=config.hp_order)
    clean_base = apply_highpass(notch, fs_hz=config.fs_hz, cutoff_hz=config.hp_clean_hz, order=config.hp_order)
    rpeaks = detect_rpeaks(detect_sig, config=config)
    return clean_base, detect_sig, rpeaks


def detect_rpeaks(signal: Sequence[float], *, config: PreprocessConfig) -> list[int]:
    x = [float(v) for v in signal]
    n = len(x)
    if n == 0:
        return []

    # Detection stages mirror Pan-Tompkins style flow in lightweight form:
    # 1) emphasize slope changes using first difference and squaring,
    # 2) smooth the energy with a moving window (default 150 ms),
    # 3) enforce a refractory period from min RR interval,
    # 4) recenter accepted events to local maxima of the original signal.
    # This keeps robust peak timing without relying on external libraries.
    dx = [0.0] + [x[i] - x[i - 1] for i in range(1, n)]
    energy = [v * v for v in dx]
    win_n = max(3, int(round(float(config.detect_window_s) * float(config.fs_hz))))
    envelope = moving_average(energy, win_n)

    mu = sum(envelope) / float(max(1, len(envelope)))
    var = sum((v - mu) * (v - mu) for v in envelope) / float(max(1, len(envelope)))
    sigma = math.sqrt(max(0.0, var))
    threshold = float(mu + float(config.detect_threshold_scale) * sigma)
    refractory = max(1, int(round(float(config.min_rr_s) * float(config.fs_hz))))

    peaks = [0] * n
    last_peak = -refractory
    i = 1
    while i < (n - 1):
        if envelope[i] >= threshold and envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]:
            if (i - last_peak) >= refractory:
                left = max(0, i - 5)
                right = min(n, i + 6)
                loc = _argmax_abs(x[left:right]) + left
                peaks[loc] = 1
                last_peak = loc
                i = loc + 1
                continue
        i += 1
    return peaks


def apply_notch(signal: Sequence[float], *, fs_hz: float, f0_hz: float, q: float = 30.0, passes: int = 2) -> list[float]:
    x = [float(v) for v in signal]
    if len(x) < 3:
        return x

    w0 = 2.0 * math.pi * float(f0_hz) / max(1e-9, float(fs_hz))
    alpha = math.sin(w0) / (2.0 * max(1e-6, float(q)))
    b0 = 1.0
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha

    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    y = x
    for _ in range(max(1, int(passes))):
        y = _filtfilt_biquad(y, b=b, a=a)
    return y


def apply_highpass(signal: Sequence[float], *, fs_hz: float, cutoff_hz: float, order: int = 3) -> list[float]:
    y = [float(v) for v in signal]
    for _ in range(max(1, int(order))):
        y = _filtfilt_onepole_highpass(y, fs_hz=fs_hz, cutoff_hz=cutoff_hz)
    return y


def moving_average(signal: Sequence[float], win_n: int) -> list[float]:
    n = len(signal)
    w = max(1, int(win_n))
    out = [0.0] * n
    acc = 0.0
    for i in range(n):
        acc += float(signal[i])
        if i >= w:
            acc -= float(signal[i - w])
        den = min(i + 1, w)
        out[i] = acc / float(max(1, den))
    return out


def _filtfilt_biquad(x: list[float], *, b: list[float], a: list[float]) -> list[float]:
    y = _lfilter_biquad(x, b=b, a=a)
    y_rev = list(reversed(y))
    y2 = _lfilter_biquad(y_rev, b=b, a=a)
    return list(reversed(y2))


def _lfilter_biquad(x: list[float], *, b: list[float], a: list[float]) -> list[float]:
    out = [0.0] * len(x)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for i, v in enumerate(x):
        y0 = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        out[i] = float(y0)
        x2 = x1
        x1 = float(v)
        y2 = y1
        y1 = float(y0)
    return out


def _filtfilt_onepole_highpass(x: list[float], *, fs_hz: float, cutoff_hz: float) -> list[float]:
    y = _onepole_highpass(x, fs_hz=fs_hz, cutoff_hz=cutoff_hz)
    y_rev = list(reversed(y))
    y2 = _onepole_highpass(y_rev, fs_hz=fs_hz, cutoff_hz=cutoff_hz)
    return list(reversed(y2))


def _onepole_highpass(x: list[float], *, fs_hz: float, cutoff_hz: float) -> list[float]:
    if not x:
        return []
    dt = 1.0 / max(1e-9, float(fs_hz))
    rc = 1.0 / (2.0 * math.pi * max(1e-9, float(cutoff_hz)))
    alpha = rc / (rc + dt)
    y = [0.0] * len(x)
    prev_x = float(x[0])
    prev_y = 0.0
    for i, xi in enumerate(x):
        yi = float(alpha * (prev_y + float(xi) - prev_x))
        y[i] = yi
        prev_x = float(xi)
        prev_y = yi
    return y


def _argmax_abs(values: Sequence[float]) -> int:
    if not values:
        return 0
    best_i = 0
    best_v = abs(float(values[0]))
    for i, v in enumerate(values[1:], start=1):
        av = abs(float(v))
        if av > best_v:
            best_v = av
            best_i = i
    return best_i
