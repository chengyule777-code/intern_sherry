from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .config import AtsConfig


@dataclass(frozen=True, slots=True)
class AtsResult:
    cleaned: list[float]
    subtraction_template: list[float]
    artifact: list[float]
    processed_beats: int


def adaptive_template_subtraction(
    signal: Sequence[float],
    rpeak_mask: Sequence[int],
    *,
    config: AtsConfig,
) -> AtsResult:
    x = [float(v) for v in signal]
    n = len(x)
    if n == 0:
        return AtsResult(cleaned=[], subtraction_template=[], artifact=[], processed_beats=0)

    rpeak_indices = _find_rpeak_indices(rpeak_mask, n)
    if len(rpeak_indices) < 2:
        return AtsResult(cleaned=list(x), subtraction_template=[0.0] * n, artifact=[0.0] * n, processed_beats=0)

    centers = _rr_centers(rpeak_indices, n)
    subtraction = [0.0] * n
    qrs_half_len = int(math.ceil(float(config.qrs_half_window_s) * float(config.fs_hz)))
    processed = 0

    # ATS per-beat processing is order-sensitive and branch-heavy:
    # 1) build an RR-bounded template around each R peak from neighboring beats,
    # 2) sweep 21 QRS width variants (default) and align each by correlation lag,
    # 3) scale left/QRS/right sections independently using linear regression,
    # 4) choose the minimum baseline-compensated L2 error variant.
    # This mirrors the MATLAB control flow while keeping deterministic behavior.
    for i, r_idx in enumerate(rpeak_indices):
        left_len = int(r_idx - centers[i])
        right_len = int(centers[i + 1] - r_idx)
        if left_len < 0 or right_len < 0:
            continue
        seg_start = int(r_idx - left_len)
        seg_end = int(r_idx + right_len)
        curr_seg = x[seg_start : seg_end + 1]
        curr_template = _build_template(
            rpeak_indices=rpeak_indices,
            length_left=left_len,
            length_right=right_len,
            signal=x,
            num=config.template_neighbor_beats,
            beat_idx=i,
        )
        if len(curr_seg) != len(curr_template):
            continue

        qrs_left_end = int(left_len + 1 - qrs_half_len)
        qrs_right_end = int(left_len + 1 + qrs_half_len)
        qrs_inc_max = int(config.qrs_inc_max_samples)
        if not (qrs_left_end > qrs_inc_max and qrs_right_end < (len(curr_template) - qrs_inc_max)):
            subtraction[seg_start : seg_end + 1] = curr_template
            processed += 1
            continue

        qrs = curr_template[qrs_left_end : qrs_right_end + 1]
        best_error = float("inf")
        best_template = curr_template

        for qrs_inc in range(-qrs_inc_max, qrs_inc_max + 1):
            mod_template, left_part_len, right_part_len = _modify_qrs_width(
                current_template=curr_template,
                qrs=qrs,
                qrs_inc=qrs_inc,
                qrs_left_end=qrs_left_end,
                qrs_right_end=qrs_right_end,
            )
            lag = _best_lag_by_corr(curr_seg, mod_template, max_pos=left_len, max_neg=right_len)
            shifted = _circular_shift(mod_template, lag)

            left_mod = int(left_part_len + lag)
            right_mod = int(right_part_len - lag)
            scaled, baseline = _scale_sections(shifted, curr_seg, left_mod, right_mod)
            err = _sq_error(curr_seg, scaled, baseline)
            if err < best_error:
                best_error = err
                best_template = scaled

        subtraction[seg_start : seg_end + 1] = best_template
        processed += 1

    cleaned = [xv - sv for xv, sv in zip(x, subtraction)]
    artifact = [xv - yv for xv, yv in zip(x, cleaned)]
    return AtsResult(
        cleaned=cleaned,
        subtraction_template=subtraction,
        artifact=artifact,
        processed_beats=processed,
    )


def _find_rpeak_indices(mask: Sequence[int], n: int) -> list[int]:
    out: list[int] = []
    for idx, v in enumerate(mask):
        if idx >= n:
            break
        if int(v) != 0:
            out.append(idx)
    return out


def _rr_centers(rpeak_indices: list[int], n: int) -> list[int]:
    centers = [0]
    for i in range(len(rpeak_indices) - 1):
        d = int(rpeak_indices[i + 1] - rpeak_indices[i])
        centers.append(int(math.floor(d / 2.0) + rpeak_indices[i]))
    centers.append(max(0, n - 1))
    return centers


def _build_template(
    *,
    rpeak_indices: list[int],
    length_left: int,
    length_right: int,
    signal: list[float],
    num: int,
    beat_idx: int,
) -> list[float]:
    beat_start = int(beat_idx - math.ceil(num / 2.0))
    beat_stop = int(beat_start + num - 1)

    out_len = int(length_left + length_right + 1)
    template = [0.0] * out_len
    beat_count = [0] * out_len

    for b in range(beat_start, beat_stop + 1):
        if b < 0 or b >= len(rpeak_indices):
            continue
        left_idx = int(rpeak_indices[b] - length_left)
        right_idx = int(rpeak_indices[b] + length_right)
        if left_idx < 0:
            mask_start = -left_idx
            mask_end = out_len
            src_start = 0
            src_end = right_idx + 1
        elif right_idx < len(signal):
            mask_start = 0
            mask_end = out_len
            src_start = left_idx
            src_end = right_idx + 1
        else:
            mask_start = 0
            mask_end = out_len - (right_idx - (len(signal) - 1))
            src_start = left_idx
            src_end = len(signal)
        if mask_end <= mask_start or src_end <= src_start:
            continue
        seg = signal[src_start:src_end]
        for j in range(mask_start, mask_end):
            template[j] += seg[j - mask_start]
            beat_count[j] += 1

    for i in range(out_len):
        if beat_count[i] > 0:
            template[i] /= float(beat_count[i])
        else:
            template[i] = 0.0
    return template


def _modify_qrs_width(
    *,
    current_template: list[float],
    qrs: list[float],
    qrs_inc: int,
    qrs_left_end: int,
    qrs_right_end: int,
) -> tuple[list[float], int, int]:
    target_qrs_len = len(qrs) + 2 * int(qrs_inc)
    if target_qrs_len < 2:
        target_qrs_len = 2
    qrs_mod = _resample_linear(qrs, target_qrs_len)

    if qrs_inc > 0:
        left_part = current_template[0 + qrs_inc : qrs_left_end]
        right_part = current_template[qrs_right_end + 1 : len(current_template) - qrs_inc]
    else:
        z = [0.0] * (-qrs_inc)
        left_part = z + current_template[0:qrs_left_end]
        right_part = current_template[qrs_right_end + 1 :] + z
    out = left_part + qrs_mod + right_part
    if len(out) < len(current_template):
        out = out + [0.0] * (len(current_template) - len(out))
    elif len(out) > len(current_template):
        out = out[: len(current_template)]
    return out, len(left_part), len(right_part)


def _resample_linear(values: list[float], target_len: int) -> list[float]:
    if target_len <= 0:
        return []
    if not values:
        return [0.0] * target_len
    if len(values) == 1:
        return [float(values[0])] * target_len
    if target_len == len(values):
        return [float(v) for v in values]

    out: list[float] = []
    src_max = len(values) - 1
    dst_max = target_len - 1
    for i in range(target_len):
        pos = (float(i) / float(dst_max)) * float(src_max)
        lo = int(math.floor(pos))
        hi = min(src_max, lo + 1)
        frac = float(pos - lo)
        out.append(float(values[lo] * (1.0 - frac) + values[hi] * frac))
    return out


def _best_lag_by_corr(seg: list[float], template: list[float], *, max_pos: int, max_neg: int) -> int:
    best_lag = 0
    best_score = -float("inf")
    for lag in range(-int(max_neg), int(max_pos) + 1):
        shifted = _circular_shift(template, lag)
        score = _corr_coeff(seg, shifted)
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def _corr_coeff(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n <= 1:
        return 0.0
    ma = sum(a[:n]) / float(n)
    mb = sum(b[:n]) / float(n)
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        xa = float(a[i] - ma)
        xb = float(b[i] - mb)
        num += xa * xb
        da += xa * xa
        db += xb * xb
    den = math.sqrt(max(1e-12, da * db))
    return float(num / den)


def _circular_shift(values: list[float], lag: int) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    l = int(lag) % n
    if l == 0:
        return list(values)
    return values[-l:] + values[:-l]


def _scale_sections(template: list[float], seg: list[float], left_mod: int, right_mod: int) -> tuple[list[float], list[float]]:
    n = len(template)
    if n == 0:
        return [], []
    if left_mod >= 0 and right_mod >= 0:
        qrs_start = int(min(n, max(0, left_mod)))
        qrs_end = int(max(qrs_start, min(n, n - right_mod)))
        if qrs_start == qrs_end:
            return _scale_whole(template, seg)

        scaled = [0.0] * n
        baseline = [0.0] * n

        left_tpl = template[:qrs_start]
        left_sig = seg[:qrs_start]
        sl, bl = _fit_scale_offset(left_sig, left_tpl)
        for i, v in enumerate(left_tpl):
            scaled[i] = sl * v
            baseline[i] = bl

        qrs_tpl = template[qrs_start:qrs_end]
        qrs_sig = seg[qrs_start:qrs_end]
        sq, bq = _fit_scale_offset(qrs_sig, qrs_tpl)
        for j, v in enumerate(qrs_tpl, start=qrs_start):
            scaled[j] = sq * v
            baseline[j] = bq

        right_tpl = template[qrs_end:]
        right_sig = seg[qrs_end:]
        sr, br = _fit_scale_offset(right_sig, right_tpl)
        for j, v in enumerate(right_tpl, start=qrs_end):
            scaled[j] = sr * v
            baseline[j] = br
        return scaled, baseline
    return _scale_whole(template, seg)


def _scale_whole(template: list[float], seg: list[float]) -> tuple[list[float], list[float]]:
    s, b = _fit_scale_offset(seg, template)
    scaled = [s * v for v in template]
    baseline = [b] * len(template)
    return scaled, baseline


def _fit_scale_offset(y: list[float], x: list[float]) -> tuple[float, float]:
    n = min(len(y), len(x))
    if n <= 0:
        return 0.0, 0.0
    mx = sum(x[:n]) / float(n)
    my = sum(y[:n]) / float(n)
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        dx = float(x[i] - mx)
        dy = float(y[i] - my)
        sxx += dx * dx
        sxy += dx * dy
    if sxx <= 1e-12:
        return 0.0, float(my)
    slope = float(sxy / sxx)
    offset = float(my - slope * mx)
    return slope, offset


def _sq_error(seg: list[float], template: list[float], baseline: list[float]) -> float:
    n = min(len(seg), len(template), len(baseline))
    s = 0.0
    for i in range(n):
        d = float(seg[i] - template[i] - baseline[i])
        s += d * d
    return s
