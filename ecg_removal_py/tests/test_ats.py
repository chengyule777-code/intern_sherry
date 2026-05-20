from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from ecg_removal_py.ats import adaptive_template_subtraction
from ecg_removal_py.cli import main as cli_main
from ecg_removal_py.config import AtsConfig, PreprocessConfig
from ecg_removal_py.preprocess import detect_rpeaks, preprocess_emg_for_ats


def _synthetic_emg_with_ecg(fs_hz: int = 1000, duration_s: float = 8.0) -> tuple[list[float], list[float], list[float], list[int]]:
    n = int(fs_hz * duration_s)
    t = [i / float(fs_hz) for i in range(n)]
    clean = [
        0.08 * math.sin(2.0 * math.pi * 80.0 * ti) + 0.03 * math.sin(2.0 * math.pi * 130.0 * ti + 0.2)
        for ti in t
    ]

    rpeaks = [0] * n
    beat_times = [0.7 + i * 0.95 for i in range(int(duration_s / 0.9))]
    for bt in beat_times:
        idx = int(round(bt * fs_hz))
        if 0 <= idx < n:
            rpeaks[idx] = 1

    artifact = [0.0] * n
    sigma = 0.014
    for rp in [i for i, v in enumerate(rpeaks) if v]:
        for k in range(-120, 121):
            idx = rp + k
            if idx < 0 or idx >= n:
                continue
            dt = k / float(fs_hz)
            artifact[idx] += 0.18 * math.exp(-0.5 * (dt / sigma) ** 2)

    mixed = [c + a for c, a in zip(clean, artifact)]
    return clean, artifact, mixed, rpeaks


def _window_energy(signal: list[float], rpeaks: list[int], half_win: int = 40) -> float:
    idxs = [i for i, v in enumerate(rpeaks) if int(v) != 0]
    if not idxs:
        return 0.0
    acc = 0.0
    cnt = 0
    for rp in idxs:
        s = max(0, rp - half_win)
        e = min(len(signal), rp + half_win + 1)
        for v in signal[s:e]:
            acc += float(v) * float(v)
            cnt += 1
    return acc / float(max(1, cnt))


class AtsRemovalTests(unittest.TestCase):
    def test_ats_reduces_qrs_window_energy(self) -> None:
        clean, _artifact, mixed, rpeaks = _synthetic_emg_with_ecg()
        cfg = AtsConfig(fs_hz=1000.0, template_neighbor_beats=20, qrs_half_window_s=0.055, qrs_inc_max_samples=8)
        result = adaptive_template_subtraction(mixed, rpeaks, config=cfg)
        residual_before = [m - c for m, c in zip(mixed, clean)]
        residual_after = [y - c for y, c in zip(result.cleaned, clean)]
        e_before = _window_energy(residual_before, rpeaks, half_win=45)
        e_after = _window_energy(residual_after, rpeaks, half_win=45)
        self.assertLess(e_after, e_before)

    def test_preprocess_and_peak_detection_shape(self) -> None:
        _clean, _artifact, mixed, _ = _synthetic_emg_with_ecg()
        p_cfg = PreprocessConfig(fs_hz=1000.0)
        pre, detect_sig, peaks = preprocess_emg_for_ats(mixed, config=p_cfg)
        self.assertEqual(len(pre), len(mixed))
        self.assertEqual(len(detect_sig), len(mixed))
        self.assertEqual(len(peaks), len(mixed))
        self.assertGreater(sum(1 for v in peaks if v), 0)
        peaks2 = detect_rpeaks(detect_sig, config=p_cfg)
        self.assertEqual(len(peaks2), len(mixed))

    def test_cli_smoke_csv(self) -> None:
        _clean, _artifact, mixed, rpeaks = _synthetic_emg_with_ecg(duration_s=4.0)
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = Path(tmp) / "in.csv"
            out_csv = Path(tmp) / "out.csv"
            with in_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["emg_raw_mV", "rpeak"])
                for x, rp in zip(mixed, rpeaks):
                    w.writerow([x, rp])
            rc = cli_main(
                [
                    "--input",
                    str(in_csv),
                    "--output",
                    str(out_csv),
                    "--signal-column",
                    "emg_raw_mV",
                    "--rpeak-column",
                    "rpeak",
                    "--fs",
                    "1000",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_csv.exists())
            rows = out_csv.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(rows[0].split(",")[:3], ["index", "raw", "preprocessed"])


if __name__ == "__main__":
    unittest.main()
