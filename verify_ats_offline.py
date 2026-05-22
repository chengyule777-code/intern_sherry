#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ECG_REMOVAL_SRC = REPO_ROOT / "ecg_removal_py"
if str(ECG_REMOVAL_SRC) not in sys.path:
    sys.path.insert(0, str(ECG_REMOVAL_SRC))

from ecg_removal_py.ats import adaptive_template_subtraction
from ecg_removal_py.config import AtsConfig, PreprocessConfig
from ecg_removal_py.preprocess import preprocess_emg_for_ats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline ATS validation on rr_app EMG CSV.")
    p.add_argument("--input", required=True, help="Input rr_app CSV path.")
    p.add_argument("--output-csv", required=True, help="Output CSV path with raw/preprocessed/cleaned/artifact/rpeak.")
    p.add_argument("--output-json", required=True, help="Output JSON metrics document path.")
    p.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    p.add_argument("--signal-column", default="emg_raw_mV")
    p.add_argument("--clean-column", default="emg_clean_ecg_mV")
    p.add_argument("--artifact-column", default="emg_ecg_artifact_mV")
    p.add_argument("--template-neighbor-beats", type=int, default=20)
    p.add_argument("--qrs-half-window-s", type=float, default=0.055)
    p.add_argument("--qrs-inc-max", type=int, default=10)
    p.add_argument("--rpeak-half-window-ms", type=float, default=60.0)
    return p.parse_args()


def _safe_float(v: str | None) -> float:
    if v is None:
        return math.nan
    s = str(v).strip()
    if s == "":
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def load_rr_csv(path: Path, signal_col: str, clean_col: str, artifact_col: str) -> tuple[list[float], list[float], list[float]]:
    raw: list[float] = []
    clean_ref: list[float] = []
    artifact_ref: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        headers = list(r.fieldnames or [])
        if signal_col not in headers:
            raise ValueError(f"Missing signal column: {signal_col}; available={headers}")
        has_clean = clean_col in headers
        has_artifact = artifact_col in headers
        for row in r:
            raw.append(_safe_float(row.get(signal_col)))
            clean_ref.append(_safe_float(row.get(clean_col)) if has_clean else math.nan)
            artifact_ref.append(_safe_float(row.get(artifact_col)) if has_artifact else math.nan)
    return raw, clean_ref, artifact_ref


def _mean(vals: list[float]) -> float:
    return sum(vals) / float(len(vals)) if vals else math.nan


def _rms(vals: list[float]) -> float:
    if not vals:
        return math.nan
    return math.sqrt(sum(v * v for v in vals) / float(len(vals)))


def _pearson(x: list[float], y: list[float]) -> float:
    if not x or not y or len(x) != len(y):
        return math.nan
    mx = _mean(x)
    my = _mean(y)
    sxy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    for a, b in zip(x, y):
        dx = a - mx
        dy = b - my
        sxy += dx * dy
        sx2 += dx * dx
        sy2 += dy * dy
    den = math.sqrt(max(0.0, sx2) * max(0.0, sy2))
    if den <= 1e-12:
        return math.nan
    return sxy / den


def _valid_pairs(a: list[float], b: list[float]) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for av, bv in zip(a, b):
        if math.isnan(av) or math.isnan(bv):
            continue
        x.append(av)
        y.append(bv)
    return x, y


def _rpeak_indices(mask: list[int]) -> list[int]:
    return [i for i, v in enumerate(mask) if int(v) != 0]


def _window_values(sig: list[float], idxs: list[int], half_n: int) -> list[float]:
    used: set[int] = set()
    out: list[float] = []
    n = len(sig)
    for i in idxs:
        a = max(0, i - half_n)
        b = min(n, i + half_n + 1)
        for j in range(a, b):
            if j in used:
                continue
            used.add(j)
            out.append(sig[j])
    return out


def write_offline_csv(
    path: Path,
    raw: list[float],
    preprocessed: list[float],
    cleaned: list[float],
    artifact: list[float],
    rpeak: list[int],
    clean_ref: list[float],
    artifact_ref: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(raw), len(preprocessed), len(cleaned), len(artifact), len(rpeak), len(clean_ref), len(artifact_ref))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "raw",
                "preprocessed",
                "cleaned_offline",
                "artifact_offline",
                "rpeak",
                "cleaned_rr_app",
                "artifact_rr_app",
            ]
        )
        for i in range(n):
            w.writerow([i, raw[i], preprocessed[i], cleaned[i], artifact[i], int(rpeak[i]), clean_ref[i], artifact_ref[i]])


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    raw, clean_ref, artifact_ref = load_rr_csv(
        input_path,
        signal_col=str(args.signal_column),
        clean_col=str(args.clean_column),
        artifact_col=str(args.artifact_column),
    )
    if not raw:
        raise RuntimeError("No samples found in input CSV.")
    if any(math.isnan(v) for v in raw):
        raise RuntimeError("Raw signal contains non-numeric values.")

    pre_cfg = PreprocessConfig(fs_hz=float(args.fs), powerline_hz=50.0)
    preprocessed, _detect, rpeaks = preprocess_emg_for_ats(raw, config=pre_cfg)
    ats_cfg = AtsConfig(
        fs_hz=float(args.fs),
        template_neighbor_beats=int(args.template_neighbor_beats),
        qrs_half_window_s=float(args.qrs_half_window_s),
        qrs_inc_max_samples=int(args.qrs_inc_max),
    )
    ats = adaptive_template_subtraction(preprocessed, rpeaks, config=ats_cfg)

    x, y = _valid_pairs(ats.cleaned, clean_ref)
    mae = _mean([abs(a - b) for a, b in zip(x, y)]) if x else math.nan
    rmse = math.sqrt(_mean([(a - b) * (a - b) for a, b in zip(x, y)])) if x else math.nan
    corr = _pearson(x, y) if x else math.nan

    peaks = _rpeak_indices(rpeaks)
    half_n = max(1, int(round((float(args.rpeak_half_window_ms) / 1000.0) * float(args.fs))))
    raw_w = _window_values(raw, peaks, half_n)
    clean_w = _window_values(ats.cleaned, peaks, half_n)
    n_w = min(len(raw_w), len(clean_w))
    raw_rms = _rms(raw_w[:n_w]) if n_w > 0 else math.nan
    cleaned_rms = _rms(clean_w[:n_w]) if n_w > 0 else math.nan
    drop_db = math.nan
    if not math.isnan(raw_rms) and raw_rms > 1e-12 and not math.isnan(cleaned_rms):
        drop_db = -20.0 * math.log10(max(1e-12, cleaned_rms / raw_rms))

    write_offline_csv(
        output_csv,
        raw=raw,
        preprocessed=preprocessed,
        cleaned=ats.cleaned,
        artifact=ats.artifact,
        rpeak=rpeaks,
        clean_ref=clean_ref,
        artifact_ref=artifact_ref,
    )

    report = {
        "input": str(input_path),
        "output_csv": str(output_csv),
        "fs_hz": float(args.fs),
        "samples": len(raw),
        "detected_rpeaks": sum(1 for v in rpeaks if int(v) != 0),
        "processed_beats": int(ats.processed_beats),
        "agreement": {
            "samples_used": len(x),
            "mae": mae,
            "rmse": rmse,
            "corr": corr,
        },
        "rpeak_window_energy": {
            "half_window_samples": half_n,
            "raw_rms": raw_rms,
            "cleaned_rms": cleaned_rms,
            "drop_db": drop_db,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"input={input_path}")
    print(f"output_csv={output_csv}")
    print(f"output_json={output_json}")
    print(f"samples={report['samples']}")
    print(f"detected_rpeaks={report['detected_rpeaks']}")
    print(f"processed_beats={report['processed_beats']}")
    print(f"agreement_mae={mae}")
    print(f"agreement_rmse={rmse}")
    print(f"agreement_corr={corr}")
    print(f"rpeak_drop_db={drop_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
