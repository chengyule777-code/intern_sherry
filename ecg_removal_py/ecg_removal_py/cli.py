from __future__ import annotations

import argparse
from pathlib import Path

from .ats import adaptive_template_subtraction
from .config import AtsConfig, PreprocessConfig
from .io import load_signal, write_output_csv, write_output_npz
from .preprocess import preprocess_emg_for_ats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATS ECG artifact removal for EMG signals.")
    p.add_argument("--input", required=True, help="Input CSV or NPZ path.")
    p.add_argument("--output", required=True, help="Output CSV or NPZ path.")
    p.add_argument("--signal-column", default="emg_raw_mV", help="Signal column/key name.")
    p.add_argument("--rpeak-column", default=None, help="Optional existing R-peak column/key (0/1).")
    p.add_argument("--fs", type=float, default=1024.0, help="Sampling frequency in Hz.")
    p.add_argument("--powerline-hz", type=float, default=50.0, help="Powerline notch center frequency.")
    p.add_argument("--template-neighbor-beats", type=int, default=40, help="ATS neighbor beats for template building.")
    p.add_argument("--qrs-half-window-s", type=float, default=0.055, help="QRS half-window in seconds.")
    p.add_argument("--qrs-inc-max", type=int, default=10, help="QRS width sweep half range in samples.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw, rpeaks_in, fs_from_file = load_signal(
        args.input,
        signal_column=args.signal_column,
        rpeak_column=args.rpeak_column,
        fs_hz=args.fs,
    )
    fs_hz = float(fs_from_file if fs_from_file is not None else args.fs)

    pre_cfg = PreprocessConfig(fs_hz=fs_hz, powerline_hz=float(args.powerline_hz))
    preprocessed, _detect_sig, rpeaks_auto = preprocess_emg_for_ats(raw, config=pre_cfg)
    rpeaks = list(rpeaks_in) if rpeaks_in is not None else list(rpeaks_auto)

    ats_cfg = AtsConfig(
        fs_hz=fs_hz,
        template_neighbor_beats=int(args.template_neighbor_beats),
        qrs_half_window_s=float(args.qrs_half_window_s),
        qrs_inc_max_samples=int(args.qrs_inc_max),
    )
    result = adaptive_template_subtraction(preprocessed, rpeaks, config=ats_cfg)

    ext = Path(args.output).suffix.lower()
    if ext == ".csv":
        write_output_csv(
            args.output,
            raw=list(raw),
            preprocessed=preprocessed,
            cleaned=result.cleaned,
            artifact=result.artifact,
            rpeaks=rpeaks,
        )
    elif ext == ".npz":
        write_output_npz(
            args.output,
            raw=list(raw),
            preprocessed=preprocessed,
            cleaned=result.cleaned,
            artifact=result.artifact,
            rpeaks=rpeaks,
            fs_hz=fs_hz,
        )
    else:
        raise ValueError("Output extension must be .csv or .npz")

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"fs_hz={fs_hz}")
    print(f"samples={len(raw)}")
    print(f"rpeaks={sum(1 for v in rpeaks if int(v) != 0)}")
    print(f"processed_beats={result.processed_beats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
