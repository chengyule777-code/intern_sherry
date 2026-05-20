# ecg_removal_py

Standalone Python implementation of ECG artifact removal from EMG using **Adaptive Template Subtraction (ATS)** inspired by:

- `ecg-removal_matlab/code/template_subtraction/adaptive_template_subtraction.m`

## Scope

- Standalone package and CLI (no direct `rr_app` integration in this phase).
- Input from CSV (default) and optional NPZ.
- Output cleaned EMG, estimated ECG artifact, and detected/provided R-peaks.

## Quick start

```bash
python3 -m ecg_removal_py.cli \
  --input /path/to/input.csv \
  --output /path/to/cleaned.csv \
  --signal-column emg_raw_mV \
  --fs 1024
```

## Input assumptions

- Single EMG stream, evenly sampled.
- Units should be mV (or consistent arbitrary unit).
- If no `--rpeak-column` is provided, R-peaks are detected internally.

## Output columns (CSV)

- `index`
- `raw`
- `preprocessed`
- `cleaned`
- `artifact`
- `rpeak`

## Notes on MATLAB parity

- ATS flow, width sweep, correlation alignment, and section-wise scaling follow MATLAB logic.
- QRS stretching uses linear interpolation in this first Python version.
- Filtering and peak detection are deterministic and lightweight, designed for pure-Python environments.
