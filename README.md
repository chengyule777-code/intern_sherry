# intern_sherry

Utilities and notes for EMG/ECG artifact-removal workflow experiments.

## Included items

### `ecg_removal_py/`

Standalone Python ECG-removal package using Adaptive Template Subtraction (ATS).

- Removes ECG artifacts from EMG by preprocessing, R-peak detection, adaptive cardiac template fitting, and subtraction.
- Main CLI:
  - `python3 -m ecg_removal_py.cli --input <input.csv> --output <cleaned.csv> --signal-column emg_raw_mV --fs 1024`
- Output includes: `raw`, `preprocessed`, `cleaned`, `artifact`, `rpeak`.

### `emg_sim/`

Wire-compatible EMG simulator (based on `emg_sim/README.md`) for generating parser-compatible synthetic data.

- Emits `S/E/I` frames that match rr_app parser expectations.
- Generates repeatable EMG streams with controllable ECG contamination.
- Useful scenarios:
  - `clean_emg`
  - `mixed_low_ecg`
  - `mixed_high_ecg`
- Example:
  - `python3 -m emg_sim.run --duration-s 8 --scenario mixed_low_ecg --output-dir ./sim_out`

### `check_rms.py`

Quick RMS comparison helper for one selected segment of cleaned output.

- Reads `cleaned_result.csv`.
- Computes and prints RMS of:
  - `raw` signal
  - `cleaned` signal
- Used as a simple sanity check that ECG removal changes signal energy in a target interval.

## Notes

- If you pass wildcard input paths to the ATS CLI, let the shell expand them (do not glue option name and path). Example:
  - `--input ./test_data/rr_*/rr_emg*.csv`

## Full pipeline example

Three concrete steps for simulate -> clean -> RMS check:

```bash
# 1) Generate simulated EMG with strong ECG contamination
python3 -m emg_sim.run --duration-s 15 --scenario mixed_high_ecg --output-dir ./test_data

# 2) Run ATS ECG removal on one generated EMG CSV
PYTHONPATH=./ecg_removal_py python3 -m ecg_removal_py.cli \
  --input ./test_data/rr_YYYYMMDD_HHMMSS/rr_emg_SIM0101_YYYYMMDD_HHMMSS.csv \
  --output ./test_data/rr_YYYYMMDD_HHMMSS/cleaned_result.csv \
  --signal-column emg_raw_mV \
  --fs 1024

# 3) Check RMS before/after cleaning
python3 ./intern_sherry/check_rms.py
```

Replace `YYYYMMDD_HHMMSS` with the actual session folder and file timestamp printed by step 1.
