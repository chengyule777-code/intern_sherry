# emg_sim

Wire-compatible EMG simulator for `rr_app`.

This folder simulates an EMG sensor by emitting parser-compatible `S/E/I` frames and then routing them through `rr_app`'s existing `DeviceState.decode_notification()` path. The resulting buffers are recorded using `rr_app.recording` so output CSV files match normal rr_app recorder schema.

## What this is for

- Generate repeatable EMG streams with optional ECG contamination artifacts.
- Exercise `rr_app` parser + recorder end-to-end without hardware.
- Produce rr_app-compatible logs for ECG-removal algorithm prototyping.

## Packet compatibility

- EMG frame: `S E <payload_len=216> ...`
  - Battery, charging flag, packet id (LE), timestamp ms (LE), mode/reserved.
  - Data region:
    - RMS (`uint16`, big-endian)
    - SNR (`uint8`)
    - 100 raw samples (`uint16`, big-endian)
- IMU frame: `S I ...` compatible with `rr_app` `_decode_imu_fw()`.

## Quick start

Run a short stream and save rr_app-format CSV output:

```bash
python -m emg_sim.run --duration-s 8 --scenario mixed_low_ecg --output-dir ./sim_out
```

The command writes a session folder like:
- `rr_YYYYMMDD_HHMMSS/rr_emg_SIM0101_YYYYMMDD_HHMMSS.csv`
- `rr_YYYYMMDD_HHMMSS/rr_imu_SIM0101_YYYYMMDD_HHMMSS.csv`
- `rr_YYYYMMDD_HHMMSS/rr_meta_SIM0101_YYYYMMDD_HHMMSS.json`

## Scenarios

- `clean_emg`: minimal ECG artifact.
- `mixed_low_ecg`: moderate ECG contamination.
- `mixed_high_ecg`: strong ECG contamination.

All scenarios support reproducible generation via `--seed`.
