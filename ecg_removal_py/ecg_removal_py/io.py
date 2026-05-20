from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_signal(
    path: str | Path,
    *,
    signal_column: str = "emg_raw_mV",
    rpeak_column: str | None = None,
    fs_hz: float | None = None,
) -> tuple[list[float], list[int] | None, float | None]:
    src = Path(path)
    ext = src.suffix.lower()
    if ext == ".csv":
        return _load_csv(src, signal_column=signal_column, rpeak_column=rpeak_column, fs_hz=fs_hz)
    if ext == ".npz":
        return _load_npz(src, signal_column=signal_column, rpeak_column=rpeak_column, fs_hz=fs_hz)
    raise ValueError(f"Unsupported input extension: {ext}")


def write_output_csv(
    path: str | Path,
    *,
    raw: list[float],
    preprocessed: list[float],
    cleaned: list[float],
    artifact: list[float],
    rpeaks: list[int],
) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "raw", "preprocessed", "cleaned", "artifact", "rpeak"])
        n = min(len(raw), len(preprocessed), len(cleaned), len(artifact), len(rpeaks))
        for i in range(n):
            w.writerow([i, raw[i], preprocessed[i], cleaned[i], artifact[i], int(rpeaks[i])])


def write_output_npz(
    path: str | Path,
    *,
    raw: list[float],
    preprocessed: list[float],
    cleaned: list[float],
    artifact: list[float],
    rpeaks: list[int],
    fs_hz: float,
) -> None:
    np = _require_numpy()
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dst),
        raw=np.asarray(raw, dtype=float),
        preprocessed=np.asarray(preprocessed, dtype=float),
        cleaned=np.asarray(cleaned, dtype=float),
        artifact=np.asarray(artifact, dtype=float),
        rpeak=np.asarray(rpeaks, dtype=int),
        fs_hz=float(fs_hz),
    )


def _load_csv(
    path: Path,
    *,
    signal_column: str,
    rpeak_column: str | None,
    fs_hz: float | None,
) -> tuple[list[float], list[int] | None, float | None]:
    raw: list[float] = []
    rpeaks: list[int] | None = [] if rpeak_column else None
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if signal_column not in (rdr.fieldnames or []):
            known = ", ".join(rdr.fieldnames or [])
            raise ValueError(f"Signal column '{signal_column}' not found. Available columns: {known}")
        if rpeak_column and rpeak_column not in (rdr.fieldnames or []):
            known = ", ".join(rdr.fieldnames or [])
            raise ValueError(f"R-peak column '{rpeak_column}' not found. Available columns: {known}")
        for row in rdr:
            raw.append(float(row[signal_column]))
            if rpeaks is not None and rpeak_column is not None:
                rpeaks.append(int(float(row[rpeak_column])))
    return raw, rpeaks, fs_hz


def _load_npz(
    path: Path,
    *,
    signal_column: str,
    rpeak_column: str | None,
    fs_hz: float | None,
) -> tuple[list[float], list[int] | None, float | None]:
    np = _require_numpy()
    payload = np.load(str(path))
    if signal_column in payload:
        raw = payload[signal_column]
    elif "emg" in payload:
        raw = payload["emg"]
    elif "raw" in payload:
        raw = payload["raw"]
    else:
        raise ValueError(f"NPZ missing signal key '{signal_column}', and fallback keys emg/raw not found.")

    rpeaks: list[int] | None = None
    if rpeak_column and rpeak_column in payload:
        rpeaks = [int(v) for v in payload[rpeak_column].reshape(-1).tolist()]
    elif "rpeak" in payload:
        rpeaks = [int(v) for v in payload["rpeak"].reshape(-1).tolist()]

    fs_out = fs_hz
    if fs_out is None and "fs_hz" in payload:
        fs_out = float(payload["fs_hz"].reshape(()))
    return [float(v) for v in raw.reshape(-1).tolist()], rpeaks, fs_out


def _require_numpy() -> Any:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("NumPy is required for NPZ operations. Install numpy or use CSV input/output.") from exc
    return np
