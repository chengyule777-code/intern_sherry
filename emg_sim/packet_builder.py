from __future__ import annotations

import math
import struct
from typing import Iterable

from rr_app.config import emgs as emgs_config


def mv_to_u16(sample_mv: float) -> int:
    # Inverse of rr_app DeviceState._emg_u16_to_mv conversion.
    v_out = (float(sample_mv) / float(emgs_config.EMG_OUTPUT_SCALE)) * float(emgs_config.EMG_GAIN_V_PER_V)
    adc_v = v_out + float(emgs_config.EMG_ADC_MID_V)
    raw = int(round((adc_v / float(emgs_config.EMG_ADC_FULL_SCALE_V)) * float(emgs_config.EMG_ADC_MAX)))
    return max(0, min(65535, raw))


def build_emg_frame(
    *,
    packet_id: int,
    timestamp_ms: int,
    mixed_samples_mv: Iterable[float],
    snr: int = 80,
    battery_raw: int = 180,
    charge_state: int = 0,
    mode_byte: int = 0,
) -> bytes:
    samples_u16 = [mv_to_u16(v) for v in mixed_samples_mv]
    if len(samples_u16) != 100:
        raise ValueError(f"EMG frame requires exactly 100 samples, got {len(samples_u16)}")

    centered = [float(x) - 32768.0 for x in samples_u16]
    rms_u16 = int(round(math.sqrt(sum(v * v for v in centered) / float(len(centered)))))

    payload = bytearray()
    payload.append(int(battery_raw) & 0xFF)
    payload.append(int(charge_state) & 0xFF)
    payload.extend(int(packet_id & 0xFFFF).to_bytes(2, "little"))
    payload.extend(int(timestamp_ms).to_bytes(8, "little", signed=False))
    payload.append(int(mode_byte) & 0xFF)
    payload.extend(struct.pack(">H", int(rms_u16) & 0xFFFF))
    payload.append(int(snr) & 0xFF)
    payload.extend(struct.pack(">100H", *samples_u16))

    if len(payload) != 216:
        raise RuntimeError(f"Unexpected EMG payload size: {len(payload)}")

    frame = bytearray()
    frame.extend(b"SE")
    frame.append(len(payload) & 0xFF)
    frame.extend(payload)
    return bytes(frame)


def build_imu_frame(
    *,
    packet_id: int,
    sensor_type: int,
    timestamp_ms: int,
    sampling_hz: int,
    samples_xyz: list[tuple[float, float, float]],
) -> bytes:
    payload = bytearray()
    payload.extend(int(packet_id & 0xFFFF).to_bytes(2, "little"))
    payload.append(int(sensor_type) & 0xFF)
    payload.extend(int(timestamp_ms).to_bytes(8, "little", signed=False))
    payload.append(int(sampling_hz) & 0xFF)
    for x, y, z in samples_xyz:
        payload.extend(struct.pack("<3f", float(x), float(y), float(z)))

    frame = bytearray()
    frame.extend(b"SI")
    frame.append(len(payload) & 0xFF)
    frame.extend(payload)
    return bytes(frame)
