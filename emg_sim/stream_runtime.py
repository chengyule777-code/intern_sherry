from __future__ import annotations

import dataclasses
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

from .packet_builder import build_emg_frame, build_imu_frame
from .signal_model import ScenarioConfig, SignalModel


@dataclass(frozen=True, slots=True)
class StreamConfig:
    duration_s: float = 10.0
    emg_packet_period_s: float = 0.1
    emg_samples_per_packet: int = 100
    emg_sample_rate_hz: int = 1000
    include_imu: bool = True
    imu_sampling_hz: int = 100
    imu_samples_per_packet: int = 3
    imu_packet_drop_prob: float = 0.10
    realtime: bool = True
    timestamp_quantum_ms: int = 10
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class StreamStats:
    emg_packets: int
    imu_packets: int
    started_timestamp_ms: int
    ended_timestamp_ms: int


class StreamRuntime:
    def __init__(
        self,
        *,
        scenario: ScenarioConfig,
        config: StreamConfig,
        emit_frame: Callable[[bytes], None],
        on_packet: Callable[[], None] | None = None,
    ) -> None:
        self._scenario = scenario
        self._config = config
        self._emit_frame = emit_frame
        self._on_packet = on_packet
        self._signal = SignalModel(scenario, seed=config.seed)
        self._rng = random.Random(config.seed)

    def run(self) -> StreamStats:
        cfg = self._config
        start_ms = self._quantized_now_ms(cfg.timestamp_quantum_ms)
        current_ms = int(start_ms)
        emg_packet_id = 0
        imu_packet_id = 0
        emg_packets = 0
        imu_packets = 0
        packet_period_ms = int(round(float(cfg.emg_packet_period_s) * 1000.0))
        deadline_s = time.monotonic() + max(0.0, float(cfg.duration_s))
        max_packets = max(1, int(round(max(0.0, float(cfg.duration_s)) / max(1e-6, float(cfg.emg_packet_period_s)))))
        packets_sent = 0

        while packets_sent < max_packets:
            clean, artifact, mixed = self._signal.sample_packet(
                start_timestamp_ms=current_ms,
                sample_count=cfg.emg_samples_per_packet,
                sample_rate_hz=cfg.emg_sample_rate_hz,
            )
            emg_packet_id = (emg_packet_id + 1) & 0xFFFF
            snr = self._estimate_snr_db(clean_mv=clean, artifact_mv=artifact)
            frame = build_emg_frame(
                packet_id=emg_packet_id,
                timestamp_ms=current_ms,
                mixed_samples_mv=mixed,
                snr=snr,
            )
            self._emit_frame(frame)
            emg_packets += 1

            if cfg.include_imu:
                imu_frames = self._build_imu_frames(
                    base_timestamp_ms=current_ms,
                    packet_id_start=imu_packet_id,
                )
                imu_packet_id = (imu_packet_id + len(imu_frames)) & 0xFFFF
                for imu_frame in imu_frames:
                    self._emit_frame(imu_frame)
                    imu_packets += 1

            if self._on_packet is not None:
                self._on_packet()
            packets_sent += 1

            current_ms += packet_period_ms
            if cfg.realtime:
                now = time.monotonic()
                if now >= deadline_s:
                    break
                sleep_s = min(float(cfg.emg_packet_period_s), max(0.0, deadline_s - now))
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        return StreamStats(
            emg_packets=emg_packets,
            imu_packets=imu_packets,
            started_timestamp_ms=start_ms,
            ended_timestamp_ms=current_ms,
        )

    def _build_imu_frames(self, *, base_timestamp_ms: int, packet_id_start: int) -> list[bytes]:
        cfg = self._config
        out: list[bytes] = []
        sensor_types = (1, 4, 6)  # acc, gyr, mag

        # IMU emission is intentionally branchy to mimic on-device variability:
        # 1) randomly drop entire sensor packets to emulate BLE/radio loss,
        # 2) apply bounded timestamp jitter while preserving 10 ms quantization,
        # 3) build small vector batches compatible with rr_app _decode_imu_fw.
        for idx, sensor_type in enumerate(sensor_types):
            if self._rng.random() < cfg.imu_packet_drop_prob:
                continue
            packet_id = (int(packet_id_start) + idx + 1) & 0xFFFF
            jitter_ms = self._rng.choice((-10, 0, 0, 0, 10))
            ts_ms = int(base_timestamp_ms + jitter_ms)
            ts_ms = int(ts_ms - (ts_ms % max(1, cfg.timestamp_quantum_ms)))
            samples = self._imu_samples(sensor_type=sensor_type, count=cfg.imu_samples_per_packet)
            out.append(
                build_imu_frame(
                    packet_id=packet_id,
                    sensor_type=sensor_type,
                    timestamp_ms=ts_ms,
                    sampling_hz=cfg.imu_sampling_hz,
                    samples_xyz=samples,
                )
            )
        return out

    def _imu_samples(self, *, sensor_type: int, count: int) -> list[tuple[float, float, float]]:
        t = time.monotonic()
        vals: list[tuple[float, float, float]] = []
        for i in range(max(1, int(count))):
            dt = float(i) / 100.0
            if int(sensor_type) == 1:
                vals.append(
                    (
                        0.06 * math.sin(2.0 * math.pi * 1.2 * (t + dt)),
                        0.05 * math.sin(2.0 * math.pi * 0.9 * (t + dt + 0.2)),
                        1.0 + 0.02 * math.sin(2.0 * math.pi * 0.5 * (t + dt)),
                    )
                )
            elif int(sensor_type) == 4:
                vals.append(
                    (
                        2.0 * math.sin(2.0 * math.pi * 0.8 * (t + dt)),
                        2.2 * math.sin(2.0 * math.pi * 1.1 * (t + dt + 0.1)),
                        1.8 * math.sin(2.0 * math.pi * 0.6 * (t + dt + 0.3)),
                    )
                )
            else:
                vals.append(
                    (
                        30.0 + 0.5 * math.sin(2.0 * math.pi * 0.2 * (t + dt)),
                        5.0 + 0.4 * math.sin(2.0 * math.pi * 0.15 * (t + dt + 0.1)),
                        -40.0 + 0.5 * math.sin(2.0 * math.pi * 0.25 * (t + dt + 0.25)),
                    )
                )
        return vals

    @staticmethod
    def _estimate_snr_db(*, clean_mv: list[float], artifact_mv: list[float]) -> int:
        p_signal = sum(v * v for v in clean_mv) / max(1.0, float(len(clean_mv)))
        p_noise = sum(v * v for v in artifact_mv) / max(1.0, float(len(artifact_mv)))
        if p_noise <= 1e-12:
            return 99
        snr = 10.0 * math.log10(max(1e-12, p_signal / p_noise))
        return max(0, min(99, int(round(snr + 45.0))))

    @staticmethod
    def _quantized_now_ms(quantum_ms: int) -> int:
        q = max(1, int(quantum_ms))
        ms = int(time.time() * 1000.0)
        return int(ms - (ms % q))
