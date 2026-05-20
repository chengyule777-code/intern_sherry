from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    emg_noise_mv: float
    emg_burst_mv: float
    emg_burst_hz: float
    ecg_amplitude_mv: float
    ecg_hr_bpm: float
    mains_hum_mv: float = 0.0


_SCENARIOS: dict[str, ScenarioConfig] = {
    "clean_emg": ScenarioConfig(
        name="clean_emg",
        emg_noise_mv=0.035,
        emg_burst_mv=0.22,
        emg_burst_hz=82.0,
        ecg_amplitude_mv=0.01,
        ecg_hr_bpm=72.0,
        mains_hum_mv=0.005,
    ),
    "mixed_low_ecg": ScenarioConfig(
        name="mixed_low_ecg",
        emg_noise_mv=0.04,
        emg_burst_mv=0.22,
        emg_burst_hz=80.0,
        ecg_amplitude_mv=0.045,
        ecg_hr_bpm=70.0,
        mains_hum_mv=0.01,
    ),
    "mixed_high_ecg": ScenarioConfig(
        name="mixed_high_ecg",
        emg_noise_mv=0.045,
        emg_burst_mv=0.21,
        emg_burst_hz=78.0,
        ecg_amplitude_mv=0.10,
        ecg_hr_bpm=68.0,
        mains_hum_mv=0.012,
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    key = str(name or "").strip().lower()
    if key not in _SCENARIOS:
        valid = ", ".join(sorted(_SCENARIOS.keys()))
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid}")
    return _SCENARIOS[key]


class SignalModel:
    """Generate clean EMG, ECG artifact, and mixed stream samples in mV."""

    def __init__(self, scenario: ScenarioConfig, *, seed: int | None = None) -> None:
        self.scenario = scenario
        self._rng = random.Random(seed)
        self._phase_emg = self._rng.random() * 2.0 * math.pi
        self._phase_hum = self._rng.random() * 2.0 * math.pi
        self._burst_phase = self._rng.random() * 2.0 * math.pi

    @staticmethod
    def available_scenarios() -> tuple[str, ...]:
        return tuple(sorted(_SCENARIOS.keys()))

    def sample_packet(
        self,
        *,
        start_timestamp_ms: int,
        sample_count: int = 100,
        sample_rate_hz: int = 1000,
    ) -> tuple[list[float], list[float], list[float]]:
        """Return (emg_clean_mv, ecg_artifact_mv, mixed_mv) for one packet."""
        clean: list[float] = []
        artifact: list[float] = []
        mixed: list[float] = []

        # Signal synthesis is intentionally staged:
        # 1) generate EMG-like high-frequency content + smooth burst envelope,
        # 2) generate ECG contamination using beat-synchronous PQRST-like pulses,
        # 3) add mains hum and white noise,
        # 4) sum into final mixed sample.
        # Keeping each component explicit makes later ECG-removal evaluation simpler.
        for i in range(sample_count):
            t_s = (float(start_timestamp_ms) + float(i)) / 1000.0
            emg_clean_mv = self._emg_component(t_s=t_s)
            ecg_mv = self._ecg_component(t_s=t_s)
            noise_mv = self.scenario.emg_noise_mv * self._rng.gauss(0.0, 1.0)
            hum_mv = self.scenario.mains_hum_mv * math.sin(2.0 * math.pi * 50.0 * t_s + self._phase_hum)

            mixed_mv = emg_clean_mv + ecg_mv + noise_mv + hum_mv
            clean.append(float(emg_clean_mv))
            artifact.append(float(ecg_mv))
            mixed.append(float(mixed_mv))
        return clean, artifact, mixed

    def _emg_component(self, *, t_s: float) -> float:
        # Burst envelope emulates short contractions around the baseline.
        env = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 0.35 * t_s + self._burst_phase))
        carrier = math.sin(2.0 * math.pi * self.scenario.emg_burst_hz * t_s + self._phase_emg)
        sideband = math.sin(2.0 * math.pi * (self.scenario.emg_burst_hz * 0.5) * t_s + 0.7)
        return float(self.scenario.emg_burst_mv * env * (0.8 * carrier + 0.2 * sideband))

    def _ecg_component(self, *, t_s: float) -> float:
        beat_period_s = 60.0 / max(20.0, float(self.scenario.ecg_hr_bpm))
        phase = (t_s / beat_period_s) % 1.0
        amp = float(self.scenario.ecg_amplitude_mv)

        # Compact PQRST template built from Gaussian bumps.
        p = 0.15 * math.exp(-((phase - 0.18) / 0.030) ** 2)
        q = -0.20 * math.exp(-((phase - 0.38) / 0.010) ** 2)
        r = 1.00 * math.exp(-((phase - 0.40) / 0.008) ** 2)
        s = -0.25 * math.exp(-((phase - 0.43) / 0.012) ** 2)
        t = 0.35 * math.exp(-((phase - 0.68) / 0.060) ** 2)
        return float(amp * (p + q + r + s + t))
