from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AtsConfig:
    fs_hz: float = 1024.0
    template_neighbor_beats: int = 40
    qrs_half_window_s: float = 0.055
    qrs_inc_max_samples: int = 10


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    fs_hz: float = 1024.0
    powerline_hz: float = 50.0
    notch_q: float = 30.0
    notch_passes: int = 2
    hp_detect_hz: float = 5.0
    hp_clean_hz: float = 20.0
    hp_order: int = 3
    min_rr_s: float = 0.22
    detect_window_s: float = 0.15
    detect_threshold_scale: float = 0.5
