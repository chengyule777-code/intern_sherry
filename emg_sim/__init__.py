"""Wire-compatible EMG stream simulator for rr_app."""

from .packet_builder import build_emg_frame, build_imu_frame
from .rr_sink import RrParserSink, make_emgs_state
from .signal_model import ScenarioConfig, SignalModel, get_scenario
from .stream_runtime import StreamConfig, StreamRuntime

__all__ = [
    "RrParserSink",
    "ScenarioConfig",
    "SignalModel",
    "StreamConfig",
    "StreamRuntime",
    "build_emg_frame",
    "build_imu_frame",
    "get_scenario",
    "make_emgs_state",
]
