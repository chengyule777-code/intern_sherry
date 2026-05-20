from __future__ import annotations

import argparse
from pathlib import Path

from rr_app.config import recording as recording_config
from rr_app.recording.core import EmgsDeviceRecorder, default_recording_session_paths, device_recording_paths

from .rr_sink import RrParserSink, make_emgs_state
from .signal_model import SignalModel, get_scenario
from .stream_runtime import StreamConfig, StreamRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wire-compatible EMG stream simulator for rr_app.")
    parser.add_argument("--duration-s", type=float, default=10.0, help="Stream duration in seconds.")
    parser.add_argument("--scenario", type=str, default="mixed_low_ecg", choices=SignalModel.available_scenarios())
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic generation.")
    parser.add_argument("--output-dir", type=str, default="./sim_out", help="Base output folder.")
    parser.add_argument("--address", type=str, default="SIM:01:01", help="Logical simulated device address.")
    parser.add_argument("--name", type=str, default="EMGS_SIM_01", help="Logical simulated device name.")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU packet emission.")
    parser.add_argument("--fast", action="store_true", help="Run without real-time sleeps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scenario = get_scenario(args.scenario)
    dev = make_emgs_state(address=args.address, name=args.name)
    sink = RrParserSink(dev=dev)

    out_dir = Path(args.output_dir).expanduser().resolve()
    session = default_recording_session_paths(out_dir, prefix="rr")
    paths = device_recording_paths(
        session,
        addr=args.address,
        include_emg=True,
        include_imu=not bool(args.no_imu),
        include_robot=False,
        prefix="rr",
    )
    recorder = EmgsDeviceRecorder(
        addr=args.address,
        paths=paths,
        origin_s=0.0,  # patched after first packet timestamp is known
        emg_fields=list(recording_config.REC_EMG_FIELDS_ALL),
        imu_fields=list(recording_config.REC_IMU_FIELDS_ALL),
    )

    first_ts_s: float | None = None

    def on_frame(frame: bytes) -> None:
        nonlocal first_ts_s
        sink.push_notify(frame)
        ts_ms = dev.last_emg_packet_ts_ms if frame[1:2] == b"E" else dev.last_imu_packet_ts_ms
        if first_ts_s is None and ts_ms is not None:
            first_ts_s = float(ts_ms) / 1000.0
            recorder.origin_s = float(first_ts_s)

    runtime = StreamRuntime(
        scenario=scenario,
        config=StreamConfig(
            duration_s=max(0.0, float(args.duration_s)),
            include_imu=not bool(args.no_imu),
            realtime=not bool(args.fast),
            seed=args.seed,
        ),
        emit_frame=on_frame,
        on_packet=lambda: recorder.capture(dev),
    )
    stats = runtime.run()
    recorder.capture(dev)
    recorder.close()

    print(f"scenario={scenario.name} duration_s={args.duration_s} seed={args.seed}")
    print(f"emg_packets={stats.emg_packets} imu_packets={stats.imu_packets}")
    print(f"session_dir={session.root_dir}")
    if paths.emg_csv is not None:
        print(f"emg_csv={paths.emg_csv}")
    if paths.imu_csv is not None:
        print(f"imu_csv={paths.imu_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
