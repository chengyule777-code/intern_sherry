from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rr_app.config import recording as recording_config
from rr_app.recording.core import EmgsDeviceRecorder, default_recording_session_paths, device_recording_paths

from emg_sim.packet_builder import build_emg_frame
from emg_sim.rr_sink import RrParserSink, make_emgs_state
from emg_sim.signal_model import SignalModel, get_scenario
from emg_sim.stream_runtime import StreamConfig, StreamRuntime


class EmgSimTests(unittest.TestCase):
    def test_packet_builder_emg_frame_decodes(self) -> None:
        dev = make_emgs_state(address="SIM:TEST:01")
        sink = RrParserSink(dev=dev)
        model = SignalModel(get_scenario("mixed_low_ecg"), seed=1)
        _, _, mixed = model.sample_packet(start_timestamp_ms=1_000, sample_count=100)
        frame = build_emg_frame(packet_id=9, timestamp_ms=1_000, mixed_samples_mv=mixed, snr=81)
        sink.push_notify(frame)

        self.assertEqual(dev.last_emg_packet_id, 9)
        self.assertEqual(dev.last_emg_packet_ts_ms, 1_000)
        self.assertEqual(len(dev.emg_buffer), 100)
        self.assertEqual(len(dev.emg_snr_buffer), 1)

    def test_scenario_ecg_amplitude_order(self) -> None:
        t_ms = 10_000
        rms_vals: dict[str, float] = {}
        for scenario in ("clean_emg", "mixed_low_ecg", "mixed_high_ecg"):
            model = SignalModel(get_scenario(scenario), seed=42)
            _, artifact, _ = model.sample_packet(start_timestamp_ms=t_ms, sample_count=100)
            rms = (sum(v * v for v in artifact) / max(1.0, float(len(artifact)))) ** 0.5
            rms_vals[scenario] = rms
        self.assertLess(rms_vals["clean_emg"], rms_vals["mixed_low_ecg"])
        self.assertLess(rms_vals["mixed_low_ecg"], rms_vals["mixed_high_ecg"])

    def test_runtime_to_rr_recorder_smoke(self) -> None:
        dev = make_emgs_state(address="SIM:TEST:02")
        sink = RrParserSink(dev=dev)

        with tempfile.TemporaryDirectory() as tmp:
            session = default_recording_session_paths(tmp, prefix="rr")
            paths = device_recording_paths(
                session,
                addr=dev.params.address,
                include_emg=True,
                include_imu=False,
                include_robot=False,
                prefix="rr",
            )
            recorder = EmgsDeviceRecorder(
                addr=dev.params.address,
                paths=paths,
                origin_s=0.0,
                emg_fields=list(recording_config.REC_EMG_FIELDS_ALL),
                imu_fields=list(recording_config.REC_IMU_FIELDS_ALL),
            )
            first_ts_s: float | None = None

            def emit_frame(frame: bytes) -> None:
                nonlocal first_ts_s
                sink.push_notify(frame)
                if first_ts_s is None and dev.last_emg_packet_ts_ms is not None:
                    first_ts_s = float(dev.last_emg_packet_ts_ms) / 1000.0
                    recorder.origin_s = float(first_ts_s)

            runtime = StreamRuntime(
                scenario=get_scenario("mixed_low_ecg"),
                config=StreamConfig(duration_s=0.5, include_imu=False, realtime=False, seed=3),
                emit_frame=emit_frame,
                on_packet=lambda: recorder.capture(dev),
            )
            stats = runtime.run()
            recorder.capture(dev)
            recorder.close()

            self.assertGreaterEqual(stats.emg_packets, 1)
            self.assertIsNotNone(paths.emg_csv)
            emg_csv = Path(str(paths.emg_csv))
            self.assertTrue(emg_csv.exists())
            lines = emg_csv.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0].split(",")[:3], ["t_ms", "state", "emg_raw_mV"])


if __name__ == "__main__":
    unittest.main()
