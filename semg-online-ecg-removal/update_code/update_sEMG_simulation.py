"""
Updated single-channel simulation app for online sEMG ECG removal.

The processing chain intentionally imports only ``update_*.py`` modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PySide6 import QtCharts, QtCore
from PySide6.QtCharts import QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QFrame, QMainWindow, QVBoxLayout

from update_code.update_heartbeat_calculation import UpdatedHeartRateCalculator
from update_code.update_online_envelope import UpdatedEnvelopeCalculator
from update_code.update_online_semg_ecg_removal import UpdatedSwtEmgDenoise
from update_code.update_qrs_detection import UpdatedQrsDetector


DEFAULT_SAMPLING_RATE = 1024
DEFAULT_DELAY_S = 300 / 1024
DEFAULT_ENVELOPE_WINDOW_S = 256 / 1024


class SingleChannelCsvReader:
    """
    Memory-backed stream-like reader for single-column CSV data.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.samples = self._load_samples(csv_path)
        self.next_index = 0

    def get_next_sample(self) -> float:
        sample = self.samples[self.next_index]
        self.next_index = (self.next_index + 1) % len(self.samples)
        return sample

    @staticmethod
    def _load_samples(csv_path: Path) -> list[float]:
        samples = []
        with csv_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    samples.append(float(stripped.split(",", maxsplit=1)[0].strip()))
                except ValueError:
                    continue

        if not samples:
            raise ValueError(f"No numeric samples found in input file: {csv_path}")

        return samples


class SingleChannelUpdatedPipeline:
    """
    Explicit single-channel processing chain for updated simulations.
    """

    def __init__(
        self,
        fs: int,
        delay_s: float,
        envelope_window_s: float = DEFAULT_ENVELOPE_WINDOW_S,
    ):
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if delay_s < 0:
            raise ValueError("delay_s must be non-negative.")
        if envelope_window_s <= 0:
            raise ValueError("envelope_window_s must be positive.")

        self.fs = fs
        self.dt_s = 1.0 / fs
        self.delay_s = delay_s
        self.envelope_window_s = envelope_window_s
        # The QRS detector still exposes its delay as samples, so convert the
        # physical delay at this boundary and keep the rest of the pipeline in seconds.
        self.delay_samples = int(round(delay_s * fs))

        self.qrs_detector = UpdatedQrsDetector(self.delay_samples, fs)
        self.heart_rate_calculator = UpdatedHeartRateCalculator(fs)
        self.swt_denoising = UpdatedSwtEmgDenoise(fs, delay_s)
        self.envelope_calculator = UpdatedEnvelopeCalculator(fs, envelope_window_s)

    def process_sample(self, measured_value: float) -> tuple[float, float]:
        peak = self.qrs_detector.qrs_detection(measured_value, self.dt_s)
        rr_interval_s = self.heart_rate_calculator.get_rr_interval_s(peak, self.dt_s)
        denoised_value = self.swt_denoising.swt_emg_denoising(
            measured_value,
            peak,
            rr_interval_s,
            self.dt_s,
        )
        envelope_value = self.envelope_calculator.calculate_envelope(denoised_value)

        return denoised_value, envelope_value


class MainWindow(QMainWindow):
    def __init__(
        self,
        csv_path: Path,
        fs: int = DEFAULT_SAMPLING_RATE,
        delay_s: float = DEFAULT_DELAY_S,
        envelope_window_s: float = DEFAULT_ENVELOPE_WINDOW_S,
    ):
        super().__init__()

        self.sampling_rate = fs
        self.length_of_signal = 10
        self.delay_s = delay_s
        self.envelope_window_s = envelope_window_s
        self.length_window = self.sampling_rate * self.length_of_signal
        self.timer_interval_ms = 35
        self.samples_per_tick = max(1, int(round(self.sampling_rate * self.timer_interval_ms / 1000)))

        self.reader = SingleChannelCsvReader(csv_path)
        self.pipeline = SingleChannelUpdatedPipeline(
            self.sampling_rate,
            self.delay_s,
            self.envelope_window_s,
        )

        self.y_ranges = [(-0.2, 0.2), (-0.08, 0.08)]
        self._initialize_plotter()
        self._initialize_timer()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

    def _initialize_plotter(self):
        self.setWindowTitle("Updated Single-Channel EMG Plot")
        self.iteration = 0

        blue = QColor(61, 125, 212)
        orange = QColor(228, 141, 39)

        blue_pen = QPen()
        blue_pen.setColor(blue)
        blue_pen.setWidth(1)

        orange_pen = QPen()
        orange_pen.setColor(orange)
        orange_pen.setWidth(2)

        self.raw_signal = QLineSeries()
        self.raw_signal.setName("raw sEMG")
        self.raw_signal.setPen(blue_pen)

        self.filtered_signal = QLineSeries()
        self.filtered_signal.setName("filtered sEMG")
        self.filtered_signal.setPen(blue_pen)

        self.envelope_signal = QLineSeries()
        self.envelope_signal.setName("envelope of filtered sEMG")
        self.envelope_signal.setPen(orange_pen)

        self.charts = []
        self.chart_views = []
        self._create_chart("raw sEMG", self.y_ranges[0], self.raw_signal)
        self._create_chart("filtered sEMG", self.y_ranges[1], self.filtered_signal, self.envelope_signal)

        self.temp_raw = [QPointF(i / self.sampling_rate, 0) for i in range(self.length_window)]
        self.temp_filtered = [QPointF(i / self.sampling_rate, 0) for i in range(self.length_window)]
        self.temp_env = [QPointF(i / self.sampling_rate, 0) for i in range(self.length_window)]

        central_frame = QFrame()
        main_layout = QVBoxLayout()
        for chart_view in self.chart_views:
            main_layout.addWidget(chart_view)

        central_frame.setLayout(main_layout)
        self.setCentralWidget(central_frame)

    def _initialize_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._process_timer_tick)
        self.timer.start(self.timer_interval_ms)

    def _process_timer_tick(self):
        # Each timer tick performs a bounded processing batch before one chart refresh.
        # This keeps sample throughput close to real time without a busy worker thread
        # or cross-thread chart updates.
        for _ in range(self.samples_per_tick):
            self._process_next_sample()

        self.raw_signal.replace(self.temp_raw)
        self.filtered_signal.replace(self.temp_filtered)
        self.envelope_signal.replace(self.temp_env)

    def _process_next_sample(self):
        current_value = self.reader.get_next_sample()
        denoised_value, envelope_value = self.pipeline.process_sample(current_value)

        x_value = self.iteration / self.sampling_rate
        self.temp_raw[self.iteration] = QPointF(x_value, current_value)
        self.temp_filtered[self.iteration] = QPointF(x_value, denoised_value)
        self.temp_env[self.iteration] = QPointF(x_value, envelope_value * 3)

        self.iteration = (self.iteration + 1) % self.length_window

    def _create_chart(self, name: str, y_range: tuple[float, float], main_signal: QLineSeries,
                      extra_signal: QLineSeries | None = None):
        chart = QtCharts.QChart()
        chart.setTitle(name)
        chart.createDefaultAxes()

        x_axis = QValueAxis()
        x_axis.setRange(0, self.length_of_signal)
        x_axis.setTickCount(self.length_of_signal + 1)

        y_axis = QValueAxis()
        y_axis.setRange(y_range[0], y_range[1])

        chart.addAxis(x_axis, QtCore.Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, QtCore.Qt.AlignmentFlag.AlignLeft)

        chart.addSeries(main_signal)
        main_signal.attachAxis(x_axis)
        main_signal.attachAxis(y_axis)

        if extra_signal is not None:
            chart.addSeries(extra_signal)
            extra_signal.attachAxis(x_axis)
            extra_signal.attachAxis(y_axis)

        chart_view = QChartView()
        chart_view.setChart(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)

        self.charts.append(chart)
        self.chart_views.append(chart_view)


def parse_args():
    parser = argparse.ArgumentParser(description="Updated single-channel sEMG simulation.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/rr_emg_0602.csv"),
        help="Path to single-column CSV/TXT file containing raw EMG values.",
    )
    parser.add_argument(
        "--print-pipeline",
        action="store_true",
        help="Print which updated processing modules are used before launching the UI.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=DEFAULT_SAMPLING_RATE,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=DEFAULT_DELAY_S,
        help="QRS output delay in seconds.",
    )
    parser.add_argument(
        "--envelope-window-s",
        type=float,
        default=DEFAULT_ENVELOPE_WINDOW_S,
        help="RMS envelope window duration in seconds.",
    )
    return parser.parse_args()


def print_pipeline_sources():
    print("Processing modules:")
    print("- QRS detection: code.update_qrs_detection.UpdatedQrsDetector")
    print("- heart rate: code.update_heartbeat_calculation.UpdatedHeartRateCalculator")
    print("- SWT denoising: code.update_online_semg_ecg_removal.UpdatedSwtEmgDenoise")
    print("- envelope: code.update_online_envelope.UpdatedEnvelopeCalculator")


if __name__ == "__main__":
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv}")

    if args.print_pipeline:
        print_pipeline_sources()

    app = QApplication([])
    window = MainWindow(
        args.csv,
        fs=args.fs,
        delay_s=args.delay_s,
        envelope_window_s=args.envelope_window_s,
    )
    window.show()
    app.exec()
