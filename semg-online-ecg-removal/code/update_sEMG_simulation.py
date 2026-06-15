"""
Updated single-channel simulation app for online sEMG ECG removal.

The script prefers available ``update_*.py`` processing modules and falls back
to original implementations for stages that have not been rewritten yet.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from PySide6 import QtCharts, QtCore
from PySide6.QtCharts import QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QFrame, QMainWindow, QVBoxLayout


def _load_processing_class(
    preferred_module: str,
    fallback_module: str,
    preferred_class_name: str,
    fallback_class_name: str | None = None,
):
    """
    Prefer an updated processing module, but fall back only when that module does not exist.

    Import errors raised from inside an existing updated module are allowed to surface, because
    silently falling back would hide a broken future implementation.
    """
    try:
        module = import_module(preferred_module)
        class_name = preferred_class_name
    except ModuleNotFoundError as exc:
        if exc.name != preferred_module:
            raise
        module = import_module(fallback_module)
        class_name = fallback_class_name or preferred_class_name

    return getattr(module, class_name), module.__name__


QrsDetectorClass, QRS_SOURCE = _load_processing_class(
    "code.update_qrs_detection",
    "code.online_qrs_detection",
    "UpdatedQrsDetector",
    "QrsDetector",
)
HeartRateCalculatorClass, HEART_RATE_SOURCE = _load_processing_class(
    "code.update_heartbeat_calculating",
    "code.heartbeat_calculating",
    "HeartRateCalculator",
)
SwtEmgDenoiseClass, SWT_SOURCE = _load_processing_class(
    "code.update_online_semg_ecg_removal_multi_channel",
    "code.online_semg_ecg_removal_multi_channel",
    "SwtEmgDenoise",
)
EnvelopeCalculatorClass, ENVELOPE_SOURCE = _load_processing_class(
    "code.update_online_envelope",
    "code.online_envelope",
    "EnvelopeCalculator",
)


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

    def __init__(self, delay: int, fs: int, envelope_window: int = 256):
        if QRS_SOURCE == "code.online_qrs_detection":
            self.qrs_detector = QrsDetectorClass(delay)
        else:
            self.qrs_detector = QrsDetectorClass(delay, fs)
        self.heart_rate_calculator = HeartRateCalculatorClass(delay)
        self.swt_denoising = SwtEmgDenoiseClass(fs, delay, 1)
        self.envelope_calculator = EnvelopeCalculatorClass(False, envelope_window)

    def process_sample(self, measured_value: float) -> tuple[float, float]:
        peak = self.qrs_detector.qrs_detection(measured_value)
        heart_rate = self.heart_rate_calculator.get_heartrate(peak)
        denoised_value = self.swt_denoising.swt_emg_denoising([measured_value], peak, heart_rate)[0]
        envelope_value = self.envelope_calculator.calculate_envelope(denoised_value)

        return denoised_value, envelope_value


class MainWindow(QMainWindow):
    def __init__(self, csv_path: Path):
        super().__init__()

        self.sampling_rate = 1024
        self.length_of_signal = 10
        self.delay = 300
        self.length_window = self.sampling_rate * self.length_of_signal
        self.samples_per_tick = 70
        self.timer_interval_ms = 35

        self.reader = SingleChannelCsvReader(csv_path)
        self.pipeline = SingleChannelUpdatedPipeline(self.delay, self.sampling_rate)

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
        help="Print which updated/fallback processing modules are used before launching the UI.",
    )
    return parser.parse_args()


def print_pipeline_sources():
    print("Processing modules:")
    print(f"- QRS detection: {QRS_SOURCE}")
    print(f"- heart rate: {HEART_RATE_SOURCE}")
    print(f"- SWT denoising: {SWT_SOURCE}")
    print(f"- envelope: {ENVELOPE_SOURCE}")


if __name__ == "__main__":
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv}")

    if args.print_pipeline:
        print_pipeline_sources()

    app = QApplication([])
    window = MainWindow(args.csv)
    window.show()
    app.exec()
