"""
MIT License

Copyright (c) 2024 Josefine Petrick, Institute for Electrical Engineering in Medicine - Universität zu Lübeck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Single-channel simulation app for online sEMG ECG removal.
It reads one raw EMG stream from a CSV file and visualizes:
- raw sEMG
- filtered sEMG
- filtered envelope
"""

import argparse
import threading
import time
from pathlib import Path

from PySide6 import QtCharts, QtCore
from PySide6.QtCharts import QLineSeries, QChartView, QValueAxis
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout

from code.sEMG_online_filter import SEMGOnlineFilter


class SingleChannelCsvReader:
    """
    Stream-like reader for single-column CSV data.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.file = open(csv_path, "r", encoding="utf-8")

    def close(self):
        self.file.close()

    def get_next_sample(self) -> float:
        """
        Return next sample from file. At EOF it restarts from the beginning
        so the simulation can keep running continuously.
        """
        wrapped_once = False

        # The loop below handles typical real-world CSV edge cases:
        # 1) skip blank rows,
        # 2) parse first column only (single-channel input),
        # 3) on EOF, rewind once and continue streaming for simulation.
        while True:
            line = self.file.readline()
            if line == "":
                if wrapped_once:
                    return 0.0
                self.file.seek(0)
                wrapped_once = True
                continue

            stripped = line.strip()
            if not stripped:
                continue

            first_column = stripped.split(",")[0].strip()
            try:
                return float(first_column)
            except ValueError:
                continue


class MainWindow(QMainWindow):
    def __init__(self, csv_path: Path):
        super().__init__()

        self.sampling_rate = 1024
        self.length_of_signal = 10
        self.delay = 300
        self.length_window = self.sampling_rate * self.length_of_signal

        self.reader = SingleChannelCsvReader(csv_path)
        self.semg_filter = SEMGOnlineFilter(1, self.delay, self.sampling_rate)

        # [raw, filtered]
        self.y_ranges = [(-0.2, 0.2), (-0.08, 0.08)]

        self._initialize_plotter()

        self.thread = threading.Thread(target=self.shimmer_simulation, daemon=True)
        self.thread.start()

    def closeEvent(self, event):
        self.reader.close()
        super().closeEvent(event)

    def shimmer_simulation(self):
        while not self.isVisible():
            time.sleep(0.01)
        while self.isVisible():
            self._simulation_callback()

    def _simulation_callback(self):
        current_value = self.reader.get_next_sample()
        denoised_value, envelope_value = self.semg_filter.filter_sEMG_online(current_value)

        x_value = self.iteration / self.sampling_rate
        self.temp_raw[self.iteration] = QPointF(x_value, current_value)
        self.temp_filtered[self.iteration] = QPointF(x_value, denoised_value)
        self.temp_env[self.iteration] = QPointF(x_value, envelope_value * 3)

        if self.iteration % 70 == 0:
            with self.lock:
                self.raw_signal.replace(self.temp_raw)
                self.filtered_signal.replace(self.temp_filtered)
                self.envelope_signal.replace(self.temp_env)

                for chart in self.charts:
                    chart.update()
            time.sleep(0.035)

        self.iteration = (self.iteration + 1) % self.length_window

    def _initialize_plotter(self):
        self.lock = threading.Lock()
        self.setWindowTitle("Single-Channel EMG Plot")
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

    def _create_chart(self, name: str, y_range: tuple, main_signal: QLineSeries, extra_signal: QLineSeries = None):
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
    parser = argparse.ArgumentParser(description="Single-channel sEMG simulation.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/rr_emg_0602.csv"),
        help="Path to single-column CSV/TXT file containing raw EMG values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv}")

    app = QApplication([])
    window = MainWindow(args.csv)
    window.show()
    app.exec()
