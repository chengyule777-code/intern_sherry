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


This is an example of the usage of this sEMG ECG removal package. This script simulates an sEMG amplifier.
However, it can be used the same way for real-time measurements by adapting the callback method.
The sEMG signals, which are used for simulation were also used in the paper of:
Petersen, E., Sauer, J., Graßhoff, J., and Rostalski, P.(2022). Removing Cardiac Artifacts From Single-Channel
    Respiratory Electromyograms. IEEE Access, 8, 30905-30917.
"""

import threading
import time

from PySide6 import QtCharts, QtCore
from PySide6.QtCharts import QLineSeries, QChartView, QValueAxis
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QMainWindow, QFrame, QVBoxLayout

from code.sEMG_online_filter import SEMGOnlineFilter


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.sampling_rate = 1024  # in Hz
        self.length_of_signal = 10  # in seconds
        self.num_channels = 2

        # hard-coded y-ranges of the signals for plotting ([[raw, filtered] of channel 1, [raw, filtered] of channel 2])
        # extend if you want to use more channels
        self.y_ranges = [[(-7, -5.5), (-0.025, 0.025)], [(-7, -5.5), (-0.01, 0.01)]]

        assert len(self.y_ranges) == self.num_channels, "you have to set the y-ranges for all channels."

        delay = 300

        self.semg_filter = SEMGOnlineFilter(self.num_channels, delay, self.sampling_rate)

        self.length_window = self.sampling_rate * self.length_of_signal

        self.data_raw_opened = []
        # if you want to use your own dataa, you might have to adjust the y-ranges above
        self.data_raw_opened.append(open("data/rr_emg_0602.csv", 'r'))
        self.data_raw_opened.append(open("data/rr_emg_0602.csv", 'r'))
        

        self._initialize_plotter(self.num_channels)

        self.thread = threading.Thread(target=self.shimmer_simulation)
        self.thread.start()

    def shimmer_simulation(self):
        while not self.isVisible():
            continue
        while self.isVisible():
            self._simulation_callback()

    def _simulation_callback(self):
        # simulating receiving data
        current_values = []
        for i in range(self.num_channels):
            current_values.append(get_next_data(1, self.data_raw_opened[i])[0])

        cur_pressure_value = get_next_data(1, self.pressure_opened)[0]

        # from here the same for a real measurement
        denoised_values, envelope_values = self.semg_filter.filter_sEMG_online(current_values)

        for i in range(self.num_channels):
            self.temp_raw[i][self.iteration] = QPointF(self.iteration / 1024, current_values[i])
            self.temp_filtered[i][self.iteration] = QPointF(self.iteration / 1024, denoised_values[i])
            self.temp_env[i][self.iteration] = QPointF(self.iteration / 1024, envelope_values[i] * 3)
        self.temp_pres[self.iteration] = QPointF(self.iteration / 1024, cur_pressure_value)

        # for real measurements this value can be lower
        if self.iteration % 70 == 0:
            with self.lock:
                for i in range(self.num_channels):
                    self.raw_signals[i].replace(self.temp_raw[i])
                    self.filtered_signals[i].replace(self.temp_filtered[i])
                    self.envelopes[i].replace(self.temp_env[i])
                self.pressure[0].replace(self.temp_pres)

                for chart in self.charts:
                    chart.update()
            # this line can be omitted in real measurements, only for performance in simulation
            time.sleep(0.035)

        self.iteration = (self.iteration + 1) % self.length_window

    # initializes the plotting window
    def _initialize_plotter(self, num_channels):
        self.lock = threading.Lock()
        self.setWindowTitle("EMG Plot")

        self.iteration = 0

        self.blue = QColor(61, 125, 212)
        self.orange = QColor(228, 141, 39)

        self.pen = QPen()
        self.pen.setColor(self.blue)
        self.pen.setWidth(1)

        self.orange_pen = QPen()
        self.orange_pen.setColor(self.orange)
        self.orange_pen.setWidth(2)

        # signals
        self.raw_signals = []
        self.filtered_signals = []
        self.envelopes = []
        self.pressure = []
        self._create_all_signals()

        # charts
        self.charts = []
        self.xAxes = []
        self.yAxes = []
        self.chart_views = []
        self._create_all_charts()

        # initialize temps
        self.temp_raw = [[] for _ in range(self.num_channels)]
        self.temp_filtered = [[] for _ in range(self.num_channels)]
        self.temp_env = [[] for _ in range(self.num_channels)]
        self.temp_pres = []
        for i in range(self.length_window):
            for j in range(2):
                self.temp_raw[j].append(QPointF(i / 1024, 0))
                self.temp_filtered[j].append(QPointF(i / 1024, 0))
                self.temp_env[j].append(QPointF(i / 1024, 0))
            self.temp_pres.append(QPointF(i / 1024, 0))

        centralFrame = QFrame()
        self.main_layout = QVBoxLayout()
        for chart_view in self.chart_views:
            self.main_layout.addWidget(chart_view)

        centralFrame.setLayout(self.main_layout)

        self.setCentralWidget(centralFrame)

    def _create_all_signals(self):
        for i in range(self.num_channels):
            self._create_signal(0, f"raw sEMG channel {i + 1}", self.pen)
            self._create_signal(1, f"filtered sEMG channel {i + 1}", self.pen)
            self._create_signal(2, f"envelope of filtered sEMG channel {i + 1}", self.orange_pen)
        self._create_signal(3, "pressure signal", self.pen)

    def _create_signal(self, signal_type, name, pen):
        list_to_append = self._get_corresponding_list_to_signal_type(signal_type)
        list_to_append.append(QLineSeries())
        list_to_append[-1].setName(name)
        list_to_append[-1].setPen(pen)

    def _create_all_charts(self):
        ranges = self.y_ranges
        for i, y_ranges in zip(range(self.num_channels), ranges):
            self._create_chart(f"raw sEMG channel {i + 1}", y_ranges[0], 0, i)
            self._create_chart(f"filtered sEMG channel {i + 1}", y_ranges[1], 1, i)
        self._create_chart("pressure", (130, 210), 3)

    def _create_chart(self, name, y_range, signal_type, channel=0):
        self.charts.append(QtCharts.QChart())
        self.charts[-1].setTitle(name)
        self.charts[-1].createDefaultAxes()

        self.xAxes.append(QValueAxis())
        self.xAxes[-1].setRange(0, self.length_of_signal)
        self.xAxes[-1].setTickCount(self.length_of_signal + 1)
        self.yAxes.append(QValueAxis())
        self.yAxes[-1].setRange(y_range[0], y_range[1])

        self.charts[-1].addAxis(self.xAxes[-1], QtCore.Qt.AlignmentFlag.AlignBottom)
        self.charts[-1].addAxis(self.yAxes[-1], QtCore.Qt.AlignmentFlag.AlignLeft)
        signal = self._get_corresponding_list_to_signal_type(signal_type)[channel]
        self.charts[-1].addSeries(signal)

        signal.attachAxis(self.xAxes[-1])
        signal.attachAxis(self.yAxes[-1])

        if signal_type == 1:
            signal = self._get_corresponding_list_to_signal_type(2)[channel]
            self.charts[-1].addSeries(signal)
            signal.attachAxis(self.xAxes[-1])
            signal.attachAxis(self.yAxes[-1])

        self.chart_views.append(QChartView())
        self.chart_views[-1].setChart(self.charts[-1])
        self.chart_views[-1].setRenderHint(QPainter.Antialiasing)

    def _get_corresponding_list_to_signal_type(self, signal_type: int):
        list_to_append = self.raw_signals
        if signal_type == 1:
            list_to_append = self.filtered_signals
        elif signal_type == 2:
            list_to_append = self.envelopes
        elif signal_type == 3:
            list_to_append = self.pressure
        return list_to_append


# This method is only used for simulating a measuring sEMG amplifier
def get_next_data(samples, file):
    """
    returns the next {sample} values from a file
    :param samples: the number of values to be returned
    :param file: the file you want to read from
    :return: the next {sample} values in an array
    """
    new_values = []
    for lines in range(0, samples):
        data_in_line = file.readline()  # read next line
        if data_in_line == '':
            new_values.append(-1)  # if eof return data and add a -1 at the end to communicate eof
            break
        else:
            new_values.append(float(data_in_line))  # add number at end of list (wrong way round)
    return new_values
