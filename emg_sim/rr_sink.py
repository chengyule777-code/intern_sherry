from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Callable

# rr_app's package import chain loads Qt controller modules from
# rr_app.devices.core.__init__. In headless simulator runs we only need
# parser/state classes, so we provide tiny stubs to avoid hard requiring PyQt6.
if "PyQt6" not in sys.modules:
    class _DummySignal:
        def emit(self, *_args: object, **_kwargs: object) -> None:
            return None

    qtcore = types.SimpleNamespace(
        QObject=type("QObject", (), {}),
        pyqtSignal=lambda *_args, **_kwargs: _DummySignal(),
        pyqtSlot=lambda *_args, **_kwargs: (lambda fn: fn),
    )
    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qtcore
    sys.modules["PyQt6"] = pyqt6

if "qasync" not in sys.modules:
    qasync = types.ModuleType("qasync")
    qasync.asyncSlot = lambda *_args, **_kwargs: (lambda fn: fn)
    sys.modules["qasync"] = qasync

# Some environments used for simulator-only runs do not have numpy installed.
# Parser-in-the-loop flow here only relies on raw packet decode paths that do
# not execute numpy-backed feature extraction unless those toggles are enabled.
if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_stub

from rr_app.devices.core.state import DeviceParams
from rr_app.devices.emgs.state import DeviceState


def make_emgs_state(*, address: str, name: str = "EMGS_SIM_01") -> DeviceState:
    params = DeviceParams(
        name=str(name),
        address=str(address),
        kind="EMGS",
        status="Connected",
        mode="EMG:RMS+RAW",
    )
    dev = DeviceState(params=params)
    dev.is_connected = True
    dev.is_streaming = True
    return dev


@dataclass(slots=True)
class RrParserSink:
    """Push raw notify bytes through rr_app parser state."""

    dev: DeviceState
    on_frame: Callable[[bytes], None] | None = None

    def push_notify(self, frame: bytes) -> None:
        if self.on_frame is not None:
            self.on_frame(bytes(frame))
        self.dev.decode_notification(bytes(frame))
