"""
Readable and efficient online FIR/IIR filters.

This module separates FIR and IIR execution and uses a Direct Form II
Transposed structure for IIR filters, which is efficient for sample-by-sample
filtering such as Butterworth filters designed by scipy.signal.butter.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum

import numpy as np
from scipy import signal


class FilterType(IntEnum):
    """Supported online filter categories."""

    FIR = 0
    IIR = 1


def _as_float_coefficients(name: str, coefficients: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(coefficients), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} coefficients must be a non-empty 1D sequence.")
    return values


class OnlineFIR:
    """Sample-by-sample FIR filter using input history only."""

    def __init__(self, coefficients: Iterable[float]):
        self.b = _as_float_coefficients("FIR", coefficients)
        self.input_history = np.zeros(len(self.b), dtype=float)

    def filter(self, input_value: float) -> float:
        self.input_history[1:] = self.input_history[:-1]
        self.input_history[0] = float(input_value)

        return float(np.dot(self.b, self.input_history))

    def filter_array(self, input_values: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(input_values), dtype=float)
        if values.ndim != 1:
            raise ValueError("FIR input values must be a 1D sequence.")
        if len(values) == 0:
            return np.array([], dtype=float)

        if len(self.b) == 1:
            output = self.b[0] * values
        else:
            zi = self._lfilter_state_from_history()
            output, _ = signal.lfilter(self.b, [1.0], values, zi=zi)

        recent = np.concatenate((values[::-1], self.input_history))
        self.input_history = recent[: len(self.b)].copy()
        return output

    def reset(self) -> None:
        self.input_history.fill(0.0)

    def _lfilter_state_from_history(self) -> np.ndarray:
        state = np.zeros(len(self.b) - 1, dtype=float)
        for state_index in range(len(state)):
            coeffs = self.b[state_index + 1 :]
            history = self.input_history[: len(coeffs)]
            state[state_index] = float(np.dot(coeffs, history))
        return state


class OnlineIIR:
    """Sample-by-sample IIR filter using Direct Form II Transposed state."""

    def __init__(
        self,
        numerator: Iterable[float],
        denominator: Iterable[float],
    ):
        self.b = _as_float_coefficients("IIR numerator", numerator)
        self.a = _as_float_coefficients("IIR denominator", denominator)
        if self.a[0] == 0:
            raise ValueError("IIR denominator coefficient a[0] must not be zero.")

        norm = self.a[0]
        self.b = self.b / norm
        self.a = self.a / norm

        self.order = max(len(self.a), len(self.b)) - 1
        self.state = np.zeros(self.order, dtype=float)
        self._b_padded = np.pad(self.b, (0, self.order + 1 - len(self.b)))
        self._a_padded = np.pad(self.a, (0, self.order + 1 - len(self.a)))

    def filter(self, input_value: float) -> float:
        input_value = float(input_value)
        if self.order == 0:
            return float(self.b[0] * input_value)

        y = self.b[0] * input_value + self.state[0]

        # Direct Form II Transposed keeps one compact state vector. Each state
        # combines the delayed feedforward input term and feedback output term
        # needed to produce the next filtered sample.
        self.state[:-1] = (
            self.state[1:]
            + self._b_padded[1:self.order] * input_value
            - self._a_padded[1:self.order] * y
        )
        self.state[-1] = self._b_padded[self.order] * input_value - self._a_padded[self.order] * y

        return float(y)

    def filter_array(self, input_values: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(input_values), dtype=float)
        if values.ndim != 1:
            raise ValueError("IIR input values must be a 1D sequence.")
        if len(values) == 0:
            return np.array([], dtype=float)

        if self.order == 0:
            return self.b[0] * values

        output, self.state = signal.lfilter(self.b, self.a, values, zi=self.state)
        return output

    def reset(self) -> None:
        self.state.fill(0.0)


class OnlineFilter:
    """
    Compatibility wrapper around clearer FIR/IIR implementations.

    type_of_filter:
        0 or FilterType.FIR -> FIR filter
        1 or FilterType.IIR -> IIR filter
    """

    def __init__(
        self,
        type_of_filter: int | FilterType,
        coefficients_in: Iterable[float],
        coefficients_out: Iterable[float] | None = None,
    ):
        self.type_of_filter = FilterType(type_of_filter)

        if self.type_of_filter == FilterType.FIR:
            self._filter = OnlineFIR(coefficients_in)
        elif self.type_of_filter == FilterType.IIR:
            if coefficients_out is None:
                raise ValueError("IIR filters require denominator coefficients.")
            self._filter = OnlineIIR(coefficients_in, coefficients_out)
        else:
            raise ValueError(f"Unsupported filter type: {type_of_filter}")

    def filter(self, input_value: float) -> float:
        return self._filter.filter(input_value)

    def filter_array(self, input_values: Iterable[float]) -> np.ndarray:
        return self._filter.filter_array(input_values)

    def reset(self) -> None:
        self._filter.reset()
