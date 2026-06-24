"""Intensity sources — the central seam of the screaming computer.

A driver's entire contract with the core is `target() -> float` in 0..1: how hard
the underlying system is working right now. The audio callback smooths this; sources
report the instantaneous (un-smoothed) target only.

Every work source (manual, chess telemetry, CPU load, ...) is just an implementation
of `IntensitySource`. The synth and audio engine never import any of them directly —
they only ever see the smoothed float in shared state.
"""

from __future__ import annotations

import math
import time
from typing import Callable


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class IntensitySource:
    def target(self) -> float:
        """Return the desired intensity in 0..1 (un-smoothed)."""
        raise NotImplementedError


class ManualIntensity(IntensitySource):
    """Intensity set by hand (keyboard/slider). Used to test audio before any real
    work source exists."""

    def __init__(self, value: float = 0.0, step: float = 0.05) -> None:
        self._value = _clamp01(value)
        self.step = step

    def target(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = _clamp01(value)

    def increase(self) -> None:
        self._value = _clamp01(self._value + self.step)

    def decrease(self) -> None:
        self._value = _clamp01(self._value - self.step)


class FakeTurnIntensity(IntensitySource):
    """A timer that mimics engine 'turns' so we can exercise the per-turn envelope
    before any real work source exists.

    Each cycle is an idle gap (target 0) followed by a 'search' during which the
    target rises monotonically to ``peak`` — just as chess node-counts will. At the
    end of the search the target snaps back to 0; the audible *release* is produced
    by the audio callback's smoothing, not by ramping down here (spec milestone 3).
    """

    def __init__(
        self,
        idle: float = 2.0,
        search: float = 4.0,
        peak: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle = idle
        self.search = search
        self.peak = _clamp01(peak)
        self._clock = clock
        self._t0 = clock()

    def target(self) -> float:
        elapsed = (self._clock() - self._t0) % (self.idle + self.search)
        if elapsed < self.idle:
            return 0.0
        frac = (elapsed - self.idle) / self.search  # 0..1 rising ramp
        return self.peak * frac


class TelemetryIntensity(IntensitySource):
    """Maps an external, monotonically-rising work metric to 0..1 on a log scale.

    A driver feeds raw work counts (chess search nodes, files processed, ...) via
    ``update``; the audio side only ever sees ``target``. A value at/below ``lo``
    reads as idle (0), at/above ``hi`` as full effort (1). Log scaling because real
    work counts span orders of magnitude within a single burst (chess nodes climb
    from tens to millions in one search), and a linear map would sit pinned near 0
    for almost the whole swell.

    This source is deliberately work-agnostic: it knows nothing about chess. The
    driver that owns it picks ``lo``/``hi`` and decides what metric to feed.
    """

    def __init__(self, lo: float, hi: float) -> None:
        if lo <= 0.0 or hi <= lo:
            raise ValueError("require 0 < lo < hi")
        self._log_lo = math.log(lo)
        self._log_hi = math.log(hi)
        self._intensity = 0.0

    def update(self, value: float) -> None:
        """Feed the latest raw work metric; recompute the mapped intensity."""
        if value <= 0.0:
            return
        frac = (math.log(value) - self._log_lo) / (self._log_hi - self._log_lo)
        self._intensity = _clamp01(frac)

    def reset(self) -> None:
        """Back to idle — call when a burst of work ends (e.g. move resolved)."""
        self._intensity = 0.0

    def target(self) -> float:
        return self._intensity
