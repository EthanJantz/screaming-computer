"""Shared state between the (future) driver thread and the audio callback.

Cross-thread communication is deliberately tiny: a single float target the audio
callback smooths toward. Assigning a Python float is atomic under the GIL, so no
lock is needed for `target_intensity`. `telemetry` is a free-form dict for display
only and must never be read inside the audio callback.

`params` is the experimentation escape hatch: when a driver (the param panel)
publishes a full `SynthParams` here, the audio callback uses it verbatim instead
of smoothing/deriving from `target_intensity`. Publishers must assign a fresh
object rather than mutating in place, so each block reads a consistent snapshot
(reference assignment is atomic under the GIL, field-by-field mutation is not).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth import SynthParams


class State:
    def __init__(self) -> None:
        self.target_intensity: float = 0.0
        self.telemetry: dict = {}
        self.params: SynthParams | None = None
