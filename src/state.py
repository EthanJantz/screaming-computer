"""Shared state between the (future) driver thread and the audio callback.

Cross-thread communication is deliberately tiny: a single float target the audio
callback smooths toward. Assigning a Python float is atomic under the GIL, so no
lock is needed for `target_intensity`. `telemetry` is a free-form dict for display
only and must never be read inside the audio callback.
"""


class State:
    def __init__(self) -> None:
        self.target_intensity: float = 0.0
        self.telemetry: dict = {}
