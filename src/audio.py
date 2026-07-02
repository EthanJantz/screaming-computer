"""Live audio output: a sounddevice OutputStream whose callback owns all DSP.

Each block the callback: reads the shared target intensity, one-pole-smooths the
current intensity toward it (this smoothing *is* the attack/release envelope),
derives synth params, and renders. Keep it pure arithmetic — no allocation we can
avoid, no I/O, no locks (spec Section 5.5).
"""

from __future__ import annotations

import sounddevice as sd

import config
from state import State
from synth import Voice, derive_params


class AudioEngine:
    def __init__(self, state: State) -> None:
        self.state = state
        self.voice = Voice()
        self.intensity = 0.0  # smoothed; lives in the audio thread
        self.stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time, status) -> None:
        # `status` flags underruns etc.; we can't safely print here, so ignore.
        params = self.state.params  # single read: the panel may swap it any time
        if params is None:
            target = self.state.target_intensity
            self.intensity += (target - self.intensity) * config.SMOOTH_COEFF
            params = derive_params(self.intensity)
        outdata[:, 0] = self.voice.render_block(frames, params)

    def start(self) -> None:
        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE,
            channels=1,
            dtype="float32",
            blocksize=config.BLOCKSIZE,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
