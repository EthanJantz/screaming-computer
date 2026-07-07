"""Live audio output: a sounddevice OutputStream whose callback owns all DSP.

Each block the callback: reads the shared target intensity, one-pole-smooths the
current intensity toward it (this smoothing *is* the attack/release envelope),
derives synth params, and renders. Keep it pure arithmetic — no allocation we can
avoid, no I/O, no locks (spec Section 5.5).
"""

from __future__ import annotations

from dataclasses import replace

import sounddevice as sd

import config
from state import State
from synth import GateEnvelope, Voice, derive_params


class AudioEngine:
    def __init__(self, state: State) -> None:
        self.state = state
        self.voice = Voice()
        self.intensity = 0.0  # smoothed; lives in the audio thread
        self.envelope = GateEnvelope(config.SAMPLERATE)  # note gate, audio thread
        self.stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time, status) -> None:
        # `status` flags underruns etc.; we can't safely print here, so ignore.
        params = self.state.params  # single read: the panel may swap it any time
        if params is None:
            target = self.state.target_intensity
            self.intensity += (target - self.intensity) * config.SMOOTH_COEFF
            params = derive_params(self.intensity)
        gate = self.state.gate  # single read: None means no note gating
        if gate is None:
            outdata[:, 0] = self.voice.render_block(frames, params)
            return
        # Note mode: pitch the voice by the (glided) note ratio, then shape the
        # whole block — breath floor included — with the gate envelope, so rests
        # are true silence.
        ratio, ramp = self.envelope.step(frames, gate, self.state.note_ratio)
        if ratio != 1.0:
            params = replace(params, pitch_scale=params.pitch_scale * ratio)
        outdata[:, 0] = self.voice.render_block(frames, params) * ramp

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
