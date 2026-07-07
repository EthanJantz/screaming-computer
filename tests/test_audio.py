"""Tests for the audio callback's note-gating path (no stream: the callback is pure)."""

import numpy as np

import config
from audio import AudioEngine
from state import State
from synth import derive_params
from test_synth import BLOCK, _line_level, make_voice, neutral_params


def _callback_blocks(engine: AudioEngine, blocks: int) -> np.ndarray:
    out = np.zeros((BLOCK, 1), dtype=np.float32)
    ys = []
    for _ in range(blocks):
        engine._callback(out, BLOCK, None, None)
        ys.append(out[:, 0].copy())
    return np.concatenate(ys).astype(np.float64)


def test_gate_none_matches_direct_render():
    """With gate unset (every existing source), the callback is today's pipeline."""
    state = State()
    state.target_intensity = 0.7
    engine = AudioEngine(state)
    engine.voice = make_voice()
    y = _callback_blocks(engine, 50)

    reference = make_voice()
    intensity = 0.0
    ref_blocks = []
    for _ in range(50):
        intensity += (0.7 - intensity) * config.SMOOTH_COEFF
        ref_blocks.append(reference.render_block(BLOCK, derive_params(intensity)))
    assert np.array_equal(y, np.concatenate(ref_blocks).astype(np.float64))


def test_gate_closed_is_true_silence():
    state = State()
    state.params = neutral_params(amplitude=0.8, breath_level=0.05)
    state.gate = 0.0
    engine = AudioEngine(state)
    engine.voice = make_voice()
    y = _callback_blocks(engine, 20)
    assert np.max(np.abs(y)) == 0.0  # breath floor is gated too


def test_note_ratio_pitches_the_voice():
    state = State()
    state.params = neutral_params(amplitude=0.5)
    state.gate = 1.0
    state.note_ratio = 2.0
    engine = AudioEngine(state)
    engine.voice = make_voice()
    _callback_blocks(engine, 20)  # let the attack finish
    y = _callback_blocks(engine, 200)
    assert _line_level(y, 2.0 * config.F0) > 20.0 * _line_level(y, config.F0)
