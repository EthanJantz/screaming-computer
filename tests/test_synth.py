"""Tests for the synth voice: rendering sanity, param derivation, roughness."""

import numpy as np
import pytest

import config
from synth import SynthParams, Voice, derive_params

SEED = 42
BLOCK = config.BLOCKSIZE


def make_voice(seed: int = SEED) -> Voice:
    np.random.seed(seed)  # ThinkDSP's breath noise draws from the global RNG
    voice = Voice()
    voice._rng = np.random.default_rng(seed)
    return voice


def render(voice: Voice, params: SynthParams, blocks: int = 100) -> np.ndarray:
    ys = [voice.render_block(BLOCK, params) for _ in range(blocks)]
    return np.concatenate(ys).astype(np.float64)


def neutral_params(**overrides) -> SynthParams:
    """A clean sustained tone; overrides switch individual ingredients on."""
    base = dict(
        amplitude=0.5,
        pitch_scale=1.0,
        rolloff_p=config.P_IDLE,
        breath_level=0.0,
        jitter_amount=0.0,
        shimmer_amount=0.0,
        sub_amount=0.0,
        chaos_amount=0.0,
        drive=1.0,
        drive_mix=0.0,
    )
    base.update(overrides)
    return SynthParams(**base)


# --- rendering sanity ---


@pytest.mark.parametrize("intensity", [0.0, 0.5, 1.0])
def test_render_finite_and_hard_limited(intensity, monkeypatch):
    monkeypatch.setitem(config.ROUGHNESS, "chaos", True)  # exercise every path
    voice = make_voice()
    y = render(voice, derive_params(intensity), blocks=200)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0


def test_block_boundaries_click_free():
    """Sample deltas across block seams must look like interior deltas."""
    voice = make_voice()
    y = render(voice, neutral_params(amplitude=0.8), blocks=100)
    deltas = np.abs(np.diff(y))
    seam_idx = np.arange(BLOCK - 1, deltas.size, BLOCK)
    seams = deltas[seam_idx]
    interior = np.delete(deltas, seam_idx)
    assert seams.max() <= interior.max() * 1.2


# --- param derivation ---


def test_roughness_toggles_gate_params(monkeypatch):
    gated = [
        ("jitter", "jitter_amount", 0.0),
        ("shimmer", "shimmer_amount", 0.0),
        ("subharmonics", "sub_amount", 0.0),
        ("chaos", "chaos_amount", 0.0),
        ("drive", "drive_mix", 0.0),
    ]
    for key, field, off_value in gated:
        monkeypatch.setitem(config.ROUGHNESS, key, False)
        assert getattr(derive_params(1.0), field) == off_value
        monkeypatch.setitem(config.ROUGHNESS, key, True)
        assert getattr(derive_params(1.0), field) != off_value
    monkeypatch.setitem(config.ROUGHNESS, "brightening", False)
    assert derive_params(1.0).rolloff_p == config.P_IDLE


def test_intensity_extremes():
    idle, full = derive_params(0.0), derive_params(1.0)
    assert idle.amplitude == pytest.approx(config.AMP_FLOOR)
    assert full.amplitude == pytest.approx(config.AMP_PEAK)
    assert idle.pitch_scale == pytest.approx(1.0)
    assert full.pitch_scale == pytest.approx(
        2.0 ** (config.F0_RISE_SEMITONES / 12.0)
    )
    assert derive_params(-1.0) == idle  # clipped
    assert derive_params(2.0) == full
