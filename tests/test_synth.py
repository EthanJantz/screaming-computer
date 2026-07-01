"""Tests for the synth voice: rendering sanity, param derivation, roughness."""

import time

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


def test_jitter_control_continuous_across_blocks():
    """The F0 wobble must not step at block seams (soundgen-style smooth jitter)."""
    voice = make_voice()
    params = neutral_params(jitter_amount=config.JITTER_MAX_ST)
    m1, _ = voice._f0_multiplier(BLOCK, params)
    m2, _ = voice._f0_multiplier(BLOCK, params)
    steps = np.abs(np.diff(np.log2(np.concatenate([m1, m2]))))
    seam = steps[BLOCK - 1]
    interior = np.delete(steps, BLOCK - 1)
    # A discontinuous control would jump by a whole anchor at the seam, ~hold
    # times larger than any interpolated per-sample step.
    assert seam <= interior.max() * 2.0


def test_jitter_is_semitone_scaled_and_symmetric():
    """log2(m) should be zero-mean with sd tracking the semitone depth.

    Linear interpolation of unit-normal anchors has variance 2/3, so the
    expected sd is depth * sqrt(2/3) ~= 0.816 * depth semitones.
    """
    voice = make_voice()
    depth = 2.0
    params = neutral_params(jitter_amount=depth)
    ms = [voice._f0_multiplier(BLOCK, params)[0] for _ in range(500)]
    log_st = np.log2(np.concatenate(ms)) * 12.0  # deviation in semitones
    assert abs(log_st.mean()) < 0.1
    assert depth * 0.7 < log_st.std() < depth * 0.95


def _line_level(y: np.ndarray, freq_hz: float) -> float:
    """Peak spectral magnitude within a couple of bins of ``freq_hz``."""
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, 1.0 / config.SAMPLERATE)
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    return float(spec[max(0, idx - 2) : idx + 3].max())


def test_subharmonic_sidebands_span_the_spectrum():
    """With sub_amount > 0, sideband lines appear at (n + 1/2) * F0 (SUB_RATIO=2)."""
    f0 = config.F0
    y_on = render(make_voice(), neutral_params(sub_amount=0.4), blocks=200)
    y_off = render(make_voice(), neutral_params(sub_amount=0.0), blocks=200)
    for ratio in (0.5, 1.5, 2.5, 6.5):
        assert _line_level(y_on, ratio * f0) > 20.0 * _line_level(y_off, ratio * f0)


def test_sidebands_follow_the_spectral_envelope():
    """A low sideband should tower over one far up the rolloff."""
    y = render(make_voice(), neutral_params(sub_amount=0.4), blocks=200)
    assert _line_level(y, 1.5 * config.F0) > 10.0 * _line_level(y, 40.5 * config.F0)


def test_render_meets_realtime_budget():
    """Full roughness (doubled partial bank) must render well inside real time."""
    voice = make_voice()
    params = derive_params(1.0)
    render(voice, params, blocks=20)  # warm-up
    blocks = 200
    t0 = time.perf_counter()
    render(voice, params, blocks=blocks)
    elapsed = time.perf_counter() - t0
    budget = blocks * BLOCK / config.SAMPLERATE
    assert elapsed < 0.5 * budget


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
