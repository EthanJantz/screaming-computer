"""Tests for the param panel: coverage of SynthParams, key handling, publishing."""

from dataclasses import fields

import numpy as np
import pytest

import config
from audio import AudioEngine
from panel import SPECS, ParamPanel
from state import State
from synth import SynthParams, derive_params


def make_panel(start: float = 0.5) -> ParamPanel:
    return ParamPanel(State(), start_intensity=start)


def test_specs_cover_every_synth_param_in_order():
    assert [name for name, *_ in SPECS] == [f.name for f in fields(SynthParams)]


def test_specs_ranges_contain_derived_extremes():
    """Every derivable value must be reachable (and representable) on the panel."""
    for params in (derive_params(0.0), derive_params(1.0)):
        for name, lo, hi, _ in SPECS:
            assert lo <= getattr(params, name) <= hi


def test_adjust_publishes_fresh_snapshot():
    panel = make_panel()
    panel.handle_key("j")  # first param row (amplitude); row 0 is intensity
    before = panel.state.params
    assert panel.handle_key("l")
    after = panel.state.params
    assert after is not before  # a new object, never in-place mutation
    _, _, _, step = SPECS[0]
    assert after.amplitude == pytest.approx(before.amplitude + step)


def test_adjust_clamps_at_range_edges():
    panel = make_panel()
    panel.handle_key("j")
    name, lo, hi, step = SPECS[0]
    for _ in range(int((hi - lo) / step) + 10):
        panel.handle_key("L")  # coarse steps overshoot without clamping
    assert getattr(panel.state.params, name) == hi
    for _ in range(int((hi - lo) / step) + 10):
        panel.handle_key("H")
    assert getattr(panel.state.params, name) == lo


def test_selection_wraps_and_targets_the_selected_row():
    panel = make_panel()
    panel.handle_key("k")  # wrap up from the intensity row to the last param
    assert panel.row == len(SPECS)
    name, _, _, step = SPECS[-1]
    before = getattr(panel.state.params, name)
    panel.handle_key("h")
    assert getattr(panel.state.params, name) == pytest.approx(before - step)


def test_digits_load_derived_params():
    panel = make_panel(start=0.0)
    panel.handle_key("9")
    assert panel.state.params == derive_params(1.0)
    assert panel.intensity == 1.0
    panel.handle_key("0")
    assert panel.state.params == derive_params(0.0)


def test_intensity_row_rederives_all_params():
    panel = make_panel(start=0.5)
    panel.handle_key("j")
    panel.handle_key("L")  # knock amplitude away from its derived value
    assert panel.state.params != derive_params(0.5)
    panel.handle_key("k")  # back to the master intensity row
    panel.handle_key("l")
    assert panel.intensity == pytest.approx(0.52)
    assert panel.state.params == derive_params(panel.intensity)


def test_intensity_clamps_to_unit_range():
    panel = make_panel(start=1.0)
    panel.handle_key("L")
    assert panel.intensity == 1.0
    for _ in range(20):
        panel.handle_key("H")
    assert panel.intensity == 0.0
    assert panel.state.params == derive_params(0.0)


def test_regime_toggle_flips_config(monkeypatch):
    monkeypatch.setitem(config.ROUGHNESS, "regimes", True)
    panel = make_panel()
    panel.handle_key("g")
    assert config.ROUGHNESS["regimes"] is False
    panel.handle_key("g")
    assert config.ROUGHNESS["regimes"] is True


def test_quit_keys_end_the_panel():
    panel = make_panel()
    assert panel.handle_key("q") is False
    assert panel.handle_key("\x03") is False


def test_lines_render_every_row_plus_header_and_intensity():
    panel = make_panel()
    lines = panel.lines()
    assert len(lines) == len(SPECS) + 2  # header + intensity + params
    assert lines[1 + panel.row].startswith(">")
    assert "intensity" in lines[1]


def test_audio_callback_uses_published_params_verbatim():
    """With `state.params` set, the callback must bypass smoothing/derivation."""
    state = State()
    engine = AudioEngine(state)
    state.target_intensity = 1.0  # would be loud if the callback derived params
    state.params = SynthParams(
        amplitude=0.0,
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
    out = np.empty((config.BLOCKSIZE, 1), dtype=np.float32)
    engine._callback(out, config.BLOCKSIZE, None, None)
    assert np.all(out == 0.0)  # zero amplitude and breath -> silence
    assert engine.intensity == 0.0  # smoothing never ran
