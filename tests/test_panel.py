"""Tests for the param panel: coverage of SynthParams, key handling, publishing."""

from dataclasses import fields

import numpy as np
import pytest

import config
from audio import AudioEngine
from panel import SPECS, ParamPanel, PlayView
from state import State
from synth import SynthParams, derive_params


def make_panel(start: float = 0.5) -> ParamPanel:
    return ParamPanel(State(), start_intensity=start)


@pytest.fixture(autouse=True)
def _restore_strain_cap(monkeypatch):
    """PlayView.enter() caps config.F0_RISE_SEMITONES; undo it after each test."""
    monkeypatch.setattr(config, "F0_RISE_SEMITONES", config.F0_RISE_SEMITONES)


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_play(start: float = 0.5) -> tuple[PlayView, FakeClock]:
    clock = FakeClock()
    play = PlayView(make_panel(start), clock=clock)
    play.enter()
    return play, clock


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


# --- play view ---


def test_enter_arms_silence_and_caps_strain(monkeypatch):
    monkeypatch.setattr(config, "F0_RISE_SEMITONES", 18.0)
    play, _ = make_play()
    assert play.state.gate == 0.0  # silent until a key is struck
    assert config.F0_RISE_SEMITONES == config.PLAY_STRAIN_SEMITONES
    play.leave()
    assert config.F0_RISE_SEMITONES == 18.0
    assert play.state.gate is None


def test_note_press_publishes_ratio_and_opens_gate():
    play, _ = make_play()
    play.handle_key("a")  # C3 at the default octave
    c3 = 440.0 * 2.0 ** ((48 - 69) / 12.0)
    assert play.state.note_ratio == pytest.approx(c3 / config.F0)
    assert play.state.gate == 1.0
    play.handle_key("w")  # C#3: one semitone up
    assert play.state.note_ratio == pytest.approx(c3 * 2 ** (1 / 12) / config.F0)


def test_octave_shift_scales_the_ratio():
    play, _ = make_play()
    play.handle_key("a")
    low = play.state.note_ratio
    play.handle_key("x")
    play.handle_key("a")
    assert play.state.note_ratio == pytest.approx(2.0 * low)
    play.handle_key("z")
    play.handle_key("z")
    play.handle_key("a")
    assert play.state.note_ratio == pytest.approx(low / 2.0)


def test_extend_on_repeat_sustains_past_the_hold(monkeypatch):
    monkeypatch.setattr(config, "NOTE_HOLD_MS", 600.0)
    play, clock = make_play()
    play.handle_key("a")
    clock.t = 0.5
    assert not play.tick() and play.state.gate == 1.0  # before the deadline
    play.handle_key("a")  # OS key-repeat extends the deadline
    clock.t = 1.0
    assert not play.tick() and play.state.gate == 1.0
    clock.t = 1.2  # 0.5 + 0.6 < 1.2: deadline lapsed
    assert play.tick() and play.state.gate == 0.0
    assert not play.tick()  # already closed; nothing to redraw


def test_legato_switches_pitch_without_closing_the_gate():
    play, clock = make_play()
    play.handle_key("a")
    clock.t = 0.3
    play.handle_key("h")  # A3 while C3 still sounds
    assert play.state.gate == 1.0
    a3 = 440.0 * 2.0 ** ((57 - 69) / 12.0)
    assert play.state.note_ratio == pytest.approx(a3 / config.F0)


def test_play_intensity_keys_drive_the_master_derive():
    play, _ = make_play(start=0.5)
    play.handle_key("=")
    assert play.panel.intensity == pytest.approx(0.52)
    assert play.state.params == derive_params(0.52)
    play.handle_key("-")
    assert play.panel.intensity == pytest.approx(0.5)


def test_play_quit_keys_end_the_panel():
    play, _ = make_play()
    assert play.handle_key("q") is False
    assert play.handle_key("\x03") is False


def test_play_lines_render_keyboard_and_status():
    play, _ = make_play()
    play.handle_key("d")  # E3
    lines = play.lines()
    assert len(lines) == 5  # header + 2 key rows + note names + status
    assert "\x1b[7md\x1b[27m" in lines[2]  # sounding key is highlighted
    assert "note E3" in lines[-1]


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
