"""Tests for the preset system: application, merging, validation, drift."""

import copy
from types import SimpleNamespace

import pytest

import config
import presets


def config_copy() -> SimpleNamespace:
    """A detached copy of the config module's constants."""
    ns = SimpleNamespace()
    for key in dir(config):
        if key.isupper():
            setattr(ns, key, copy.deepcopy(getattr(config, key)))
    return ns


def test_all_shipped_presets_apply_cleanly():
    for name in presets.PRESETS:
        presets.apply(name, config_copy())


def test_apply_sets_values_and_merges_roughness():
    cfg = config_copy()
    presets.apply("scream", cfg)
    assert cfg.F0 == 490.0
    assert cfg.SUB_WIDTH_HZ == 100.0
    assert cfg.ROUGHNESS["brightening"] is False
    # Merged, not replaced: unlisted ingredients keep their config defaults.
    assert set(cfg.ROUGHNESS) == set(config.ROUGHNESS)
    assert cfg.ROUGHNESS["jitter"] == config.ROUGHNESS["jitter"]


def test_roar_pins_the_current_defaults():
    """If this fails, config.py drifted from the "roar" preset — update one."""
    for key, value in presets.PRESETS["roar"].items():
        if key == "ROUGHNESS":
            for rkey, rvalue in value.items():
                assert config.ROUGHNESS[rkey] == rvalue
        else:
            assert getattr(config, key) == value, key


def test_unknown_config_key_raises():
    with pytest.raises(AttributeError, match="NOT_A_KEY"):
        presets.apply_overrides({"NOT_A_KEY": 1}, config_copy())


def test_unknown_preset_name_raises():
    with pytest.raises(KeyError, match="available"):
        presets.apply("whisper", config_copy())


def test_real_config_module_untouched_by_tests():
    assert config.F0 == 110.0
