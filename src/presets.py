"""Named scream recipes: bundles of config overrides selectable from the CLI.

A preset is a dict of config attribute overrides, so switching sound characters
is `--preset <name>` instead of hand-editing config.py. `ROUGHNESS` entries are
merged key-by-key (a preset can flip one ingredient without restating the rest);
every other entry replaces the config value wholesale. Both shipped presets list
the same keys, so they read as side-by-side recipes.
"""

from __future__ import annotations

from typing import Any

import config

PRESETS: dict[str, dict[str, Any]] = {
    # The original screaming-computer voice, pinned explicitly so "roar" stays
    # stable even if the config.py defaults drift later.
    "roar": {
        "F0": 110.0,
        "F0_RISE_SEMITONES": 18.0,
        "P_IDLE": 2.0,
        "ROUGHNESS": {"brightening": True},
        "JITTER_MAX_ST": 1.4,
        "SHIMMER_MAX": 0.18,
        "SUB_RATIO": 2,
        "SUB_WIDTH_HZ": 4000.0,
        "REGIME_MIN_BLOCKS": 8,
        "BREATH_IDLE": 0.012,
        "FORMANTS": [
            (730, 90),
            (1090, 110),
            (2440, 170),
            (3500, 250),
            (4500, 300),
        ],
        "CONTOUR": {
            "times": (0.0, 0.8, 1.0),
            "intensity": (0.0, 1.0, 0.6),
            "duration": 3.0,
            "pause": 2.0,
        },
    },
    # Approximation of the first syllable of the reference soundgen call (see the
    # comment on each line for the soundgen parameter it came from), elongated.
    "scream": {
        "F0": 490.0,  # pitch anchor floor (490 Hz)
        "F0_RISE_SEMITONES": 10.9,  # 490 -> 920 Hz peak is ~10.9 semitones
        "P_IDLE": 1.0,  # rolloff = -6 dB/oct
        "ROUGHNESS": {"brightening": False},  # hold the -6 dB/oct rolloff fixed
        "JITTER_MAX_ST": 2.0,  # jitterDep = 2 semitones
        "SHIMMER_MAX": 0.10,  # shimmerDep = 10 (%)
        "SUB_RATIO": 4,  # subFreq = 180 Hz at the contour's mean pitch (~720 Hz)
        "SUB_WIDTH_HZ": 100.0,  # subWidth = 100
        "REGIME_MIN_BLOCKS": 4,  # shortestEpoch = 50 ms (4 blocks ~ 46 ms)
        "BREATH_IDLE": 0.018,  # noise = -35 dB
        "FORMANTS": [  # formants = c(850, 1600, 2900, 4500, 5400); bandwidths
            (850, 100),  # estimated, since soundgen infers them from frequency
            (1600, 150),
            (2900, 220),
            (4500, 320),
            (5400, 400),
        ],
        # The call's first syllable: pitch (490, 800, 920, 840, 650) Hz at times
        # (0, .1, .5, .95, 1), mapped to intensity via the semitone rise and
        # elongated from 600 ms to 2.5 s.
        "CONTOUR": {
            "times": (0.0, 0.1, 0.5, 0.95, 1.0),
            "intensity": (0.0, 0.78, 1.0, 0.86, 0.45),
            "duration": 2.5,
            "pause": 1.5,
        },
    },
}


def apply_overrides(
    overrides: dict[str, Any], cfg: Any = config, origin: str = "--set"
) -> None:
    """Apply config overrides onto ``cfg``, validating every key.

    ``origin`` names the source of the overrides in error messages. ``cfg`` is
    injectable so tests can apply against a copy instead of the real module.
    """
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"{origin} sets unknown config key {key!r}")
        if key == "ROUGHNESS":
            cfg.ROUGHNESS.update(value)
        else:
            setattr(cfg, key, value)


def apply(name: str, cfg: Any = config) -> None:
    """Apply the named preset's overrides onto the config module."""
    try:
        overrides = PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; available: {sorted(PRESETS)}"
        ) from None
    apply_overrides(overrides, cfg, origin=f"preset {name!r}")
