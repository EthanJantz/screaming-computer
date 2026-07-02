"""Interactive SynthParams panel: hand-drive the renderer, bypassing intensity.

Every other source moves a single 0..1 intensity that `derive_params` fans out
into `SynthParams`. This panel is the opposite experiment: it publishes a full
`SynthParams` into shared state, which the audio callback uses verbatim (no
smoothing, no derivation), so each renderer control can be played by ear in
isolation. The master `intensity` row (and the digit keys) load `derive_params(i)`
wholesale — what a real driver would sound like at that effort — as a known
starting point to push individual knobs away from.
"""

from __future__ import annotations

import select
import sys
from dataclasses import replace

import config
from state import State
from synth import SynthParams, derive_params
from term import RawTerminal, read_key

_TICK = 0.05  # seconds the key poll blocks; the display only changes on keys

# One row per SynthParams field: (name, lo, hi, step). Ranges give
# experimentation room around what derive_params can produce; they are not hard
# synth limits.
SPECS: list[tuple[str, float, float, float]] = [
    ("amplitude", 0.0, 1.0, 0.02),
    ("pitch_scale", 0.25, 4.0, 0.05),
    ("rolloff_p", 0.25, 4.0, 0.05),
    ("breath_level", 0.0, 0.2, 0.002),
    ("jitter_amount", 0.0, 4.0, 0.05),
    ("shimmer_amount", 0.0, 1.0, 0.02),
    ("sub_amount", 0.0, 1.5, 0.02),
    ("chaos_amount", 0.0, 1.0, 0.02),
    ("drive", 1.0, 12.0, 0.1),
    ("drive_mix", 0.0, 1.0, 0.02),
]

_UP = {"\x1b[A", "k"}
_DOWN = {"\x1b[B", "j"}
_LESS = {"\x1b[D", "h"}
_MORE = {"\x1b[C", "l"}
_QUIT = {"q", "\x03"}  # q, Ctrl-C
_BIG = 10.0  # step multiplier for the shifted adjust keys
_INTENSITY_STEP = 0.02  # master-row step; re-derives every param


def _bar(frac: float, width: int = 20) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "-" * (width - filled)


class ParamPanel:
    """Selection + adjustment model for the panel; the run loop just feeds keys.

    Row 0 is the master `intensity` slider: moving it (or pressing a digit)
    replaces every param with `derive_params(intensity)`, discarding manual
    tweaks. The remaining rows edit one `SynthParams` field each. Publishes a
    fresh `SynthParams` into `state.params` on every change (see `State` for
    why a fresh object, not in-place mutation).
    """

    def __init__(self, state: State, start_intensity: float = 0.0) -> None:
        self.state = state
        self.row = 0
        self._derive(start_intensity)

    def _publish(self, params: SynthParams) -> None:
        self.params = params
        self.state.params = params

    def _derive(self, intensity: float) -> None:
        self.intensity = min(max(intensity, 0.0), 1.0)
        self._publish(derive_params(self.intensity))

    def _adjust(self, direction: float) -> None:
        if self.row == 0:
            self._derive(self.intensity + direction * _INTENSITY_STEP)
            return
        name, lo, hi, step = SPECS[self.row - 1]
        value = getattr(self.params, name) + direction * step
        self._publish(replace(self.params, **{name: min(max(value, lo), hi)}))

    def handle_key(self, key: str) -> bool:
        """Apply one keypress; return False when the panel should quit."""
        if not key or key in _QUIT:  # '' is EOF (stdin closed)
            return False
        if key in _UP:
            self.row = (self.row - 1) % (len(SPECS) + 1)
        elif key in _DOWN:
            self.row = (self.row + 1) % (len(SPECS) + 1)
        elif key in _LESS:
            self._adjust(-1.0)
        elif key in _MORE:
            self._adjust(+1.0)
        elif key == "H":
            self._adjust(-_BIG)
        elif key == "L":
            self._adjust(+_BIG)
        elif key.isdigit():
            self._derive(int(key) / 9.0)
        elif key == "g":
            config.ROUGHNESS["regimes"] = not config.ROUGHNESS["regimes"]
        return True

    def lines(self) -> list[str]:
        regimes = "on" if config.ROUGHNESS["regimes"] else "off"
        header = (
            "j/k select   h/l adjust (H/L coarse)   0-9 load derive_params(n/9)   "
            f"g regimes[{regimes}]   q quit"
        )
        marker = ">" if self.row == 0 else " "
        rows = [
            f"{marker} {'intensity':<15}{self.intensity:7.3f}  "
            f"[{_bar(self.intensity)}]  (master: re-derives all params)"
        ]
        for i, (name, lo, hi, _) in enumerate(SPECS, start=1):
            value = getattr(self.params, name)
            frac = (value - lo) / (hi - lo)
            marker = ">" if i == self.row else " "
            rows.append(f"{marker} {name:<15}{value:7.3f}  [{_bar(frac)}]")
        return [header, *rows]


def _draw(panel: ParamPanel, first: bool) -> None:
    lines = panel.lines()
    if not first:
        sys.stdout.write(f"\x1b[{len(lines)}A")  # cursor back to the header line
    for line in lines:
        sys.stdout.write(f"\r\x1b[K{line}\n")  # clear the line, then redraw it
    sys.stdout.flush()


def run(state: State, start_intensity: float = 0.0) -> None:
    """Drive `state.params` from the keyboard until quit; restores derive mode."""
    panel = ParamPanel(state, start_intensity)
    try:
        with RawTerminal():
            _draw(panel, first=True)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], _TICK)
                if not ready:
                    continue
                if not panel.handle_key(read_key()):
                    break
                _draw(panel, first=False)
    finally:
        state.params = None  # hand the callback back to intensity-derived params
