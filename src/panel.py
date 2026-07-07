"""Interactive SynthParams panel: hand-drive the renderer, bypassing intensity.

Every other source moves a single 0..1 intensity that `derive_params` fans out
into `SynthParams`. This panel is the opposite experiment: it publishes a full
`SynthParams` into shared state, which the audio callback uses verbatim (no
smoothing, no derivation), so each renderer control can be played by ear in
isolation. The master `intensity` row (and the digit keys) load `derive_params(i)`
wholesale — what a real driver would sound like at that effort — as a known
starting point to push individual knobs away from.

The panel has two views, toggled with `]`:

  params  the SynthParams rows above — the workbench, where the voice drones.
  play    a musical keyboard: home-row keys play mono notes through the gate
          envelope (`GateEnvelope` in synth.py), so the voice sculpted in the
          params view becomes an instrument. Terminals never report key-up, so
          notes sustain by extend-on-repeat: each press holds the gate open for
          config.NOTE_HOLD_MS and the OS key-repeat keeps extending that
          deadline while the key is physically held.
"""

from __future__ import annotations

import select
import sys
import time
from dataclasses import replace
from typing import Callable

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
_TOGGLE = "]"  # switch between the params and play views (q stays quit)
_BIG = 10.0  # step multiplier for the shifted adjust keys
_INTENSITY_STEP = 0.02  # master-row step; re-derives every param

# Play view: standard DAW layout (chosen to avoid q) — home row = white keys,
# the row above = sharps. Values are semitone offsets from the base octave's C.
_NOTE_OFFSETS: dict[str, int] = {
    "a": 0, "w": 1, "s": 2, "e": 3, "d": 4, "f": 5, "t": 6,
    "g": 7, "y": 8, "h": 9, "u": 10, "j": 11,
    "k": 12, "o": 13, "l": 14, "p": 15, ";": 16, "'": 17,
}  # fmt: skip
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KEY_ART = ["   w e   t y u   o p", "  a s d f g h j k l ; '"]
_NAME_ART = "  C D E F G A B C D E F"  # aligned under the white-key row


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
            f"g regimes[{regimes}]   ] play   q quit"
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


class PlayView:
    """Musical-keyboard model: computer keys play mono notes (last press wins).

    Presses publish `state.note_ratio`/`state.gate`; the audio callback's
    `GateEnvelope` turns those into click-free attacks, releases, and legato
    glides. `tick()` (called from the run loop's poll) closes the gate once a
    press's extend-on-repeat deadline lapses. Intensity stays playable via -/=
    through the panel's master `_derive`, same as the params view's row 0. The
    clock is injectable so tests control time (as with the intensity sources).
    """

    def __init__(
        self, panel: ParamPanel, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.panel = panel
        self.state = panel.state
        self._clock = clock
        self.octave = 3  # 'a' = C3 ~ 130.8 Hz, near the default 110 Hz voice
        self.offset: int | None = None  # sounding note's semitone offset
        self._deadline = 0.0
        self._saved_rise = config.F0_RISE_SEMITONES

    def enter(self) -> None:
        """Arm play mode: silence until a key is struck, strain rise capped."""
        self._saved_rise = config.F0_RISE_SEMITONES
        config.F0_RISE_SEMITONES = min(
            self._saved_rise, config.PLAY_STRAIN_SEMITONES
        )
        self.state.gate = 0.0
        self.offset = None

    def leave(self) -> None:
        """Back to the droning workbench: gate off, strain rise restored."""
        config.F0_RISE_SEMITONES = self._saved_rise
        self.state.gate = None

    def _press(self, offset: int) -> None:
        midi = 12 * (self.octave + 1) + offset  # MIDI note number (C4 = 60)
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        self.state.note_ratio = freq / config.F0
        self.state.gate = 1.0
        self._deadline = self._clock() + config.NOTE_HOLD_MS / 1000.0
        self.offset = offset

    def tick(self) -> bool:
        """Close the gate once the last press's deadline lapses; True if changed."""
        if self.state.gate == 1.0 and self._clock() >= self._deadline:
            self.state.gate = 0.0
            return True
        return False

    def handle_key(self, key: str) -> bool:
        """Apply one keypress; return False when the panel should quit."""
        if not key or key in _QUIT:  # '' is EOF (stdin closed)
            return False
        if key in _NOTE_OFFSETS:
            self._press(_NOTE_OFFSETS[key])
        elif key == "z":
            self.octave = max(self.octave - 1, 0)
        elif key == "x":
            self.octave = min(self.octave + 1, 7)
        elif key == "-":
            self.panel._derive(self.panel.intensity - _INTENSITY_STEP)
        elif key == "=":
            self.panel._derive(self.panel.intensity + _INTENSITY_STEP)
        return True

    def _note_label(self) -> str:
        if self.offset is None:
            return "--"
        name = _NOTE_NAMES[self.offset % 12]
        return f"{name}{self.octave + self.offset // 12}"

    def lines(self) -> list[str]:
        header = (
            "keys play notes (hold to sustain)   z/x octave   -/= intensity   "
            "] params   q quit"
        )
        sounding = self.state.gate == 1.0
        art = []
        for row in _KEY_ART:
            chars = []
            for ch in row:
                if sounding and _NOTE_OFFSETS.get(ch) == self.offset:
                    chars.append(f"\x1b[7m{ch}\x1b[27m")  # reverse-video the key
                else:
                    chars.append(ch)
            art.append("".join(chars))
        status = (
            f"  note {self._note_label():<4} gate {'on ' if sounding else 'off'}  "
            f"octave {self.octave}   intensity {self.panel.intensity:4.2f} "
            f"[{_bar(self.panel.intensity)}]"
        )
        return [header, *art, _NAME_ART, status]


def _draw(lines: list[str], prev: int) -> int:
    """Redraw in place, coping with views of different heights; returns height."""
    if prev:
        sys.stdout.write(f"\x1b[{prev}A")  # cursor back to the first line
    for line in lines:
        sys.stdout.write(f"\r\x1b[K{line}\n")  # clear the line, then redraw it
    extra = prev - len(lines)
    if extra > 0:  # blank the leftover lines of a taller previous view
        sys.stdout.write("\r\x1b[K\n" * extra + f"\x1b[{extra}A")
    sys.stdout.flush()
    return len(lines)


def run(state: State, start_intensity: float = 0.0) -> None:
    """Drive `state.params` from the keyboard until quit; restores derive mode."""
    panel = ParamPanel(state, start_intensity)
    play = PlayView(panel)
    view: ParamPanel | PlayView = panel
    try:
        with RawTerminal():
            height = _draw(panel.lines(), 0)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], _TICK)
                redraw = False
                if ready:
                    key = read_key()
                    if key == _TOGGLE:
                        if view is panel:
                            view = play
                            play.enter()
                        else:
                            view = panel
                            play.leave()
                    elif not view.handle_key(key):
                        break
                    redraw = True
                if view is play and play.tick():
                    redraw = True
                if redraw:
                    height = _draw(view.lines(), height)
    finally:
        if view is play:
            play.leave()  # restores the strain cap; sets gate back to None
        state.gate = None
        state.params = None  # hand the callback back to intensity-derived params
