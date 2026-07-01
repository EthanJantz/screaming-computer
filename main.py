"""Entry point.

Drives `state.target_intensity` from a selectable intensity source while the audio
callback smooths it into the attack/sustain/release envelope. The smoothing lives in
the audio thread (see `audio.py`); here we just feed the un-smoothed target.

Sources (`--source`):
    fake    a timer that fakes engine 'turns' (swell then release). Milestone 3.
    manual  keyboard control (up/k/+ louder, down/j/- softer). Milestone 2.
    chess   play Stockfish; its search effort drives the scream. Milestone 6.

Run with `uv run main.py`. Press q or Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from pathlib import Path

# Implementation modules live as flat files under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from audio import AudioEngine  # noqa: E402
from intensity import FakeTurnIntensity, IntensitySource, ManualIntensity  # noqa: E402
from state import State  # noqa: E402

_TICK = 0.02  # seconds between target updates / display refreshes (~50 Hz)


class _RawTerminal:
    """Put the TTY in cbreak mode so we read single keypresses without Enter.

    cbreak (not full raw) leaves signal keys working, so Ctrl-C still interrupts.
    Restores the original terminal settings on exit no matter what.
    """

    def __enter__(self) -> "_RawTerminal":
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def _read_key() -> str:
    """Read one keypress, collapsing arrow-key escape sequences into e.g. '\\x1b[A'."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        ready, _, _ = select.select([sys.stdin], [], [], 0.001)
        if ready:
            ch += sys.stdin.read(2)
    return ch


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def _render_status(target: float, smoothed: float) -> None:
    print(
        f"\rtarget {target:4.2f} [{_bar(target)}]   voice {smoothed:4.2f} [{_bar(smoothed)}]   ",
        end="",
        flush=True,
    )


def _drive_loop(state: State, source: IntensitySource, engine: AudioEngine) -> None:
    """Feed the source's target into shared state and show a live meter.

    A single loop polls the keyboard (non-blocking), updates the target, and renders.
    This loop is the 'driver thread' — separate from the audio callback thread.
    """
    up = {"\x1b[A", "k", "+", "="}
    down = {"\x1b[B", "j", "-", "_"}
    quit_keys = {"q", "\x03"}  # q, Ctrl-C
    manual = isinstance(source, ManualIntensity)

    if manual:
        print("up/k/+ louder   down/j/- softer   q quit")
    else:
        print("fake turns: swell then release on a timer   q quit")

    with _RawTerminal():
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], _TICK)
            if ready:
                key = _read_key()
                if key in quit_keys:
                    break
                if manual and key in up:
                    source.increase()
                elif manual and key in down:
                    source.decrease()
            state.target_intensity = source.target()
            _render_status(source.target(), engine.intensity)
    print()


def _run_chess(state: State) -> None:
    """Play Stockfish; the engine's search effort drives the scream."""
    # Imported lazily so the other sources don't pay the chess/engine import cost.
    from chess_driver import ChessDriver

    try:
        driver = ChessDriver(state)
    except FileNotFoundError as e:
        print(e)  # no Stockfish installed — print the install hint, not a traceback
        return
    try:
        driver.play()
    finally:
        driver.close()


def run(source_name: str = "fake", start: float = 0.0) -> None:
    state = State()
    engine = AudioEngine(state)
    engine.start()
    source: IntensitySource = (
        ManualIntensity(start) if source_name == "manual" else FakeTurnIntensity()
    )
    state.target_intensity = source.target()
    try:
        if source_name == "chess":
            _run_chess(state)
            return
        else:
            _drive_loop(state, source, engine)
            return
    finally:
        engine.stop()
        print("Stopping.")


def main() -> None:
    parser = argparse.ArgumentParser(description="screaming computer")
    parser.add_argument(
        "--source",
        choices=("fake", "manual", "chess"),
        default="fake",
        help="intensity source (default: fake — the milestone 3 swell/release timer)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.0,
        help="starting value for the manual source",
    )
    args = parser.parse_args()
    run(args.source, args.intensity)


if __name__ == "__main__":
    main()
