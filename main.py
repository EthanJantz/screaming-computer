"""Entry point.

Drives `state.target_intensity` from a selectable intensity source while the audio
callback smooths it into the attack/sustain/release envelope. The smoothing lives in
the audio thread (see `audio.py`); here we just feed the un-smoothed target.

Sources (`--source`):
    fake     a timer that fakes engine 'turns' (swell then release). Milestone 3.
    manual   keyboard control (up/k/+ louder, down/j/- softer). Milestone 2.
    contour  a scripted intensity gesture from config.CONTOUR, repeating.
    chess    play Stockfish; its search effort drives the scream. Milestone 6.
    panel    hand-drive every SynthParams field live (bypasses intensity).

Sound character comes from `--preset <name>` (see src/presets.py), with one-off
tweaks via repeatable `--set KEY=VALUE`, e.g.:

    uv run main.py --preset scream --source contour --set SUB_WIDTH_HZ=250

Run with `uv run main.py`. Press q or Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import ast
import select
import sys
from pathlib import Path

# Implementation modules live as flat files under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config  # noqa: E402
import presets  # noqa: E402
from audio import AudioEngine  # noqa: E402
from intensity import (  # noqa: E402
    ContourIntensity,
    FakeTurnIntensity,
    IntensitySource,
    ManualIntensity,
)
from state import State  # noqa: E402
from term import RawTerminal, read_key  # noqa: E402

_TICK = 0.02  # seconds between target updates / display refreshes (~50 Hz)


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
        print(f"{type(source).__name__} driving the scream   q quit")

    with RawTerminal():
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], _TICK)
            if ready:
                key = read_key()
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


def _make_source(source_name: str, start: float) -> IntensitySource:
    if source_name == "manual":
        return ManualIntensity(start)
    if source_name == "contour":
        return ContourIntensity(**config.CONTOUR)
    return FakeTurnIntensity()


def _parse_overrides(pairs: list[str]) -> dict:
    """Parse repeated --set KEY=VALUE flags into a config-override dict."""
    overrides = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        try:
            overrides[key.strip()] = ast.literal_eval(raw.strip())
        except (SyntaxError, ValueError):
            raise SystemExit(f"--set {key}: can't parse value {raw!r}") from None
    return overrides


def run(source_name: str = "fake", start: float = 0.0) -> None:
    state = State()
    engine = AudioEngine(state)
    engine.start()
    source = _make_source(source_name, start)
    state.target_intensity = source.target()
    try:
        if source_name == "chess":
            _run_chess(state)
            return
        elif source_name == "panel":
            import panel

            panel.run(state, start)
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
        choices=("fake", "manual", "contour", "chess", "panel"),
        default="fake",
        help="intensity source (default: fake — the milestone 3 swell/release timer)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.0,
        help="starting value for the manual source / panel's initial derive_params",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(presets.PRESETS),
        help="named scream recipe (see src/presets.py)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="one-off config override on top of the preset (repeatable)",
    )
    args = parser.parse_args()
    # Apply sound-character overrides before the engine (and its Voice) exist.
    if args.preset:
        presets.apply(args.preset)
    if args.overrides:
        presets.apply_overrides(_parse_overrides(args.overrides))
    run(args.source, args.intensity)


if __name__ == "__main__":
    main()
