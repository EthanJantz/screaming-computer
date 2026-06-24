"""Chess work source: a human-vs-Stockfish game whose *search effort* drives the scream.

This is the project's first real "driver". It is one facade over the screaming
computer — the synth and audio engine know nothing about it. The contract with the
core is, as always, a single 0..1 intensity in shared state.

What screams here is the engine *doing work*: while Stockfish searches, we stream its
live node count and map it (log scale) into intensity via a `TelemetryIntensity`. The
voice swells as the search deepens and snaps back to idle once the move resolves. The
scream is NOT about who is winning — a losing position the engine breezes through is
quiet; a hard position it grinds on is loud. Work, not opinion.

Everything chess-specific (Stockfish path, think time, node->intensity calibration,
the TUI) lives in this file. To add another work source, write a sibling driver that
feeds its own metric into a `TelemetryIntensity`; nothing else changes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import chess
import chess.engine

from intensity import TelemetryIntensity
from state import State

# --- Chess-specific tuning (stays with the driver, never in config.py) ---
# Repo-local build, used as a last-resort fallback for development. Cloners are
# expected to install Stockfish via their package manager instead (see _find_stockfish).
LOCAL_STOCKFISH = Path(__file__).resolve().parent.parent / "stockfish"

_INSTALL_HINT = (
    "Stockfish chess engine not found.\n"
    "Install it with your package manager, then re-run:\n"
    "    macOS:          brew install stockfish\n"
    "    Debian/Ubuntu:  sudo apt install stockfish\n"
    "    Arch:           sudo pacman -S stockfish\n"
    "Or point STOCKFISH_PATH at a binary you built:\n"
    "    STOCKFISH_PATH=/path/to/stockfish uv run main.py --source chess"
)


def _find_stockfish() -> str:
    """Locate a Stockfish binary: $STOCKFISH_PATH, then PATH, then the repo-local build.

    We never ship the binary (it's ~91 MB and platform-specific); users install their
    own. Raises FileNotFoundError with an install hint if none is found.
    """
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    if LOCAL_STOCKFISH.is_file():
        return str(LOCAL_STOCKFISH)
    raise FileNotFoundError(_INSTALL_HINT)


THINK_TIME = 2.0  # seconds Stockfish searches per move (its "turn" length)

# Node counts mapping to idle/full effort. Calibrated by ear on this machine: a 2 s
# search reaches ~1M nodes here, so the swell tops out near full effort at the end of
# the think. Tune these if the scream peaks too early/late or never reaches the top.
N_MIN = 2_000  # search just started -> idle
N_MAX = 1_500_000  # deep search -> full-effort scream

HUMAN_PLAYS_WHITE = True


class ChessDriver:
    """Runs the game loop and feeds search effort into shared `State`."""

    def __init__(self, state: State) -> None:
        stockfish = _find_stockfish()  # raises FileNotFoundError with an install hint
        self.state = state
        self.board = chess.Board()
        self.intensity = TelemetryIntensity(N_MIN, N_MAX)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)

    def close(self) -> None:
        """Tear down the Stockfish subprocess (so tuning doesn't orphan engines)."""
        try:
            self.engine.quit()
        except Exception:
            pass  # already dead / never started — nothing to clean up

    # --- the work: stream Stockfish's search, scream by its node count ---
    def _engine_move(self) -> chess.Move:
        best: chess.Move | None = None
        limit = chess.engine.Limit(time=THINK_TIME)
        with self.engine.analysis(self.board, limit) as analysis:
            for info in analysis:
                nodes = info.get("nodes")
                if nodes:
                    self.intensity.update(nodes)
                    # Single atomic float write — the only cross-thread contact point.
                    self.state.target_intensity = self.intensity.target()
                pv = info.get("pv")
                if pv:
                    best = pv[0]  # deepest principal variation = engine's choice
        # Move resolved: release the scream back to idle.
        self.intensity.reset()
        self.state.target_intensity = 0.0
        if best is None:  # pathological (no pv reported) — fall back to any legal move
            best = next(iter(self.board.legal_moves))
        return best

    # --- terminal UI (line-based; swap for a GUI in the final product) ---
    def _render_board(self) -> str:
        # Board from the human's point of view (their pieces at the bottom).
        orientation = chess.WHITE if HUMAN_PLAYS_WHITE else chess.BLACK
        return self.board.unicode(
            borders=True, invert_color=True, orientation=orientation
        )

    def _parse_move(self, raw: str) -> chess.Move | None:
        for parse in (self.board.parse_san, self.board.parse_uci):
            try:
                move = parse(raw)
            except ValueError:  # Invalid/Illegal/Ambiguous all subclass ValueError
                continue
            if move in self.board.legal_moves:
                return move
        return None

    def _human_move(self) -> bool:
        """Prompt for the human's move. Returns False if they want to quit."""
        while True:
            try:
                raw = input(
                    "your move (e.g. e4, Nf3, or g1f3) — 'moves' to list, 'q' to quit: "
                ).strip()
            except EOFError:
                return False
            if raw in ("q", "quit"):
                return False
            if raw in ("moves", "l"):
                print("  " + ", ".join(self.board.san(m) for m in self.board.legal_moves))
                continue
            move = self._parse_move(raw)
            if move is None:
                print("  not a legal move here — type 'moves' to see the options")
                continue
            self.board.push(move)
            return True

    def play(self) -> None:
        human = chess.WHITE if HUMAN_PLAYS_WHITE else chess.BLACK
        side = "White" if HUMAN_PLAYS_WHITE else "Black"
        print(f"You are {side}. Stockfish thinks for {THINK_TIME:.0f}s per move.")
        print("Listen for it to strain on the hard moves.\n")
        while not self.board.is_game_over():
            print(self._render_board())
            if self.board.turn == human:
                if not self._human_move():
                    print("Resigning. Good game.")
                    return
            else:
                print("Stockfish is thinking...")
                move = self._engine_move()
                print(f"Stockfish plays {self.board.san(move)}\n")
                self.board.push(move)
        print(self._render_board())
        print(f"Game over — {self.board.result()} ({self.board.outcome().termination.name.lower()})")
