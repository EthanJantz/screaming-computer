"""Chess work source: a human-vs-Stockfish game whose *search effort* drives the scream.

This is the project's first real "driver". It is one facade over the screaming
computer — the synth and audio engine know nothing about it. The contract with the
core is, as always, a single 0..1 intensity in shared state.

What screams here is the engine *doing work*: Stockfish searches every move to a
fixed depth, and we map the *time it takes to reach that depth* (log scale) into
intensity via a `TelemetryIntensity`. An easy position hits the depth almost
instantly and stays quiet; a hard one grinds for seconds and swells loud, snapping
back to idle once the move resolves. The scream is NOT about who is winning — a
losing position the engine breezes through is quiet; a hard position it grinds on is
loud. Work, not opinion.

Everything chess-specific (Stockfish path, search depth, time->intensity calibration,
the TUI) lives in this file. To add another work source, write a sibling driver that
feeds its own metric into a `TelemetryIntensity`; nothing else changes.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine

from intensity import TelemetryIntensity
from state import State
from term import (
    AltScreen,
    MouseTracking,
    RawTerminal,
    flush_input,
    parse_mouse,
    read_key,
)

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


# Approach A: fix the *search depth* so every move is the same "amount of thinking",
# and let the TIME to reach that depth be the difficulty signal — an easy position
# hits SEARCH_DEPTH in a blink (quiet), a hard one grinds for seconds (loud). This is
# what makes "the computer is working harder" audible, where a fixed think-time (every
# move burning ~the same node count) did not.
SEARCH_DEPTH = 20  # plies searched per move (its "turn"); tune for your CPU's speed
TIME_CAP = 8.0  # seconds: safety cap so a pathological position can't grind forever

# Elapsed search time (seconds) mapping to idle/full effort, log-scaled because think
# times span orders of magnitude across positions. Calibrate by ear to the spread
# SEARCH_DEPTH actually produces here: lower T_MAX if hard moves never reach the top,
# raise it if easy moves already peak.
T_MIN = 0.1  # reached the depth almost instantly -> idle
T_MAX = 4.0  # a long grind to the depth -> full-effort scream

HUMAN_PLAYS_WHITE = True

# --- board colors (256-color ANSI backgrounds/foregrounds) ---
# Squares use tan/brown so both piece colors stay legible; the destination square
# of the last move gets a green background so it pops without reading as an error.
_LIGHT_BG = "\x1b[48;5;180m"
_DARK_BG = "\x1b[48;5;94m"
_LAST_MOVE_BG = "\x1b[48;5;107m"  # green: destination of the last move
_SELECTED_BG = "\x1b[48;5;179m"  # amber: the piece the mouse has picked up
_TARGET_BG = "\x1b[48;5;108m"  # muted green: a legal destination for it
_WHITE_FG = "\x1b[38;5;231m"
_BLACK_FG = "\x1b[38;5;16m"
_RESET = "\x1b[0m"

# The board scales up on roomier terminals (no font change needed). Squares are
# drawn CELL_W columns by CELL_H rows; ~2:1 keeps them visually square on a
# terminal's tall cells. _LABEL_W is the left gutter that holds the rank digits.
_LABEL_W = 3
# A piece is a single terminal glyph that can't grow with the cell, so the cap is
# deliberately small: past ~4x2 the glyph starts to look lost in the square.
_MAX_CELL_H = 2


def _center(text: str, width: int) -> str:
    """`text` centered in `width` columns (slightly left-biased for odd padding)."""
    left = (width - len(text)) // 2
    return " " * left + text + " " * (width - len(text) - left)


class ChessDriver:
    """Runs the game loop and feeds search effort into shared `State`."""

    def __init__(self, state: State) -> None:
        stockfish = _find_stockfish()  # raises FileNotFoundError with an install hint
        self.state = state
        self.board = chess.Board()
        self.human = chess.WHITE if HUMAN_PLAYS_WHITE else chess.BLACK
        self.selected: chess.Square | None = None  # square the mouse has picked up
        # Cell size of the last drawn frame; _square_at hit-tests against it. Set for
        # real on every _draw from the terminal size; this default matches a minimal
        # board (and is what the unit tests exercise directly).
        self.cell_w, self.cell_h = 3, 1
        self.intensity = TelemetryIntensity(T_MIN, T_MAX)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)

    def close(self) -> None:
        """Tear down the Stockfish subprocess (so tuning doesn't orphan engines)."""
        try:
            self.engine.quit()
        except Exception:
            pass  # already dead / never started — nothing to clean up

    # --- the work: fixed-depth search; scream by how long it takes to get there ---
    def _engine_move(self) -> chess.Move:
        best: chess.Move | None = None
        limit = chess.engine.Limit(depth=SEARCH_DEPTH, time=TIME_CAP)
        start = time.monotonic()
        with self.engine.analysis(self.board, limit) as analysis:
            for info in analysis:
                # Elapsed time to the current depth is the effort signal: harder
                # positions take longer to reach SEARCH_DEPTH, so they swell louder.
                self.intensity.update(time.monotonic() - start)
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

    # --- terminal UI (fixed-position frame on the alt screen, redrawn in place) ---
    def _render_board(self) -> list[str]:
        """Colored board lines from the human's point of view (their pieces at the
        bottom), scaled to `self.cell_w` x `self.cell_h` per square. Highlights the
        last move's destination and, while a piece is picked up by the mouse, that
        square and its legal destinations. Each rank spans cell_h lines; the piece
        glyph / dot / rank digit sit on the block's middle line."""
        w, h = self.cell_w, self.cell_h
        mid = h // 2
        last_to = self.board.peek().to_square if self.board.move_stack else None
        targets = set()
        if self.selected is not None:
            targets = {
                m.to_square
                for m in self.board.legal_moves
                if m.from_square == self.selected
            }
        ranks = range(7, -1, -1) if HUMAN_PLAYS_WHITE else range(8)
        files = range(8) if HUMAN_PLAYS_WHITE else range(7, -1, -1)
        lines = []
        for rank in ranks:
            for sub in range(h):
                label = _center(str(rank + 1), _LABEL_W) if sub == mid else " " * _LABEL_W
                cells = [label]
                for file in files:
                    square = chess.square(file, rank)
                    if square == self.selected:
                        bg = _SELECTED_BG
                    elif square in targets:
                        bg = _TARGET_BG
                    elif square == last_to:
                        bg = _LAST_MOVE_BG
                    else:
                        bg = _LIGHT_BG if (file + rank) % 2 else _DARK_BG
                    piece = self.board.piece_at(square)
                    if sub == mid and piece:
                        fg = _WHITE_FG if piece.color == chess.WHITE else _BLACK_FG
                        # Always the filled glyph; side is carried by the fg color.
                        glyph = chess.UNICODE_PIECE_SYMBOLS[piece.symbol().lower()]
                        cells.append(f"{bg}{fg}{_center(glyph, w)}{_RESET}")
                    elif sub == mid and square in targets:
                        cells.append(f"{bg}{_BLACK_FG}{_center('·', w)}{_RESET}")
                    else:
                        cells.append(f"{bg}{' ' * w}{_RESET}")
                lines.append("".join(cells))
        labels = "".join(_center(chess.FILE_NAMES[f], w) for f in files)
        lines.append(" " * _LABEL_W + labels)
        return lines

    def _square_at(self, col: int, row: int) -> chess.Square | None:
        """Map a 1-based terminal (col, row) to a board square, or None if off-board.

        Inverts the fixed layout in `_draw`: header on row 1, blank row 2, then the
        board from row 3, each rank cell_h rows tall; a cell_w-wide square after the
        _LABEL_W-column rank gutter. Uses the same cell size the last frame drew with,
        and reverses the orientation logic `_render_board` draws with.
        """
        line = (row - 3) // self.cell_h
        if not 0 <= line <= 7 or col <= _LABEL_W:
            return None
        file_idx = (col - _LABEL_W - 1) // self.cell_w
        if not 0 <= file_idx <= 7:
            return None
        if HUMAN_PLAYS_WHITE:
            return chess.square(file_idx, 7 - line)
        return chess.square(7 - file_idx, line)

    def _move_between(
        self, frm: chess.Square, to: chess.Square
    ) -> chess.Move | None:
        """The legal move from `frm` to `to`, or None. Auto-queens a promotion;
        type the move (e.g. e8=N) to underpromote."""
        candidates = [
            m
            for m in self.board.legal_moves
            if m.from_square == frm and m.to_square == to
        ]
        for move in candidates:
            if move.promotion in (None, chess.QUEEN):
                return move
        return candidates[0] if candidates else None

    def _click(self, col: int, row: int) -> tuple[chess.Move | None, str]:
        """Resolve a left-click into a move via two clicks (from, then to).

        Returns (move-or-None, new status line). First click picks up one of the
        human's pieces; the second either completes a legal move, re-picks another
        own piece, or clears the selection.
        """
        square = self._square_at(col, row)
        if square is None:  # clicked the border/labels — drop any selection
            self.selected = None
            return None, ""
        piece = self.board.piece_at(square)
        if self.selected is None:
            if piece is not None and piece.color == self.human:
                self.selected = square
                return None, f"{chess.square_name(square)} — click a destination"
            return None, ""
        if square == self.selected:  # click the picked-up piece again to drop it
            self.selected = None
            return None, ""
        move = self._move_between(self.selected, square)
        if move is not None:
            self.selected = None
            return move, ""
        if piece is not None and piece.color == self.human:  # switch pieces
            self.selected = square
            return None, f"{chess.square_name(square)} — click a destination"
        self.selected = None
        return None, "not a legal destination — 'moves' lists the options"

    def _cell_size(self, cols: int, rows: int) -> tuple[int, int]:
        """Largest CELL_W x CELL_H (in terminal cells) that fits the board, its
        gutter, and the surrounding chrome in a `cols` x `rows` terminal, keeping a
        ~2:1 aspect so squares look square. Never smaller than the minimal 3x1."""
        # Chrome around the 8*h board lines: header, blank, file labels, blank,
        # status, prompt = 6 rows; the gutter + 8 squares must fit `cols`.
        h = (rows - 6) // 8
        w_fit = (cols - _LABEL_W) // 8
        h = max(1, min(h, w_fit // 2, _MAX_CELL_H))
        w = max(3, min(2 * h, w_fit))
        return w, h

    def _draw(self, status: str, prompt: str = "") -> None:
        """Repaint the whole frame at a fixed position (top of the alt screen).

        Cursor-home + clear-each-line instead of a full screen wipe, so there is
        no flicker; the cursor lands after `prompt`, where typed input echoes. The
        board is sized to the terminal each frame, so nothing scrolls off.
        """
        cols, rows = shutil.get_terminal_size(fallback=(80, 24))
        self.cell_w, self.cell_h = self._cell_size(cols, rows)
        side = "White" if HUMAN_PLAYS_WHITE else "Black"
        header = (
            f"You are {side}. Stockfish searches to depth {SEARCH_DEPTH} — the harder "
            "the position, the longer (and louder) it strains."
        )
        # Keep header/status to one line: a wrapped header would push the board down
        # and desync mouse hit-testing (see _square_at).
        out = ["\x1b[H"]
        for line in (header[:cols], "", *self._render_board(), "", status[:cols]):
            out.append(f"\x1b[K{line}\n")
        out.append(f"\x1b[K{prompt}\x1b[J")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _parse_move(self, raw: str) -> chess.Move | None:
        for parse in (self.board.parse_san, self.board.parse_uci):
            try:
                move = parse(raw)
            except ValueError:  # Invalid/Illegal/Ambiguous all subclass ValueError
                continue
            if move in self.board.legal_moves:
                return move
        return None

    def _typed_move(self, raw: str) -> tuple[chess.Move | None, str | None]:
        """Interpret a typed line: (move, None) to play, (None, status) to report,
        or (None, None) to quit. 'moves'/'l' lists; a bad move explains itself."""
        if raw in ("q", "quit"):
            return None, None
        if raw in ("moves", "l"):
            return None, ", ".join(self.board.san(m) for m in self.board.legal_moves)
        move = self._parse_move(raw)
        if move is None:
            return None, f"{raw!r} is not a legal move here — 'moves' lists the options"
        return move, ""

    def _human_move(self, status: str) -> bool:
        """The human's turn: play by clicking (two clicks: piece, then destination)
        or by typing a move. Cbreak mode reads keys one at a time, so we echo the
        typed buffer by redrawing. Returns False if they resign/quit."""
        prompt = "click a piece or type a move (e.g. e4, g1f3) — 'moves', 'q' quit: "
        flush_input()  # drop any clicks that landed while Stockfish was thinking
        self.selected = None
        buf = ""
        while True:
            self._draw(status, prompt + buf)
            key = read_key()
            if not key:  # EOF (stdin closed)
                return False
            mouse = parse_mouse(key)
            if mouse is not None:
                col, row, button, pressed = mouse
                # Left-button press only; ignore releases, other buttons, the wheel.
                if pressed and (button & 0b11) == 0 and not (button & 64):
                    move, status = self._click(col, row)
                    if move is not None:
                        self.board.push(move)
                        return True
                continue
            if key in ("\r", "\n"):
                raw, buf = buf.strip(), ""
                if not raw:
                    continue
                move, status = self._typed_move(raw)
                if move is not None:
                    self.board.push(move)
                    return True
                if status is None:  # q/quit
                    return False
            elif key in ("\x7f", "\x08"):  # backspace
                buf = buf[:-1]
            elif len(key) == 1 and key.isprintable():
                buf += key

    def play(self) -> None:
        status = "Your move." if self.board.turn == self.human else ""
        result = "Resigned. Good game."
        try:
            with AltScreen(), MouseTracking(), RawTerminal():
                while not self.board.is_game_over():
                    if self.board.turn == self.human:
                        if not self._human_move(status):
                            break
                        status = ""
                    else:
                        self._draw("Stockfish is thinking...")
                        move = self._engine_move()
                        status = f"Stockfish plays {self.board.san(move)} — your move."
                        self.board.push(move)
                if self.board.is_game_over():
                    result = (
                        f"Game over — {self.board.result()} "
                        f"({self.board.outcome().termination.name.lower()})"
                    )
                    self._draw(f"{result}   press any key to exit")
                    read_key()
        except KeyboardInterrupt:
            pass  # treated as resign; the context managers unwind the screen
        # The alt screen vanishes on exit, so leave the outcome on the normal one.
        print(result)
