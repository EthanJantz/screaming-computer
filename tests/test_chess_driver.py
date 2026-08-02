"""Tests for the chess driver's mouse/keyboard UI: coordinate mapping, the
two-click move state machine, promotion, board highlighting, and typed input.

These exercise only the pure UI/logic — none touch Stockfish — so the driver is
built with `__new__`, skipping `__init__` (which would spawn the engine). That
keeps the tests fast and hermetic, and unlike the real game they don't need a
Stockfish binary installed.
"""

import itertools
import re

import chess

import chess_driver
from chess_driver import ChessDriver, _wrap

# The board's fixed-position colors, as emitted by `_render_board` (see chess_driver).
_LAST_MOVE_BG = "\x1b[48;5;107m"
_SELECTED_BG = "\x1b[48;5;179m"
_TARGET_BG = "\x1b[48;5;108m"
_WHITE_FG = "\x1b[38;5;231m"
_BLACK_FG = "\x1b[38;5;16m"


def make_driver(fen: str | None = None) -> ChessDriver:
    d = ChessDriver.__new__(ChessDriver)  # no engine: __init__ would spawn Stockfish
    d.board = chess.Board(fen) if fen else chess.Board()
    d.human = chess.WHITE
    d.selected = None
    d.cell_w, d.cell_h = 3, 1  # minimal board; _draw would size this to the terminal
    d.pad_x, d.pad_y = 0, 0  # unpadded; _draw would center this in the terminal
    return d


# A long, already-wrapped status: what 'moves' hands `_draw` from the opening.
_MOVES_STATUS = _wrap(
    ", ".join(chess.Board().san(m) for m in chess.Board().legal_moves)
)


def _plain(line: str) -> str:
    """`line` with its color/cursor escapes stripped, so painted columns line up with
    the columns a terminal reports clicks in."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)


# --- coordinate mapping: terminal (col, row) -> board square ---
# Layout (human plays White, pieces at the bottom): _HEADER_ROWS of chrome, then the
# board on rows 5..12, a 3-column rank label then 3 columns per square.
# row = 5 + (7 - rank_index); col = 4 + 3*file.


def test_square_at_maps_cells_from_the_human_pov():
    d = make_driver()
    assert d._square_at(16, 11) == chess.E2  # file e (col 16), rank 2 (row 11)
    assert d._square_at(16, 9) == chess.E4  # rank 4 is two rows up from rank 2
    assert d._square_at(4, 12) == chess.A1  # bottom-left corner
    assert d._square_at(27, 5) == chess.H8  # top-right corner


def test_square_at_is_stable_across_a_squares_three_columns():
    d = make_driver()
    assert d._square_at(16, 11) == d._square_at(17, 11) == d._square_at(18, 11)


def test_square_at_rejects_off_board_clicks():
    d = make_driver()
    assert d._square_at(1, 11) is None  # rank-label column (col < 4)
    assert d._square_at(16, 4) is None  # header row (above the board)
    assert d._square_at(16, 13) is None  # below the board
    assert d._square_at(28, 5) is None  # right of the h-file (file_idx > 7)


def test_square_at_uses_the_current_cell_size():
    """At a larger cell size, hit-testing scales: a1 fills a 6x3 block from the
    bottom-left, and clicks anywhere inside it map to the same square."""
    d = make_driver()
    d.cell_w, d.cell_h = 6, 3
    # a1 occupies cols 4..9 and the three board lines for rank 1 (rows 26..28).
    for row in (26, 27, 28):
        for col in (4, 6, 9):
            assert d._square_at(col, row) == chess.A1
    assert d._square_at(10, 28) == chess.B1  # next file over
    assert d._square_at(4, 29) is None  # one row below the board


def test_square_at_follows_the_centered_board():
    """Centering shifts the board right by pad_x and down by pad_y, and hit-testing
    moves with it — the same click that was a1 unpadded is off-board once shifted."""
    d = make_driver()
    assert d._square_at(4, 12) == chess.A1
    d.pad_x, d.pad_y = 10, 5
    assert d._square_at(4 + 10, 12 + 5) == chess.A1
    assert d._square_at(27 + 10, 5 + 5) == chess.H8  # far corner shifts too
    assert d._square_at(4, 12) is None  # left of the shifted gutter


def test_square_at_agrees_with_the_frame_actually_drawn(monkeypatch, capsys):
    """The regression guard for click/paint desync: hit-test the real frame rather
    than a remembered offset. Every rank digit and file letter `_draw` paints must
    map back, through `_square_at`, to the square it labels — at each terminal size,
    since cell size and centering (and so the board's top row) change with it.

    Clicks silently drifted a rank when chrome was added above the board and only
    `_square_at`'s hard-coded row offset was left behind, which no test pinning
    literal rows could catch: they encode the same offset the code does.
    """
    sizes = ((80, 24), (100, 30), (120, 45), (60, 20), (200, 50))
    # A wrapped 'moves' list is the tall case: the status block grows and the board
    # shrinks to make room, so every row above the board moves with it.
    for (cols, rows), status in itertools.product(sizes, ("status", _MOVES_STATUS)):
        d = make_driver()
        size = (cols, rows)  # bound now, not captured, so each pass draws its own size
        monkeypatch.setattr(
            chess_driver.shutil, "get_terminal_size", lambda fallback=None: size
        )
        d._draw(status, "> ")
        lines = [_plain(line) for line in capsys.readouterr().out.split("\n")]
        assert len(lines) <= rows, (
            f"frame overflows a {cols}x{rows} terminal ({len(lines)} lines)"
        )

        label_row = max(  # the file-letter row closes the board
            i
            for i, line in enumerate(lines, 1)
            if re.fullmatch(r"\s*a(\s+[b-h]){7}\s*", line)
        )
        files = {
            m.group(): m.start() + 1  # 1-based column, as the terminal reports clicks
            for m in re.finditer(r"[a-h]", lines[label_row - 1])
        }
        assert len(files) == 8
        for row, line in enumerate(lines[: label_row - 1], 1):
            digit = re.match(r"\s*([1-8])\s", line[: d.pad_x + 3 + 1])
            if digit is None:  # a board line without the rank label, or chrome
                continue
            rank = int(digit.group(1)) - 1
            for name, col in files.items():
                want = chess.square(chess.FILE_NAMES.index(name), rank)
                got = d._square_at(col, row)
                assert got == want, (
                    f"{cols}x{rows}: click on the {name}{rank + 1} label "
                    f"selected {got and chess.square_name(got)}"
                )


def test_draw_trims_a_status_too_tall_for_the_terminal(monkeypatch, capsys):
    """A status block can't be allowed to push the frame past the last row — that
    scrolls the terminal and leaves every click a rank off. It is cut to fit and
    marked, rather than overflowing."""
    monkeypatch.setattr(
        chess_driver.shutil, "get_terminal_size", lambda fallback=None: (80, 24)
    )
    d = make_driver()
    d._draw(_wrap(", ".join(f"Qx{i}" for i in range(200))), "# ")
    lines = capsys.readouterr().out.split("\n")
    assert len(lines) <= 24
    assert "…" in _plain(lines[-2])  # the cut is marked, just above the prompt


def test_draw_paints_a_wrapped_status_one_line_per_row(monkeypatch, capsys):
    """A newline-joined status becomes that many rows of the frame, each aligned under
    the board — not a single truncated line."""
    monkeypatch.setattr(
        chess_driver.shutil, "get_terminal_size", lambda fallback=None: (100, 40)
    )
    d = make_driver()
    d._draw(_MOVES_STATUS, "# ")
    painted = [_plain(line).strip() for line in capsys.readouterr().out.split("\n")]
    for line in _MOVES_STATUS.split("\n"):
        assert line in painted
    assert painted[-1] == "#"  # the prompt still lands last, below the whole block


# --- _cell_size: scale the board to fill the terminal, keeping a ~2:1 aspect ---


def test_cell_size_grows_on_a_roomy_terminal():
    d = make_driver()
    assert d._cell_size(80, 24) == (4, 2)  # bigger than the minimal 3x1 board


def test_cell_size_caps_so_the_glyph_stays_proportionate():
    d = make_driver()
    # A lone glyph can't grow, so cells stop enlarging past the cap even on a
    # huge terminal (bounded by _MAX_CELL_H).
    assert d._cell_size(400, 200) == (4, 2)


def test_cell_size_never_below_the_minimal_board():
    d = make_driver()
    assert d._cell_size(20, 10) == (3, 1)  # clamps up so the board still renders


# --- _move_between: the legal move joining two squares, auto-queening promotions ---


def test_move_between_returns_the_legal_move():
    d = make_driver()
    assert d._move_between(chess.E2, chess.E4) == chess.Move.from_uci("e2e4")


def test_move_between_auto_queens_a_promotion():
    d = make_driver("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = d._move_between(chess.A7, chess.A8)
    assert move is not None and move.promotion == chess.QUEEN


def test_move_between_returns_none_for_an_illegal_pair():
    d = make_driver()
    assert d._move_between(chess.E2, chess.E5) is None


# --- _click: the two-click (pick up a piece, then a destination) state machine ---


def test_click_picks_up_an_own_piece():
    d = make_driver()
    move, _ = d._click(16, 11)  # e2 pawn
    assert d.selected == chess.E2 and move is None


def test_click_ignores_empty_and_opponent_squares():
    d = make_driver()
    assert d._click(16, 9)[0] is None and d.selected is None  # empty e4
    assert d._click(4, 5)[0] is None and d.selected is None  # a8 (Black's rook)


def test_two_clicks_complete_a_move_and_clear_the_selection():
    d = make_driver()
    d._click(16, 11)  # pick up e2
    move, _ = d._click(16, 9)  # drop on e4
    assert move == chess.Move.from_uci("e2e4")
    assert d.selected is None


def test_clicking_the_picked_up_piece_again_drops_it():
    d = make_driver()
    d._click(16, 11)
    move, _ = d._click(17, 11)  # same e2 square, different column of the cell
    assert move is None and d.selected is None


def test_clicking_another_own_piece_switches_the_selection():
    d = make_driver()
    d._click(22, 12)  # g1 knight
    move, _ = d._click(7, 12)  # b1 knight — no g1->b1 move, so re-pick instead
    assert move is None and d.selected == chess.B1


def test_illegal_destination_clears_the_selection():
    d = make_driver()
    d._click(16, 11)  # e2
    move, status = d._click(16, 8)  # e5: empty, not a legal e2 destination
    assert move is None and d.selected is None
    assert "not a legal destination" in status


def test_clicking_off_board_clears_the_selection():
    d = make_driver()
    d._click(16, 11)
    move, _ = d._click(1, 11)  # the rank label, off the playable grid
    assert move is None and d.selected is None


# --- _render_board: fixed frame with move/selection highlighting ---


def test_render_has_eight_ranks_plus_a_file_label_row():
    lines = make_driver()._render_board()
    assert len(lines) == 9


def test_render_orients_white_at_the_bottom():
    lines = make_driver()._render_board()
    assert _BLACK_FG in lines[0]  # rank 8 across the top = Black's back rank
    assert _WHITE_FG in lines[7]  # rank 1 along the bottom = White's back rank


def test_render_highlights_only_the_last_moves_destination():
    d = make_driver()
    d.board.push_san("e4")
    joined = "\n".join(d._render_board())
    assert joined.count(_LAST_MOVE_BG) == 1


def test_render_marks_the_selection_and_its_legal_targets():
    d = make_driver()
    d.selected = chess.E2  # a pawn: legal targets are exactly e3 and e4
    joined = "\n".join(d._render_board())
    assert joined.count(_SELECTED_BG) == 1
    assert joined.count(_TARGET_BG) == 2
    assert "·" in joined  # empty legal squares get a dot marker


def test_render_scales_to_the_cell_size():
    d = make_driver()
    d.cell_w, d.cell_h = 6, 3
    lines = d._render_board()
    assert len(lines) == 8 * 3 + 1  # each rank is cell_h lines, plus the label row
    d.board.push_san("e4")
    joined = "\n".join(d._render_board())
    assert joined.count(_LAST_MOVE_BG) == 3  # the e4 square spans all cell_h lines


# --- _typed_move: interpreting a typed line into a move/status/quit ---


def test_typed_quit_words_signal_quit():
    d = make_driver()
    assert d._typed_move("q") == (None, None)
    assert d._typed_move("quit") == (None, None)


def test_typed_moves_lists_the_legal_moves():
    d = make_driver()
    move, status = d._typed_move("moves")
    assert move is None
    assert "e4" in status and "Nf3" in status


def test_typed_moves_is_wrapped_onto_several_lines():
    """The 20 opening moves are one unreadable run-on line otherwise, and `_draw`
    would just truncate it at the terminal's edge — no move is lost to wrapping."""
    d = make_driver()
    _, status = d._typed_move("moves")
    lines = status.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= chess_driver._MIN_BOARD_W for line in lines)
    # Lines break after a comma, so the trailing empty token is expected.
    listed = {m.strip() for line in lines for m in line.split(",") if m.strip()}
    assert listed == {d.board.san(m) for m in d.board.legal_moves}


def test_wrap_keeps_a_short_status_on_one_line():
    assert _wrap("check!") == "check!"
    assert _wrap("") == ""


def test_typed_legal_move_is_played():
    d = make_driver()
    move, status = d._typed_move("e4")
    assert move == chess.Move.from_uci("e2e4") and status == ""


def test_typed_illegal_move_explains_itself():
    d = make_driver()
    move, status = d._typed_move("e9")
    assert move is None and "not a legal move" in status
