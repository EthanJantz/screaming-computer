"""Tests for the chess driver's mouse/keyboard UI: coordinate mapping, the
two-click move state machine, promotion, board highlighting, and typed input.

These exercise only the pure UI/logic — none touch Stockfish — so the driver is
built with `__new__`, skipping `__init__` (which would spawn the engine). That
keeps the tests fast and hermetic, and unlike the real game they don't need a
Stockfish binary installed.
"""

import chess

from chess_driver import HUMAN_PLAYS_WHITE, ChessDriver

# The board's fixed-position colors, as emitted by `_render_board` (see chess_driver).
_LAST_MOVE_BG = "\x1b[48;5;107m"
_SELECTED_BG = "\x1b[48;5;179m"
_TARGET_BG = "\x1b[48;5;108m"
_WHITE_FG = "\x1b[38;5;231m"
_BLACK_FG = "\x1b[38;5;16m"


def make_driver(fen: str | None = None) -> ChessDriver:
    d = ChessDriver.__new__(ChessDriver)  # no engine: __init__ would spawn Stockfish
    d.board = chess.Board(fen) if fen else chess.Board()
    d.human = chess.WHITE if HUMAN_PLAYS_WHITE else chess.BLACK
    d.selected = None
    return d


# --- coordinate mapping: terminal (col, row) -> board square ---
# Layout (human plays White, pieces at the bottom): board on rows 3..10, a 3-column
# rank label then 3 columns per square. row = 3 + (7 - rank_index); col = 4 + 3*file.


def test_square_at_maps_cells_from_the_human_pov():
    d = make_driver()
    assert d._square_at(16, 9) == chess.E2  # file e (col 16), rank 2 (row 9)
    assert d._square_at(16, 7) == chess.E4  # rank 4 is two rows up from rank 2
    assert d._square_at(4, 10) == chess.A1  # bottom-left corner
    assert d._square_at(27, 3) == chess.H8  # top-right corner


def test_square_at_is_stable_across_a_squares_three_columns():
    d = make_driver()
    assert d._square_at(16, 9) == d._square_at(17, 9) == d._square_at(18, 9)


def test_square_at_rejects_off_board_clicks():
    d = make_driver()
    assert d._square_at(1, 9) is None  # rank-label column (col < 4)
    assert d._square_at(16, 2) is None  # header row (above the board)
    assert d._square_at(16, 11) is None  # below the board
    assert d._square_at(28, 3) is None  # right of the h-file (file_idx > 7)


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
    move, _ = d._click(16, 9)  # e2 pawn
    assert d.selected == chess.E2 and move is None


def test_click_ignores_empty_and_opponent_squares():
    d = make_driver()
    assert d._click(16, 7)[0] is None and d.selected is None  # empty e4
    assert d._click(4, 3)[0] is None and d.selected is None  # a8 (Black's rook)


def test_two_clicks_complete_a_move_and_clear_the_selection():
    d = make_driver()
    d._click(16, 9)  # pick up e2
    move, _ = d._click(16, 7)  # drop on e4
    assert move == chess.Move.from_uci("e2e4")
    assert d.selected is None


def test_clicking_the_picked_up_piece_again_drops_it():
    d = make_driver()
    d._click(16, 9)
    move, _ = d._click(17, 9)  # same e2 square, different column of the cell
    assert move is None and d.selected is None


def test_clicking_another_own_piece_switches_the_selection():
    d = make_driver()
    d._click(22, 10)  # g1 knight
    move, _ = d._click(7, 10)  # b1 knight — no g1->b1 move, so re-pick instead
    assert move is None and d.selected == chess.B1


def test_illegal_destination_clears_the_selection():
    d = make_driver()
    d._click(16, 9)  # e2
    move, status = d._click(16, 6)  # e5: empty, not a legal e2 destination
    assert move is None and d.selected is None
    assert "not a legal destination" in status


def test_clicking_off_board_clears_the_selection():
    d = make_driver()
    d._click(16, 9)
    move, _ = d._click(1, 9)  # the rank label, off the playable grid
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


def test_typed_legal_move_is_played():
    d = make_driver()
    move, status = d._typed_move("e4")
    assert move == chess.Move.from_uci("e2e4") and status == ""


def test_typed_illegal_move_explains_itself():
    d = make_driver()
    move, status = d._typed_move("e9")
    assert move is None and "not a legal move" in status
