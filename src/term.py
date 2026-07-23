"""Raw-terminal helpers shared by the interactive drivers (main loop, panel)."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty


class AltScreen:
    """Switch to the terminal's alternate screen buffer for full-screen TUIs.

    Drivers that own the whole screen (e.g. the chess board) draw here; the main
    screen and its scrollback come back untouched on exit.
    """

    def __enter__(self) -> "AltScreen":
        sys.stdout.write("\x1b[?1049h\x1b[H")
        sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.write("\x1b[?1049l")
        sys.stdout.flush()


class RawTerminal:
    """Put the TTY in cbreak mode so we read single keypresses without Enter.

    cbreak (not full raw) leaves signal keys working, so Ctrl-C still interrupts.
    Restores the original terminal settings on exit no matter what.
    """

    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class MouseTracking:
    """Ask the terminal to report mouse clicks as SGR escape sequences.

    Enables button tracking (?1000) with SGR extended coordinates (?1006, so
    columns past 223 don't overflow), and turns both back off on exit so the
    shell is left clean. Reports arrive through `read_key`; decode with
    `parse_mouse`. Needs a terminal that supports xterm mouse reporting.
    """

    def __enter__(self) -> "MouseTracking":
        sys.stdout.write("\x1b[?1000h\x1b[?1006h")
        sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.write("\x1b[?1000l\x1b[?1006l")
        sys.stdout.flush()


def flush_input() -> None:
    """Discard unread terminal input (e.g. clicks that piled up mid-computation)."""
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def _input_ready(fd: int, timeout: float = 0.05) -> bool:
    """True if `fd` has a byte to read within `timeout` seconds."""
    return bool(select.select([fd], [], [], timeout)[0])


def _read_char(fd: int, first: bytes) -> str:
    """Complete a UTF-8 character whose lead byte is `first` (single bytes stay fast)."""
    b0 = first[0]
    if b0 < 0x80:
        return first.decode()
    length = 4 if b0 >= 0xF0 else 3 if b0 >= 0xE0 else 2 if b0 >= 0xC0 else 1
    while len(first) < length:
        more = os.read(fd, 1)
        if not more:
            break
        first += more
    return first.decode("utf-8", "replace")


def read_key() -> str:
    """Read one keypress. Arrow keys collapse to e.g. '\\x1b[A'; an SGR mouse report
    (when tracking is on) comes back whole as '\\x1b[<b;x;yM' for `parse_mouse`.

    Reads from the raw fd with `os.read` rather than the buffered text `sys.stdin`:
    a buffered `read(1)` pulls a whole escape sequence into Python's own buffer and
    then `select` on the fd sees nothing pending, so a mouse report would arrive as a
    bare Escape followed by its bytes as separate keys. Going through the fd keeps
    `select` honest so the full sequence is assembled here.
    """
    fd = sys.stdin.fileno()
    first = os.read(fd, 1)
    if not first:
        return ""  # EOF (stdin closed)
    if first != b"\x1b":
        return _read_char(fd, first)
    if not _input_ready(fd):
        return "\x1b"  # a lone Escape keypress
    seq = first + os.read(fd, 1)
    if not seq.endswith(b"["):
        return seq.decode("utf-8", "replace")  # ESC + non-CSI (e.g. Alt-key)
    third = os.read(fd, 1)
    seq += third
    if third != b"<":
        return seq.decode("utf-8", "replace")  # ordinary CSI: arrows are ESC [ A ...
    while not seq[-1:].isalpha():  # SGR mouse: consume through the final 'M'/'m'
        if not _input_ready(fd):
            break
        seq += os.read(fd, 1)
    return seq.decode("utf-8", "replace")


def parse_mouse(seq: str) -> tuple[int, int, int, bool] | None:
    """Decode an SGR mouse report into (col, row, button, pressed); None if not one.

    col/row are 1-based terminal cells; `pressed` is True for a press ('M'), False
    for a release ('m'). `button`'s low two bits pick the button (0 = left).
    """
    if not seq.startswith("\x1b[<") or seq[-1:] not in ("M", "m"):
        return None
    try:
        button, col, row = (int(p) for p in seq[3:-1].split(";"))
    except ValueError:
        return None
    return col, row, button, seq[-1] == "M"
