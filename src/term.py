"""Raw-terminal helpers shared by the interactive drivers (main loop, panel)."""

from __future__ import annotations

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


def read_key() -> str:
    """Read one keypress, collapsing arrow-key escape sequences into e.g. '\\x1b[A'."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        ready, _, _ = select.select([sys.stdin], [], [], 0.001)
        if ready:
            ch += sys.stdin.read(2)
    return ch
