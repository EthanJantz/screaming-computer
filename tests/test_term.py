"""Tests for the shared terminal helpers: assembling keypresses (including
variable-length SGR mouse reports) and decoding those reports into events.
"""

import ast
import os
import select
import signal
import time

import pytest

import term
from term import parse_mouse

pty = pytest.importorskip("pty")  # POSIX only; the pty test needs os.fork


class FakeTTY:
    """Backs read_key's raw-fd reads: pops queued bytes, b'' at end (EOF)."""

    def __init__(self, data: str) -> None:
        self.buf = bytearray(data.encode())

    def fileno(self) -> int:
        return 0

    def read(self, fd: int, n: int = 1) -> bytes:  # matches os.read(fd, n)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out


def feed(monkeypatch, data: str) -> None:
    """Wire read_key to scripted bytes; select is 'ready' while bytes remain."""
    fake = FakeTTY(data)
    monkeypatch.setattr(term.sys, "stdin", fake)
    monkeypatch.setattr(term.os, "read", fake.read)
    monkeypatch.setattr(
        term.select,
        "select",
        lambda r, w, x, t: ([fake.fileno()] if fake.buf else [], [], []),
    )


# --- read_key: single keys, arrow escapes, and whole mouse sequences ---


def test_read_key_returns_a_plain_character(monkeypatch):
    feed(monkeypatch, "e")
    assert term.read_key() == "e"


def test_read_key_collapses_an_arrow_escape(monkeypatch):
    feed(monkeypatch, "\x1b[A")
    assert term.read_key() == "\x1b[A"


def test_read_key_reads_a_whole_mouse_press(monkeypatch):
    feed(monkeypatch, "\x1b[<0;16;7M")
    assert term.read_key() == "\x1b[<0;16;7M"


def test_read_key_reads_a_whole_mouse_release(monkeypatch):
    feed(monkeypatch, "\x1b[<0;16;7m")
    assert term.read_key() == "\x1b[<0;16;7m"


def test_read_key_returns_a_bare_escape_when_nothing_follows(monkeypatch):
    feed(monkeypatch, "\x1b")
    assert term.read_key() == "\x1b"


def test_read_key_returns_empty_string_at_eof(monkeypatch):
    feed(monkeypatch, "")
    assert term.read_key() == ""


def test_read_key_does_not_over_consume_after_a_mouse_event(monkeypatch):
    """A mouse report must stop at its final letter, leaving the next key intact."""
    feed(monkeypatch, "\x1b[<0;16;7Mq")
    assert term.read_key() == "\x1b[<0;16;7M"
    assert term.read_key() == "q"


# --- read_key end to end: a real pty, where a buffered read would over-consume ---


def _read_keys_over_pty(inject: bytes, count: int) -> list[str]:
    """Run read_key `count` times in a child whose stdin is a real pty, injecting
    `inject` all at once, and return what read_key produced.

    Guards the specific bug that a buffered `sys.stdin.read` caused: reading the ESC
    pulls the whole sequence into Python's buffer, `select` on the fd then sees
    nothing, and the mouse report arrives as a bare Escape plus stray keys. The
    result travels back over a dedicated pipe, so the terminal's echo of the injected
    bytes (cbreak leaves echo on) can't pollute it.
    """
    result_r, result_w = os.pipe()
    pid, fd = pty.fork()
    if pid == 0:  # child: its stdin/stdout/stderr are the pty slave
        os.close(result_r)
        # pytest swaps sys.stdin for a capture object whose fileno() raises; the
        # child inherited it. Restore a real stdin on fd 0 (the pty slave).
        term.sys.stdin = os.fdopen(0, "rb", buffering=0)
        from term import RawTerminal, read_key

        with RawTerminal():
            keys = [read_key() for _ in range(count)]
        os.write(result_w, repr(keys).encode())
        os._exit(0)

    os.close(result_w)
    time.sleep(0.3)  # let the child reach its first blocking read
    os.write(fd, inject)
    out = b""
    deadline = time.monotonic() + 5.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise AssertionError("read_key did not return in time")
        ready, _, _ = select.select([result_r], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(result_r, 4096)
        if not chunk:  # child closed the pipe on exit
            break
        out += chunk
    os.close(result_r)
    os.waitpid(pid, 0)
    return ast.literal_eval(out.decode("utf-8", "replace"))


def test_read_key_assembles_a_mouse_click_over_a_real_pty():
    keys = _read_keys_over_pty(b"\x1b[<0;16;7Mq", count=2)
    assert keys == ["\x1b[<0;16;7M", "q"]


# --- parse_mouse: decode an SGR report into (col, row, button, pressed) ---


def test_parse_mouse_decodes_a_left_press():
    assert parse_mouse("\x1b[<0;16;7M") == (16, 7, 0, True)


def test_parse_mouse_decodes_a_release():
    assert parse_mouse("\x1b[<0;16;7m") == (16, 7, 0, False)


def test_parse_mouse_rejects_non_mouse_sequences():
    assert parse_mouse("\x1b[A") is None  # an arrow key
    assert parse_mouse("q") is None  # a plain key
    assert parse_mouse("\x1b[<1;2;3") is None  # unterminated (no M/m)


def test_parse_mouse_rejects_a_malformed_body():
    assert parse_mouse("\x1b[<a;b;cM") is None  # non-integer coordinates
