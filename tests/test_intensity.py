"""Tests for the contour intensity source."""

import pytest

from intensity import ContourIntensity


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_source(**overrides) -> tuple[ContourIntensity, FakeClock]:
    clock = FakeClock()
    kwargs = dict(
        times=(0.0, 0.5, 1.0),
        intensity=(0.0, 1.0, 0.4),
        duration=2.0,
        pause=1.0,
        clock=clock,
    )
    kwargs.update(overrides)
    return ContourIntensity(**kwargs), clock


def test_contour_hits_and_interpolates_anchors():
    src, clock = make_source()
    clock.t = 0.0
    assert src.target() == 0.0
    clock.t = 1.0  # frac 0.5: the peak anchor
    assert src.target() == pytest.approx(1.0)
    clock.t = 0.5  # frac 0.25: halfway up the first segment
    assert src.target() == pytest.approx(0.5)
    clock.t = 1.5  # frac 0.75: halfway down toward 0.4
    assert src.target() == pytest.approx(0.7)


def test_contour_pauses_then_repeats():
    src, clock = make_source()
    clock.t = 2.5  # inside the pause
    assert src.target() == 0.0
    clock.t = 4.0  # wrapped: elapsed 1.0 -> frac 0.5 -> peak again
    assert src.target() == pytest.approx(1.0)


def test_contour_clamps_out_of_range_anchors():
    src, clock = make_source(intensity=(-0.5, 2.0, 0.4))
    clock.t = 0.0
    assert src.target() == 0.0
    clock.t = 1.0
    assert src.target() == 1.0


def test_contour_rejects_bad_anchors():
    with pytest.raises(ValueError):
        make_source(times=(0.0, 1.0))  # length mismatch with 3 intensities
    with pytest.raises(ValueError):
        make_source(times=(0.0, 0.8, 0.5))  # not sorted
    with pytest.raises(ValueError):
        make_source(duration=0.0)
