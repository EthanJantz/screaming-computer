"""Additive source-filter voice synthesizer.

Renders a controllable sustained vowel ("ahh"). It knows nothing about where its
parameters come from — every driver, no matter the work source, ultimately just
moves a `SynthParams`. This is the core of the screaming computer.

Milestone 1 implements the voiced glottal source shaped by fixed formants, with a
per-harmonic phase accumulator that persists across blocks (no clicks at block
boundaries). Roughness ingredients (jitter, subharmonics, drive, breath) arrive in
later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import config


@dataclass
class SynthParams:
    """Per-block synth control values, derived from a 0..1 intensity."""

    amplitude: float  # overall output level
    pitch_scale: float  # multiplies F0 (rises with intensity); 1.0 == idle pitch
    rolloff_p: float  # source amplitude rolloff exponent (lower = brighter)
    breath_level: float  # mixed-in filtered-noise breath
    jitter_amount: float  # depth of fast sample-and-hold F0 wobble (hoarseness)
    shimmer_amount: float  # depth of per-sample amplitude wobble (roughness)
    sub_amount: float  # level of F0/2 and F0/3 subharmonic oscillators (growl)
    chaos_amount: float  # likelihood/strength of held pitch-jump "breaks"
    drive: float  # tanh waveshaper sharpness (grit); 1.0 == no shaping
    drive_mix: float  # dry->wet crossfade for the waveshaper (0..1)


def derive_params(intensity: float) -> SynthParams:
    """Map a smoothed 0..1 intensity to synth parameters.

    Roughness ingredients are gated by ``config.ROUGHNESS``; a disabled ingredient
    derives to a no-op value (e.g. 0), so the toggle never reaches the renderer.
    """
    i = float(np.clip(intensity, 0.0, 1.0))
    # amplitude can end up larger than 1?
    amplitude = (
        config.AMP_FLOOR + (config.AMP_PEAK - config.AMP_FLOOR) * i**config.AMP_CURVE
    )
    pitch_scale = 2.0 ** (config.F0_RISE_SEMITONES * i / 12.0)
    if config.ROUGHNESS.get("brightening"):
        rolloff_p = config.P_IDLE - (config.P_IDLE - config.P_BRIGHT) * i
    else:
        rolloff_p = config.P_IDLE  # constant timbre, no brightening
    breath_level = config.BREATH_IDLE  # constant floor for v1
    jitter_amount = config.JITTER_MAX * i if config.ROUGHNESS.get("jitter") else 0.0
    shimmer_amount = config.SHIMMER_MAX * i if config.ROUGHNESS.get("shimmer") else 0.0
    # Subharmonics kick in only past mid-intensity.
    sub_ramp = min(max((i - 0.5) / 0.5, 0.0), 1.0)
    sub_amount = (
        config.SUB_MAX * sub_ramp if config.ROUGHNESS.get("subharmonics") else 0.0
    )
    # Chaos (pitch breaks) ramps in only at high intensity.
    chaos_ramp = min(
        max((i - config.CHAOS_THRESH) / (1.0 - config.CHAOS_THRESH), 0.0), 1.0
    )
    chaos_amount = chaos_ramp if config.ROUGHNESS.get("chaos") else 0.0
    if config.ROUGHNESS.get("drive"):
        drive = 1.0 + config.DRIVE_MAX * i
        drive_mix = i  # dry at idle, full wet at peak
    else:
        drive, drive_mix = 1.0, 0.0
    return SynthParams(
        amplitude=amplitude,
        pitch_scale=pitch_scale,
        rolloff_p=rolloff_p,
        breath_level=breath_level,
        jitter_amount=jitter_amount,
        shimmer_amount=shimmer_amount,
        sub_amount=sub_amount,
        chaos_amount=chaos_amount,
        drive=drive,
        drive_mix=drive_mix,
    )


class Voice:
    def __init__(
        self,
        samplerate: int = config.SAMPLERATE,
        f0: float = config.F0,
        formants: list[tuple[float, float]] | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.f0 = f0
        formants: list[tuple[Any, Any]] = (
            formants if formants is not None else config.FORMANTS
        )

        self._formants = formants
        self._nyquist = samplerate / 2.0

        # Harmonics 1..H up to (but not including) Nyquist at the *idle* pitch. As
        # the pitch rises these slide up; harmonics that cross Nyquist get masked.
        h_max = int((self._nyquist - 1e-6) // f0)
        self.harmonics = np.arange(1, h_max + 1, dtype=np.float64)
        self.freqs = self.harmonics * f0

        # Per-sample phase increment per harmonic (at idle pitch; scaled per block).
        self.omega = 2.0 * np.pi * self.freqs / samplerate

        # Formant gain at the idle harmonic frequencies (cached fast-path for the
        # no-pitch-shift case). Recomputed per block when the pitch rises.
        self.formant_gain = self._formant_gain(self.freqs, formants)

        # Persistent phase accumulator — the key to click-free block boundaries.
        self.phase = np.zeros_like(self.freqs)

        # Subharmonic oscillators at F0/2 and F0/3 (period-doubled growl), each with
        # its own persistent phase. They ride the same per-block jitter as the voice.
        self._sub_omega = 2.0 * np.pi * np.array([f0 / 2.0, f0 / 3.0]) / samplerate
        self._sub_phase = np.zeros(2, dtype=np.float64)

        # Reusable sample-index ramp (blocksize is constant in practice).
        self._t = np.arange(config.BLOCKSIZE, dtype=np.float64)

        # Fast roughness control: sample-and-hold segment lengths in samples.
        self._jitter_hold = max(1, int(config.JITTER_LEN_MS * samplerate / 1000.0))
        self._shimmer_hold = max(1, int(config.SHIMMER_LEN_MS * samplerate / 1000.0))
        self._shimmer_prev = 0.0  # carries shimmer env across blocks (no click)

        # Pitch-chaos state: current (glided) multiplier, the target it's gliding
        # toward, and how many blocks the target is held before re-rolling.
        self._chaos_cur = 1.0
        self._chaos_target = 1.0
        self._chaos_hold = 0

        # Breath: white noise shaped by the *vowel formants* — i.e. a whispered
        # "ahh", coherent with the voiced tone, rather than generic hiss. We design
        # a linear-phase FIR from the formant magnitude response once, then convolve
        # the noise through it per block (vectorized, carrying a tail between blocks
        # for continuity — no per-sample Python loop in the audio callback).
        self._rng = np.random.default_rng()
        self._breath_kernel = self._design_breath_fir(samplerate, formants)
        self._breath_tail = np.zeros(self._breath_kernel.size - 1, dtype=np.float64)

    @classmethod
    def _design_breath_fir(
        cls, samplerate: int, formants: list[tuple[float, float]], length: int = 256
    ) -> np.ndarray:
        """Linear-phase FIR whose magnitude ≈ formant envelope * gentle air-rolloff.

        Convolving white noise with this yields breath colored like the vowel. The
        kernel is normalized to unit energy so `breath_level` sets the breath's RMS
        directly.
        """
        freqs = np.fft.rfftfreq(length, d=1.0 / samplerate)
        mag = cls._formant_gain(freqs, formants)
        # Gentle high-frequency rolloff so it reads as airy breath, not bright hiss.
        mag = mag / np.sqrt(1.0 + (freqs / config.BREATH_CUTOFF) ** 2)
        # Zero-phase impulse response -> shift to the centre + window -> linear phase.
        h = np.fft.fftshift(np.fft.irfft(mag, n=length))
        h *= np.hanning(length)
        norm = np.sqrt(np.sum(h**2))
        if norm > 0.0:
            h = h / norm
        return h

    @staticmethod
    def _formant_gain(
        freqs: np.ndarray, formants: list[tuple[float, float]]
    ) -> np.ndarray:
        """Product of resonant formant responses g_i(f) (spec Section 5.1)."""
        gain = np.ones_like(freqs)
        for fc, bandwidth in formants:
            q = fc / bandwidth
            x = freqs / fc
            gain *= 1.0 / np.sqrt((1.0 - x**2) ** 2 + (x / q) ** 2)
        return gain

    def _amplitudes(self, rolloff_p: float, f0_factor: float) -> np.ndarray:
        """Harmonic amplitudes: source rolloff * (fixed) formant gain, normalized.

        ``f0_factor`` slides the harmonics up/down under the *stationary* formants;
        the formant gain is re-evaluated at the shifted frequencies (source-filter
        synthesis), and harmonics above Nyquist are dropped to avoid aliasing.

        Normalizing so the amplitudes sum to 1 bounds the summed sine block to
        [-1, 1] before the overall amplitude is applied.
        """
        a_source = 1.0 / self.harmonics**rolloff_p
        if f0_factor == 1.0:
            a = a_source * self.formant_gain
        else:
            freqs = self.freqs * f0_factor
            gain = self._formant_gain(freqs, self._formants)
            a = a_source * gain * (freqs < self._nyquist)
        total = a.sum()
        if total > 0.0:
            a = a / total
        return a

    def _sah(self, n: int, hold: int) -> np.ndarray:
        """Sample-and-hold random control in [-1, 1], piecewise-constant.

        Steps are fine for *frequency* (jitter): phase stays continuous through a
        frequency step, so it warbles rather than clicks.
        """
        reps = n // hold + 1
        vals = self._rng.uniform(-1.0, 1.0, size=reps)
        return np.repeat(vals, hold)[:n]

    def _shimmer_control(self, n: int, hold: int) -> np.ndarray:
        """Linearly-interpolated random control in [-1, 1], continuous across blocks.

        Used for *amplitude* (shimmer): a sudden amplitude step would click, so we
        interpolate between control points and carry the last value into the next
        block.
        """
        npts = n // hold + 2
        pts = self._rng.uniform(-1.0, 1.0, size=npts)
        pts[0] = self._shimmer_prev
        xs = np.arange(npts) * hold
        env = np.interp(np.arange(n), xs, pts)
        self._shimmer_prev = float(env[-1])
        return env

    def _chaos_segment(self, n: int, amount: float) -> np.ndarray:
        """Per-sample F0 multiplier for pitch chaos, glided (not stepped).

        Re-rolls a target ratio occasionally (scaled by `amount`), then glides the
        current multiplier toward it across the block. Gliding turns a jump into a
        fast vocal-sounding portamento instead of an instant, distortion-like step.
        Continuous across blocks (each block's ramp starts where the last ended).
        """
        if self._chaos_hold > 0:
            self._chaos_hold -= 1
        elif self._rng.random() < amount * config.CHAOS_PROB:
            self._chaos_target = float(self._rng.choice(config.CHAOS_RATIOS))
            lo, hi = config.CHAOS_HOLD_BLOCKS
            self._chaos_hold = int(self._rng.integers(lo, hi + 1))
        else:
            self._chaos_target = 1.0  # settle back to normal pitch

        start = self._chaos_cur
        end = start + (self._chaos_target - start) * config.CHAOS_GLIDE
        seg = np.linspace(start, end, n, endpoint=False)
        self._chaos_cur = end
        return seg

    def _breath(self, frames: int, level: float) -> np.ndarray:
        """Low-passed white noise, scaled by `level`, continuous across blocks."""
        white = self._rng.standard_normal(frames)
        # Prepend the previous block's tail so the filter sees past samples.
        x = np.concatenate([self._breath_tail, white])
        noise = np.convolve(x, self._breath_kernel, mode="valid")  # length == frames
        if self._breath_tail.size:
            self._breath_tail = white[-self._breath_tail.size :]
        return noise * level

    def render_block(self, frames: int, params: SynthParams) -> np.ndarray:
        """Render one mono block of `frames` samples as float32 in [-1, 1].

        Order: build per-sample F0 (pitch rise * chaos * jitter) -> voiced harmonics
        -> subharmonics -> shimmer (amplitude wobble) -> waveshape -> overall
        amplitude (the swell) -> add constant breath floor -> hard-limit.

        Frequency varies *within* the block now, so phase is accumulated per sample
        (a cumulative sum of the instantaneous F0 multiplier) instead of using a
        single per-block frequency. The breath is added *after* the amplitude so it
        stays a fixed floor: at idle the breath is what you mostly hear; as the voice
        swells it dominates and the breath recedes underneath.
        """
        n = frames

        # Per-sample F0 multiplier m[k] = pitch rise * chaos glide * fast jitter.
        m = np.full(n, params.pitch_scale, dtype=np.float64)
        if params.chaos_amount:
            m *= self._chaos_segment(n, params.chaos_amount)
        # Central (pre-jitter) factor drives the formant re-evaluation for the block.
        central_factor = float(m.mean())
        if params.jitter_amount:
            m *= 1.0 + params.jitter_amount * self._sah(n, self._jitter_hold)

        # Cumulative phase factor: c_excl[k] = sum_{j<k} m[j] (so c_excl[0] == 0);
        # total advance over the block is sum(m). This keeps phase continuous even as
        # the instantaneous frequency changes every sample.
        cum = np.cumsum(m)
        c_excl = cum - m
        total_advance = float(cum[-1])

        a = self._amplitudes(params.rolloff_p, central_factor)

        # phases[h, k] = phase_h + omega_h * c_excl[k]. (H x N temporary — known hot
        # spot; fine at 44.1k/512, revisit if we see dropouts.)
        phases = self.phase[:, None] + self.omega[:, None] * c_excl[None, :]
        block = (a[:, None] * np.sin(phases)).sum(axis=0)
        self.phase = (self.phase + self.omega * total_advance) % (2.0 * np.pi)

        # Subharmonics: F0/2 and F0/3 sines, riding the same per-sample F0.
        if params.sub_amount:
            sub_phases = (
                self._sub_phase[:, None] + self._sub_omega[:, None] * c_excl[None, :]
            )
            block += params.sub_amount * np.sin(sub_phases).sum(axis=0)
            self._sub_phase = (self._sub_phase + self._sub_omega * total_advance) % (
                2.0 * np.pi
            )

        # Shimmer: per-sample amplitude wobble on the voiced source.
        if params.shimmer_amount:
            block *= 1.0 + params.shimmer_amount * self._shimmer_control(
                n, self._shimmer_hold
            )

        # Drive: tanh waveshaping for high-harmonic grit. Crossfade dry->wet by
        # intensity so it's a no-op at idle and fades in with no level jump (the
        # /tanh(drive) normalization keeps unit-level peaks roughly unit).
        if params.drive_mix > 0.0:
            wet = np.tanh(params.drive * block) / np.tanh(params.drive)
            block += params.drive_mix * (wet - block)

        block *= params.amplitude  # the swell applies to the voice only
        block += self._breath(n, params.breath_level)  # constant idle floor
        np.clip(block, -1.0, 1.0, out=block)
        return block.astype(np.float32)
