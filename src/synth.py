"""Additive source-filter voice synthesizer, built on ThinkDSP.

Renders a controllable sustained vowel ("ahh"). It knows nothing about where its
parameters come from — every driver, no matter the work source, ultimately just
moves a `SynthParams`. This is the core of the screaming computer.

The voiced source is expressed with ThinkDSP's ``Signal`` -> ``Wave`` model: each
sound component is a ``Signal`` rendered over a continuous absolute-time axis (a
running integer sample counter) and mixed as ``Wave`` data per block. See
``AdditiveVoice`` for how phase stays continuous while the fundamental bends every
sample. Roughness ingredients (jitter, shimmer, subharmonics, chaos, drive, breath)
layer on top of that voiced source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from thinkdsp import PI2, Signal, UncorrelatedGaussianNoise, Wave


@dataclass
class SynthParams:
    """Per-block synth control values, derived from a 0..1 intensity."""

    amplitude: float  # overall output level
    pitch_scale: float  # multiplies F0 (rises with intensity); 1.0 == idle pitch
    rolloff_p: float  # source amplitude rolloff exponent (lower = brighter)
    breath_level: float  # mixed-in filtered-noise breath
    jitter_amount: float  # sd of the fast F0 wobble, in semitones (hoarseness)
    shimmer_amount: float  # depth of per-sample amplitude wobble (roughness)
    sub_amount: float  # level of the subharmonic sideband partials (growl)
    chaos_amount: float  # likelihood/strength of held pitch-jump "breaks"
    drive: float  # tanh waveshaper sharpness (grit); 1.0 == no shaping
    drive_mix: float  # dry->wet crossfade for the waveshaper (0..1)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ramp(i: float, threshold: float) -> float:
    """Clamped 0->1 ramp: 0 at or below ``threshold``, 1 at full intensity."""
    return min(max((i - threshold) / (1.0 - threshold), 0.0), 1.0)


def derive_params(intensity: float) -> SynthParams:
    """Map a smoothed 0..1 intensity to synth parameters.

    Roughness ingredients are gated by ``config.ROUGHNESS``; a disabled ingredient
    derives to a no-op value (e.g. 0), so the toggle never reaches the renderer.
    """
    i = float(np.clip(intensity, 0.0, 1.0))
    on = config.ROUGHNESS.get
    return SynthParams(
        amplitude=_lerp(config.AMP_FLOOR, config.AMP_PEAK, i**config.AMP_CURVE),
        pitch_scale=2.0 ** (config.F0_RISE_SEMITONES * i / 12.0),
        rolloff_p=(
            _lerp(config.P_IDLE, config.P_BRIGHT, i)
            if on("brightening")
            else config.P_IDLE
        ),
        breath_level=config.BREATH_IDLE,  # constant floor for v1
        jitter_amount=config.JITTER_MAX_ST * i if on("jitter") else 0.0,
        shimmer_amount=config.SHIMMER_MAX * i if on("shimmer") else 0.0,
        # Subharmonics kick in past mid-intensity; chaos (pitch breaks) only at
        # high intensity.
        sub_amount=config.SUB_MAX * _ramp(i, 0.5) if on("subharmonics") else 0.0,
        chaos_amount=_ramp(i, config.CHAOS_THRESH) if on("chaos") else 0.0,
        drive=1.0 + config.DRIVE_MAX * i if on("drive") else 1.0,
        drive_mix=i if on("drive") else 0.0,  # dry at idle, full wet at peak
    )


def _formant_gain(freqs: np.ndarray, formants: list[tuple[int, int]]) -> np.ndarray:
    """Product of resonant formant responses g_i(f) (spec Section 5.1)."""
    gain = np.ones_like(freqs)
    for fc, bandwidth in formants:
        q = fc / bandwidth
        x = freqs / fc
        gain *= 1.0 / np.sqrt((1.0 - x**2) ** 2 + (x / q) ** 2)
    return gain


def _design_breath_fir(
    samplerate: int, formants: list[tuple[int, int]], length: int = 256
) -> np.ndarray:
    """Linear-phase FIR whose magnitude ≈ formant envelope * gentle air-rolloff.

    Convolving white noise with this yields breath colored like the vowel. The
    kernel is normalized to unit energy so `breath_level` sets the breath's RMS
    directly.
    """
    freqs = np.fft.rfftfreq(length, d=1.0 / samplerate)
    mag = _formant_gain(freqs, formants)
    # Gentle high-frequency rolloff so it reads as airy breath, not bright hiss.
    mag = mag / np.sqrt(1.0 + (freqs / config.BREATH_CUTOFF) ** 2)
    # Zero-phase impulse response -> shift to the centre + window -> linear phase.
    h = np.fft.fftshift(np.fft.irfft(mag, n=length))
    h *= np.hanning(length)
    norm = np.sqrt(np.sum(h**2))
    if norm > 0.0:
        h = h / norm
    return h


class AdditiveVoice(Signal):
    """A bank of harmonically-related sinusoids, summed internally.

    This is a ThinkDSP ``Signal``, but its ``evaluate`` does the additive sum in one
    numpy matrix op rather than composing a ``SumSignal`` of N ``Sinusoid`` objects —
    the project's deliberate trade-off (a few dozen partials per block, in the audio
    callback).

    ThinkDSP's ``Sinusoid`` gets click-free block boundaries for free from absolute
    time, but only while frequency is constant. Here the fundamental is modulated
    every sample by ``f_mult`` (pitch rise * chaos * jitter), so — exactly as
    ThinkDSP's ``Chirp`` does — we integrate the instantaneous frequency into phase
    via ``cumsum``. Continuity across blocks then comes from ``base_phases`` (the
    phase each partial holds at the block's first sample, carried by the ``Voice``),
    not from the absolute ``ts`` values.
    """

    def __init__(
        self,
        f0: float,
        ratios: np.ndarray,
        amps: np.ndarray,
        base_phases: np.ndarray,
        f_mult: np.ndarray,
        framerate: int,
    ) -> None:
        self.f0 = f0
        self.ratios = np.asarray(ratios, dtype=np.float64)  # partial freq / f0
        self.amps = np.asarray(amps, dtype=np.float64)
        self.base_phases = np.asarray(base_phases, dtype=np.float64)
        self.f_mult = f_mult  # per-sample fundamental multiplier (len == n)
        self.framerate = framerate

    def evaluate(self, ts: np.ndarray) -> np.ndarray:
        """Sum the partials over the block whose times are ``ts``."""
        dt = 1.0 / self.framerate
        # Exclusive cumulative fundamental cycles: cyc_excl[k] = sum_{j<k} f0*m[j]*dt.
        # The first sample then sits exactly at base_phases (no boundary discontinuity).
        cyc = self.f0 * dt * np.cumsum(self.f_mult)
        cyc_excl = cyc - self.f0 * dt * self.f_mult
        phase = (
            self.base_phases[:, None] + PI2 * self.ratios[:, None] * cyc_excl[None, :]
        )
        return (self.amps[:, None] * np.sin(phase)).sum(axis=0)


class Voice:
    def __init__(
        self,
        samplerate: int = config.SAMPLERATE,
        f0: float = config.F0,
        formants: list[tuple[float, float]] | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.f0 = f0
        self._formants: list[tuple[int, int]] = (
            formants if formants is not None else config.FORMANTS
        )
        self._nyquist = samplerate / 2.0

        # Harmonics 1..H up to (but not including) Nyquist at the *idle* pitch. As
        # the pitch rises these slide up; harmonics that cross Nyquist get masked.
        h_max = int((self._nyquist - 1e-6) // f0)
        self.harmonics = np.arange(1, h_max + 1, dtype=np.float64)

        # One bank of partials: subharmonic sidebands between the harmonics
        # (soundgen addSubh) first, then the harmonics themselves. For period
        # multiplier G, sidebands sit at ratios n + s/G in every harmonic gap
        # (including 1/G below the fundamental), putting period-G growl energy
        # across the whole spectrum. Each partial's phase is carried across
        # blocks for click-free boundaries.
        g = config.SUB_RATIO
        offsets = np.arange(1, g, dtype=np.float64) / g
        sub = (np.arange(0, h_max)[:, None] + offsets[None, :]).ravel()
        self._sub_ratios = sub[sub * f0 < self._nyquist]
        self._ratios = np.concatenate([self._sub_ratios, self.harmonics])
        self._phases = np.zeros(self._ratios.size, dtype=np.float64)
        # Each sideband's distance to its nearest harmonic, as a fraction of F0.
        frac = self._sub_ratios % 1.0
        self._sub_dist_frac = np.minimum(frac, 1.0 - frac)

        # Formant gain at the idle-pitch partial frequencies (cached fast-path for
        # the no-pitch-shift case). Recomputed per block when the pitch rises.
        self.formant_gain = _formant_gain(self._ratios * f0, self._formants)

        # Continuous absolute-time axis: the running sample index handed to each
        # block's ThinkDSP Wave (integer counter -> no float drift over long runs).
        self._n = 0

        # Fast roughness controls: anchor segment lengths in samples, plus each
        # control's last value, carried across blocks so the controls never step.
        self._jitter_hold = max(1, int(config.JITTER_LEN_MS * samplerate / 1000.0))
        self._shimmer_hold = max(1, int(config.SHIMMER_LEN_MS * samplerate / 1000.0))
        self._control_prev = {"jitter": 0.0, "shimmer": 0.0}

        # Pitch-chaos state: current (glided) multiplier, the target it's gliding
        # toward, and how many blocks the target is held before re-rolling.
        self._chaos_cur = 1.0
        self._chaos_target = 1.0
        self._chaos_hold = 0

        # Breath: white noise shaped by the *vowel formants* — i.e. a whispered
        # "ahh", coherent with the voiced tone, rather than generic hiss. We design
        # a linear-phase FIR from the formant magnitude response once, then convolve
        # the noise through it per block (vectorized, carrying a tail between blocks
        # for continuity — no per-sample Python loop in the audio callback). The
        # white noise itself comes from a ThinkDSP noise Signal.
        self._rng = np.random.default_rng()
        self._breath_kernel = _design_breath_fir(samplerate, self._formants)
        self._breath_tail = np.zeros(self._breath_kernel.size - 1, dtype=np.float64)

    def _envelope(self, rolloff_p: float, f0_factor: float) -> np.ndarray:
        """Source rolloff * formant gain at every partial, Nyquist-masked.

        ``f0_factor`` slides the partials up/down under the *stationary* formants;
        the formant gain is re-evaluated at the shifted frequencies (source-filter
        synthesis), and partials above Nyquist are dropped to avoid aliasing. The
        rolloff ratio is clamped at 1 so sub-fundamental sidebands take the
        fundamental's source level instead of exceeding it.
        """
        a_source = np.maximum(self._ratios, 1.0) ** -rolloff_p
        if f0_factor == 1.0:
            return a_source * self.formant_gain
        freqs = self._ratios * self.f0 * f0_factor
        gain = _formant_gain(freqs, self._formants)
        return a_source * gain * (freqs < self._nyquist)

    def _amplitudes(
        self, rolloff_p: float, f0_factor: float, sub_amount: float
    ) -> np.ndarray:
        """Amplitudes for the full partial bank at the current pitch.

        Harmonic amplitudes are normalized to sum to 1, bounding the voiced sum
        to [-1, 1] before the overall amplitude is applied. Sidebands ride on
        top: the same envelope level, scaled by ``sub_amount`` and a Gaussian
        falloff with Hz-distance from the nearest harmonic (soundgen's subWidth
        weighting), so they follow the vowel's spectral shape.
        """
        amps = self._envelope(rolloff_p, f0_factor)
        nsub = self._sub_ratios.size
        total = amps[nsub:].sum()
        if total > 0.0:
            amps = amps / total
        if sub_amount > 0.0:
            dist_hz = self._sub_dist_frac * self.f0 * f0_factor
            weight = np.exp(-0.5 * (dist_hz / config.SUB_WIDTH_HZ) ** 2)
            amps[:nsub] *= sub_amount * weight
        else:
            amps[:nsub] = 0.0
        return amps

    def _smooth_control(self, n: int, hold: int, name: str) -> np.ndarray:
        """Linearly-interpolated random control, continuous across blocks.

        Random anchors every ``hold`` samples — unit-normal for jitter (soundgen's
        rnorm), uniform [-1, 1] for shimmer — joined by linear interpolation, with
        the last value carried into the next block so the control never steps.
        """
        npts = n // hold + 2
        if name == "jitter":
            pts = self._rng.normal(0.0, 1.0, size=npts)
        else:
            pts = self._rng.uniform(-1.0, 1.0, size=npts)
        pts[0] = self._control_prev[name]
        xs = np.arange(npts) * hold
        env = np.interp(np.arange(n), xs, pts)
        self._control_prev[name] = float(env[-1])
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

    def _breath(self, frames: int, level: float, start: float) -> np.ndarray:
        """Formant-shaped breath: ThinkDSP white noise through the FIR, continuous.

        The white noise is a ThinkDSP ``UncorrelatedGaussianNoise`` Wave on the same
        absolute-time axis as the voice. We prepend the previous block's tail so the
        FIR sees past samples (no edge click), then carry the new tail forward.
        """
        white = (
            UncorrelatedGaussianNoise(amp=1.0)
            .make_wave(
                duration=frames / self.samplerate,
                start=start,
                framerate=self.samplerate,
            )
            .ys
        )
        x = np.concatenate([self._breath_tail, white])
        noise = np.convolve(x, self._breath_kernel, mode="valid")  # length == frames
        if self._breath_tail.size:
            self._breath_tail = white[-self._breath_tail.size :]
        return noise * level

    def _f0_multiplier(self, n: int, params: SynthParams) -> tuple[np.ndarray, float]:
        """Per-sample F0 multiplier m[k] = pitch rise * chaos glide * fast jitter.

        Also returns the central (pre-jitter) factor, which drives the block's
        formant re-evaluation.
        """
        m = np.full(n, params.pitch_scale, dtype=np.float64)
        if params.chaos_amount:
            m *= self._chaos_segment(n, params.chaos_amount)
        central_factor = float(m.mean())
        if params.jitter_amount:
            # Multiplicative in semitone space (soundgen: 2^(rnorm(sd=dep)/12)).
            z = self._smooth_control(n, self._jitter_hold, "jitter")
            m *= 2.0 ** (params.jitter_amount * z / 12.0)
        return m, central_factor

    def render_block(self, frames: int, params: SynthParams) -> np.ndarray:
        """Render one mono block of `frames` samples as float32 in [-1, 1].

        Pipeline: per-sample F0 -> voiced partials (harmonics + subharmonics) ->
        shimmer (amplitude wobble) -> waveshape -> overall amplitude (the swell) ->
        add constant breath floor -> hard-limit. The breath is added *after* the
        amplitude so it stays a fixed floor: at idle the breath is most of what you
        hear; as the voice swells it dominates and the breath recedes underneath.
        """
        n = frames
        fs = self.samplerate

        m, central_factor = self._f0_multiplier(n, params)

        # Continuous absolute-time window for this block (ThinkDSP make_wave style).
        start = self._n / fs
        duration = n / fs

        # Voiced partials: the formant-shaped source-filter "ahh" plus the
        # subharmonic sidebands, summed in one bank riding the same per-sample F0.
        # Zero-amplitude rows (e.g. all sidebands when sub_amount == 0) are sliced
        # out before the matrix op so the tonal voice doesn't pay for them.
        amps = self._amplitudes(params.rolloff_p, central_factor, params.sub_amount)
        active = amps > 0.0
        voiced = AdditiveVoice(
            self.f0, self._ratios[active], amps[active], self._phases[active], m, fs
        ).make_wave(duration, start, fs)
        block = voiced.ys

        # Advance the carried phases by this block's fundamental cycle count.
        block_cycles = self.f0 * float(m.sum()) / fs
        self._phases = (self._phases + PI2 * self._ratios * block_cycles) % PI2

        # Shimmer: per-sample amplitude wobble on the voiced source.
        if params.shimmer_amount:
            shimmer = self._smooth_control(n, self._shimmer_hold, "shimmer")
            block = block * (1.0 + params.shimmer_amount * shimmer)

        # Drive: tanh waveshaping for high-harmonic grit. Crossfade dry->wet by
        # intensity so it's a no-op at idle and fades in with no level jump (the
        # /tanh(drive) normalization keeps unit-level peaks roughly unit).
        if params.drive_mix > 0.0:
            wet = np.tanh(params.drive * block) / np.tanh(params.drive)
            block = block + params.drive_mix * (wet - block)

        block = block * params.amplitude  # the swell applies to the voice only
        block = block + self._breath(n, params.breath_level, start)  # constant floor

        # Hand the mixed block back as a ThinkDSP Wave, then hard-limit and emit.
        out = Wave(block, framerate=fs)
        np.clip(out.ys, -1.0, 1.0, out=out.ys)
        self._n += n
        return out.ys.astype(np.float32)
