"""All tunable constants for the synth and audio engine.

Starting points from the spec (Section 6). Expect to tune by ear. Anything
specific to a particular work source (e.g. chess node-count bounds) belongs with
that driver, not here.
"""

# --- Audio ---
SAMPLERATE = 44100
BLOCKSIZE = 512

# --- Voice fundamental ---
F0 = 110.0  # Hz, fundamental at idle

# A scream rises in pitch as it strains. F0 climbs this many semitones from idle to
# full intensity (set to 0.0 for the original constant-pitch behaviour). 18 semitones
# = 1.5 octaves, taking a 110 Hz idle groan up to a ~311 Hz scream at full effort.
F0_RISE_SEMITONES = 18.0

# Formants for "ahh" (/ɑ/): (center_Hz, bandwidth_Hz)
FORMANTS = [
    (730, 90),
    (1090, 110),
    (2440, 170),
    (3500, 250),
    (4500, 300),
]

# --- Source rolloff / brightening ---
P_IDLE = 2.0  # source rolloff exponent at idle (~ -12 dB/oct)
P_BRIGHT = 1.0  # rolloff at full intensity (brighter/harsher)

# --- Amplitude envelope ---
AMP_FLOOR = 0.05  # idle voice level
AMP_PEAK = 0.9  # full-load level (leave headroom under 1.0)
AMP_CURVE = 1.5

# --- Roughness ---
# Each ingredient is individually toggleable so it can be A/B'd by ear. Grown one
# entry at a time as ingredients are added.
ROUGHNESS = {
    "jitter": True,
    "shimmer": True,
    "subharmonics": True,
    "chaos": False,
    "drive": True,
    "brightening": True,
    "regimes": True,
}

# Jitter: fast random F0 wobble (hoarseness). Deep + fast like soundgen's
# jitterDep/jitterLen — slow/shallow jitter reads as vibrato instead of roughness.
# Normal anchors in semitone space, like soundgen's 2^(rnorm(sd=jitterDep)/12).
JITTER_MAX_ST = 1.4  # sd of the F0 wobble at full intensity, in semitones
JITTER_LEN_MS = 2.0  # anchor segment length (soundgen jitterLen)

# Shimmer: per-sample random amplitude perturbation (soundgen shimmerDep).
SHIMMER_MAX = 0.18  # fractional amplitude wobble at full intensity
SHIMMER_LEN_MS = 3.0  # control segment length (linearly interpolated)

# Subharmonics: sideband partials between the harmonics (soundgen addSubh). For a
# period multiplier SUB_RATIO=G, sidebands sit at ratios n + s/G (s = 1..G-1), so
# G=2 puts period-doubled energy at F0/2, 1.5*F0, 2.5*F0, ... across the spectrum.
SUB_MAX = 0.4  # sideband level at full intensity, relative to the local envelope
SUB_RATIO = 2  # period multiplier G (soundgen subRatio)
SUB_WIDTH_HZ = 4000.0  # Gaussian sd of sideband strength vs Hz-distance to the
# nearest harmonic (soundgen subWidth); large = flat, small
# = growl concentrated where harmonics are dense/low
DRIVE_MAX = 6.0  # tanh drive amount added at full intensity

# Roughness regimes: a bounded random walk gates subharmonics/jitter/shimmer in
# episodes (soundgen getIntegerRandomWalk + nonlinBalance), so roughness breaks in
# and out instead of sitting at a constant texture. Thresholds drop with intensity.
REGIME_STEP = 0.04  # random-walk step (sd) per block
REGIME_MIN_BLOCKS = 8  # minimum blocks per regime episode (~90 ms; clumper)
REGIME_GATE_SMOOTH = 0.15  # per-block one-pole coefficient for the effect gates

# Pitch chaos: occasional held jumps to a multiple of F0 — deterministic-chaos
# "breaks" that give screams their wild quality (soundgen chaos_freq). High
# intensity only.
CHAOS_THRESH = 0.72  # chaos only above this intensity
CHAOS_PROB = 0.18  # per-decision jump probability at full intensity
CHAOS_RATIOS = [0.5, 1.5]  # F0 multipliers a jump can land on
CHAOS_HOLD_BLOCKS = (2, 6)  # how many blocks a jump persists (min, max)
CHAOS_GLIDE = 0.35  # per-block glide rate toward the target ratio (0..1);
# lower = slower portamento. Instant jumps sound like
# digital distortion, so we glide instead.
BREATH_IDLE = 0.012  # constant breath/noise floor (RMS); ~0.7x the idle voice RMS
BREATH_CUTOFF = (
    4000.0  # Hz, gentle air-rolloff above this in the breath's formant shaping
)

# --- Intensity smoothing (per audio block) ---
SMOOTH_COEFF = 0.02  # tune for an attack/release feel of ~0.2-0.6 s

# --- Keyboard play mode (the panel's play view) ---
# Terminals report key presses only (no key-up), so a note holds its gate open
# NOTE_HOLD_MS past the most recent press; the OS key-repeat keeps extending that
# deadline while the key is physically held (hold a key = sustain the note).
GATE_ATTACK_MS = 8.0  # gate envelope attack time constant
GATE_RELEASE_MS = 200.0  # gate envelope release time constant
NOTE_HOLD_MS = 600.0  # gate-open time per press (must exceed the OS repeat delay)
GLIDE_MS = 30.0  # legato portamento time constant (0 = pitch snaps instantly)
PLAY_STRAIN_SEMITONES = 2.0  # cap on F0_RISE_SEMITONES while playing notes, so
# intensity strains held notes sharp without wrecking
# the pitch you played

# --- Contour source (main.py --source contour) ---
# A scripted intensity gesture: `times` are 0..1 fractions of `duration` (seconds),
# `intensity` the anchor values (linearly interpolated between anchors). After the
# gesture the target drops to 0 for `pause` seconds, then it repeats. Presets
# override this with specific cries.
CONTOUR = {
    "times": (0.0, 0.8, 1.0),
    "intensity": (0.0, 1.0, 0.6),
    "duration": 3.0,
    "pause": 2.0,
}
