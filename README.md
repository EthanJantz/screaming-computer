# Screaming Computer

What if your computer screamed? This is a real time scream synthesization engine designed for integration with external sources that drive scream intensity — plus a play view that turns the voice into a playable instrument.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Clone the repository and run `uv sync`.

## Run

```bash
uv run main.py                   # "fake" source that loops the intensity ramp
uv run main.py --source manual   # drive the intensity via the keyboard
uv run main.py --source contour  # a scripted intensity gesture, repeating
uv run main.py --source chess    # play chess, screams are driven by the AI
uv run main.py --source panel    # hand-drive every synth parameter / play notes
```

Press `q` (or Ctrl-C) to quit any source.

## Sound character

`--preset <name>` selects a named scream recipe from `src/presets.py` — `roar` (the original screaming-computer voice) or `scream` (an approximation of a reference [soundgen](https://cran.r-project.org/package=soundgen) scream). One-off tweaks layer on top with repeatable `--set KEY=VALUE` overrides of anything in `src/config.py`:

```bash
uv run main.py --preset scream --source contour --set SUB_WIDTH_HZ=250
```

## Playing it as an instrument

`--source panel` opens the parameter workbench: `j`/`k` select a row, `h`/`l` adjust it (`H`/`L` coarse), digits load `derive_params(n/9)` wholesale, `g` toggles roughness regimes. Press `]` to flip to the play view, where the same voice becomes a mono keyboard instrument:

```
   w e   t y u   o p
  a s d f g h j k l ; '
  C D E F G A B C D E F
```

Hold a key to sustain its note and release to let it fade; play legato to glide between pitches. `z`/`x` shift the octave, `-`/`=` ride the scream intensity (which strains held notes a couple of semitones sharp, like a real voice), `]` returns to the workbench, `q` quits. Sculpt the scream in the workbench, then perform it. Envelope and portamento timings live in the `Keyboard play mode` section of `src/config.py` (e.g. `--set GLIDE_MS=80` for a slower slide).

## Chess

Screaming Computer comes with support for sourcing intensity from the [Stockfish](https://github.com/official-stockfish/Stockfish) chess engine. Intensity is driven by search effort — the voice swells as the engine grinds on a hard position, not when it is winning.

```bash
brew install stockfish        # macOS
sudo apt install stockfish    # Debian/Ubuntu
sudo pacman -S stockfish      # Arch
```

The app will find Stockfish on your PATH, via the `STOCKFISH_PATH` environment variable, or by building it and moving the binary to the root of this repository.
