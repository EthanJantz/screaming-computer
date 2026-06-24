# Screaming Computer

What if your computer screamed? This is a real time scream synthesization engine designed for integration with external sources to drive scream intensity. 

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Clone the repository and run `uv sync`. 

Screaming Computer comes with support for sourcing intensity from the [Stockfish](https://github.com/official-stockfish/Stockfish) chess engine. Intensity is driven by search effort.

```bash
brew install stockfish        # macOS
sudo apt install stockfish    # Debian/Ubuntu
sudo pacman -S stockfish      # Arch
```

The app will find Stockfish on your PATH, via the `STOCKFISH_PATH` environment variable, or by building it and moving the binary to the root of this repository.

## Run 

```bash
uv run main.py                 # "fake" source that loops the intensity ramp 
uv run main.py --source manual # drive the intensity via the keyboard 
uv run main.py --source chess  # play chess, screams are driven by the AI
```
