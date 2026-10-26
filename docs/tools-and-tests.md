The Python tools need [python-chess](https://github.com/niklasf/python-chess) (`pip install chess`); the
trainer also needs numpy and torch (`requirements.txt`).

| | |
|---|---|
| `tools/pipeline.sh` | the training loop ([Training](training.md)) |
| `tools/match.py` | engine against engine under selfmate rules: random, book or mixed openings played with both colours, fixed nodes, depth or a time control, concurrency, PGN, Elo with its error bar; `--e2 random` plays a random mover |
| `tools/play.py` | play against the engine in the terminal (SAN or UCI moves, `hint`, `undo`) |
| `tools/make_puzzles.py`, `tools/puzzles.py` | make selfmate puzzles with an independent python-chess solver, and check that the engine solves them |
| `tools/gen_book.py` | opening books for datagen: random-play positions, with the start position or positions drawn from another book |
| `tools/release_net.py` | name a network for publication after its SHA-256, and make it the engine's default |
| `tools/fairy-stockfish-selfmate.ini` | selfmate for [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish), as a sparring partner |
| `train/train.py` | the trainer |
| `train/calibrate.py` | how well a data set's search scores predict its results |
| `train/verify_net.py` | the engine's evaluations against numpy, bit for bit |
| `train/ftperm.py` | reorder a network's units for the engine's sparse first layer (same evaluations, faster); `train.py` does it for every network it writes |
| `tests/run_all.sh` | the tests: perft, move generation against python-chess, puzzles, network updates, engine against numpy, bench |
| `tests/universal.sh` | the universal build: every kernel set gives the same results, and with `qemu-user` the binary runs on emulated older CPUs |

`tests/run_all.sh` runs after `make` and `make featlib`. It needs a network (`NET=file`, else `nets/best.nnue`,
the first `.nnue` in `nets/`, or the one `make net` downloaded) and, for the check against numpy, data
(`DATA=file`, else the first file in `data/`). Every tool prints its options with `--help`.
