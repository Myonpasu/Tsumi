"""The network's inputs, for the trainer and the verifier.

They come from the engine's own code (src/nnue_features.h and its move generator), built as a shared library
(`make featlib` -> train/libfeat.so) and loaded here with ctypes, so the trainer and the engine never disagree
about an input.  src/nnue_features.h describes the features.
"""
import ctypes
import os

import numpy as np

QA = 255          # an accumulator value of 1.0 (src/nnue.h)
SCALE = 1000      # internal units (millipawns) per 1.0 of network output: an output of 1.0 shows as +1.00

# The king bucket of each king square after mirroring (only files a-d occur; the e-h half mirrors them): one bucket
# per square of the half board, as Stockfish's HalfKAv2_hm.  The engine reads the table from the network file.
KING_BUCKETS = np.array([r * 4 + min(f, 7 - f) for r in range(8) for f in range(8)], dtype=np.int64)
NUM_KING_BUCKETS = int(KING_BUCKETS.max()) + 1
FEATURES_PER_BUCKET = 768
FEATLIB_VERSION = 5                 # src/nnue_features.h: the library must be rebuilt when it changes
THREAT_FEATURES = 10 * 13 * 64      # src/nnue_features.h: attacker x (empty + 12 pieces) x target square
HEAD_EXTRAS = 32                    # the head's legal-move inputs
OUTPUT_BUCKETS = 16
MAX_PIECES = 32
MAX_THREATS = 128                   # threat indices kept per position (self-play positions have about 17 on average and
                                    # rarely more than 60; longer lists are cut short)

RECORD_DTYPE = np.dtype([           # src/datagen.h TrainingRecord
    ("occ", "<u8"), ("pcs", "u1", 16), ("score", "<i2"), ("result", "i1"),
    ("stm", "u1"), ("ply", "u1"), ("flags", "u1"), ("next", "<i2"),
])

_featlib: "ctypes.CDLL | None" = None


def load_featlib() -> ctypes.CDLL:
    """train/libfeat.so, built by `make featlib`; exits with a message if it is missing or out of date."""
    global _featlib
    if _featlib is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfeat.so")
        if not os.path.exists(path):
            raise SystemExit("train/libfeat.so not found: run `make featlib` (the engine's feature code, built for the trainer)")
        lib = ctypes.CDLL(path)
        if not hasattr(lib, "tsumi_featlib_version") or lib.tsumi_featlib_version() != FEATLIB_VERSION:
            raise SystemExit("train/libfeat.so is out of date: run `make featlib`")
        for name, value in (("tsumi_threat_features", THREAT_FEATURES), ("tsumi_head_extras", HEAD_EXTRAS),
                            ("tsumi_output_buckets", OUTPUT_BUCKETS)):
            getattr(lib, name).restype = ctypes.c_int
            if getattr(lib, name)() != value:
                raise SystemExit("train/libfeat.so and train/features.py disagree on the network's inputs: run `make featlib`")
        lib.tsumi_extract.restype = ctypes.c_int
        lib.tsumi_extract.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int] \
            + [ctypes.c_void_p] * 6 + [ctypes.c_int]
        _featlib = lib
    return _featlib


EXTRACT_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))


def batch_features(recs, king_buckets=KING_BUCKETS, max_threats=MAX_THREATS, threads=None):
    """The inputs of a batch of records, from the side to move's perspective (us) and the opponent's (them):
    (psq_us, psq_them, thr_us, thr_them, extras, bucket, truncated).  The index lists are int64 with padding
    indices where unused (NUM_KING_BUCKETS * 768 for the piece-square lists, THREAT_FEATURES for the threat lists),
    extras the head's legal-move inputs as uint8 0..127, bucket the output buckets, and `truncated` the number of
    positions with more than max_threats threats, whose lists were cut short."""
    lib = load_featlib()
    recs = np.ascontiguousarray(recs)
    n = len(recs)
    kb = np.ascontiguousarray(np.asarray(king_buckets, dtype=np.uint8))
    nb = int(kb.max()) + 1
    psq_us = np.empty((n, MAX_PIECES), dtype=np.int64)
    psq_them = np.empty((n, MAX_PIECES), dtype=np.int64)
    thr_us = np.empty((n, max_threats), dtype=np.int64)
    thr_them = np.empty((n, max_threats), dtype=np.int64)
    bucket = np.empty(n, dtype=np.int64)
    extras = np.empty((n, HEAD_EXTRAS), dtype=np.uint8)
    rc = lib.tsumi_extract(recs.ctypes.data, n, kb.ctypes.data, nb * FEATURES_PER_BUCKET, THREAT_FEATURES, max_threats,
                           psq_us.ctypes.data, psq_them.ctypes.data, thr_us.ctypes.data, thr_them.ctypes.data,
                           bucket.ctypes.data, extras.ctypes.data, EXTRACT_THREADS if threads is None else threads)
    if rc < 0:
        raise ValueError("malformed training record")
    return psq_us, psq_them, thr_us, thr_them, extras, bucket, rc


def record_fen(rec):
    """The FEN of a record's position (no castling rights, no en-passant square, as the record holds neither)."""
    pieces = "PNBRQKpnbrqk"
    occ, board, i = int(rec["occ"]), {}, 0
    for sq in range(64):
        if occ >> sq & 1:
            byte = int(rec["pcs"][i >> 1])
            board[sq] = pieces[(byte >> 4) if i & 1 else (byte & 15)]
            i += 1
    rows = []
    for r in range(7, -1, -1):
        row, empty = "", 0
        for f in range(8):
            p = board.get(r * 8 + f)
            if p is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + p
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows) + (" b" if rec["stm"] else " w") + " - - 0 1"
