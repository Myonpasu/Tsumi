"""The network (src/nnue.h) in PyTorch, its quantised export in the engine's file format, and an integer
implementation of the engine's evaluation, which verify_net.py checks the engine against.

Training-only additions (feature factorisation, as in Stockfish's trainer): two dense tables of virtual features
are added to the real ones while training, one per oriented piece-square without the king bucket and one per
(attacker, content) of a threat without the target square.  Every real feature shares the weight of its virtual
feature, so that rare features start from sensible values instead of waiting to be seen.  The export folds the
virtual weights into the real ones, so neither the network file nor the engine knows of them.
"""
import struct

import numpy as np
import torch
from torch import nn

from features import FEATURES_PER_BUCKET, HEAD_EXTRAS, KING_BUCKETS, OUTPUT_BUCKETS, QA, SCALE, THREAT_FEATURES

# Quantisation (src/nnue.h).  The accumulator rows are int8 at scale QA, so their weights stay within +-127/QA; the
# head's first layer has int8 weights at scale 128, its second int8 at scale 64, and its output layer int16 at 4096.
ROW_CLAMP = 127.0 / QA
HEAD_ACT = 127
HEAD_W0SCALE, HEAD_WSCALE, HEAD_OUT_WSCALE = 128, 64, 4096
HEAD_UNIT0, HEAD_UNIT, HEAD_OUT_UNIT = HEAD_ACT * HEAD_W0SCALE, HEAD_ACT * HEAD_WSCALE, HEAD_ACT * HEAD_OUT_WSCALE
W0_CLAMP, W1_CLAMP, W2_CLAMP = 127.0 / HEAD_W0SCALE, 127.0 / HEAD_WSCALE, 32767.0 / HEAD_OUT_WSCALE
L2 = L3 = 32                      # widths of the head's hidden layers

MAGIC = b"TSUMINET"
FORMAT_VERSION = 4
# The header after the magic (src/nnue.cpp): format version, H, NB, output buckets, QA, 64 (unused), SCALE, L2, L3,
# then the fixed values 1, 32 (HEAD_EXTRAS), 1, 1.
HEADER = struct.Struct("<13I")


class NNUE(nn.Module):
    king_bucket_table: torch.Tensor   # buffers registered in __init__ (annotated for static checkers)
    output_scale: torch.Tensor
    extras_mask_table: torch.Tensor
    bucket_scheme_table: torch.Tensor

    def __init__(self, hidden=1024):
        super().__init__()
        if hidden % 64 or not 0 < hidden <= 4096:
            raise ValueError("the hidden size must be a multiple of 64, at most 4096")
        self.hidden = hidden
        self.num_king_buckets = int(KING_BUCKETS.max()) + 1
        self.num_features = self.num_king_buckets * FEATURES_PER_BUCKET
        # Saved with the checkpoint, so that a run resuming it can check that it is one of this network: the king-bucket
        # table, the output's unit, and the codes (both 1) of the legal-move inputs and of the output buckets.
        self.register_buffer("king_bucket_table", torch.as_tensor(KING_BUCKETS))
        self.register_buffer("output_scale", torch.tensor(SCALE, dtype=torch.int64))
        self.register_buffer("extras_mask_table", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("bucket_scheme_table", torch.tensor(1, dtype=torch.int64))
        # The last row of each table is its padding index: always zero, never trained.
        self.ft = nn.EmbeddingBag(self.num_features + 1, hidden, mode="sum", padding_idx=self.num_features)
        self.ft_virtual = nn.EmbeddingBag(FEATURES_PER_BUCKET + 1, hidden, mode="sum", padding_idx=FEATURES_PER_BUCKET)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.tft = nn.EmbeddingBag(THREAT_FEATURES + 1, hidden, mode="sum", padding_idx=THREAT_FEATURES)
        self.tft_virtual = nn.EmbeddingBag(THREAT_FEATURES // 64 + 1, hidden, mode="sum", padding_idx=THREAT_FEATURES // 64)
        n0, OB = hidden + HEAD_EXTRAS, OUTPUT_BUCKETS
        self.fc0_w = nn.Parameter(torch.zeros(OB, L2, n0))
        self.fc0_b = nn.Parameter(torch.zeros(OB, L2))
        self.fc1_w = nn.Parameter(torch.zeros(OB, L3, 2 * L2))
        self.fc1_b = nn.Parameter(torch.zeros(OB, L3))
        self.fc2_w = nn.Parameter(torch.zeros(OB, 2 * L2 + 2 * L3))
        self.fc2_b = nn.Parameter(torch.zeros(OB))
        # The virtual tables start as no-ops, and the units' bias at zero: about half the units are then clipped off,
        # so the products are sparse from the start, which the engine's sparse first layer relies on.
        nn.init.normal_(self.ft.weight, std=0.05)
        nn.init.normal_(self.tft.weight, std=0.05)
        nn.init.normal_(self.fc0_w, std=n0 ** -0.5)
        nn.init.normal_(self.fc1_w, std=(2 * L2) ** -0.5)
        with torch.no_grad():
            for table in (self.ft, self.ft_virtual, self.tft, self.tft_virtual):
                table.weight[-1].zero_()
            self.ft_virtual.weight.zero_()
            self.tft_virtual.weight.zero_()

    def accumulate(self, psq, thr):
        """A perspective's units before clipping: the bias and the rows of its active features, real and virtual."""
        vpsq = torch.where(psq == self.num_features, torch.full_like(psq, FEATURES_PER_BUCKET), psq % FEATURES_PER_BUCKET)
        vthr = torch.where(thr == THREAT_FEATURES, torch.full_like(thr, THREAT_FEATURES // 64), thr // 64)
        psq_sum = self.ft(psq) + self.ft_bias + self.ft_virtual(vpsq)
        thr_sum = self.tft(thr) + self.tft_virtual(vthr)
        return psq_sum + thr_sum

    def forward(self, psq_us, psq_them, thr_us, thr_them, extras, bucket):
        """The output y of each position (an expected result through the anchored curve, see wdl.py; the engine
        reports y * SCALE)."""
        half = self.hidden // 2
        a = torch.clamp(self.accumulate(psq_us, thr_us), 0.0, 1.0)
        b = torch.clamp(self.accumulate(psq_them, thr_them), 0.0, 1.0)
        u = torch.cat([a[:, :half] * a[:, half:], b[:, :half] * b[:, half:], extras], dim=1)   # the engine's order
        # Every bucket's layer is computed and each position's own picked: cheaper than gathering a weight matrix
        # per position.
        rows = torch.arange(u.shape[0], device=u.device)
        z0 = torch.einsum("bn,oln->bol", u, self.fc0_w)[rows, bucket] + self.fc0_b[bucket]
        a0 = torch.clamp(z0, 0.0, 1.0)
        h0 = torch.cat([a0, a0 * a0], dim=1)
        z1 = torch.einsum("bn,oln->bol", h0, self.fc1_w)[rows, bucket] + self.fc1_b[bucket]
        a1 = torch.clamp(z1, 0.0, 1.0)
        h1 = torch.cat([a1, a1 * a1], dim=1)
        z2 = (torch.cat([h0, h1], dim=1) * self.fc2_w[bucket]).sum(dim=1) + self.fc2_b[bucket]
        return z2 + z0[:, -2] - z0[:, -1]

    def clamp_weights(self):
        """Keep every weight within what the quantised network can hold (after each optimiser step, as Stockfish's
        trainer does).  For a factorised table the bound is on the sum real + virtual: the virtual weights are clamped
        to +-C and the real ones to [-C - virtual, C - virtual]."""
        with torch.no_grad():
            C = ROW_CLAMP
            v = self.ft_virtual.weight[:FEATURES_PER_BUCKET]
            v.clamp_(-C, C)
            real = self.ft.weight[:self.num_features].view(self.num_king_buckets, FEATURES_PER_BUCKET, self.hidden)
            real.clamp_(min=(-C - v).unsqueeze(0), max=(C - v).unsqueeze(0))
            v = self.tft_virtual.weight[:THREAT_FEATURES // 64]
            v.clamp_(-C, C)
            real = self.tft.weight[:THREAT_FEATURES].view(THREAT_FEATURES // 64, 64, self.hidden)
            real.clamp_(min=(-C - v).unsqueeze(1), max=(C - v).unsqueeze(1))
            for table in (self.ft, self.ft_virtual, self.tft, self.tft_virtual):
                table.weight[-1].zero_()
            self.fc0_w.clamp_(-W0_CLAMP, W0_CLAMP)
            self.fc1_w.clamp_(-W1_CLAMP, W1_CLAMP)
            self.fc2_w.clamp_(-W2_CLAMP, W2_CLAMP)

    def quantised(self):
        """The network as the engine holds it: a dict of integer arrays, as read_net returns and write_net writes."""
        self.clamp_weights()

        def q(t, scale, dtype):
            a = np.round(t.detach().cpu().numpy() * scale)
            info = np.iinfo(dtype)
            return np.clip(a, info.min, info.max).astype(dtype)

        def folded(real, virtual, repeat):
            w = real.weight.detach()[:-1].clone()
            v = virtual.weight.detach()[:-1]
            if v.abs().max() > 0:
                w += repeat(v)
            return w.clamp_(-ROW_CLAMP, ROW_CLAMP)

        return {
            "H": self.hidden, "NB": self.num_king_buckets, "L2": L2, "L3": L3, "SCALE": int(self.output_scale),
            "king_buckets": KING_BUCKETS.astype(np.uint8),
            "psq_w": q(folded(self.ft, self.ft_virtual, lambda v: v.repeat(self.num_king_buckets, 1)), QA, np.int8),
            "bias": q(self.ft_bias, QA, np.int16),
            "threat_w": q(folded(self.tft, self.tft_virtual, lambda v: v.repeat_interleave(64, dim=0)), QA, np.int8),
            "fc0_w": q(self.fc0_w, HEAD_W0SCALE, np.int8), "fc0_b": q(self.fc0_b, HEAD_UNIT0, np.int32),
            "fc1_w": q(self.fc1_w, HEAD_WSCALE, np.int8), "fc1_b": q(self.fc1_b, HEAD_UNIT, np.int32),
            "fc2_w": q(self.fc2_w, HEAD_OUT_WSCALE, np.int16), "fc2_b": q(self.fc2_b, HEAD_OUT_UNIT, np.int32),
        }

    def export(self, path):
        write_net(self.quantised(), path)


def check_checkpoint(state, hidden):
    """Why a checkpoint cannot be resumed by a model of `hidden` units, or None if it can."""
    if "tft_virtual.weight" not in state or "fc0_w" not in state or "extras_mask_table" not in state \
            or "bucket_scheme_table" not in state:
        return "it is not a checkpoint of this network"
    if int(state["output_scale"]) != SCALE or int(state["extras_mask_table"]) != 1 or int(state["bucket_scheme_table"]) != 1:
        return "it was trained with other settings (output scale, legal-move inputs or output buckets)"
    if not torch.equal(state["king_bucket_table"].cpu(), torch.as_tensor(KING_BUCKETS)):
        return "it has another king-bucket table"
    h, (ob, l2, n0), l3 = state["ft.weight"].shape[1], state["fc0_w"].shape, state["fc1_w"].shape[1]
    if (ob, l2, l3, n0 - h) != (OUTPUT_BUCKETS, L2, L3, HEAD_EXTRAS):
        return "its head has another shape"
    if h != hidden:
        return f"it has {h} hidden units, this run {hidden} (--hidden {h} to continue it)"
    return None


def write_net(net, path):
    """Write a network (quantised()'s or read_net's dict) in the engine's format, src/nnue.cpp."""
    H, NB = int(net["H"]), int(net["NB"])
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(HEADER.pack(FORMAT_VERSION, H, NB, OUTPUT_BUCKETS, QA, 64, int(net["SCALE"]), int(net["L2"]), int(net["L3"]),
                            1, HEAD_EXTRAS, 1, 1))
        layout = (("king_buckets", np.uint8), ("psq_w", np.int8), ("bias", "<i2"), ("threat_w", np.int8), ("fc0_w", np.int8),
                  ("fc0_b", "<i4"), ("fc1_w", np.int8), ("fc1_b", "<i4"), ("fc2_w", "<i2"), ("fc2_b", "<i4"))
        f.writelines(np.ascontiguousarray(net[key]).astype(dtype).tobytes() for key, dtype in layout)


def read_net(path):
    """A network file as a dict of numpy arrays (see write_net)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC or len(data) < 8 + HEADER.size:
        raise ValueError(f"{path} is not a Tsumi network")
    hdr = HEADER.unpack_from(data, 8)
    if hdr[0] != FORMAT_VERSION or hdr[3] != OUTPUT_BUCKETS or hdr[4] != QA or hdr[9:] != (1, HEAD_EXTRAS, 1, 1):
        raise ValueError(f"{path} is not a network of the format this version reads (format version {hdr[0]})")
    H, NB, SCALE_, L2_, L3_ = hdr[1], hdr[2], hdr[6], hdr[7], hdr[8]
    OB, n_in, n0 = OUTPUT_BUCKETS, NB * FEATURES_PER_BUCKET, H + HEAD_EXTRAS
    off = 8 + HEADER.size

    def take(dtype, count, shape=None):
        nonlocal off
        a = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += count * a.itemsize
        return a.reshape(shape) if shape else a

    net: dict = {"H": H, "NB": NB, "L2": L2_, "L3": L3_, "SCALE": SCALE_}
    net["king_buckets"] = take(np.uint8, 64)
    net["psq_w"] = take(np.int8, n_in * H, (n_in, H))
    net["bias"] = take("<i2", H)
    net["threat_w"] = take(np.int8, THREAT_FEATURES * H, (THREAT_FEATURES, H))
    net["fc0_w"] = take(np.int8, OB * L2_ * n0, (OB, L2_, n0))
    net["fc0_b"] = take("<i4", OB * L2_, (OB, L2_))
    net["fc1_w"] = take(np.int8, OB * L3_ * 2 * L2_, (OB, L3_, 2 * L2_))
    net["fc1_b"] = take("<i4", OB * L3_, (OB, L3_))
    net["fc2_w"] = take("<i2", OB * (2 * L2_ + 2 * L3_), (OB, 2 * L2_ + 2 * L3_))
    net["fc2_b"] = take("<i4", OB)
    if off != len(data):
        raise ValueError(f"{path} has trailing data")
    return net


def trunc_div(a, b):
    """C++ integer division, which truncates towards zero."""
    q = abs(a) // b
    return q if a >= 0 else -q


def evaluate_quantised(net, psq_us, psq_them, thr_us, thr_them, extras, bucket):
    """The engine's evaluation (src/nnue.cpp) in integer numpy: feature index lists of each perspective, the
    legal-move inputs (0..127) and the output bucket."""
    H, L2_ = net["H"], net["L2"]
    half = H // 2

    def products(psq, thr):
        # The engine keeps the piece-square and the threat sums as two int16 accumulators and adds them (with
        # saturation, which clips the same) before clipping each unit to 0..QA.
        p = (net["bias"].astype(np.int64) + net["psq_w"][psq].astype(np.int64).sum(axis=0)).astype(np.int16).astype(np.int64)
        t = net["threat_w"][thr].astype(np.int64).sum(axis=0).astype(np.int16).astype(np.int64)
        v = np.clip(p + t, 0, QA)
        return (v[:half] * v[half:] + 256) >> 9

    u = np.concatenate([products(psq_us, thr_us), products(psq_them, thr_them), np.asarray(extras, dtype=np.int64)])
    z0 = net["fc0_b"][bucket].astype(np.int64) + net["fc0_w"][bucket].astype(np.int64) @ u   # 127 * 128 per 1.0
    a0 = np.clip(z0 >> 7, 0, HEAD_ACT)
    h0 = np.concatenate([a0, (a0 * a0) // HEAD_ACT])
    z1 = net["fc1_b"][bucket].astype(np.int64) + net["fc1_w"][bucket].astype(np.int64) @ h0  # 127 * 64 per 1.0
    a1 = np.clip(z1 >> 6, 0, HEAD_ACT)
    h1 = np.concatenate([a1, (a1 * a1) // HEAD_ACT])
    w2 = net["fc2_w"][bucket].astype(np.int64)
    z2 = int(net["fc2_b"][bucket]) + int((h0 * w2[:2 * L2_]).sum() + (h1 * w2[2 * L2_:]).sum())
    skip = (HEAD_OUT_UNIT // HEAD_UNIT0) * int(z0[L2_ - 2] - z0[L2_ - 1])
    return trunc_div((z2 + skip) * net["SCALE"], HEAD_OUT_UNIT)
