"""Win/draw/loss models: how the trainer turns a search score, and the network's output, into an expected result.

The score model (fit_wdl) is fitted to a run's own data by maximum likelihood over the three outcomes.  For a score
s in internal units from the side to move's point of view:

    x(s) = knee * asinh(s / knee)            linear below the knee, logarithmic beyond it
    win  = sigmoid((x - win_offset) / win_scale)
    loss = sigmoid((-x - loss_offset) / loss_scale)
    draw = 1 - win - loss,   expected result = win + draw / 2

Two things about self-play scores call for this form.  The odds of winning grow with the logarithm of the score
(the search backs up ever larger scores from won positions, so the scores reach the tens of thousands), and a
model linear in the score misses either the centre or the tails.  And the side to move's score is optimistic: it
loses half its games at a smaller |score| than it wins half of them, which the separate win and loss curves
express.

The network's output y goes through a fixed curve instead (anchored_curve): the side to move wins with probability
win(y) = 1 / (1 + (1/p - 1)^(1 - y)) and loses with probability win(-y).  That is Stockfish's form, sigmoid((y -
a) / b) with a = 1 and b = 1 / ln(1/p - 1): a 50% win at y = +1 and a win probability of p at y = 0.  The network
learns, for every position, the output whose expected result on this curve is what the fitted model makes of the
search score, so its units stay put however the scale of the search scores drifts.  The smaller p, the steeper
the curve, and the less a balanced position counts in the loss.
"""
import argparse
from fractions import Fraction

import numpy as np
import torch

SCORE_CLIP = 20000   # the fit's score range (static evaluations stay within +-20000; mate scores are left out before)
ANCHOR_P = 0.01      # the anchored curve's win probability at an output of 0 (it is 1/2 at +1)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class WdlModel:
    """Win and loss probabilities of a score: "wdl" (knee, win offset, win scale, loss offset, loss scale; the
    offsets and scales in units of the compressed score x) or "linear" (offset, scale: win = sigmoid((s - offset) /
    scale), loss = sigmoid((-s - offset) / scale), the form of the anchored curve)."""

    def __init__(self, kind, params):
        assert kind in ("wdl", "linear"), kind
        self.kind, self.params = kind, tuple(float(p) for p in params)
        assert len(self.params) == (5 if kind == "wdl" else 2)

    def _win_loss_np(self, s):
        s = np.asarray(s, dtype=np.float64)
        if self.kind == "wdl":
            knee, wo, ws, lo, ls = self.params
            x = knee * np.arcsinh(s / knee)
            return sigmoid_np((x - wo) / ws), sigmoid_np((-x - lo) / ls)
        o, k = self.params
        return sigmoid_np((s - o) / k), sigmoid_np((-s - o) / k)

    def wdl_np(self, s):
        """(win, draw, loss) probabilities for the side to move."""
        win, loss = self._win_loss_np(s)
        return win, np.clip(1.0 - win - loss, 0.0, 1.0), loss

    def prob_np(self, s):
        """The expected result, win + draw / 2."""
        win, loss = self._win_loss_np(s)
        return (0.5 * (1.0 + win - loss)).astype(np.float32)

    def prob_torch(self, s):
        if self.kind == "wdl":
            knee, wo, ws, lo, ls = self.params
            x = knee * torch.asinh(s / knee)
            win, loss = torch.sigmoid((x - wo) / ws), torch.sigmoid((-x - lo) / ls)
        else:
            o, k = self.params
            win, loss = torch.sigmoid((s - o) / k), torch.sigmoid((-s - o) / k)
        return 0.5 * (1.0 + win - loss)

    def slope_np(self, s):
        """d(expected result)/d(score).  At lambda = 1 a position's weight in the loss is proportional to its square:
        an error of e costs about (slope * e)^2."""
        s = np.asarray(s, dtype=np.float64)
        win, loss = self._win_loss_np(s)
        if self.kind == "wdl":
            knee, _, ws, _, ls = self.params
            dx = 1.0 / np.sqrt(1.0 + (s / knee) ** 2)
        else:
            ws = ls = self.params[1]
            dx = np.ones_like(s)
        return 0.5 * dx * (win * (1.0 - win) / ws + loss * (1.0 - loss) / ls)

    def even_scores(self):
        """The scores at which the side to move wins half its games, and loses half of them."""
        if self.kind == "wdl":
            knee, wo, _, lo, _ = self.params
            return float(knee * np.sinh(min(wo / knee, 30.0))), float(-knee * np.sinh(min(lo / knee, 30.0)))
        return self.params[0], -self.params[0]

    def describe(self):
        def at(v):   # beyond the fitted range when the data has (almost) no wins or no losses
            return f"at {v:+.0f}" if abs(v) <= SCORE_CLIP else f"only beyond {'+' if v > 0 else '-'}{SCORE_CLIP}"
        win50, loss50 = self.even_scores()
        return (f"the side to move wins half its games {at(win50)} and loses half {at(loss50)}; balanced positions "
                f"draw {float(self.wdl_np(0.0)[1]):.0%}")


def anchored_curve(p=ANCHOR_P):
    """The fixed curve of the network's output y: win(y) = 1 / (1 + (1/p - 1)^(1 - y)), loss(y) = win(-y)."""
    if not 0.0 < p < 0.5:
        raise ValueError(f"the anchored curve's win probability at 0 must be between 0 and 1/2 (got {p})")
    return WdlModel("linear", [1.0, 1.0 / float(np.log(1.0 / p - 1.0))])


def anchor_probability(text):
    """argparse type of --anchor-p: a probability between 0 and 1/2, as a decimal or a fraction (1/4)."""
    try:
        v = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"not a number or a fraction: {text!r}")
    if not 0.0 < v < 0.5:
        raise argparse.ArgumentTypeError(f"the win probability at 0 must be between 0 and 1/2 (got {text})")
    return v


def score_counts(scores, results_stm, clip=SCORE_CLIP):
    """(win, draw, loss) counts per integer score in [-clip, clip]: the fit's sufficient statistics, so that a model
    is evaluated on at most 2 * clip + 1 bins rather than on millions of positions."""
    s = np.clip(np.rint(np.asarray(scores, dtype=np.float64)), -clip, clip).astype(np.int64) + clip
    r = np.asarray(results_stm)
    n = 2 * clip + 1
    counts = np.zeros((3, n), dtype=np.float64)       # rows: win, draw, loss for the side to move
    for row, v in ((0, 1), (1, 0), (2, -1)):
        counts[row] = np.bincount(s[r == v], minlength=n)
    return counts


def fit_wdl(scores, results_stm):
    """The maximum-likelihood wdl model of scores (the side to move's, no mate scores) and results (+1 win, 0 draw,
    -1 loss for the side to move).  Returns (model, log-loss per position).  L-BFGS in float64 from several
    starting knees, as the likelihood has more than one basin when the knee is far off."""
    counts = score_counts(scores, results_stm)
    keep = counts.sum(0) > 0
    f64 = torch.float64
    s = torch.arange(-SCORE_CLIP, SCORE_CLIP + 1, dtype=f64)[torch.from_numpy(keep)]
    C = torch.from_numpy(counts[:, keep])
    N = max(float(C.sum()), 1.0)

    def nll(p):   # p = [log knee, win offset, log win scale, loss offset, log loss scale]
        knee = p[0].exp()
        x = knee * torch.asinh(s / knee)
        win = torch.sigmoid((x - p[1]) / p[2].exp())
        loss = torch.sigmoid((-x - p[3]) / p[4].exp())
        draw = (1.0 - win - loss).clamp_min(1e-12)
        return -(C[0] * win.clamp_min(1e-12).log() + C[1] * draw.log() + C[2] * loss.clamp_min(1e-12).log()).sum() / N

    def fit_from(knee0):
        """(log-loss, parameters) of an L-BFGS fit from one starting knee, or None if it fails."""
        p = torch.tensor([np.log(knee0), 2 * knee0, np.log(knee0), 2 * knee0, np.log(knee0)], dtype=f64, requires_grad=True)
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=400, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-13)

        def closure():
            opt.zero_grad()
            v = nll(p)
            v.backward()
            return v
        try:
            opt.step(closure)
        except RuntimeError:
            return None
        with torch.no_grad():
            v = float(nll(p))
        params = p.detach().numpy().copy()
        return (v, params) if np.isfinite(v) and np.isfinite(params).all() else None

    fits = [f for f in map(fit_from, (30.0, 100.0, 300.0, 1000.0, 3000.0)) if f is not None]
    if not fits:
        raise RuntimeError("the wdl fit did not converge")
    v, p = min(fits, key=lambda f: f[0])
    return WdlModel("wdl", [np.exp(p[0]), p[1], np.exp(p[2]), p[3], np.exp(p[4])]), v
