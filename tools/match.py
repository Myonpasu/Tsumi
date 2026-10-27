#!/usr/bin/env python3
"""Engine-vs-engine matches under selfmate rules (the checkmated side wins), with an Elo estimate.

Standard GUIs and match runners score checkmate for the mating side, so this runner is needed to test Tsumi
builds or networks against each other.  Both engines talk UCI; python-chess keeps the rules.  Openings are random
legal moves from the start position, positions drawn at random from --book (FEN/EPD lines or PGN games), or a mix
(--book-share); each is played with both colours, as fishtest plays its books.  The result is reported in the form
engine tests are usually quoted in: the Elo difference with its 95% confidence interval, the configuration, the
game counts, and the pentanomial counts of the pairs of games.  With --sprt the match is a sequential probability
ratio test, which stops as soon as it is decided.

Examples:
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 100 --tc 1+0.05 --concurrency 8
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi --opt1 EvalFile=nets/gen2.nnue --opt2 EvalFile=nets/gen1.nnue --nodes 50000
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 50 --depth 8 --pgn out.pgn
  python3 tools/match.py --e1 ./tsumi --e2 random --games 20 --nodes 5000      # sanity check vs random mover
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 1000 --nodes 50000 --book UHO_Lichess_4852_v1.epd.zip --book-share 50
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --sprt 0 5 --tc 10+0.1 --concurrency 8   # until it is decided
"""
import argparse
import contextlib
import math
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_book import draw_positions


class EngineDied(RuntimeError):
    """The engine's output ended (it exited, or was killed); lines is what it wrote since the last command."""
    def __init__(self, lines):
        super().__init__("engine died" + (f" after: {lines[-1]}" if lines else ""))
        self.lines = lines


@contextlib.contextmanager
def deadline(seconds, kill):
    """Kills the engine if the block takes longer, so that waiting on an engine that hangs ends in EngineDied."""
    timer = threading.Timer(seconds, kill)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


class UciEngine:
    def __init__(self, path, options):
        # A path, or a command line ("python3 some_engine.py").  Outside the terminal's process group: a Ctrl-C
        # reaches this script alone, which ends the engines itself.
        cmd = shlex.split(path, posix=os.name != "nt")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        assert self.p.stdin is not None and self.p.stdout is not None
        self.stdin, self.stdout = self.p.stdin, self.p.stdout
        try:
            with deadline(60, self.kill):   # a program that never answers (not a UCI engine, say) is killed
                self.send("uci")
                listed = self.listed_options(self.wait_for("uciok"))
                self.settings = {k: default for k, (default, _) in listed.items() if default is not None}
                # An engine ignores an option it does not have, and a value its combo option does not list, so a
                # typo would otherwise play the whole match with the engine's defaults.
                unlisted = []   # combo options given a value the engine did not list
                for k, v in options.items():
                    if k.lower() not in listed:
                        raise ValueError(f"the engine has no option {k}")
                    values = listed[k.lower()][1]
                    if values is not None and v.lower() not in values:
                        unlisted.append((k, v))
                    self.settings[k.lower()] = v
                    self.send(f"setoption name {k} value {v}")
                if unlisted:   # an option set since may have added the value (Fairy-Stockfish's VariantPath adds
                    self.send("uci")   # variants), so list the options again and set the value again
                    listed = self.listed_options(self.wait_for("uciok"))
                    for k, v in unlisted:
                        if v.lower() not in (listed.get(k.lower(), (None, None))[1] or []):
                            raise ValueError(f"{v!r} is not one of the values of {k}")
                        self.send(f"setoption name {k} value {v}")
                self.send("isready")
                self.wait_for("readyok")
        except BaseException:
            self.quit()   # the engine is not left running
            raise

    @staticmethod
    def listed_options(lines):
        """The options in an engine's answer to "uci": lower-case name -> (default, the values of a combo option in
        lower case, else None)."""
        d = {}
        for line in lines:
            if line.startswith("option name "):
                name, _, rest = line.removeprefix("option name ").partition(" type ")
                kind, _, rest = rest.partition(" ")
                parts = rest.removeprefix("default ").split(" var ")   # the default, then a combo's values
                default = parts[0].split(" min ")[0] if rest.startswith("default ") else None
                d[name.lower()] = default, [v.lower() for v in parts[1:]] if kind == "combo" else None
        return d

    def send(self, s):
        self.stdin.write(s + "\n")
        self.stdin.flush()

    def wait_for(self, token):
        lines = []
        while True:
            line = self.stdout.readline()
            if not line:
                raise EngineDied(lines)
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    def new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")

    def bestmove(self, board, go_args):
        moves = " ".join(m.uci() for m in board.move_stack)
        start = board.root().fen()
        self.send(f"position fen {start}" + (f" moves {moves}" if moves else ""))
        self.send("go " + go_args)
        lines = self.wait_for("bestmove")
        score = None
        for line in lines:
            toks = line.split()
            if toks[:1] == ["info"] and "score" in toks[:-2]:
                i = toks.index("score")
                try:
                    score = (toks[i + 1], int(toks[i + 2]))
                except ValueError:
                    pass
        toks = lines[-1].split()
        return (toks[1] if len(toks) > 1 else "0000"), score

    def quit(self):
        """Ends the engine: by "quit", or killed if it does not answer or is gone already."""
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()
        for f in (self.stdin, self.stdout):
            try:
                f.close()
            except OSError:   # nothing reads the pipe any more
                pass

    def kill(self):
        self.p.kill()


class RandomEngine:
    """Pseudo-engine that plays uniformly random legal moves (sanity baseline: --e2 random)."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.settings = {}

    def new_game(self):
        pass

    def bestmove(self, board, go_args):
        return self.rng.choice(list(board.legal_moves)).uci(), None

    def quit(self):
        pass

    def kill(self):
        pass


def make_engine(path, options, seed=0):
    return RandomEngine(seed) if path == "random" else UciEngine(path, options)


def fmt_time(secs):
    secs = int(secs + 0.5)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def result_of(board, max_plies):
    """Return 1-0 / 0-1 / 1/2-1/2 under selfmate rules, or None if the game continues.  The draws are the engine's:
    stalemate, a dead position, the fifty-move rule (100 half-moves, unless the last of them mates) and the third
    occurrence of a position."""
    if board.is_checkmate():
        return "1-0" if board.turn == chess.WHITE else "0-1"   # the side TO MOVE is mated and therefore wins
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        return "1/2-1/2"
    if len(board.move_stack) >= max_plies:
        return "1/2-1/2"
    return None


def parse_tc(tc):
    """(base, increment) in seconds from "base+inc" or "base"."""
    base, _, inc = tc.partition("+")
    return float(base), float(inc or 0)


def parse_options(lst):
    d = {}
    for item in lst or []:
        k, eq, v = item.partition("=")
        if not eq:
            raise ValueError(f"an option is NAME=VALUE (got {item!r})")
        d[k] = v
    return d


def random_opening(rng, plies):
    while True:
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if result_of(b, 10**9) is None:
            return b


def play_game(args, e1, e2, opening, swap):
    """Play one game; returns (result string, pgn game).  swap=True means e2 plays White."""
    board = opening.copy()
    white, black = (e2, e1) if swap else (e1, e2)
    white.new_game()
    black.new_game()
    clocks = None
    inc_ms = 0.0
    if args.tc:
        base, inc = parse_tc(args.tc)
        clocks = [base * 1000, base * 1000]
        inc_ms = inc * 1000
    game = chess.pgn.Game()
    game.setup(opening)
    node = game
    res = result_of(board, args.max_plies)
    while res is None:
        eng = white if board.turn == chess.WHITE else black
        if clocks:   # at least 1 ms: a clock of 0 would read as no clock at all
            go = f"wtime {max(1, int(clocks[0]))} btime {max(1, int(clocks[1]))} winc {int(inc_ms)} binc {int(inc_ms)}"
        elif args.nodes:
            go = f"nodes {args.nodes}"
        elif args.depth:
            go = f"depth {args.depth}"
        else:
            go = f"movetime {args.movetime}"
        t0 = time.time()
        mv, score = eng.bestmove(board, go)
        dt = (time.time() - t0) * 1000
        if clocks:
            idx = 0 if board.turn == chess.WHITE else 1
            clocks[idx] -= dt
            if clocks[idx] < 0:
                # time forfeit: the side that ran out of time loses
                res = "0-1" if board.turn == chess.WHITE else "1-0"
                node.comment = "time forfeit"
                break
            clocks[idx] += inc_ms
        try:
            move = chess.Move.from_uci(mv)
        except ValueError:
            move = None
        if move is None or move not in board.legal_moves:
            print(f"illegal move {mv} from {'white' if board.turn else 'black'} in {board.fen()}", file=sys.stderr)
            res = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        node = node.add_variation(move)
        if score:
            node.comment = f"{score[0]} {score[1]}"
        board.push(move)
        res = result_of(board, args.max_plies)
    game.headers["Result"] = res
    game.headers["White"] = args.name2 if swap else args.name1
    game.headers["Black"] = args.name1 if swap else args.name2
    game.headers["Event"] = "selfmate chess: the checkmated side wins"
    return res, game


def fixed(x, decimals=0):
    """x rounded to the given decimals; a zero is written 0, never -0."""
    return f"{round(x, decimals) + 0.0:.{decimals}f}"   # + 0.0 turns -0.0 into 0.0


def signed(x, decimals=0):
    """x with an explicit sign, rounded to the given decimals; a zero is written +0, never -0."""
    return f"{round(x, decimals) + 0.0:+.{decimals}f}"


def fmt_err(err, decimals=0):
    return "?" if err is None else f"{err:.{decimals}f}"


# The statistics are the ones engine tests are usually reported with, so that results compare with other engines'.
# The unit is the pair of games of an opening, played with both colours: with a strong first-move effect the two
# games of a pair tend to cancel, and treating the games as independent overstates the error.  A pair's outcome is
# one of LL, LD, DD or WL, WD, WW, worth 0, 1/4, 1/2, 3/4, 1 of the pair's points (the pentanomial model).

def logistic_elo(score):
    """The Elo difference that gives an expected score, capped at the Elo of 0.1% and 99.9%."""
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def t_975(df):
    """The 97.5th percentile of Student's t with df degrees of freedom (the factor of a 95% interval; 1.96 in the
    limit), by its expansion in the normal percentile (Abramowitz and Stegun 26.7.5), within 0.001 from df = 4."""
    z = 1.959964
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
            + (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / (92160 * df**4))


def elo_interval(pent):
    """(Elo, half-width of its 95% confidence interval) from the pentanomial counts [LL, LD, DD+WL, WD, WW].  The
    interval is the t interval of the mean pair score mapped to Elo, and the half-width its longer side; it is
    None when there is no spread to estimate one from (every pair went the same way)."""
    n = sum(pent)
    if n == 0:
        return 0.0, None
    mean = sum(i / 4 * c for i, c in enumerate(pent)) / n
    var = sum((i / 4 - mean) ** 2 * c for i, c in enumerate(pent)) / n
    e = logistic_elo(mean)
    if var == 0:
        return e, None
    half = t_975(n - 1) * math.sqrt(var / n)
    return e, max(logistic_elo(mean + half) - e, e - logistic_elo(mean - half))


SPRT_BOUND = math.log(0.95 / 0.05)   # the LLR that decides, for 5% chances of either wrong decision


def llr(pent, elo0, elo1):
    """The log-likelihood ratio of H1 (an Elo difference of elo1) to H0 (elo0) for the pentanomial counts, a
    generalized SPRT in normalized Elo (M. Van den Bergh, "Comments on normalized Elo").  A hypothesis is a mean
    pair score of 1/2 + t sigma, with sigma the spread of the pair scores and t = Elo x ln 10 / 800 x sqrt 2 (the
    sqrt 2 for pairs).  Each is represented by the distribution over the five outcomes nearest the observed one,
    in likelihood, that has its t (the paper's maximum likelihood estimate), and the ratio is N times the
    observed mean of the log ratio of the two."""
    counts = [max(1e-3, c) for c in pent]   # no empty outcome, since the estimates divide by them
    n = sum(counts)
    probs = [c / n for c in counts]
    t0, t1 = (e * math.log(10) / 800 * math.sqrt(2) for e in (elo0, elo1))
    p0, p1 = (mle_t(probs, t) for t in (t0, t1))
    return n * sum(p * math.log(q1 / q0) for p, q0, q1 in zip(probs, p0, p1))


def mle_t(probs, t):
    """The distribution over the pair scores 0, 1/4, ..., 1 nearest probs (in likelihood) whose mean is 1/2 + t
    times its spread: a fixed-point iteration from the uniform distribution, each round solving its secular
    equation by bisection."""
    scores = [i / 4 for i in range(5)]
    p = [1 / 5] * 5
    for _ in range(10):
        mu = sum(a * q for a, q in zip(scores, p))
        sigma = math.sqrt(sum(q * (a - mu) ** 2 for a, q in zip(scores, p)))
        phi = [a - 0.5 - t * sigma * (1 + ((mu - a) / sigma) ** 2) / 2 for a in scores]
        if min(phi) >= 0 or max(phi) <= 0:
            raise ValueError("the SPRT's hypotheses are too far from an even score for the estimate")
        lo, hi = -1 / max(phi), -1 / min(phi)   # the secular equation falls from +inf to -inf between these
        for _ in range(60):
            x = (lo + hi) / 2
            if sum(q * f / (1 + x * f) for q, f in zip(probs, phi)) > 0:
                lo = x
            else:
                hi = x
        p, last = [q / (1 + lo * f) for q, f in zip(probs, phi)], p
        if max(abs(a - b) for a, b in zip(p, last)) < 1e-9:
            break
    return p


def limit_name(args):
    """The limit per move in the usual notation: 10.0+0.10s, N=50000 (nodes), D=8 (depth), or MT=100 (ms)."""
    if args.tc:
        base, inc = parse_tc(args.tc)
        return f"{base:.1f}+{inc:.2f}s"
    if args.nodes:
        return f"N={args.nodes}"
    if args.depth:
        return f"D={args.depth}"
    return f"MT={args.movetime}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e2", required=True)
    ap.add_argument("--name1", default="engine1")
    ap.add_argument("--name2", default="engine2")
    ap.add_argument("--opt1", action="append", help="a UCI option for engine 1, e.g. EvalFile=nets/x.nnue (one per --opt1, "
                                                   "repeated for more)")
    ap.add_argument("--opt2", action="append", help="the same for engine 2")
    ap.add_argument("--games", type=int, help="number of games (rounded up to an even number; default 20), or with --sprt "
                                              "the most to play (default 40000)")
    ap.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"),
                    help="play until a sequential probability ratio test decides between an Elo difference of ELO0 and "
                         "one of ELO1 (normalized Elo), e.g. --sprt 0 5")
    ap.add_argument("--tc", help="time control base+inc in seconds, e.g. 10+0.1")
    ap.add_argument("--nodes", type=int, default=0)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--movetime", type=int, default=100, help="ms per move when no other limit is given")
    ap.add_argument("--opening-plies", type=int, default=8, help="random plies from the start position per opening (default 8)")
    ap.add_argument("--book", help="a book to draw openings from: FEN or EPD lines, or PGN games played to their end, plain or "
                                   "in a .zip (such as the Stockfish books); each position at most once while the book lasts")
    ap.add_argument("--book-share", type=int, default=100,
                    help="percentage of the openings drawn from --book; the rest are --opening-plies random plies (default 100)")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pgn", help="write all games to this PGN file")
    args = ap.parse_args()

    if args.tc:
        try:
            parse_tc(args.tc)
        except ValueError:
            ap.error(f"--tc takes base+increment in seconds, e.g. 10+0.1 (got {args.tc!r})")
    if not 0 <= args.book_share <= 100:
        ap.error("--book-share is a percentage, 0-100")
    if args.sprt and max(abs(x) for x in args.sprt) > 200:   # the estimate breaks down beyond
        ap.error("--sprt takes Elo differences within +-200: an SPRT is for small differences")
    seed = args.seed or time.time_ns()
    rng = random.Random(seed)
    try:
        o1, o2 = parse_options(args.opt1), parse_options(args.opt2)
    except ValueError as e:
        ap.error(str(e))
    # An engine that cannot start, take its options, or search stops the match here, with what it wrote, rather than
    # failing every pair.  Fairy-Stockfish, for one, explains a network it cannot load and then hangs, hence the
    # deadline.
    settings = []   # each engine's options in effect, for the configuration line
    for path, opts in ((args.e1, o1), (args.e2, o2)):
        engine = None
        try:
            engine = make_engine(path, opts)
            with deadline(60, engine.kill):
                engine.bestmove(chess.Board(), "depth 1")
            settings.append(engine.settings)
        except EngineDied as e:
            sys.exit("\n  ".join([f"{path} failed{', after writing:' if e.lines else ''}"] + e.lines[-10:]))
        except (OSError, RuntimeError, ValueError) as e:
            sys.exit(f"{path}: {e}")
        finally:
            if engine is not None:
                engine.quit()
    games = args.games or (40000 if args.sprt else 20)
    pairs = (games + 1) // 2
    # A book position is drawn once while the book lasts: a deterministic engine (a fixed node or depth limit) plays
    # the same pair of games from the same opening, which would count twice for one game's worth of information.
    from_book = [args.book is not None and rng.randrange(100) < args.book_share for _ in range(pairs)]
    n_book = sum(from_book)
    book_openings, book_size = [], ""
    if n_book and args.book:
        if not os.path.isfile(args.book):
            sys.exit(f"{args.book} not found")
        book, book_size, bad = draw_positions(args.book, n_book, rng)
        if not book:
            sys.exit(f"{args.book} holds no position to start from (a FEN or an EPD line per position, or PGN games)")
        if bad:
            print(f"note: {bad} of the entries drawn from {args.book} held no position to start from and were left out", file=sys.stderr)
        while len(book_openings) < n_book:
            book_openings += rng.sample(book, min(len(book), n_book - len(book_openings)))
        if n_book > len(book):
            print(f"warning: {'up to ' if args.sprt else ''}{n_book} openings wanted but {args.book} has {len(book)} "
                  f"position{'s' if len(book) != 1 else ''}: they repeat, and the error bar is too small if the engines "
                  "are deterministic", file=sys.stderr)
    it = iter(book_openings)
    openings = [next(it) if b else None for b in from_book]   # a random opening is made when its pair is played

    def opening(i):
        if openings[i] is None:   # from its own generator, so the same for a --seed whatever order the pairs run in
            openings[i] = random_opening(random.Random(f"{seed}:{i}"), args.opening_plies)
        return openings[i]

    random_desc = f"{args.opening_plies} random plies from the start position"
    if n_book == 0:
        openings_desc = random_desc
    elif n_book == pairs:
        openings_desc = f"drawn from {args.book} ({book_size})"
    else:
        openings_desc = f"{n_book} drawn from {args.book} ({book_size}), {pairs - n_book} reached by {random_desc}"

    stats = {"w": 0, "l": 0, "d": 0}
    pent = [0] * 5   # pairs of games by engine 1's outcome: LL, LD, DD or WL, WD, WW
    lock = threading.Lock()
    games_out = []
    total = pairs * 2
    limit = (f"{args.tc} time control" if args.tc else f"{args.nodes:,} nodes per move" if args.nodes
             else f"depth {args.depth}" if args.depth else f"{args.movetime} ms per move")
    print(f"match: {args.name1} vs {args.name2}")
    print(f"  {'games':<16}{'up to ' if args.sprt else ''}{total} ({pairs} opening{'s' if pairs != 1 else ''}, each played "
          f"with both colours), {limit}, {args.concurrency} at a time")
    if args.sprt:
        print(f"  {'sprt':<16}H0 {args.sprt[0]:+.2f} Elo, H1 {args.sprt[1]:+.2f} Elo (normalized): the match stops when the "
              f"LLR leaves ({-SPRT_BOUND:.2f}, {SPRT_BOUND:.2f})")
    print(f"  {'openings':<16}{openings_desc}")
    print(f"  {'engines':<16}{args.name1}: {args.e1} {' '.join(args.opt1 or [])}   |   {args.name2}: {args.e2} {' '.join(args.opt2 or [])}")
    # Progress is printed at most 40 times, at a round number of games (every 1, 5, 10, 50, 100, 500, ... games),
    # or every 100 games for an SPRT, whose length is unknown, and after the last one.  The Elo and its error bar
    # are those of the pairs completed so far.
    step, five = 1, True
    while total / step > 40:
        step *= 5 if five else 2
        five = not five
    if args.sprt:
        step = min(step, 100)
    print(f"\n  {'games':>12}  {'W':>5}  {'L':>5}  {'D':>5}  {'score':>6}  {'elo':>15}{'     LLR' if args.sprt else ''}  "
          f"{'elapsed':>8}  {'ETA':>8}")
    t_start = time.time()

    failures = []
    interrupted = threading.Event()   # Ctrl-C
    decided = threading.Event()       # the SPRT has decided: the pairs not yet started are not played
    live = set()                      # the engines of the pairs in play

    def run_pair(i):
        if interrupted.is_set() or decided.is_set():
            return
        committed = []   # the games of this pair already counted: taken back if the pair fails
        try:
            play_pair(i, committed)
        except (OSError, RuntimeError, ValueError) as e:   # an engine died or broke the protocol
            with lock:
                failures.append(i)
                for key, game in committed:
                    stats[key] -= 1
                    games_out.remove(game)
            if not interrupted.is_set():   # killed by the interrupt, not a failure
                print(f"opening {i + 1}: {type(e).__name__}: {e} (the pair is left out)", file=sys.stderr, flush=True)

    def play_pair(i, committed):
        engines = []
        try:
            engines.append(make_engine(args.e1, o1, i))
            engines.append(make_engine(args.e2, o2, i + 1000))
            with lock:
                live.update(engines)
            points = 0.0
            for swap in (False, True):
                res, game = play_game(args, engines[0], engines[1], opening(i), swap)
                game.headers["Round"] = str(i + 1)
                # The result from engine 1's point of view.
                e1_white = not swap
                key = "d" if res == "1/2-1/2" else "w" if (res == "1-0") == e1_white else "l"
                points += {"w": 1.0, "d": 0.5, "l": 0.0}[key]
                with lock:
                    stats[key] += 1
                    games_out.append(game)
                    committed.append((key, game))
                    if swap:
                        pent[int(points * 2)] += 1
                        if args.sprt and abs(llr(pent, *args.sprt)) >= SPRT_BOUND:
                            decided.set()
                    wins, losses, draws = stats["w"], stats["l"], stats["d"]
                    n = wins + losses + draws
                    if n % step == 0 or n == total:
                        e, err = elo_interval(pent)
                        secs = time.time() - t_start
                        eta = fmt_time(secs * (total - n) / n) if n < total and not args.sprt else "-"
                        print(f"  {n:>6}/{total:<5}  {wins:>5}  {losses:>5}  {draws:>5}  {(wins + draws / 2) / n:>6.1%}  "
                              f"{signed(e):>7} +- {fmt_err(err):<4}{f'  {fixed(llr(pent, *args.sprt), 2):>6}' if args.sprt else ''}  "
                              f"{fmt_time(secs):>8}  {eta:>8}", flush=True)
        finally:
            with lock:
                live.difference_update(engines)
            for e in engines:
                e.quit()

    ex = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        list(ex.map(run_pair, range(pairs)))
    except KeyboardInterrupt:
        interrupted.set()
        with lock:
            for e in live:
                e.kill()
        ex.shutdown(wait=True, cancel_futures=True)
        played = stats["w"] + stats["l"] + stats["d"]
        print(f"\ninterrupted after {played} of {total} games: no result", flush=True)
        sys.exit(130)
    ex.shutdown(wait=True)

    # The result in the form engine tests are usually quoted in: the Elo and its 95% interval, the limit per move
    # with the engines' threads and hash (engine 1's / engine 2's where they differ), the SPRT's LLR with its bounds
    # and hypotheses, the games, and the pairs by outcome.
    def setting(name, unit=""):
        values = [s[name.lower()] for s in settings if name.lower() in s]
        return f" {name}={values[0] if len(set(values)) == 1 else '/'.join(values)}{unit}" if values else ""

    wins, losses, draws = stats["w"], stats["l"], stats["d"]
    e, err = elo_interval(pent)
    lines = [f"Elo   | {fixed(e, 2)} +- {fmt_err(err, 2)} (95%)",
             f"{'SPRT' if args.sprt else 'Conf'}  | {limit_name(args)}{setting('Threads')}{setting('Hash', 'MB')}"]
    if args.sprt:
        lines.append(f"LLR   | {fixed(llr(pent, *args.sprt), 2)} ({-SPRT_BOUND:.2f}, {SPRT_BOUND:.2f}) "
                     f"[{args.sprt[0]:.2f}, {args.sprt[1]:.2f}]")
    lines.append(f"Games | N: {wins + losses + draws} W: {wins} L: {losses} D: {draws}")
    lines.append(f"Penta | [{', '.join(map(str, pent))}]")
    if args.sprt:
        ratio = llr(pent, *args.sprt)
        lines.append("SPRT " + ("passed: the LLR reached the upper bound (H1 accepted)" if ratio >= SPRT_BOUND else
                                "failed: the LLR reached the lower bound (H0 accepted)" if ratio <= -SPRT_BOUND else
                                f"undecided: the limit of {total} games was reached (--games)"))
    print("\n" + "\n".join(lines))
    if args.pgn:
        with open(args.pgn, "w") as f:
            f.writelines(str(g) + "\n\n" for g in games_out)
    if failures:
        sys.exit(f"{len(failures)} of {pairs} openings failed (see above): the result counts the others only")


if __name__ == "__main__":
    main()
