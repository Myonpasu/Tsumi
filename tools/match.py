#!/usr/bin/env python3
"""Engine-vs-engine matches under selfmate rules (the checkmated side wins), with an Elo estimate.

Standard GUIs and match runners score checkmate for the mating side, so this runner is needed to test Tsumi
builds or networks against each other.  Both engines talk UCI; python-chess keeps the rules.  Openings are random
legal moves from the start position, positions drawn at random from --book (FEN/EPD lines or PGN games), or a mix
(--book-share); each is played with both colours, as fishtest plays its books.

Examples:
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 100 --tc 1+0.05 --concurrency 8
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi --opt1 EvalFile=nets/gen2.nnue --opt2 EvalFile=nets/gen1.nnue --nodes 50000
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 50 --depth 8 --pgn out.pgn
  python3 tools/match.py --e1 ./tsumi --e2 random --games 20 --nodes 5000      # sanity check vs random mover
  python3 tools/match.py --e1 ./tsumi --e2 ./tsumi-old --games 1000 --nodes 50000 --book UHO_Lichess_4852_v1.epd.zip --book-share 50
"""
import argparse
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


class UciEngine:
    def __init__(self, path, options):
        # A path, or a command line ("python3 some_engine.py").  Outside the terminal's process group: a Ctrl-C
        # reaches this script alone, which ends the engines itself.
        cmd = shlex.split(path, posix=os.name != "nt")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  start_new_session=True, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        assert self.p.stdin is not None and self.p.stdout is not None
        self.stdin, self.stdout = self.p.stdin, self.p.stdout
        self.send("uci")
        self.wait_for("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, s):
        self.stdin.write(s + "\n")
        self.stdin.flush()

    def wait_for(self, token):
        lines = []
        while True:
            line = self.stdout.readline()
            if not line:
                raise RuntimeError("engine died" + (f" after: {lines[-1]}" if lines else ""))
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
        k, v = item.split("=", 1)
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


def signed(x, decimals=0):
    """x with an explicit sign, rounded to the given decimals; a zero is written +0, never -0."""
    v = round(x, decimals) + 0.0   # + 0.0 turns -0.0 into 0.0
    return f"{v:+.{decimals}f}"


def elo_from_score(score, se):
    """Elo difference for an expected score, with the 95% half-width implied by its standard error through the
    slope of the Elo curve at that score (None when there is no spread to estimate one from: every game, or every
    pair, went the same way)."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    e = -400 * math.log10(1 / score - 1)
    err = None if se is None else 1.96 * se * 400 / (math.log(10) * score * (1 - score))
    return e, err


def fmt_err(err, decimals=0):
    return "?" if err is None else f"{err:.{decimals}f}"


def elo(wins, losses, draws):
    """Elo estimate treating the games as independent (the running estimate while a match is in progress)."""
    n = wins + losses + draws
    if n == 0:
        return 0.0, None
    score = (wins + draws / 2) / n
    var = (wins * (1 - score) ** 2 + losses * score ** 2 + draws * (0.5 - score) ** 2) / n
    return elo_from_score(score, math.sqrt(var / n) if var > 0 else None)


def elo_pairs(pair_scores):
    """Elo estimate from the per-opening pair scores (engine 1's points over the two games of a pair, 0..2).
    Each opening is played with both colours, so the pair is the independent unit, and the error bar is the
    spread of the pair scores (the pentanomial model of fishtest/cutechess): with a strong first-move effect
    the two games of a pair tend to cancel, and the game-by-game formula overstates the error."""
    n = len(pair_scores)
    if n == 0:
        return 0.0, None
    mean = sum(pair_scores) / n / 2
    var = sum((s / 2 - mean) ** 2 for s in pair_scores) / n
    return elo_from_score(mean, math.sqrt(var / n) if var > 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e2", required=True)
    ap.add_argument("--name1", default="engine1")
    ap.add_argument("--name2", default="engine2")
    ap.add_argument("--opt1", action="append", help="UCI option for engine 1, e.g. EvalFile=nets/x.nnue (repeatable)")
    ap.add_argument("--opt2", action="append")
    ap.add_argument("--games", type=int, default=20, help="number of games (rounded up to an even number)")
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
    rng = random.Random(args.seed or time.time_ns())
    o1, o2 = parse_options(args.opt1), parse_options(args.opt2)
    pairs = (args.games + 1) // 2
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
            print(f"warning: {n_book} openings wanted but {args.book} has {len(book)} position{'s' if len(book) != 1 else ''}: "
                  "they repeat, and the error bar is too small if the engines are deterministic", file=sys.stderr)
    it = iter(book_openings)
    openings = [next(it) if b else random_opening(rng, args.opening_plies) for b in from_book]
    random_desc = f"{args.opening_plies} random plies from the start position"
    if n_book == 0:
        openings_desc = random_desc
    elif n_book == pairs:
        openings_desc = f"drawn from {args.book} ({book_size})"
    else:
        openings_desc = f"{n_book} drawn from {args.book} ({book_size}), {pairs - n_book} reached by {random_desc}"

    stats = {"w": 0, "l": 0, "d": 0}
    pair_scores = []        # engine 1's points over the two games of each opening (0, 0.5, ..., 2)
    lock = threading.Lock()
    games_out = []
    total = pairs * 2
    limit = (f"{args.tc} time control" if args.tc else f"{args.nodes:,} nodes per move" if args.nodes
             else f"depth {args.depth}" if args.depth else f"{args.movetime} ms per move")
    print(f"match: {args.name1} vs {args.name2}")
    print(f"  {'games':<16}{total} ({pairs} opening{'s' if pairs != 1 else ''}, each played with both colours), {limit}, "
          f"{args.concurrency} at a time")
    print(f"  {'openings':<16}{openings_desc}")
    print(f"  {'engines':<16}{args.name1}: {args.e1} {' '.join(args.opt1 or [])}   |   {args.name2}: {args.e2} {' '.join(args.opt2 or [])}")
    # Progress is printed at most 40 times, at a round number of games (every 1, 5, 10, 50, 100, 500, ...
    # games), and after the last one; the Elo and its error bar are the running game-by-game estimate (the
    # final line uses the paired-opening estimate).
    step, five = 1, True
    while total / step > 40:
        step *= 5 if five else 2
        five = not five
    print(f"\n  {'games':>12}  {'W':>5}  {'L':>5}  {'D':>5}  {'score':>6}  {'elo':>16}  {'elapsed':>8}  {'ETA':>8}")
    t_start = time.time()

    failures = []
    interrupted = threading.Event()   # Ctrl-C
    live = set()                      # the engines of the pairs in play

    def run_pair(i):
        if interrupted.is_set():
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
                res, game = play_game(args, engines[0], engines[1], openings[i], swap)
                game.headers["Round"] = str(i + 1)
                # The result from engine 1's point of view.
                e1_white = not swap
                key = "d" if res == "1/2-1/2" else "w" if (res == "1-0") == e1_white else "l"
                points += {"w": 1.0, "d": 0.5, "l": 0.0}[key]
                with lock:
                    stats[key] += 1
                    games_out.append(game)
                    committed.append((key, game))
                    wins, losses, draws = stats["w"], stats["l"], stats["d"]
                    n = wins + losses + draws
                    if n % step == 0 or n == total:
                        e, err = elo(wins, losses, draws)
                        secs = time.time() - t_start
                        eta = secs * (total - n) / n
                        print(f"  {n:>6}/{total:<5}  {wins:>5}  {losses:>5}  {draws:>5}  {(wins + draws / 2) / n:>6.1%}  "
                              f"{signed(e):>7} +/- {fmt_err(err):<4}  {fmt_time(secs):>8}  {fmt_time(eta) if n < total else '-':>8}",
                              flush=True)
            with lock:
                pair_scores.append(points)
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

    wins, losses, draws = stats["w"], stats["l"], stats["d"]
    e, err = elo_pairs(pair_scores)
    pent = [sum(1 for s in pair_scores if s == k / 2) for k in range(5)]
    print(f"\nFinal: {args.name1} vs {args.name2}: +{wins} -{losses} ={draws}   elo {signed(e, 1)} +/- {fmt_err(err, 1)}  "
          f"(draw rate {draws / max(1, wins + losses + draws):.0%}, pairs LL/LD/DD+WL/WD/WW {'/'.join(map(str, pent))}, "
          f"game-by-game error +/- {fmt_err(elo(wins, losses, draws)[1], 1)}, {fmt_time(time.time() - t_start)})")
    if args.pgn:
        with open(args.pgn, "w") as f:
            f.writelines(str(g) + "\n\n" for g in games_out)
    if failures:
        sys.exit(f"{len(failures)} of {pairs} openings failed (see above): the result counts the others only")


if __name__ == "__main__":
    main()
