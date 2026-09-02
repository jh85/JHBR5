#!/usr/bin/env python3
"""Teacher labelling with a USI engine that prints its root visit distribution
(JHBR3 with the `RootDistOutput` patch, or JHBR5 with `RootDistOutput=true`).

Sources of positions:
  --sfens FILE            one sfen per line (result unknown -> 0 unless --results)
  --records SHARD...      existing .rec shards (keeps their game result; e.g. pack imports)
  --selfplay GAMES        the teacher plays itself from random openings

For every position the engine searches `--nodes N`; the record stores the
root score, best move, the visit distribution and the game result
(flags kTeacherJhbr3). Work is split over `--workers` engine processes, each
writing `<out>.partK.rec`.

  python tools/teacher.py --engine ../JHBR3/build-trt/jhbr3 --engine-option OnnxModel=... \
      --records data/pack/*.rec --sample 2000000 --nodes 800 --workers 8 --out data/teacher/pack
"""
import argparse
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

FLAG_TEACHER = 2
FLAG_SELFPLAY = 4


class UsiEngine:
    def __init__(self, cmd, options, cwd=None):
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("usi")
        self.wait("usiok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("setoption name RootDistOutput value true")
        self.send("isready")
        self.wait("readyok")
        self.send("usinewgame")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.strip() == token:
                return

    def search(self, sfen, nodes):
        """Returns (bestmove, score_cp, [(usi, visits)])."""
        self.send(f"position sfen {sfen}")
        self.send(f"go nodes {nodes}")
        score, dist, best = 0, [], None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "info" and "score" in parts:
                i = parts.index("score")
                if parts[i + 1] == "cp":
                    score = int(parts[i + 2])
                elif parts[i + 1] == "mate":
                    m = int(parts[i + 2]) if parts[i + 2].lstrip("-").isdigit() else 1
                    score = 30000 if m > 0 else -30000
            if parts[0] == "info" and len(parts) > 2 and parts[1] == "string" and parts[2] == "rootdist":
                dist = [(t.rsplit(":", 1)[0], int(t.rsplit(":", 1)[1])) for t in parts[3:]]
            if parts[0] == "bestmove":
                best = parts[1] if len(parts) > 1 else None
                return best, score, dist

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def worker(job):
    (wid, args, items, out_path, build) = job
    sys.path.insert(0, build)
    import jhbr5
    opts = dict(kv.split("=", 1) for kv in args.engine_option)
    eng = UsiEngine(args.engine.split(), opts, cwd=args.engine_cwd)
    w = jhbr5.RecordWriter(out_path)
    n = 0
    t0 = time.time()
    if args.selfplay:
        import cshogi
        rng = random.Random(args.seed + wid)
        for _ in range(items):
            board = cshogi.Board()
            for _ in range(rng.randint(0, args.random_plies)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            game = []
            outcome = 0  # from black
            while True:
                if board.is_game_over():
                    outcome = -1 if board.turn == cshogi.BLACK else 1
                    break
                if board.move_number > args.max_ply or board.is_draw():
                    break
                if board.is_nyugyoku():
                    outcome = 1 if board.turn == cshogi.BLACK else -1
                    break
                sfen = board.sfen()
                best, score, dist = eng.search(sfen, args.nodes)
                if not best or best in ("resign", "win"):
                    outcome = (-1 if board.turn == cshogi.BLACK else 1) if best != "win" else (1 if board.turn == cshogi.BLACK else -1)
                    break
                game.append((sfen, score, best, board.move_number, board.turn, dist))
                board.push_usi(best)
            for sfen, score, best, ply, turn, dist in game:
                r = outcome if turn == cshogi.BLACK else -outcome
                w.write(sfen, score, best, ply, r, dist, FLAG_TEACHER | FLAG_SELFPLAY)
                n += 1
    else:
        for sfen, ply, result in items:
            best, score, dist = eng.search(sfen, args.nodes)
            if not best or best in ("resign", "win"):
                continue
            w.write(sfen, score, best, ply, result, dist, FLAG_TEACHER)
            n += 1
    w.close()
    eng.close()
    return wid, n, time.time() - t0


def load_positions(args, build):
    sys.path.insert(0, build)
    import jhbr5
    items = []
    if args.sfens:
        for line in open(args.sfens):
            sfen = line.split("\t")[0].strip()
            if sfen:
                ply = int(sfen.split()[-1]) if sfen.split()[-1].isdigit() else 1
                items.append((sfen, ply, 0))
    for shard in args.records:
        reader = jhbr5.BatchReader([shard], 4096, 4096, args.seed, False, True, False)
        # BatchReader yields features, not sfens; read raw records instead.
        del reader
        items += read_records(shard, jhbr5)
    if args.sample and len(items) > args.sample:
        random.Random(args.seed).shuffle(items)
        items = items[: args.sample]
    return items


def read_records(path, jhbr5):
    """Read (sfen, ply, result) tuples straight from a .rec shard."""
    import struct
    out = []
    with open(path, "rb") as f:
        assert f.read(16)[:7] == b"JHBR5RC"
        while True:
            head = f.read(44)
            if len(head) < 44:
                break
            sfen32, score, move, ply, result, flags, n_dist, _ = struct.unpack("<32shHHbBHH", head)
            f.read(4 * n_dist)
            try:
                sfen = jhbr5.sfen_from_packed(sfen32, max(ply, 1))
            except RuntimeError:
                continue
            out.append((sfen, ply, result))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="engine command line")
    ap.add_argument("--engine-cwd")
    ap.add_argument("--engine-option", action="append", default=[], help="NAME=VALUE (repeatable)")
    ap.add_argument("--sfens")
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--selfplay", type=int, default=0, help="number of self-play games")
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=320)
    ap.add_argument("--sample", type=int, default=0, help="random subset of the positions")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.partK.rec")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
    args = ap.parse_args()
    build = os.path.abspath(args.build)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.selfplay:
        per = [args.selfplay // args.workers + (1 if i < args.selfplay % args.workers else 0) for i in range(args.workers)]
        jobs = [(i, args, per[i], f"{args.out}.part{i}.rec", build) for i in range(args.workers) if per[i] > 0]
    else:
        items = load_positions(args, build)
        if not items:
            sys.exit("no positions")
        chunks = [items[i::args.workers] for i in range(args.workers)]
        jobs = [(i, args, chunks[i], f"{args.out}.part{i}.rec", build) for i in range(args.workers) if chunks[i]]
        print(f"{len(items)} positions over {len(jobs)} workers")
    total = 0
    with mp.Pool(len(jobs)) as pool:
        for wid, n, secs in pool.imap_unordered(worker, jobs):
            total += n
            print(f"worker {wid}: {n} records in {secs:.0f}s ({n / max(secs, 1e-9):.1f} pos/s)", flush=True)
    print(f"done: {total} records -> {args.out}.part*.rec")


if __name__ == "__main__":
    main()
