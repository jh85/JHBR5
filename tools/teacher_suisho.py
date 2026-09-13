#!/usr/bin/env python3
"""Teacher labelling with a standard-USI MultiPV engine (e.g. Suisho11/YaneuraOu).

Unlike tools/teacher.py (which expects the JHBR `info string rootdist`
extension), this parses the standard `info ... multipv k score cp X ... pv M`
stream. The final MultiPV iteration's cp scores are converted to a visit
distribution with softmax(cp / temperature), written as [(usi, visits)] pairs
(scaled so the max is 65535). Positions with a mate score (|cp| >= --mate-cp)
are written as value-only records (empty distribution).

Sources of positions:
  --records SHARD...      existing .rec shards (keeps game result); shards are
                          distributed whole to workers (no global load), so the
                          input should already be random (see sample_teacher.py)
  --sample N              take a random N-record subset (loads into memory; use
                          only for small pilots)

Each worker runs one engine process and writes <out>.partK.rec.

  python tools/teacher_suisho.py --engine /workspace/Suisho11/Suisho11 \
      --engine-cwd /workspace/Suisho11 --engine-option Threads=4 \
      --records /workspace/data/teacher_in/pack/*.rec --multipv 8 --nodes 20000 \
      --temperature 60 --workers 40 --out /workspace/data/teacher/pack
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import struct
import subprocess
import sys
import time

FLAG_TEACHER = 2


class UsiEngine:
    def __init__(self, cmd, options, cwd=None):
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("usi")
        self.wait("usiok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
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
        """Returns (bestmove, score_cp, {multipv_idx: (cp, move)})."""
        self.send(f"position sfen {sfen}")
        self.send(f"go nodes {nodes}")
        pv = {}
        score, best = 0, None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            parts = line.split()
            if not parts or parts[0] != "info" or "multipv" not in parts:
                if parts and parts[0] == "bestmove":
                    best = parts[1] if len(parts) > 1 else None
                    return best, score, pv
                continue
            try:
                k = int(parts[parts.index("multipv") + 1])
                i = parts.index("score")
                if parts[i + 1] == "cp":
                    cp = int(parts[i + 2])
                elif parts[i + 1] == "mate":
                    m = parts[i + 2]
                    cp = 30000 if not m.startswith("-") else -30000
                else:
                    continue
                j = parts.index("pv")
                move = parts[j + 1]
            except (ValueError, IndexError):
                continue
            pv[k] = (cp, move)
            if k == 1:
                score = cp

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def softmax_visits(cps, temperature):
    """cps: list of cp scores (non-mate). Returns [(index, visits)] max 65535."""
    m = max(cps)
    exps = [math.exp((c - m) / temperature) for c in cps]
    s = sum(exps)
    return [max(1, int(round(65535.0 * e / s))) for e in exps]


def iter_shard(path, jhbr5):
    """Yield (sfen, ply, result) from a .rec shard without loading it."""
    with open(path, "rb") as f:
        assert f.read(16)[:7] == b"JHBR5RC"
        while True:
            head = f.read(44)
            if len(head) < 44:
                return
            sfen32, _score, _move, ply, result, _flags, n_dist, _ = struct.unpack(
                "<32shHHbBHH", head)
            if n_dist:
                f.read(4 * n_dist)
            try:
                sfen = jhbr5.sfen_from_packed(sfen32, max(ply, 1))
            except RuntimeError:
                continue
            yield sfen, ply, result


def worker(job):
    (wid, args, shards, out_path, build) = job
    sys.path.insert(0, build)
    import jhbr5
    opts = dict(kv.split("=", 1) for kv in args.engine_option)
    opts.setdefault("USI_OwnBook", "false")
    if args.multipv > 1 and not args.no_dist:
        opts.setdefault("MultiPV", str(args.multipv))
    eng = UsiEngine(args.engine.split(), opts, cwd=args.engine_cwd)
    w = jhbr5.RecordWriter(out_path)
    n, t0 = 0, time.time()

    def positions():
        if isinstance(shards, tuple) and shards[0] == "__items__":
            yield from shards[1]
        else:
            for shard in shards:
                yield from iter_shard(shard, jhbr5)

    for sfen, ply, result in positions():
            best, score, pv = eng.search(sfen, args.nodes)
            if not best or best in ("resign", "win"):
                continue
            dist = []
            if not args.no_dist and len(pv) > 1 and abs(score) < args.mate_cp:
                ordered = [pv[k] for k in sorted(pv)]
                cps = [c for c, _m in ordered]
                if all(abs(c) < args.mate_cp for c in cps):
                    visits = softmax_visits(cps, args.temperature)
                    dist = [(m, v) for (_c, m), v in zip(ordered, visits)]
            try:
                w.write(sfen, score, best, ply, result, dist, FLAG_TEACHER)
            except RuntimeError:
                continue
            n += 1
            if n % 10000 == 0:
                el = time.time() - t0
                print(f"worker {wid}: {n} records ({n / el:.1f} pos/s)", flush=True)
    w.close()
    eng.close()
    return wid, n, time.time() - t0


def load_sample(args, build):
    sys.path.insert(0, build)
    import jhbr5
    items = []
    for shard in args.records:
        items += iter_shard(shard, jhbr5)
    if args.sample and len(items) > args.sample:
        random.Random(args.seed).shuffle(items)
        items = items[: args.sample]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--engine-cwd")
    ap.add_argument("--engine-option", action="append", default=[], help="NAME=VALUE (repeatable)")
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=0, help="random subset (pilot mode; loads into RAM)")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--no-dist", action="store_true", help="value-only pass: write empty distributions")
    ap.add_argument("--temperature", type=float, default=60.0)
    ap.add_argument("--mate-cp", type=int, default=29000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.partK.rec")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"))
    args = ap.parse_args()
    build = os.path.abspath(args.build)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.sample:
        items = load_sample(args, build)
        if not items:
            sys.exit("no positions")
        random.Random(args.seed).shuffle(items)
        chunks = [items[i::args.workers] for i in range(args.workers)]
        jobs = [(i, args, ("__items__", chunks[i]), f"{args.out}.part{i}.rec", build)
                for i in range(args.workers) if chunks[i]]
        print(f"{len(items)} positions over {len(jobs)} workers (pilot mode)")
    else:
        shards = sorted(glob.glob(args.records[0])) if len(args.records) == 1 and any(
            c in args.records[0] for c in "*?[") else sorted(args.records)
        assert shards, "no input shards"
        chunks = [shards[i::args.workers] for i in range(args.workers)]
        jobs = [(i, args, chunks[i], f"{args.out}.part{i}.rec", build)
                for i in range(args.workers) if chunks[i]]
        print(f"{len(shards)} shards over {len(jobs)} workers")

    total = 0
    with mp.Pool(len(jobs)) as pool:
        for wid, n, secs in pool.imap_unordered(worker, jobs):
            total += n
            print(f"worker {wid}: done {n} records in {secs:.0f}s ({n / max(secs, 1e-9):.1f} pos/s)", flush=True)
    print(f"done: {total} records -> {args.out}.part*.rec")


if __name__ == "__main__":
    main()
