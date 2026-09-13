#!/usr/bin/env python3
"""Calibrate the softmax temperature for MultiPV score -> visit distribution.

For each position: MultiPV-8 search at --nodes (shallow) and --ref-nodes (deep).
Finds the T minimizing CE(deep_dist(T) || shallow_dist(T)), i.e. the T at which
shallow-score noise hurts least, plus top-1 stability stats.

  python tools/calibrate_temp.py --records /workspace/data/teacher_in/pack/sample_0000.rec ... \
      --sample 30000 --nodes 20000 --ref-nodes 200000 --workers 16
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_suisho import UsiEngine, iter_shard


def log_softmax(cps, T):
    m = max(cps)
    z = [math.exp(max((c - m) / T, -700.0)) for c in cps]
    s = sum(z)
    return [math.log(max(e, 1e-300) / s) for e in z]


def worker(job):
    wid, args, items, build = job
    sys.path.insert(0, build)
    import jhbr5
    opts = {"Threads": str(args.threads), "USI_Hash": "128", "MultiPV": str(args.multipv),
            "USI_OwnBook": "false"}
    eng = UsiEngine(args.engine.split(), opts, cwd=args.engine_cwd)
    out = []
    for sfen, ply, result, src in items:
        _b, _s, shallow = eng.search(sfen, args.nodes)
        _b, _s, deep = eng.search(sfen, args.ref_nodes)
        sc = [shallow[k][0] for k in sorted(shallow) if k in deep and abs(shallow[k][0]) < args.mate_cp]
        dc = [deep[k][0] for k in sorted(deep) if k in shallow and abs(deep[k][0]) < args.mate_cp]
        if len(sc) >= 2 and len(sc) == len(dc):
            out.append((sc, dc))
    eng.close()
    return wid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--engine-cwd")
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--ref-nodes", type=int, default=200000)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--mate-cp", type=int, default=29000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--build", default="/workspace/JHBR5/build")
    args = ap.parse_args()

    sys.path.insert(0, args.build)
    import jhbr5
    items = []
    for pat in args.records:
        for shard in sorted(glob.glob(pat)):
            src = "psv" if "psv" in shard else "pack"
            items += [(s, p, r, src) for s, p, r in iter_shard(shard, jhbr5)]
    random.Random(args.seed).shuffle(items)
    items = items[: args.sample]
    chunks = [items[i::args.workers] for i in range(args.workers)]
    jobs = [(i, args, c, args.build) for i, c in enumerate(chunks) if c]

    pairs = []
    with mp.Pool(len(jobs)) as pool:
        for wid, out in pool.imap_unordered(worker, jobs):
            pairs += out
            print(f"worker {wid}: {len(out)} usable pairs (total {len(pairs)})", flush=True)

    print(f"\n{len(pairs)} paired positions")
    import pickle
    with open("/workspace/runs/dist1/calib_pairs.pkl", "wb") as f:
        pickle.dump(pairs, f)
    top1_agree = sum(1 for sc, dc in pairs if max(range(len(sc)), key=lambda i: sc[i]) ==
                     max(range(len(dc)), key=lambda i: dc[i]))
    print(f"top-1 agreement {top1_agree / len(pairs):.3f}")
    print(f"{'T':>8} {'CE':>10} {'CE_pack':>10} {'CE_psv':>10}")
    best = None
    for T in [10, 20, 30, 40, 60, 80, 120, 160, 240, 320]:
        ce = 0.0
        for sc, dc in pairs:
            ls = log_softmax(sc, T)
            pd = log_softmax(dc, T)
            ce -= sum(math.exp(p) * l for p, l in zip(pd, ls))
        ce /= len(pairs)
        if best is None or ce < best[1]:
            best = (T, ce)
        print(f"{T:>8} {ce:>10.4f}")
    print(f"best T = {best[0]} (CE {best[1]:.4f})")


if __name__ == "__main__":
    main()
