#!/usr/bin/env python3
"""End-to-end trainer smoke test on synthetic data (CPU, small widths):
write a shard with RecordWriter -> train value and policy nets for a few steps
-> export -> the engine loads the files (eval_positions runs).

  test_train_smoke.py --build <dir> --eval <eval_positions> --sfens <file> [--tmp dir]
"""
import argparse
import os
import random
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--sfens", required=True)
ap.add_argument("--tmp", default="/tmp")
args = ap.parse_args()
sys.path.insert(0, args.build)
try:
    import torch  # noqa: F401
    import jhbr5
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train")
sfens = [l.split("\t")[0].strip() for l in open(args.sfens) if l.strip()]
rng = random.Random(5)
shard = os.path.join(args.tmp, "jhbr5_smoke.rec")
w = jhbr5.RecordWriter(shard)
n = 0
for sfen in sfens:
    moves = jhbr5.legal_moves(sfen)
    if not moves:
        continue
    for _ in range(4):
        dist = [(m, rng.randint(1, 100)) for m in rng.sample(moves, min(len(moves), 6))]
        try:
            w.write(sfen, rng.randint(-800, 800), moves[0], rng.randint(1, 100), rng.choice([-1, 0, 1]), dist, 4)
        except RuntimeError:
            break  # test positions without both kings cannot be packed
        n += 1
w.close()
print(f"wrote {n} records")

env = dict(os.environ, PYTHONPATH=args.build + os.pathsep + os.environ.get("PYTHONPATH", ""))
out = os.path.join(args.tmp, "jhbr5_smoke_run")
common = ["--shards", shard, "--batch-size", "64", "--steps", "6", "--workers", "0", "--device", "cpu",
          "--out", out, "--log-every", "3", "--save-every", "1000", "--shuffle-buffer", "256"]
runs = [("value", "v1", "256", "smoke_value.nn"), ("policy", "v1", "512", "smoke_policy.nn"),
        ("value", "v2", "256", "smoke_value2.nn"), ("policy", "v2", "512", "smoke_policy2.nn")]
for net, arch, l1, nn in runs:
    r = subprocess.run([sys.executable, os.path.join(train_dir, "train.py"), "--net", net, "--arch", arch,
                        "--l1", l1, "--export", os.path.join(args.tmp, nn)] + common,
                       env=env, capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        print("FAIL train", net, arch)
        sys.exit(1)
ok = True
for arch, vnn, pnn in [("v1", "smoke_value.nn", "smoke_policy.nn"), ("v2", "smoke_value2.nn", "smoke_policy2.nn")]:
    r = subprocess.run([args.eval, os.path.join(args.tmp, vnn), os.path.join(args.tmp, pnn),
                        args.sfens], capture_output=True, text=True)
    good = r.returncode == 0 and r.stdout.count("\nE\n") >= 50
    print(f"test_train_smoke arch {arch}:", "ok" if good else f"FAILED ({r.stderr[-500:]})")
    ok = ok and good
for f in [shard] + [os.path.join(args.tmp, nn) for _n, _a, _l, nn in runs]:
    if os.path.exists(f):
        os.remove(f)
sys.exit(0 if ok else 1)
