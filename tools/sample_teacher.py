#!/usr/bin/env python3
"""Sample a random subset of positions from imported .rec shards (value-only,
44-byte records) as teacher-input shards.

  python tools/sample_teacher.py --src /workspace/data/pack --records 103600000 \
      --val-records 1400000 --out /workspace/data/teacher_in/pack \
      --val-out /workspace/data/val_teacher/pack --seed 20260905
"""
import argparse
import glob
import os

import numpy as np

HEADER = b"JHBR5RC\0" + (1).to_bytes(4, "little") + (0).to_bytes(4, "little")
REC = np.dtype([("sfen", "u1", 32), ("score", "<i2"), ("move", "<u2"), ("ply", "<u2"),
                ("result", "i1"), ("flags", "u1"), ("n_dist", "<u2"), ("reserved", "<u2")])


def allocate(counts, total):
    """Proportional integer allocation of `total` across shards summing exactly."""
    raw = np.array(counts, dtype=np.float64) * total / sum(counts)
    base = np.floor(raw).astype(np.int64)
    rem = total - base.sum()
    order = np.argsort(raw - base)[::-1]
    for i in order[:rem]:
        base[i] += 1
    return base


def sample_dir(src, out, val_out, n_train, n_val, seed, shard_records):
    files = sorted(glob.glob(os.path.join(src, "*.rec")))
    counts = [(os.path.getsize(f) - 16) // 44 for f in files]
    rng = np.random.default_rng(seed)
    alloc_train = allocate(counts, n_train)
    alloc_val = allocate(counts, n_val)

    def dump(indices, path):
        assert len(indices) == 0 or indices[-1] * 44 + 16 <= os.path.getsize(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        recs = mm[16:].view(REC)
        return recs[np.sort(indices)]

    # collect sampled rows, then write rotated output shards
    def write_shards(rows, out_dir, tag, per_shard):
        os.makedirs(out_dir, exist_ok=True)
        for i in range(0, len(rows), per_shard):
            part = rows[i:i + per_shard]
            p = os.path.join(out_dir, f"{tag}_{i // per_shard:04d}.rec")
            with open(p, "wb") as o:
                o.write(HEADER)
                o.write(np.ascontiguousarray(part).tobytes())

    # stream per input shard to bound memory: accumulate into a list, these are
    # 44-byte rows; 105M rows = 4.6 GB, fine for this box.
    train_rows, val_rows = [], []
    for f, c, at, av in zip(files, counts, alloc_train, alloc_val):
        if at == 0 and av == 0:
            continue
        perm = rng.permutation(c)
        take = np.concatenate([perm[:at], perm[at:at + av]])
        rows = dump(take, f)
        train_rows.append(rows[:at])
        val_rows.append(rows[at:])
        print(f"{os.path.basename(f)}: {c} records -> {at} train + {av} val", flush=True)
    train = np.concatenate(train_rows) if train_rows else np.empty(0, dtype=REC)
    val = np.concatenate(val_rows) if val_rows else np.empty(0, dtype=REC)
    rng.shuffle(train)
    rng.shuffle(val)
    write_shards(train, out, "sample", shard_records)
    write_shards(val, val_out, "val", max(shard_records // 4, 1))
    print(f"train: {len(train)} -> {out}; val: {len(val)} -> {val_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--records", type=int, required=True)
    ap.add_argument("--val-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-records", type=int, default=4_000_000)
    args = ap.parse_args()
    sample_dir(args.src, args.out, args.val_out, args.records, args.val_records,
               args.seed, args.shard_records)


if __name__ == "__main__":
    main()
