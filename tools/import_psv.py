#!/usr/bin/env python3
"""Re-frame YaneuraOu PSV (.psv/.bin, 40-byte PackedSfenValue) files as JHBR5
record shards. No decoding is needed: the first 40 bytes of a JHBR5 record are
a PackedSfenValue; each record gets a 4-byte `n_dist = 0, reserved = 0` tail
(docs/DATA_FORMAT.md). YaneuraOu's move16 has the same bit layout as JHBR3's.

  python tools/import_psv.py --out data/psv file1.bin file2.psv ...
"""
import argparse
import os
import sys

import numpy as np

HEADER = b"JHBR5RC\0" + (1).to_bytes(4, "little") + (0).to_bytes(4, "little")
FLAG_IMPORTED = 16
PSV = np.dtype([("sfen", "u1", 32), ("score", "<i2"), ("move", "<u2"), ("ply", "<u2"),
                ("result", "i1"), ("pad", "u1")])
REC = np.dtype([("sfen", "u1", 32), ("score", "<i2"), ("move", "<u2"), ("ply", "<u2"),
                ("result", "i1"), ("flags", "u1"), ("n_dist", "<u2"), ("reserved", "<u2")])


def convert(path, out_path, chunk=1 << 20):
    n = 0
    with open(path, "rb") as f, open(out_path, "wb") as o:
        o.write(HEADER)
        while True:
            buf = f.read(chunk * PSV.itemsize)
            if not buf:
                break
            psv = np.frombuffer(buf[: len(buf) // PSV.itemsize * PSV.itemsize], dtype=PSV)
            rec = np.zeros(len(psv), dtype=REC)
            for k in ("sfen", "score", "move", "ply", "result"):
                rec[k] = psv[k]
            rec["flags"] = FLAG_IMPORTED
            o.write(rec.tobytes())
            n += len(psv)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for path in args.files:
        out = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + ".rec")
        n = convert(path, out)
        total += n
        print(f"{path}: {n} records -> {out}")
    print(f"done: {total} records")


if __name__ == "__main__":
    sys.exit(main())
