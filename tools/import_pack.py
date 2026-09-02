#!/usr/bin/env python3
"""Import YaneuraOu gensfen `.pack` game records into JHBR5 record shards.

Format (YaneuraOu-ScriptCollection/GenSfen/readme.md): per game
  start_flag u8 (1 = hirate, 0 = Apery HCP 32 bytes + game_ply u16),
  then (move16 u16, eval i16)* and an end marker u16 (0x0000 draw,
  0x0081 black wins, 0x0102 white wins) followed by a reason byte.
Evals are from the side to move. Decoding uses cshogi (HCP + move16).

  python tools/import_pack.py --pack-dir ../2000000 --out data/pack --workers 8 [--limit N]

Writes one .rec shard per pack file (value-only records, flags kImportedPsv).
"""
import argparse
import glob
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

try:
    import cshogi
except ImportError:
    sys.exit("cshogi is required: python3 -m pip install cshogi")

REASON_NAMES = {0: "resign", 1: "draw_rep", 2: "max_moves", 3: "interrupt", 4: "timeup",
                5: "illegal", 6: "rep_check", 10: "win24", 11: "draw24", 12: "win27", 13: "try"}
FLAG_IMPORTED = 16


def iter_games(data):
    pos = 0
    n = len(data)
    while pos < n:
        flag = data[pos]
        pos += 1
        if flag == 1:
            hcp, ply = None, 1
        elif flag == 0:
            hcp = data[pos:pos + 32]
            ply, = struct.unpack_from("<H", data, pos + 32)
            pos += 34
        else:
            raise ValueError(f"unknown start flag {flag} at {pos}")
        moves = []
        while True:
            mv, = struct.unpack_from("<H", data, pos)
            if mv in (0x0000, 0x0081, 0x0102):
                reason = data[pos + 2]
                pos += 3
                break
            ev, = struct.unpack_from("<h", data, pos + 2)
            pos += 4
            moves.append((mv, ev))
        result = {0x0000: 0, 0x0081: 1, 0x0102: -1}[mv]  # from black
        yield hcp, ply, moves, result, reason


def convert_file(args):
    path, out_path, limit, min_ply, skip_interrupted, build = args
    sys.path.insert(0, build)
    import jhbr5
    data = open(path, "rb").read()
    w = jhbr5.RecordWriter(out_path)
    board = cshogi.Board()
    games = positions = skipped = 0
    for hcp, ply, moves, result, reason in iter_games(data):
        if skip_interrupted and reason == 3:
            skipped += 1
            continue
        if hcp is None:
            board.reset()
        else:
            board.set_hcp(np.frombuffer(hcp, dtype=cshogi.HuffmanCodedPos))
        board.move_number = ply
        for mv16, ev in moves:
            m = board.move_from_move16(mv16)
            if m == 0:
                break
            if board.move_number >= min_ply:
                stm_black = board.turn == cshogi.BLACK
                r = result if stm_black else -result
                score = max(-32000, min(32000, ev))
                try:
                    w.write(board.sfen(), score, cshogi.move_to_usi(m), board.move_number, r, [], FLAG_IMPORTED)
                    positions += 1
                except RuntimeError:
                    pass
            board.push(m)
        games += 1
        if limit and positions >= limit:
            break
    w.close()
    return path, games, positions, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", help="directory of .pack files")
    ap.add_argument("--pack", nargs="*", default=[], help="explicit .pack files")
    ap.add_argument("--out", required=True, help="output directory for .rec shards")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=0, help="max positions per file (0 = all)")
    ap.add_argument("--min-ply", type=int, default=1, help="skip positions before this ply")
    ap.add_argument("--keep-interrupted", action="store_true")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"),
                    help="directory containing the jhbr5 Python module")
    args = ap.parse_args()
    files = list(args.pack)
    if args.pack_dir:
        files += sorted(glob.glob(os.path.join(args.pack_dir, "*.pack")))
    if not files:
        sys.exit("no .pack files")
    os.makedirs(args.out, exist_ok=True)
    jobs = [(f, os.path.join(args.out, os.path.splitext(os.path.basename(f))[0] + ".rec"),
             args.limit, args.min_ply, not args.keep_interrupted, os.path.abspath(args.build)) for f in files]
    t0 = time.time()
    total_pos = 0
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for path, games, positions, skipped in pool.imap_unordered(convert_file, jobs):
            total_pos += positions
            print(f"{os.path.basename(path)}: {games} games, {positions} positions, {skipped} skipped "
                  f"({total_pos / (time.time() - t0):.0f} pos/s)", flush=True)
    print(f"done: {total_pos} positions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
