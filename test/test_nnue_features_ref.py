#!/usr/bin/env python3
"""Pure-Python reference for the JHBR5 NNUE feature and move-bucket mappings.

This file deliberately shares no code with the engine: it has its own sfen
parser, attack generator and table construction, written from docs/DESIGN.md
sections 2-4. It runs `dump_nnue_features` on a position file and checks every
feature index and bucket bit-for-bit. It also recomputes the feature-set id.

Usage: test_nnue_features_ref.py <dump_nnue_features binary> <sfen file>
"""
import struct
import subprocess
import sys

BLACK, WHITE = 0, 1
# JHBR3 piece type idx: P1 L2 N3 S4 B5 R6 G7 K8, promoted = idx | 8 (9..14).
CHAR_TO_IDX = {"P": 1, "L": 2, "N": 3, "S": 4, "B": 5, "R": 6, "G": 7, "K": 8}
HAND_MAX = [18, 4, 4, 4, 2, 2, 4]
HAND_OFF = [0, 18, 22, 26, 30, 32, 34]
NUM_SLOTS = 76 + 2 * 14 * 81
PAIRS_PER_OWNER = 8228
THREAT_CLASSES = 16
GROUP_A = 81 * NUM_SLOTS
GROUP_B = NUM_SLOTS + 2 * PAIRS_PER_OWNER * THREAT_CLASSES
POLICY_BOARD = 2 * 14 * 81 * 4
POLICY_INPUTS = POLICY_BOARD + 76
NUM_BUCKETS = 10043
SEE_THRESHOLD = -90
CLS = {0: 7, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 5, 13: 6}


def tid(idx):
    return 0 if idx == 8 else (idx if idx < 8 else idx - 1)


def sq_of(file, rank):
    return file * 9 + rank


def frame_sq(p, sq):
    return sq if p == BLACK else 80 - sq


# --- attacks -----------------------------------------------------------------
GOLD = [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)]
STEPS = {
    1: [(0, -1)],
    3: [(-1, -2), (1, -2)],
    4: [(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)],
    7: GOLD, 9: GOLD, 10: GOLD, 11: GOLD, 12: GOLD,
    8: [(df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0)],
}
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def attacks(idx, color, sq, occ):
    """Squares attacked by a piece of type idx/colour on sq given occupancy set."""
    f, r = divmod(sq, 9)
    sign = 1 if color == BLACK else -1
    out = set()

    def step(dirs):
        for df, dr in dirs:
            ff, rr = f + df * sign, r + dr * sign
            if 0 <= ff < 9 and 0 <= rr < 9:
                out.add(sq_of(ff, rr))

    def slide(dirs):
        for df, dr in dirs:
            ff, rr = f + df * sign, r + dr * sign
            while 0 <= ff < 9 and 0 <= rr < 9:
                s = sq_of(ff, rr)
                out.add(s)
                if s in occ:
                    break
                ff, rr = ff + df * sign, rr + dr * sign

    if idx in STEPS:
        step(STEPS[idx])
    if idx == 2:
        slide([(0, -1)])
    if idx in (5, 13):
        slide(DIAG)
    if idx in (6, 14):
        slide(ORTH)
    if idx == 13:
        step(ORTH)
    if idx == 14:
        step(DIAG)
    return out


# --- tables ------------------------------------------------------------------
def build_tables():
    dest = [[attacks(9 if t == 0 else 0, BLACK, s, set()) if False else None for s in range(81)] for t in range(14)]
    off = [[0] * 81 for _ in range(14)]
    running = 0
    for t in range(14):
        idx = 8 if t == 0 else (t if t < 8 else t + 1)
        for s in range(81):
            dest[t][s] = attacks(idx, BLACK, s, set())
            off[t][s] = running
            running += len(dest[t][s])
    assert running == PAIRS_PER_OWNER, running

    zone = {sq_of(f, r) for f in range(9) for r in range(3)}
    rank0 = {sq_of(f, 0) for f in range(9)}
    rank01 = rank0 | {sq_of(f, 1) for f in range(9)}
    plain, plain_off, promo, promo_off = [], [], [], []
    running = 0
    for t in range(14):
        plain.append([]); plain_off.append([])
        for s in range(81):
            d = set(dest[t][s])
            if t in (1, 2):
                d -= rank0
            if t == 3:
                d -= rank01
            plain[t].append(d); plain_off[t].append(running); running += len(d)
    assert running == 8115, running
    for t in range(14):
        promo.append([]); promo_off.append([])
        for s in range(81):
            d = set()
            if 1 <= t <= 6:
                d = set(dest[t][s]) if s in zone else set(dest[t][s]) & zone
            promo[t].append(d); promo_off[t].append(running); running += len(d)
    assert running == 8115 + 1397, running
    drop, drop_off = [], []
    allsq = set(range(81))
    for h in range(7):
        d = set(allsq)
        if h in (0, 1):
            d -= rank0
        if h == 2:
            d -= rank01
        drop.append(d); drop_off.append(running); running += len(d)
    assert running == NUM_BUCKETS, running
    return dict(dest=dest, off=off, plain=plain, plain_off=plain_off, promo=promo,
                promo_off=promo_off, drop=drop, drop_off=drop_off)


T = build_tables()


def rank_in(dset, to):
    return sum(1 for s in dset if s < to)


def pair_index(t, own_from, own_to):
    assert own_to in T["dest"][t][own_from]
    return T["off"][t][own_from] + rank_in(T["dest"][t][own_from], own_to)


def fnv1a(h, data):
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def feature_set_id():
    h = 2166136261
    h = fnv1a(h, b"JHBR5-FS1")
    for v in (NUM_SLOTS, GROUP_A, GROUP_B, POLICY_INPUTS, THREAT_CLASSES):
        h = fnv1a(h, struct.pack("<i", v))
    flat = b"".join(struct.pack("<i", T["off"][t][s]) for t in range(14) for s in range(81))
    return fnv1a(h, flat)


# --- position ----------------------------------------------------------------
class Pos:
    def __init__(self, sfen):
        parts = sfen.split()
        self.board = {}  # sq -> (color, idx)
        rows = parts[0].split("/")
        assert len(rows) == 9
        for rank, row in enumerate(rows):
            file = 8
            promoted = False
            for ch in row:
                if ch.isdigit():
                    file -= int(ch)
                elif ch == "+":
                    promoted = True
                else:
                    color = BLACK if ch.isupper() else WHITE
                    idx = CHAR_TO_IDX[ch.upper()] | (8 if promoted else 0)
                    self.board[sq_of(file, rank)] = (color, idx)
                    promoted = False
                    file -= 1
        self.stm = BLACK if parts[1] == "b" else WHITE
        self.hand = {BLACK: [0] * 7, WHITE: [0] * 7}
        if parts[2] != "-":
            n = 0
            for ch in parts[2]:
                if ch.isdigit():
                    n = n * 10 + int(ch)
                else:
                    c = BLACK if ch.isupper() else WHITE
                    self.hand[c][CHAR_TO_IDX[ch.upper()] - 1] += max(n, 1)
                    n = 0
        self.occ = set(self.board)
        self.king = {}
        for s, (c, idx) in self.board.items():
            if idx == 8:
                self.king[c] = s

    def attacked_by(self, color):
        out = set()
        for s, (c, idx) in self.board.items():
            if c == color:
                out |= attacks(idx, c, s, self.occ)
        return out


def slot(owner, t, fsq):
    return 76 + (owner * 14 + t) * 81 + fsq


def hand_slot(owner, h, k):
    return owner * 38 + HAND_OFF[h] + k - 1


def group_a(pos, p):
    base = (frame_sq(p, pos.king[p]) if p in pos.king else 0) * NUM_SLOTS
    out = []
    for s, (c, idx) in pos.board.items():
        if idx == 8:
            continue
        out.append(base + slot(int(c != p), tid(idx), frame_sq(p, s)))
    for c in (BLACK, WHITE):
        for h in range(7):
            for k in range(1, pos.hand[c][h] + 1):
                out.append(base + hand_slot(int(c != p), h, k))
    return sorted(out)


def group_b(pos):
    stm = pos.stm
    out = []
    for s, (c, idx) in pos.board.items():
        if idx != 8:
            out.append(slot(int(c != stm), tid(idx), frame_sq(stm, s)))
    for c in (BLACK, WHITE):
        for h in range(7):
            for k in range(1, pos.hand[c][h] + 1):
                out.append(hand_slot(int(c != stm), h, k))
    for s, (c, idx) in pos.board.items():
        owner = int(c != stm)
        for to in attacks(idx, c, s, pos.occ) & pos.occ:
            tc, tidx = pos.board[to]
            cls = (8 if tc != stm else 0) + CLS[tid(tidx)]
            pair = pair_index(tid(idx), frame_sq(c, s), frame_sq(c, to))
            out.append(NUM_SLOTS + ((owner * PAIRS_PER_OWNER + pair) * THREAT_CLASSES + cls))
    return sorted(out)


def policy(pos):
    stm = pos.stm
    att = pos.attacked_by(1 - stm)
    dfd = pos.attacked_by(stm)
    out = []
    for s, (c, idx) in pos.board.items():
        flags = (1 if s in att else 0) + (2 if s in dfd else 0)
        out.append((int(c != stm) * 14 + tid(idx)) * 81 + frame_sq(stm, s) + 2268 * flags)
    for c in (BLACK, WHITE):
        for h in range(7):
            for k in range(1, pos.hand[c][h] + 1):
                out.append(POLICY_BOARD + hand_slot(int(c != stm), h, k))
    return sorted(out)


def parse_usi_sq(s):
    return sq_of(int(s[0]) - 1, ord(s[1]) - ord("a"))


def bucket(pos, usi):
    stm = pos.stm
    if usi[1] == "*":
        h = CHAR_TO_IDX[usi[0]] - 1
        to = frame_sq(stm, parse_usi_sq(usi[2:4]))
        return T["drop_off"][h] + rank_in(T["drop"][h], to)
    frm = parse_usi_sq(usi[0:2])
    to = frame_sq(stm, parse_usi_sq(usi[2:4]))
    t = tid(pos.board[frm][1])
    frm = frame_sq(stm, frm)
    if usi.endswith("+"):
        return T["promo_off"][t][frm] + rank_in(T["promo"][t][frm], to)
    return T["plain_off"][t][frm] + rank_in(T["plain"][t][frm], to)


def main():
    binary, sfens = sys.argv[1], sys.argv[2]
    out = subprocess.run([binary, sfens], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    fs = int(lines[0].split()[1])
    fails = 0
    if fs != feature_set_id():
        print(f"FAIL feature set id: engine {fs} reference {feature_set_id()}")
        fails += 1
    i, positions = 1, 0
    while i < len(lines):
        assert lines[i].startswith("S ")
        sfen = lines[i][2:]
        pos = Pos(sfen)
        got = {}
        i += 1
        moves = []
        while lines[i] != "E":
            tag = lines[i].split()[0]
            if tag == "M":
                _, usi, bk, _see = lines[i].split()
                moves.append((usi, int(bk)))
            else:
                vals = [int(x) for x in lines[i].split()[2:]]
                got[tag] = sorted(vals)
            i += 1
        i += 1
        positions += 1
        exp = {
            "A_us": group_a(pos, pos.stm),
            "A_them": group_a(pos, 1 - pos.stm),
            "B": group_b(pos),
            "P": policy(pos),
        }
        for k, v in exp.items():
            if got[k] != v:
                fails += 1
                if fails < 10:
                    print(f"FAIL {k} {sfen}\n  engine {got[k]}\n  ref    {v}")
        for usi, bk in moves:
            rb = bucket(pos, usi)
            if rb != bk:
                fails += 1
                if fails < 10:
                    print(f"FAIL bucket {sfen} {usi}: engine {bk} ref {rb}")
    print(f"test_nnue_features_ref: {positions} positions, {'FAILED' if fails else 'ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
