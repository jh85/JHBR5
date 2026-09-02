# JHBR5 design proposal

Status: **reviewed 2026-09-02; Phase 1 implemented**. Companion notes:
`MONTY_NOTES.md`, `YANEURAOU_NNUE_NOTES.md`; the review answers are in
`open_questions.txt`.

## Review decisions (supersede the text below where they differ)

| # | Question | Decision |
|---|---|---|
| 1 | Shared vs separate group-A tables | **Shared** (closest to Monty's single table, simpler). Consequence: the PST skip connection is attached to group B only, because a shared table cannot carry a signed PST for both frames. |
| 2 | S or M profile | **M** (value L1 1024, policy L1 4096) as defaults; widths are header fields so S nets load unchanged. |
| 3 | Threat pairs in v1 | **Yes** (Monty has them). |
| 4 | SEE doubling | **On** (Monty has it); the header records which table a net uses. |
| 5 | Records store move16 | **Yes** (Monty stores moves). |
| 6 | JHBR3 `info string rootdist` patch | Allowed, opt-in only, no impact on JHBR3 performance. |
| 7 | Expansion on the second visit | **Monty's way**; `MaxNodes` keeps counting iterations. |
| 8 | MLH head | **Dropped**. |
| 9 | Butterfly / policy temperature | **Off** by default until SPRT. |
| 10 | Data | `../2000000/*.pack` (YaneuraOu pack game records, 155 MB) for Phase 4. |
| 11 | Target machines | Ryzen 9 9955HX 32 GB (M profile fits); EPYC 9115 750 GB available briefly. |
| 12 | LICENSE file | Not now. |

Implementation deviations recorded during Phase 1: L1 and policy-readout
biases are i16 (Monty: i8) for headroom; positions without a king (test
sfens) use king square 0; SEE captures with the lowest-square attacker among
equal types, which is not frame invariant in rare double-attacker positions.

Measured on the dev machine (one thread, AVX2, random M nets, 8-ply random
walks): 172k value evals/s (5.8 µs, 14.5 group-A + 66 group-B rows per
eval) and 106k policy expansions/s (9.5 µs); scalar kernels 112k / 66k.
These beat the estimates in §5.5 by 2×, mostly because group-B rows are
fewer than budgeted and hardware prefetch handles the row streams well.

Phase 2 (search integration) measured with random M nets, `bench 20000`:
50k playouts/s on one thread, 175k on four, 281k on eight (the estimate in
§5.5 was 12k–30k per thread; the shallow mate probe at depth 5 costs about a
quarter of single-thread throughput). `MaxNodes` now defaults to 100,000,000
so timed searches are clock limited (JHBR3 defaulted to 800 for GPU
self-play). Per-thread boards use make/undo (§7.7); the value cache is the
8-byte Monty-style table (§7.5); `TreeMemoryMB` counts live nodes exactly
(§7.6) rather than per-search allocation.

## 0. Summary of the recommendation

| Topic | Recommendation |
|---|---|
| Value net | Three sparse groups → three L1 accumulators → pairwise multiply → 16 → 32 → 3 WDL, plus a per-feature PST skip to the logits. Groups: `A_us`, `A_them` (king-relative HalfKP-14, 189,864 inputs each, exactly 38 active, finny-cached) and `A_thr` (stm-frame piece slots + attacker→target threat pairs, 265,640 inputs, ~100 active, recomputed). L1 = 512 per group for the first nets ("S", 330 MB), 1024 as the target ("M", 660 MB, same as Monty). |
| Policy net | Monty layout, no shared trunk: 9,148 sparse inputs (piece-square × {plain, attacked, defended, both} + hand slots) → L1 4096 → pairwise → 2048 → one i8 row per **move bucket**. |
| Move buckets | (piece type, from, to) for board moves (8,115) + (type, from, to) promotions (1,397) + (hand type, to) drops (531) = **10,043**; doubled by shogi SEE → **20,086** rows. |
| Accumulators | Finny tables (option c) per thread for `A_us`/`A_them`; from-scratch (option a) for `A_thr` and for the policy. No accumulators in tree nodes (option d) except conceptually the root. Incremental-along-path (option b) rejected with numbers. |
| Search | Synchronous leaf evaluation, no leaf batching. Value on first visit, policy/expansion on second visit (Monty). Keep JHBR3 PUCT/FPU, position-mutex tree parallelism, shallow mate probe, root df-pn/BNS guard, repetition and declaration handling. Add butterfly history and depth/Q policy temperature as options, off by default until SPRT-tested. |
| Threads | `Threads` worker threads, each with its own board, finny cache and scratch; shared tree with JHBR3's hashed position mutexes; virtual loss only for multi-thread diversification (released at immediate backup). |
| Quantisation | L1 weights i8 (×128), accumulators i16, pairwise products i16, L2 i16 with a **calibrated** power-of-two scale, rest f32. Policy readout i8 (×128) with Monty's `/4` product shift. |
| SIMD | `nnue/simd.h` with `scalar`, `avx2` implementations and an `avx512` stub, chosen by `-DJHBR5_ISA=`. |
| Training | PyTorch, `nn.EmbeddingBag` sparse layers, feature/bucket mapping from the engine C++ via pybind11, λ-blend WDL loss, visit-distribution CE for policy, HalfKP factoriser, quantised export with C++ round-trip test. |
| Data | Fixed 40-byte PSV-compatible head + variable visit-distribution tail. Stage 0: PSV value pretraining. Stage 1: JHBR3 (GPU) distillation for policy. Stage 2: JHBR5 self-play. |

## 1. Scope and principles

Kept from JHBR3 unchanged: `shogi/` (board, bitboards, movegen, packed sfen,
mate-in-1), `mate/` (shallow mate, df-pn, BNS), `usi/` (time management, root
mate state, protocol), `book/`, repetition and entering-king handling in
`mcts/`, `tools/` strength harness, all tests that do not depend on the GPU
path.

Removed: `inference/nn_tensorrt.*`, `inference/nn_eval.*` (ONNX), `shogi/
encoder*` (148-plane encoder and CUDA unpack), `pyext/jhbr2_encoder_capi.cc`,
the ONNX/TensorRT Python scripts, the MLH (moves-left) head and its search
term (no value net output for it in v1), `batch_eval.cpp`, the JHBR2/dlshogi
model-format options.

Replaced: `inference/` becomes `nnue/`; `mcts/uct_search.cc`'s batching path
becomes synchronous.

Principles: (1) one implementation of every mapping (feature index, move
bucket, SEE) in C++, exposed to Python; (2) all sizes live in the weight-file
header so the engine can load "S" and "M" nets without recompiling; (3)
everything builds and passes on AVX2 with `cmake -S . -B build
-DJHBR5_ISA=avx2 && cmake --build build && ctest --test-dir build`.

## 2. Coordinates and perspective

JHBR3 conventions (`shogi/types.h`): `Square = file*9 + rank`, rank 0 is the
far rank for BLACK, `Flip(sq) = 80 - sq`, `ShogiBoard::Flipped()` rotates 180°
and swaps colours.

**Frame of a perspective `p ∈ {BLACK, WHITE}`**: if `p == WHITE` apply
`Flipped()`. In its own frame a player always moves "up" (toward rank 0), its
own pieces are `us`, the opponent's are `them`. All feature and bucket tables
are defined in the frame of the side to move, except the `A_them` group which
is defined in the opponent's frame (section 3.2).

There is **no horizontal mirroring**. Shogi's initial position is not
left/right symmetric (bishop and rook sides), so Monty's king-file mirror does
not apply.

## 3. Value network

### 3.1 Piece slots (the shogi "BonaPiece")

In a frame, every non-king piece occupies exactly one slot:

| Slot range | Meaning | Count |
|---|---|---|
| 0 .. 75 | hand: owner (us/them) × type (P,L,N,S,G,B,R) × count `k = 1..max_t` with max = (18,4,4,4,4,2,2) → 38 per owner | 76 |
| 76 .. 2343 | board: owner (2) × type id (14) × square (81) | 2268 |

Type ids: 0 = king, 1..7 = P L N S G B R, 8..13 = +P +L +N +S +B +R (JHBR3's
`PieceType::idx` minus one for the promoted range). Group A never emits the
king id (the king keys the table), so 162 of the 2268 slots are unused there;
the policy inputs (section 4.1) use the same table with the king id live.
Promoted minors are kept distinct from gold (YaneuraOu merges them into
gold); the extra slots are cheap and the distinction matters for captures
(a +P returns to hand as a pawn).

Hand counts use the thermometer encoding: `k` pieces in hand activate slots
`1..k`, so a capture changes exactly one slot. `NUM_SLOTS = 2344`. A position
activates exactly **38** slots per frame (every non-king piece, on board or
in hand).

### 3.2 Group A — king-relative slots (`A_us`, `A_them`)

`A_p[ksq_p * 2344 + slot_p]` where `ksq_p` is the king square of `p` in `p`'s
frame and `slot_p` the slot of the piece in `p`'s frame. Size per group
`81 × 2344 = 189,864`; active 38.

`A_us` uses `p = side to move`; `A_them` uses `p = opponent`. The two groups
have **separate weight tables** (unlike YaneuraOu, which shares the HalfKP
table between perspectives and only reorders the concatenation). Sharing is
possible and halves the table; we start separate because the "them" table
must learn "how good is this for the side *not* to move", which is a
different function once threats are involved, and because separate tables
keep the export trivial. Sharing is listed as an open question.

Comparison: YaneuraOu HalfKP is `81 × 1548 = 125,388` per perspective (9
board types with promoted minors merged as gold, 90 hand slots with unused
zero slots), 38 active, shared table; HalfKA1 (kings as pieces) is 138,510,
the half-mirror variants 69,660–76,950, and HalfKPE9 (9 attack-effect
buckets) 1,128,492 (`YANEURAOU_NNUE_NOTES.md` §2). Ours is 1.5× HalfKP only
because of the 14 board types; we do not need HalfKA's king-as-piece because
the other frame's table is keyed by that king, and we do not use mirroring.

### 3.3 Group B — threat pairs and absolute slots (`A_thr`)

Frame: side to move. Inputs:

1. **Absolute slots**: the 2344 slots above, not king-relative. Active 38.
   These carry the PST/material information that Monty puts in its 768
   piece-square inputs.
2. **Threat pairs**: for each attacker of owner `o ∈ {us, them}`, type `t`
   (14, king included), on `from`, and each square `to` in
   `attacks(t, o, from, occupancy)` that is **occupied**, a feature
   `(o, t, from, to, class(target))`.

   * `pair(o, t, from, to)` is a dense index over every geometrically
     possible (type, from, to) on an empty board in the owner's frame,
     computed with prefix-sum tables exactly like Monty's
     `ValueIndices/ValueAttacks` (`attacks.rs`). Count per owner:

     | type | pairs | type | pairs |
     |---|---|---|---|
     | P | 72 | +P +L +N +S | 416 each |
     | L | 324 | B | 816 |
     | N | 112 | +B | 1104 |
     | S | 328 | R | 1296 |
     | G | 416 | +R | 1552 |
     | K | 544 | | |

     Total **8,228** per owner (Monty: 39,928 per side, because chess sliders
     on 8×8 with queens produce more pairs and Monty keeps 6–12 target types).
   * `class(target)` = target owner (2) × type class (8): {P, L, N, S,
     G-like (G,+P,+L,+N,+S), B-like (B,+B), R-like (R,+R), K} → 16 classes.
     Monty keeps per-attacker subsets of 6–12; we use a uniform 16 to keep the
     index formula trivial.
   * Index = `2344 + ((o * 8228 + pair) * 16 + class)`. Threat inputs
     `2 × 8228 × 16 = 263,296`.

`A_thr` size `2344 + 263,296 = 265,640`. Active count is data dependent:
38 slots + number of (attacker, occupied target) pairs; in middlegame
positions from PSV data we expect 40–80 pairs, so ~80–120 active rows. Monty
budgets 128 active for its equivalent.

Unlike Monty, symmetric pairs (rook attacks rook) are not de-duplicated:
shogi has no colourless symmetry once promotion direction matters, and the
duplication costs nothing at inference (only table size).

### 3.4 Architecture and quantisation

```
A_us   (189,864 → L1)  i8 weights ×128, i16 bias ×128, i16 accumulator   [finny-cached]
A_them (189,864 → L1)  same                                              [finny-cached]
A_thr  (265,640 → L1)  same                                              [from scratch]
act_g[i] = clamp(acc_g[i], 0, 128) * clamp(acc_g[i + L1/2], 0, 128)      i16, ≤ 16384, for i < L1/2
h     = concat(act_us, act_them, act_thr)                                 3 × L1/2
L2:   h (3·L1/2) → 16   i16 weights × QB (calibrated), i32 accumulate, out = (acc/(128·128) + b)/QB  f32
L3:   16 → 32   f32, SCReLU on input
L4:   32 → 3    f32, SCReLU on input
PST:  Σ_active pst[feature] (3 × i16, ×256) added to the 3 logits
softmax → (W, D, L)
```

Sizes (i8 unless noted):

| Profile | L1 | A_us | A_them | A_thr | L2 (i16) | PST (i16) | total |
|---|---|---|---|---|---|---|---|
| S | 512 | 97 MB | 97 MB | 136 MB | 24 KB | 3.9 MB | **334 MB** |
| M | 1024 | 194 MB | 194 MB | 272 MB | 49 KB | 3.9 MB | **664 MB** |

Monty's value net is 660 MB (80,624 × 8192). YaneuraOu's common nets are
64 MB (HalfKP 256×2-32-32, i16 weights) to 257 MB (1024×2-8-32), and its
SFNN-1536 net already uses the same pairwise-product activation between the
two halves (`YANEURAOU_NNUE_NOTES.md` §3.3, §7.4). We are in Monty's memory
class but with a much smaller per-eval footprint (section 5). YaneuraOu's
i16 feature-transformer weights (`kWeightScaleBits = 6`, `FV_SCALE = 16`)
are the fallback if i8 L1 weights prove too coarse; the header's dtype field
allows both.

Overflow analysis. i16 accumulators: ≤ 120 active rows × 127 = 15,240 <
32,767 (bias adds ≤ 32,767/2 by construction: biases are clipped to ±64 in
training). Products ≤ 128·128 = 16,384. L2 i32: worst case `3·L1/2 × 16384 ×
|w|max`; with L1 = 1024 and QB = 256 (|w| ≤ 253) that is 6.4e9, above 2^31,
so QB is **calibrated at export**: the exporter measures the maximum |acc|
over a calibration set of 1M positions and picks the largest power of two QB
≤ 1024 such that `max|acc| × 8 < 2^31`; the header stores QB. A hard
guarantee (QB = 64 for L1 = 1024, QB = 128 for L1 = 512) is available as an
exporter flag. Monty (4096 × 16384 × 1014) has the same theoretical exposure
and runs with QB = 1024.

The trainer clips L1 weights to ±(127/128), L1 biases to ±64, L2 weights to
±(1 − 1/QB) (bullet/Monty do the same with ±0.99).

### 3.5 Justification of the sizes

* 81 king squares (no buckets): finny tables key on the exact king square,
  and YaneuraOu shows 81 is affordable for shogi with billions of positions.
  The data-hunger of `81 ×` is mitigated in the trainer by a factoriser
  (section 9.3), the standard nodchip remedy.
* L1 512/1024 instead of Monty's 8192: the dev machine has 256 KB L2 per
  core and ~35–40 GB/s DRAM; the per-eval byte budget (section 5.5) is what
  limits width, and the shogi active count (~150 rows across groups) is ~1.5×
  Monty's. 8192 would stream ~1.2 MB per eval. 1024 streams ~150 KB.
* Three groups instead of one: lets `A_us`/`A_them` be cached by king square
  while the attack-dependent group is recomputed. A single merged table would
  force a full recompute every eval.

## 4. Policy network

### 4.1 Inputs (frame: side to move)

| Range | Feature | Count |
|---|---|---|
| 0 .. 9071 | `sqf(o, t, sq) + 2268 × (attacked_by_them(sq) + 2 × defended_by_us(sq))`, `sqf = (o*14 + t)*81 + sq`, type ids as in 3.1 with the king id (0) live | 2268 × 4 = 9072 |
| 9072 .. 9147 | hand slots (76, as in 3.1) | 76 |

`INPUT_SIZE = 9148`. Active: one feature per piece on board (≤ 40, kings
included) + hand thermometer slots (≤ 38) → ≤ 78; typical ~55. Monty: 3072
inputs, ≤ 32 active.

`attacked_by_them(sq)` / `defended_by_us(sq)` are one bitboard each
(`AttackersTo` over all squares is too slow; we add a `ShogiBoard::
AttackedSquares(Color)` that ORs piece attacks — needed by the threat group
too).

### 4.2 Architecture

```
L1:  9148 → L1p (4096)    i8 ×128, i16 bias, i16 accumulator, from scratch per expansion
hl[i] = (clamp(a_i,0,128) * clamp(a_{i+L1p/2},0,128)) >> 2       i16 ≤ 4096, L1p/2 = 2048 entries
logit(m) = (Σ_i W[bucket(m)][i] * hl[i]) / (128·32) / 128 + b[bucket(m)] / 128     i8 rows, i32 dot
```

Sizes: L1 `9148 × 4096 = 37.5 MB`; readout `20,086 × 2048 = 41 MB`. Monty:
126 MB + 161 MB. Per expansion: ~55 rows × 4 KB = 220 KB for `hl`, then 2 KB
per legal move (shogi averages ~80 legal moves in the middlegame, max 593) →
160 KB. So an expansion (~380 KB) costs more than a value eval; this is why
expansion is deferred to the second visit (section 7.2). The i32 dot cannot
overflow: `2048 × 4096 × 127 = 1.07e9 < 2^31`.

`L1p` is a header parameter; 2048 is the "S" profile (19 MB + 21 MB).

### 4.3 Move bucket table

Frame: side to move. `bucket(m) ∈ [0, 10043)`:

| Kind | Index | Count |
|---|---|---|
| Board, no promotion | `OFF_PLAIN[t][from] + rank_of(to in DEST_PLAIN[t][from])` where `DEST_PLAIN` is the empty-board reach of type `t` from `from`, minus squares on which the piece could not legally stand unpromoted (P/L on rank 0, N on ranks 0–1); `from` ranges over squares the piece can stand on | 8,115 |
| Board, promotion | `OFF_PROMO[t][from] + rank_of(to in DEST_PROMO[t][from])`, `t ∈ {P,L,N,S,B,R}`, `to` restricted to moves where `from` or `to` is in the promotion zone | 1,397 |
| Drop | `OFF_DROP[t] + drop_rank(to)`; P/L exclude rank 0 (72), N excludes ranks 0–1 (63), S/G/B/R all 81 | 531 |

Per-type counts (plain / promo): P 63/27, L 252/189, N 80/48, S 328/123,
G 416/0, K 544/0, B 816/416, R 1296/594, +P +L +N +S 416/0, +B 1104/0,
+R 1552/0. Board total 9,512; with drops **10,043**. `rank_of` is
`popcount(DEST & below(to))` on the 81-bit JHBR3 bitboard, as in Monty's
`map_move_to_index`.

**SEE doubling**: `bucket_final = 10043 × good_see + bucket`, `good_see =
see_ge(pos, m, -90)` (a pawn is 90 in YaneuraOu's scale). Rows: **20,086**.
Applied to *all* moves including drops and quiet moves (a drop onto an
attacked, undefended square is "bad SEE"). `see_ge` is a new
`ShogiBoard::SeeGe(Move, int threshold)` re-implemented from the YaneuraOu
algorithm in JHBR3's board API (drops capture nothing and put the dropped
piece at risk; promotion adds `value(promoted) - value(base)`; captured
pieces count at their board value). Pins are handled the simple way
(YaneuraOu ignores most pin cases; Monty's near-fully-legal SEE is not
needed for a bucket selector). Whether SEE doubling is on is a
`bucket_table_version` field in the policy header, so a net trained without
it still loads.

Comparison: JHBR3's current policy head is dlshogi's 27 × 81 = 2187
(direction, to) labels; Monty's is 7,840. Ours is 20,086 rows of 2 KB; the
table is 41 MB and only the rows of legal moves are touched.

### 4.4 Softmax temperature and butterfly history

At expansion, for each legal move: `logit += butterfly[stm][from_id][to] /
ButterflyDivisor`, with `from_id ∈ [0, 88)` = from square for board moves,
`81 + hand type` for drops. Table `2 × 88 × 81` atomic i16. Update after each
completed iteration through a non-terminal child with Monty's gravity rule
(`MONTY_NOTES.md` 4.7).

Temperature `T(depth, q)` exactly as Monty's `get_pst` with the same four
parameters exposed as USI options; `p_i ∝ exp((logit_i − max)/T)`. Both are
behind `UsePolicyTemperature` and `UseButterfly` (default **off**); JHBR3's
PUCT constants were tuned for a T = 1 policy and the two features must be
SPRT-tested (Phase 5) before becoming defaults. Note that expansion-on-second-
visit makes the node's `q` available at expansion time, which `T(depth, q)`
requires.

## 5. Accumulator strategy for MCTS

### 5.1 Cost model

Per-row cost is one L1-wide i8 load + i16 add: at L1 = 1024 that is 1 KB
streamed and 32 AVX2 ops. Random 1 KB rows from a 660 MB table are DRAM
misses: ~80–100 ns each on the dev machine if the hardware prefetcher keeps
several in flight (we issue explicit `_mm_prefetch` for the next 4 rows). So
**≈ 0.1 µs per row** is the unit; a value eval with R rows costs ~0.1·R µs
plus ~1 µs for the dense tail.

### 5.2 Options

**(a) Recompute from scratch (Monty).** Rows per eval: 38 + 38 + ~100 = ~176
→ ~18 µs. Simple, order independent, no state.

**(b) Incremental along the selection path.** Each `DoMove` changes ≤ 3 slots
(moved piece from/to, captured piece to hand slot) in each of two frames,
plus a full 38-row refresh of a frame whenever that frame's king moves.
Rows per ply ≈ 8; the average selection depth in JHBR3 at 10k–100k nodes is
~15–25 plies, so 120–200 rows per iteration *before* the threat group, i.e.
no better than (a), and it needs a per-ply accumulator stack (2 × 2 KB per
ply) and a king-move detector. It only pays off if interior nodes needed
accumulators, which they do not (expansion policy is a different net).
Rejected.

**(c) Finny tables.** Per thread, per frame `p`, per king square `k`: a cached
accumulator plus the board state it was built from:

```
struct AccCacheEntry {
  alignas(64) int16_t acc[L1];            // 2 KB at L1 = 1024
  Bitboard  pieces[2][14];                // owner × type, in frame p (28 × 16 B = 448 B)
  Hand      hands[2];                     // 8 B
};
AccCacheEntry cache[2][81];               // ≈ 400 KB per thread at L1 = 1024
```

Refresh of frame `p` at eval time: `e = cache[p][ksq_p]`; for each (owner,
type) `d = e.pieces ^ cur.pieces`; subtract rows for `d & e.pieces`, add rows
for `d & cur.pieces`; for hands, add/sub thermometer slots between old and
new counts; then `e.pieces = cur.pieces`, `e.hands = cur.hands`. Entries
start as the empty board (acc = bias, no pieces), so first use is a plain
refresh. This is Stockfish's `AccumulatorCaches` mechanism. YaneuraOu carries
its own variant behind `USE_FINNY_TABLES` (`nnue_feature_transformer.h`,
keyed per trigger × perspective × king square, storing the last *active-index
list* and diffing index lists) but no build enables it
(`YANEURAOU_NNUE_NOTES.md` §4). We take the Stockfish bitboard-snapshot form:
diffing 28 bitboards is cheaper and branch-free compared with merging two
sorted index lists, and it needs no per-entry index storage.

Why it suits MCTS: it needs no make/unmake ordering, it is keyed only by the
current position, and MCTS leaves are spatially coherent — consecutive leaf
evaluations in one thread are usually a few pieces apart from the previous
leaf with the same king square. Expected rows per frame: 2–10 (a king move
lands on an entry that is stale by however long ago that king square was
last seen; worst case 76 rows, same as (a)). The diff itself is 28 128-bit
XORs plus popcounts, ~30 ns.

Rows per eval: ~5 + ~5 (groups A) + ~100 (group B, from scratch) → ~110 →
~11 µs. The threat group dominates; that is inherent to threat features
(Monty pays the same) and is the price of the feature set, not of the caching
scheme.

**(d) Accumulators in tree nodes.** 3 × L1 × 2 B = 6 KB per node at L1 =
1024 (2 KB per cached group only, 4 KB, if the threat group is excluded).
JHBR3 allocates ~100 B per unexpanded node and ~2.6 KB per expanded node (80
children × 32 B). A 10-second search on 8 threads at ~300k evals/s creates
~3M nodes → 12–18 GB. Rejected. The one node that benefits is the root
(evaluated at every tree-reuse), and it does not need special handling.

### 5.3 Recommendation

(c) for `A_us` and `A_them`; (a) for `A_thr` and for the policy L1. The
policy L1 could also be finny-cached on its piece-square part only, but the
attacked/defended flags change with almost every move, so the diff would be
large; it stays from-scratch and is amortised by expansion-on-second-visit.

### 5.4 Interaction with threads and tree reuse

Caches are per thread and never shared, so no locking. They persist across
searches and games (a stale entry is just a larger diff), and are reset only
when a new network is loaded.

### 5.5 Expected throughput (dev machine, single thread)

| Component | Cost |
|---|---|
| Value eval (finny + threats, L1 = 1024) | ~11 µs |
| Policy expansion (L1p = 4096, 80 moves) | ~40 µs, on ~40% of evals → ~16 µs amortised |
| Movegen + shallow mate probe depth 5 | 2–50 µs (mate probe dominates in tactical positions) |
| Tree ops, repetition check, board make/undo | ~2 µs |

≈ 30–80 µs per iteration → 12k–30k iterations/s/thread; 8 threads at ~80%
scaling → 80k–200k/s. JHBR3 on 2× RTX 3090 targets 60k nps. The **"S"**
profile roughly halves the network cost. These are estimates; Phase 1's
`bench` measures them and the widths are header parameters precisely so we
can trade net size for speed without code changes.

## 6. Inference engine

### 6.1 Layout

```
nnue/
  simd.h            ISA wrappers: add_rows_i8_to_i16, sub_rows, pairwise_mul, dot_i16_i16, dot_i16_i8,
                    prefetch; scalar/, avx2/, avx512/ (stub) selected by JHBR5_ISA
  features.h/.cc    slots, group A/B index functions, policy inputs, attack bitboards, active-feature lists
  move_buckets.h/.cc bucket tables (constexpr), bucket(m), see-aware wrapper
  see.cc            ShogiBoard::SeeGe
  net_format.h      header struct, tensor table, checksum
  value_net.h/.cc   ValueNet (weights), ValueScratch (per thread: finny cache + buffers)
  policy_net.h/.cc  PolicyNet, PolicyScratch
  evaluator.h       Evaluator interface used by search
```

### 6.2 Evaluator interface

```cpp
struct WDL { float win, draw, loss; };
class Evaluator {                       // one per thread; holds ValueScratch + PolicyScratch, refs to shared nets
 public:
  WDL  Evaluate(const ShogiBoard& pos);                                  // value net
  void Policy(const ShogiBoard& pos, const MoveList& moves, float* logits);  // policy net, logits[i] for moves[i]
  void Reset();                                                           // clear finny caches
};
```

Nets are loaded once (`NetworkSet::Load(value_path, policy_path)`), shared
read-only by all threads (mmap or aligned heap copy).

### 6.3 Weight file format

```
struct NetHeader {              // 256 bytes, little endian
  char     magic[8]   = "JHBR5NN\0";
  uint32_t version    = 1;
  uint32_t kind;                // 1 = value, 2 = policy
  uint32_t feature_set_id;      // hash of the C++ mapping tables, computed at build time
  uint32_t bucket_table_id;     // policy only: hash incl. SEE flag; 0 for value
  uint32_t l1[4];               // per-group L1 width (value: 3 groups; policy: 1)
  uint32_t l2, l3;              // value: 16, 32
  uint32_t qa, qb, q_pst;       // 128, calibrated QB, 256
  uint32_t n_tensors;
  uint64_t payload_bytes;
  uint32_t payload_crc32c;
  uint32_t header_crc32c;
  uint8_t  reserved[...];
};
struct TensorDesc { char name[16]; uint32_t dtype; uint32_t rank; uint64_t shape[3]; uint64_t offset; };
```

Tensors follow at 64-byte-aligned offsets. The loader rejects a file whose
`feature_set_id`/`bucket_table_id` differs from the compiled tables — this is
the mechanism that guarantees trainer and engine agree.

### 6.4 SIMD abstraction

`simd.h` exposes fixed-width primitives on `int16_t*`/`int8_t*` spans whose
length is a multiple of 32; the AVX2 file implements them with
`_mm256_cvtepi8_epi16`/`_mm256_add_epi16`/`_mm256_max_epi16`/`_mm256_min_epi16`
/`_mm256_mullo_epi16`/`_mm256_madd_epi16`. The scalar file is the reference.
The AVX-512 file is a copy of the AVX2 file with `TODO(avx512)` bodies that
`static_assert` when `JHBR5_ISA=avx512` is selected, so nobody can silently
ship an untested kernel. Tests: `test_nnue_kernels` compares avx2 and scalar
bit-for-bit on random inputs.

## 7. Search changes

### 7.1 Synchronous evaluation

`UCTSearcher` loses `batch_`, `QueuingNode`, `EvalNode`, `visitor_pool`,
`batch_max_`, `PlayoutStatus::kQueuing`. An iteration is: descend, evaluate
at the leaf, back up, all in one call. `kDiscarded` remains for the race where
another thread expanded the node first (the iteration is simply restarted).

Batching was reconsidered for CPU: the only potential gain is prefetching
rows for leaf `i+1` while computing leaf `i`, which we get more simply with
`_mm_prefetch` inside one eval; and batching would keep the virtual-loss
distortion and the deferred backup. Not worth it.

### 7.2 Node state machine (Monty's first/second-visit split)

```
kFresh      --first visit-->  terminal checks (repetition, declaration, ply cap, shallow mate, no legal moves)
                              value = Evaluate(pos) (or eval-cache hit)   -> kEvaluated, back up
kEvaluated  --second visit--> under position mutex: movegen, Policy(), allocate children with priors -> kExpanded
                              then continue the same iteration: PUCT select, descend
kExpanded   ------------->    PUCT select, descend
```

JHBR3's `move_count == kNotExpanded` sentinel becomes an explicit `state`
atomic (`kFresh/kEvaluated/kExpanded/kTerminal`). `MaxNodes` keeps counting
iterations (playouts), as now.

Terminal detection at the first visit needs "does the side to move have a
legal move"; JHBR3's `GenerateEvasionMoves(stop_after_one)` answers it in
check, and out of check a full `GenerateLegalMoves` (~1–2 µs) is used and
discarded. The shallow mate probe (`HasMateWithin`) stays at the first visit
so a position that is mate-in-N for the mover is never evaluated by the net.

### 7.3 What stays

PUCT (`SelectPuctChild` with `CInit/CBase/FpuReduction` root and non-root),
`BackupTrajectory`, proven win/loss/draw propagation, virtual loss add/sub
(now spanning only the descent of one iteration), position mutexes,
`RejectRootMates` with BNS/df-pn, `search_repetition.h`, declaration win,
`MaxMovesToDraw`, time management (`AdaptiveTimeController` unchanged: it
consumes root snapshots), tree reuse (`NodeTree::ResetToPosition`), info
output, PV.

The moves-left utility is removed from `PuctParameters` (no MLH output).

### 7.4 Value post-processing

Backup value is `W + 0.5·D` from the leaf's side to move, exactly JHBR3's
`(value + 1)/2` with `value = W − L`. Terminal draws use `DrawValueBlack/
White` as today. Monty's draw inflation ("sharpness") is exposed as
`DrawScale`/`DrawQuadratic` (default 0 = off); contempt is not carried over.

### 7.5 Evaluation cache

Replace `inference/nn_cache.h` (which stores full policy vectors) with a
Monty-style 16-byte `{key32, w16, d16, visits32}` table keyed by
`MakeNNCacheKey(hash, is_repetition)`; hit ⇒ skip the value net. Policy is
never cached. Option `EvalCacheMB` (default 64). Cleared on `isready` as now.

### 7.6 Tree memory

`TreeMemoryMB` (default 4096): the tree counts bytes allocated for nodes and
child arrays; when exceeded the search stops (reported as
`TimeStopReason::kTreeFull`). Shrinking `child_node_t` (move u16, prior u16
(policy as 1/65535 like Monty), visits i32, win f32 → 12 B) is a follow-up.

### 7.7 Board handling per iteration

JHBR3 copies `root_board_` (including its heap `history_` vector) for every
playout. At 20k+ iterations/s/thread that allocation shows up; each thread
keeps one board, descends with `DoMove` and returns with `UndoMove` (JHBR3's
`UndoInfo` already supports this; the mate solvers use it).

### 7.8 Optional Monty search features (off by default, USI-switchable)

Butterfly history, policy temperature (4.4), `DrawScale/DrawQuadratic`.
Monty's CPUCT visit/variance scaling and gini exploration are **not** ported
in Phase 2: they replace the PUCT formula JHBR3's SPSA tooling was built
around; they can be evaluated later as a separate experiment.

## 8. Multithreading

* `Threads` (default: physical cores) `UCTSearcher` threads; the
  `UCTSearcherGroup`/GPU/slot structure is deleted. Each thread owns:
  `ShogiBoard`, `Evaluator` (finny caches ≈ 0.4 MB, policy scratch ≈ 20 KB
  + logits), `MoveList`s, RNG.
* Shared: tree, eval cache, butterfly table (atomic i16), `playout_count_`,
  stop flags, time controller (already lock-free with `time_check_busy_`).
* Locks: the 65,536 hashed position mutexes protect expansion (movegen +
  policy + child allocation, now ~40 µs instead of ~2 µs). Contention is low
  because expansions of the *same* position by two threads are rare and the
  discard path handles it. Node statistics remain atomics.
* Virtual loss: JHBR3's `+1 move_count` during descent stays; it only exists
  to steer other threads away for the ~50 µs an iteration takes. Monty's
  "threads in subtree" Q-scaling is an alternative if scaling tests show root
  collisions.
* Root contention: Monty's per-thread root accumulator flush is a known fix
  if 8–16 threads show `fetch_add` contention on root children; deferred.
* Datagen (section 10.3) runs independent games per thread with 1 search
  thread each, so it scales linearly and needs no tree sharing.

Changes vs JHBR3: no per-GPU groups, no minibatch, no pinned buffers, no
`in_flight_playouts_` for time management (set to `Threads`).

## 9. Training pipeline (Phase 3)

### 9.1 Bindings

`pyext/jhbr5_py.cc` (pybind11) exposes: `value_features(sfen|packed) ->
(idx_us, idx_them, idx_thr)`, `policy_features(...) -> idx`, `move_bucket(
pos, move16) -> int`, `see_ge`, `legal_moves`, `feature_set_id`,
`bucket_table_id`, and a `RecordReader` that decodes a shard file into padded
numpy batches (indices, targets, masks) entirely in C++. The Python trainer
contains no index arithmetic.

### 9.2 Model

```python
class SparseGroup(nn.Module):        # EmbeddingBag(num_inputs, l1, mode="sum") + bias; clipped to [0,1] then pairwise
class ValueNet(nn.Module):           # 3 SparseGroups -> concat -> Linear(3*l1/2,16) -> SCReLU -> Linear(16,32) -> SCReLU -> Linear(32,3) + PST(EmbeddingBag(num_inputs,3))
class PolicyNet(nn.Module):          # SparseGroup(9148, l1p) -> pairwise -> gather rows W[buckets] -> logits, masked softmax over legal moves
```

Weight clipping hooks enforce the quantisation ranges (3.4). Policy logits
are computed as a gathered batched matmul (`W[buckets]` in bf16, batch ≤
4096, moves padded to the batch max ≤ 593) — a Triton kernel like Monty's
`SelectAffine` is a follow-up if memory limits the batch size.

### 9.3 Factoriser for group A

Training-only virtual inputs whose weights are folded into the real tables
at export (nodchip's trick): `P` (slot only, 2344) and `KPrel` (piece slot ×
king-relative offset, 2344 × 289 … capped by dropping the king square to a
17×17 offset). Every real feature `A_p[k][s]` receives `W_P[s] + W_KPrel[s,
rel(k)]` added to its own row. This is what makes 81 king squares trainable
on ~10^8 positions rather than 10^9.

### 9.4 Losses

* Value: `target = λ·wdl_from_score(s) + (1 − λ)·onehot(result)`,
  `wdl_from_score` = `w = σ((s − δ)/κ)`, `l = σ((−s − δ)/κ)`, `d = 1 − w − l`
  with `κ, δ` fitted once on the data (defaults κ = 600 cp, YaneuraOu's
  convention); loss = cross-entropy against the 3-way softmax. λ configurable
  per stage (0.7 for PSV pretraining, 0.5 for self-play).
* Policy: cross-entropy between the softmax over legal-move logits and the
  normalised visit distribution. Positions without a distribution (PSV) are
  skipped by the policy trainer.
* One trainer script, `train/train.py --net value|policy`, shared loader and
  export.

### 9.5 Export and round-trip

`train/export.py` rounds to the integer types of 3.4/4.2, calibrates QB,
writes the `JHBR5NN` file. `test_net_roundtrip` (C++) loads it and compares
`Evaluate`/`Policy` on 256 fixed positions against outputs dumped by PyTorch
with the same quantised weights (tolerance: WDL ±0.005, policy logits ±0.05).
A second check compares the quantised C++ outputs to the *float* PyTorch
outputs to report quantisation loss.

### 9.6 Data loading at scale

Shards of ≤ 1 GB; a manifest lists shards; each `DataLoader` worker owns a
`RecordReader` over a subset of shards and a 1M-record shuffle buffer;
epochs are shard-permuted. The C++ reader produces batches at > 1M
positions/s/core (feature extraction is the same code as the engine's).

## 10. Data pipeline (Phase 4)

### 10.1 Record format (`docs/DATA_FORMAT.md` will be the normative spec)

```
struct RecordHead {                   // 44 bytes; first 40 bytes == YaneuraOu PackedSfenValue
  uint8_t  sfen[32];                  // packed sfen (JHBR3 shogi/packed_sfen)
  int16_t  score;                     // search score, cp, side to move
  uint16_t move;                      // best/played move, JHBR3 Move16
  uint16_t game_ply;
  int8_t   result;                    // +1 win / 0 draw / -1 loss, side to move
  uint8_t  flags;                     // bit0: has distribution, bit1: teacher=JHBR3, bit2: self-play, bit3: resign-adjudicated ...
  uint16_t n_dist;                    // number of (move16, visits) pairs that follow
  uint16_t reserved;
};
struct DistEntry { uint16_t move; uint16_t visits; };   // visits scaled so max = 65535
```

The distribution stores **move16, not bucket id**: buckets depend on the
SEE flag and table version, moves do not; the loader maps move → bucket with
the engine code. This deviates from the task text and is listed as a
decision to confirm. A record with `n_dist = 0` is a PSV record plus 4
bytes, so importing YaneuraOu PSV is a trivial re-framing.

### 10.2 Warm start

`tools/teacher_jhbr3.py`: drives JHBR3 (GPU) over positions sampled from PSV
shards and from JHBR3 self-play, `go nodes N`, and records root Q (→ score)
and the root visit distribution. JHBR3 does not print its root distribution
today; a **small patch to JHBR3** (an `info string rootdist m:v m:v …` line
behind an option) is the simplest route and is an open question below.

`tools/import_psv.py` re-frames PSV into JHBR5 records (value only).

### 10.3 Self-play datagen (`jhbr5 datagen …`)

In-engine, `--threads T` independent games per thread (Monty's design;
"multi-process" in the task is satisfied by threads plus a driver script that
can also launch several processes for NUMA hosts). Per game: opening from a
book file or `R ~ U[0,16]` uniformly random legal plies; per move `go nodes
N` with KLD-gain early stop; root Dirichlet noise (α = 0.15, ε = 0.25);
move sampled ∝ visits^(1/τ) with τ from 1.0 decaying ×0.9 per move to 0.2
then greedy; resign when root Q < 0.03 for 8 consecutive plies with 10% of
games exempt (to measure false resignations); draws by sennichite rules,
declaration, and a 320-ply cap; output records with distributions.

### 10.4 Loop

`tools/pipeline.py`: generate → train → export → test (round-trip, bench,
SPRT at equal nodes vs the incumbent) → promote (copy into `nets/` and
update `nets/manifest.json`). Fully reproducible from a config file and seed.

### 10.5 Data plan

| Stage | Source | Positions | Purpose |
|---|---|---|---|
| 0 | existing PSV (YaneuraOu teachers) | ≥ 200M | value pretraining, λ = 0.7 |
| 1 | JHBR3 teacher over PSV samples + JHBR3 self-play | 20–50M with distributions | policy warm start + value fine-tune |
| 2+ | JHBR5 self-play, generations of 2–5M positions, replay window of the last ~5 generations | rolling | reinforcement |

Throughput estimate for stage 1: JHBR3 at 1000 nodes/position on 8 × RTX
5090 ≈ 400 positions/s ≈ 35M/day.

## 11. Testing (Phase 5)

* `tools/run_strength_test.sh` / `strength_test.py` already handle two USI
  engines with option lists; JHBR5 adds `--engine-option Threads=…` presets
  and three wrappers: `vs_jhbr3_equal_nodes.sh`, `vs_yaneuraou_equal_time.sh`
  (same threads, same clock; JHBR5 `Threads=n` vs YaneuraOu `Threads=n`),
  `sprt.sh` (JHBR5 vs JHBR5).
* `tools/sprt.py`: pentanomial GSPRT (pairs), bounds `[0, 5]` Elo for
  feature tests, `[-5, 0]` for non-regression, wired into `strength_test.py`
  as a stop condition.
* `bench` USI command: fixed 32 positions × N nodes, prints nodes/s,
  evals/s, expansions/s, cache hit rate; `bench eval` prints pure evals/s
  and policy expansions/s; `tools/thread_scaling.sh` runs 1/2/4/8 threads.

## 12. Build

`CMakeLists.txt`: `option(JHBR5_ISA "scalar|avx2|avx512" avx2)`; sets
`-mavx2 -mfma -mbmi2` or `-mavx512bw -mavx512vnni`, and `-DJHBR5_ISA_*`.
Targets: `jhbr5`, `nnue` (static), `jhbr5_py` (pybind11, optional
`-DJHBR5_BUILD_PY=ON`), tests via `ctest`. `-march=native` is dropped in
favour of the explicit ISA flag so the AVX-512 machine's binary is
reproducible.

## 13. Risks

1. **Bandwidth** on the dev machine: if 11 µs/eval is optimistic, the "S"
   profile and a 4-class target compression (halving `A_thr`) are the knobs.
2. **Data hunger** of 81-king-square tables: factoriser + PSV volume.
3. **Shallow mate probe** cost relative to a 10 µs eval: `LeafMateDepth` may
   need to drop to 3; measure in `bench`.
4. **Policy per-move cost** at 593 legal moves (2 KB × 593 = 1.2 MB): rare,
   accepted.
5. **JHBR3 patch** for teacher distributions: minor, but touches the
   frozen engine.
6. **Deferred expansion** subtly changes `MaxNodes`-limited strength
   comparisons vs JHBR3 (fewer expansions per iteration); equal-nodes tests
   should be read with that in mind.

## 14. Open questions for review

1. Share the HalfKP table between `A_us` and `A_them` (YaneuraOu style, halves
   the value net) or keep two tables (proposed)?
2. Start with the "S" profile (L1 512 / L1p 2048) or go straight to "M"?
3. Threat pairs in v1 (proposed) or ship v1 with groups A + absolute slots
   only and add threats as v2 once the pipeline works end to end?
4. SEE doubling of the bucket table on from the start (proposed) or off?
5. Records store `move16` rather than bucket ids (proposed) — acceptable?
6. May I add an opt-in `info string rootdist` output to JHBR3 for the teacher
   tool, or do you prefer a Python driver that reconstructs distributions
   from `MultiPV` output / another route?
7. Expansion on the second visit (Monty) — OK with the change to node
   accounting?
8. Drop the MLH head entirely in v1 (proposed)?
9. Butterfly and policy temperature default **off** until SPRT (proposed).
10. Which PSV corpora are available for stage 0 (paths, sizes, teacher
    depths)? Any ban on using YaneuraOu-generated data for the final nets?
11. Target AVX-512 machine: RAM and core count, to size the "M"/"L" profiles.
12. Licence: JHBR3 has no top-level LICENSE file although its headers say
    GPL-3.0; shall I add `LICENSE` (GPL-3.0) to JHBR5?

## 15. Decisions in this document you might disagree with

* Three separate value input groups instead of Monty's single table.
* 14 distinct board types (promoted minors ≠ gold) and compact 76-slot hands.
* Uniform 16 target classes for threats (no per-attacker subsets).
* No horizontal mirroring; full 81 king squares; two frames with separate
  tables.
* i8 L1 weights (Monty) rather than i16 (YaneuraOu).
* Calibrated QB instead of a fixed constant.
* Keeping JHBR3's PUCT rather than Monty's; Monty extras off by default.
* Policy expansion deferred to the second visit.
* Value-only eval cache; policy never cached.
* `move16` in records; in-engine threaded datagen rather than processes.
* Removing the MLH head and all TensorRT/ONNX/encoder code in Phase 1.
