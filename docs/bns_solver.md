# BNS (Branch Number Search) mate solver

`mate/bns.{h,cc}` implements 岡部文洋「経路分枝数を用いた詰め将棋解図について」
(Fumihiro Okabe, "Application for solving tsume shogi problem by route
branch number") as a second mate solver beside `MateDfpnSolver`, sharing
all of JHBR3's board, move-generation, hashing, and repetition machinery.

## Algorithm

Route branch numbers replace df-pn's proof/disproof numbers. Where df-pn
sums a value over siblings, BNS counts unresolved siblings:

```
unresolved leaf:  abn = 1,   obn = 1
mate proved:      abn = 0,   obn = INF
no-mate proved:   abn = INF, obn = 0
OR  (attacker, k active children, best = first min abn):
    abn = best.abn,   obn = best.obn + (k-1)
AND (defender, k active children, best = first min obn):
    obn = best.obn,   abn = best.abn + (k-1)
child thresholds (c2 = second-smallest relevant number, multiset):
    OR:  ABN' = min(abn(c2)+1, ABN),  OBN' = OBN - k + 1
    AND: ABN' = ABN - k + 1,          OBN' = min(obn(c2)+1, OBN)
iterate while ABN > abn && OBN > obn; root starts at (INF, INF)
```

All formulas were checked against the paper (§3–5.1); the arithmetic is
in pure inline functions (`bns::Summarize`, `bns::ChildThresholds`) unit
tested on hand-built AND/OR graphs in `test/test_bns.cc`. The engine is
shared between BNS and a pn/dn control mode (`Arith::kPnDn`) that differs
*only* in the sibling aggregation — the paper's own comparison
methodology.

## Search architecture

TT-based recursive threshold iteration in the shape of cshogi/dlshogi's
`dfpn_inner` (cshogi `src/dfpn.cpp`, GPLv3, adapted not copied), because
BNS is designed for merged search graphs. Per node visit: probe own
entry, mate-in-1 shortcut on first visit, generate checks/evasions,
compute child hashes once (`ShogiBoard::HashAfter`, added for this), then
iterate summarize → store → threshold-check → descend.

### Transposition table

Fixed-size 2-way clustered table with 16-byte hot entries and
generation-based lazy clearing, keyed by **(position hash, ply from
root)**. Each entry is a 64-bit mixed fingerprint/generation tag plus
the two 32-bit search numbers. The experimental paper-GHI effort value
lives in a cold sidecar and is not written by the normal solver.

The ply in the key is essential, not cosmetic. With hash-only keying the
search **livelocks**: a position's entry written deep inside its own
subtree is later read as a direct child's view (a transposition at a
different depth), creating circular value dependencies; the threshold
iteration then enters a limit cycle (observed: root values frozen from
1k to 5M nodes on mate5 problems). Depth-keying layers the graph
acyclically, as in cshogi/dlshogi. The paper describes the same
phenomenon (§5.2, 展開履歴依存ループ) — see the experimental mode below.

### Repetitions and GHI

Rule-aware repetition uses the host's `CheckRepetition` (16-ply window,
kWin/kLoss/kDraw with continuous-check counters; same mapping as
`MateDfpnSolver`), plus a full-path hash scan for longer cycles
(conservative no-mate, as in df-pn).

Route-dependent verdicts never enter the TT. They live in a path-scoped
override list: each override is anchored at the ancestor occurrence that
makes it true and is dropped when the search unwinds past that ancestor.
Verdicts derived from overrides are themselves route-dependent — with
one exception (the **loop-head rule**): a dependency anchored at the
node itself dissolves there, because a cycle back to a position exists
below every occurrence of that position; the verdict becomes
unconditional and is stored normally. Resource-limited verdicts (depth
cap) never dissolve and can never surface as a root "no mate": the root
answer degrades to UNSOLVED instead.

### Experimental paper-GHI mode (`set_paper_ghi_mode`)

Follows §5.2 literally: hash-only keying (maximum sharing), Figure-3
on-path marking (while a node is on the path its entry shows
`abn = child threshold, obn = 1`, so attackers avoid and defenders seek
the cycle), and a Figure-4-style escape that widens a hammered node's
child thresholds to the parent's window (sequential expansion; the paper
triggers on accumulated subtree depth > 1600, we trigger on entry effort
— a documented deviation). Correct on all tested sets and competitive
with the default mode; kept experimental because, unlike depth-keying,
it has no convergence argument, only the escape heuristic.

## Interfaces

`MateBnsSolver` mirrors `MateDfpnSolver`: `search(board, nodes_limit,
deadline)` returning the mating move / `NoMateMove()` / null, `get_pv()`
(attacker shortest, defender longest, validated), `get_mate_ply()`,
sticky `stop()`. PV extraction walks the TT (cshogi `get_pv` style) with
cycle avoidance and bounded re-search when proof chains were evicted.

## Measured optimization results (mate3–mate11 generated sets)

Kept (defaults):
- mate-in-1 shortcut at fresh OR nodes; fast `GenerateCheckingMoves`.
- TT prefetch of child clusters; 48-bit key verification.
- Incremental sibling refresh: with depth-keyed entries, sibling views
  are frozen while the search is inside one child, so re-summarize
  re-probes only the returned child unless the override list changed
  (~3% total).
- `DfpnNodePool::Alloc` reuse in the baseline df-pn (allocation
  amortization; behavior identical).

Negative results (implemented, measured, default **off**):
- `set_use_mate3_probe`: 3-ply probe at fresh OR nodes — ~2x slower
  overall; JHBR3's `MateIn3Ply` costs more than the iteration it saves.
- `set_use_new_node_block`: full cshogi-style new-node block (defense
  collapse detection, grandchild mate-1 proofs, evasion-count child
  inits) — ~2x slower despite persisting all byproducts.
- `set_use_dominance`: hand-dominance finals table (優越関係;
  board-only key + attacker hand, mate valid for dominating hands,
  no-mate for dominated) — +15-20% time on these sets; the per-child
  probe overhead exceeds the reuse. Sound (uses the new
  `Hand::Dominates`, `ShogiBoard::BoardKey/BoardKeyAfter`).
- TT victim policy (Small-Tree-GC-style effort priority vs first-slot
  replacement): effort retention was slower under the final small-table
  configuration, so first-slot replacement is the normal default. The
  effort policy remains available for paper-GHI experiments.

These knobs remain available for longer/composed problems, where the
trade-offs may reverse.

## Board-primitive optimization (stage 2 of the project)

Measured head-to-head against cshogi's primitives on identical
positions, JHBR3's board core was already at parity (DoMove 28 vs 27 ns;
Qugiy sliders 7.0 vs 5.6 ns) — the mate-solver gap lived in three
specialized routines. All three were rewritten with full oracle
validation (39k-position movegen tests, 61k-position mate-1 soundness):

| primitive | before | after | cshogi |
|---|---|---|---|
| mate-in-1 probe | 830 ns (complete) | 133 ns (`FindMateInOneApprox`) | 63 ns |
| evasion generation | 354 ns | 219 ns | 132 ns |
| checking-move generation | 269 ns | 194 ns | 69 ns |

- `shogi/mate1ply.cc`: Apery/cshogi `mateMoveIn1Ply` port — sound,
  deliberately incomplete (near-king mates only, 96.6% coverage of
  exact mates); the search absorbs misses one expansion later. The
  solvers use it at non-check nodes; the complete routine remains for
  in-check nodes, PV extraction, and as the test oracle.
- Evasions: non-king evasions found by enumerating our attackers of
  each capture/interpose square (reverse lookup) instead of computing
  attacks of every own piece.
- Checks: slider-drop candidate squares taken from the occupancy-aware
  slider effect of the king's square (attack symmetry) instead of
  empty-board zones with per-square occlusion tests; check zones by
  const reference; pin test inlined.

Solver effect: mate11 total 19.3 s → 14.3 s (−26%), mate9 −26%,
mate5 −22% (2000-problem subsets). The dlshogi new-node block was
re-tested with the cheap probes and still loses (~1.5x) — its
evasion-generation cost per fresh node remains the bottleneck at
JHBR3's remaining constant factors; the knob stays off.

### SIMD Bitboard (stage 4)

`lczero::Bitboard` now stores a `union { uint64_t p_[2]; __m128i m_; }`
(YaneuraOu's layout): and/or/xor/andnot/shift/equality run as single
SSE2/SSE4.1 instructions, `byte_reverse` uses the SSSE3 shuffle, and the
Qugiy `Unpack`/`Decrement` pairs use the SSE forms (`cmpeq`+`add` borrow
trick). `Bitboard256` is a real `__m256i` under AVX2 (the Qugiy bishop
path). The public API and all constexpr entry points are unchanged —
constant evaluation takes the scalar branch via
`std::is_constant_evaluated()`, and every SIMD block has a scalar
fallback keyed off compiler macros (`__SSE2__`/`__SSSE3__`/`__SSE4_1__`/
`__AVX2__`), so no build flags changed and a plain `-march=x86-64`
build still passes the whole test battery.

Measured effect (Zen 1): sliders 7.0→6.0 ns, `InCheck` 16→11 ns,
mate-1 probe 133→96 ns, checks 194→163 ns, evasions 219→185 ns,
blockers 11.3→7.8 ns; solver totals another ~3% (mate11 14.3→13.8 s).
Perft node counts are bit-identical to the scalar build.

On AVX-512: YaneuraOu's own bitboard has no AVX-512 code (one
speculative comment only); AVX-512/VNNI is used by its NNUE evaluator,
which JHBR3 does not have (ONNX/TensorRT NN instead). This port
therefore targets SSE/AVX2 and compiles cleanly for
`-march=icelake-server`/`sapphirerapids` (verified; AVX2 paths run
as-is on AVX-512 hardware).

### Gap-closing round (stage 5)

- **pn/dn is the class default arithmetic** (`Arith::kPnDn`): measured
  consistently faster on these sets; `kBns` remains the paper mode and
  is what benchmark label `bns` runs explicitly.
- **Move cache** (`set_move_cache_mb`, default 2): memoizes
  (moves, child hashes) per (position, ply) — movegen output is
  position-pure, so reuse across visits is exact. 24-move slots,
  direct-mapped, generation-stamped. It helps the pn/dn configuration
  ~4-5% on deep sets and remains a smaller net win for the final strict
  BNS configuration.
- **Check generator v2**: the slider-symmetry trick extended to board
  moves — per source piece, direct-check destinations are the slider
  effect cast from the king over source-vacated occupancy (exact; see
  the proof sketch in the code), removing all per-target attack tests.
  194→175 ns; ~1% solver.
- **Negative results** (implemented, measured, reverted/off):
  OR-node move ordering (captures/board/drops tiers) cost 5-6% —
  matching YaneuraOu's own data, where their ordering variant is slower
  than their plain one on these sets; the evasion king-move x-ray
  pre-ban cost ~1% at solver level (checker-loop overhead exceeds the
  saved attack queries on real position mixes).

### YaneuraOu comparison round (stage 6, 2026-08-04)

Reviewing current YaneuraOu `mate_dfpn.hpp` showed that its fastest
matebench mode uses compact 16-byte `Node<u32, false>` records in a
contiguous tree arena. Generated children remain attached to the node,
so revisits use direct child access instead of regenerating moves or
probing a transposition table. JHBR3 retains its merged-graph design but
now follows the same cache-locality lesson:

- the TT hot record is 24→16 bytes; unused stored best moves were removed;
- first lookup/insertion is fused, stale clusters terminate scans early,
  and the default table is 2-way;
- checking/evasion generators can fill the per-ply `MoveList` in place,
  removing a 1192-byte result copy from each cache miss;
- optional move ordering/new-node scratch was moved out of the recursive
  hot frame, and normal route-independent child refresh bypasses the
  dominance/override bookkeeping;
- PV extraction no longer repeats an exhaustive 3-ply search at every OR
  node unless a frontier option actually created such a proof;
- an approximate but sound root mate-in-3 probe checks the first 12 root
  checks and reuses its generated move list on failure;
- cache sweeps selected a 4 MiB TT and 2 MiB move cache. Larger tables
  lose locality on this host and were consistently slower.

One controlled pass over the report's identical subsets (5M nodes, 10 s,
single pinned core, all PVs replayed) compared the production pn/dn
configuration with the current local YaneuraOu Node32/no-ordering mode:

| set | JHBR3 BNS-family | YaneuraOu | faster |
|---|---:|---:|---|
| mate3 (10k) | 0.11 s | 0.35 s | JHBR3 |
| mate5 (10k) | 1.61 s | 1.34 s | YaneuraOu |
| mate7 (5k) | 2.70 s | 2.37 s | YaneuraOu |
| mate9 (3k) | 4.77 s | 4.88 s | JHBR3 |
| mate11 (2k) | 7.60 s | 7.94 s | JHBR3 |
| **aggregate** | **16.79 s** | **16.88 s** | **JHBR3 (0.6%)** |

The aggregate edge is within run-to-run noise, so this is not a claim
that JHBR3 is universally faster. The mate9/mate11 lead is repeatable;
mate5/mate7 still expose a search-order/node-count deficit. The literal
paper branch-number arithmetic also remains slower than the production
pn/dn arithmetic on these short generated problems. For that mode,
keeping generator order when branch numbers tie (rather than using the
opposite number as a secondary key) reduced the mate11 run from 21.57M
nodes / 10.15 s to 20.08M nodes / 9.45 s; asymmetric and reverse tie
policies all exposed worse hard-tail positions.

## Host additions

- `ShogiBoard::HashAfter(Move)`, `BoardKey()`, `BoardKeyAfter(Move)`
  (property-tested against `DoMove`).
- `CheckRepetition(max_back_plies, int* distance)` overload.
- `Hand::Dominates` (borrow-bit trick; was declared but undefined).
- `DfpnNodePool::Alloc` buffer reuse; `MateDfpnSolver::
  set_use_fast_check_movegen` (benchmark-only, default off).

## Benchmarking

`test/bench_mate_solvers.cc` runs any solver over an SFEN problem file
with node/time budgets, PV validation, and CSV output; external drivers
live in `/data/new_jhbr2/dfpn/cshogi_bench/`. See the benchmark report
in `bench_results/` for current numbers.

## Engine integration and strength test

The engine's root mate thread selects its solver via the USI option
`RootMateSolver` (combo, default `bns`, var `dfpn`): `MateBnsSolver` in
its pn/dn configuration (4 MB lazy-allocated table, 2 MB move cache)
or the original tree df-pn. Result handling, the repetition boundary
check, and info output are unchanged; both solvers share the same
interface by construction.

Root mate time is owned by the main MCTS lifecycle. The worker is released
after `Search::Run()` has reset its stop state, a proved mate immediately stops
MCTS, and every MCTS exit immediately stops and joins the root mate worker.
There is no post-MCTS grace period or independent wall-time limit. BNS uses an
effectively unlimited node count because its tables are fixed-size; the
optional tree df-pn keeps a 2,000,000-node cap as a linear-memory guard. The
move watchdog stops both workers at the shared response deadline.

The retired `DfPnMaxTime` option is accepted but ignored for compatibility
with existing engine configuration files.

A/B strength test (2026-08-04): 200 opening pairs (400 games, colors
swapped) from a book-derived balanced suite, byoyomi 500 ms, TensorRT
epoch-3 148-plane model on an RTX 3090, zero failed games.

    RootMateSolver=bns vs =dfpn: 51.5% score,
    +10.4 Elo, 95% CI [-11.6, +32.3], LOS 82%

The BNS side led at every 25-pair milestone and the point estimate sits
in the expected small-positive range for a root-mate-solver upgrade,
but 400 games cannot certify an effect of this size (confirming
~10 Elo at 95% confidence needs roughly 1500-2000 games). Verdict:
suggestive positive, not proven; the match harness output is resumable
for extension.
