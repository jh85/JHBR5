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
OR  (attacker, k active children, best = min abn, tie: min obn):
    abn = best.abn,   obn = best.obn + (k-1)
AND (defender, k active children, best = min obn, tie: min abn):
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

Fixed-size 4-way clustered table, 24-byte entries, generation-based
lazy clearing, keyed by **(position hash, ply from root)**.

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
- TT victim policy (Small-Tree-GC-style effort priority vs naive
  replacement): no measurable difference at any table size; in a fixed
  clustered table the eviction decision barely matters — table *size*
  is what matters. OSL/cshogi's sweep GC targets unbounded tables and
  much longer searches; a sweep has no role in a fixed table, so GC
  here reduces to the replacement policy (documented negative result).

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
- **Move cache** (`set_move_cache_mb`, default 64): memoizes
  (moves, child hashes) per (position, ply) — movegen output is
  position-pure, so reuse across visits is exact. 24-move slots,
  direct-mapped, generation-stamped. Helps the pn/dn config ~4-5% on
  deep sets; roughly neutral-to-slightly-negative for the paper mode,
  whose benchmark rows therefore run with `--mcache=0`.
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
