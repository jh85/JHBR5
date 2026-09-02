# Monty study notes

Reference tree: `../Monty/` at commit `0950aff` (2026-05-12, "Fix comparator-ordering
bug (#154)"). Monty is AGPL-3.0. These notes describe *ideas and numbers*; no code
was copied into JHBR5. Every claim below cites the file and function it was read
from. Derived sizes were recomputed with a script that replicates Monty's `const`
tables (see "Derived constants").

All shapes are for the release build (`cfg(not(feature = "datagen"))`). The datagen
build uses a smaller policy net and different CPUCT defaults; these differences are
called out where they matter.

## 1. File map

| Path | Contents |
|---|---|
| `src/networks/common.rs` | `Accumulator<T,N>`, `Layer`, `TransposedLayer`, `SCReLU`, `add_multi_i8` (sparse L1 accumulation) |
| `src/networks/value.rs` | `ValueNetwork` struct and `eval()` |
| `src/networks/value/threats.rs` | value feature extraction `map_features`, per-piece threat index functions |
| `src/networks/value/attacks.rs` | empty-board attack tables and prefix-sum index tables used by the threat mapping |
| `src/networks/policy.rs` | `PolicyNetwork`, `hl()` (hidden layer per node) and `get()` (one logit per move) |
| `src/networks/policy/inputs.rs` | policy feature extraction `map_features` |
| `src/networks/policy/outputs.rs` | `map_move_to_index` (move bucket table), `NUM_MOVES_INDICES` |
| `src/networks/policy/see.rs` | static exchange evaluation `greater_or_equal_to` (used to double the buckets) |
| `src/chess.rs` | `ChessState` wrapper: `EvalWdl`, contempt, sharpness ("material") adjustment, `map_moves_with_policies` |
| `src/mcts.rs` | `Searcher::search`, search loop, limits, reporting, `calibrate_wdl` (display only) |
| `src/mcts/iteration.rs` | `perform_one` (one MCTS iteration), `pick_action` (PUCT), `get_utility` |
| `src/mcts/helpers.rs` | CPUCT scaling, exploration scaling, PST (policy softmax temperature), FPU, time management |
| `src/mcts/params.rs` | all tunable parameters with defaults |
| `src/tree.rs` | `Tree`: two halves, hash table, butterfly table, root accumulator, `expand_node`, `relabel_policy`, subtree reuse |
| `src/tree/node.rs` | `Node` (64 bytes), `NodeStatsDelta`, u48 counters |
| `src/tree/half.rs` | `TreeHalf`: bump allocator with per-thread 1024-node chunks, cross-links |
| `src/tree/hash.rs` | 16-byte `HashEntry` table keyed by position hash |
| `src/tree/lock.rs` | `CustomLock`: spin write-lock protecting a node's `actions` pointer |
| `src/lib.rs` | `read_into_struct_unchecked`: mmap the weight file straight onto the struct |
| `crates/montyformat/src/format.rs` | `MontyFormat` (policy data: per-move visit distribution) |
| `crates/montyformat/src/value.rs` | `MontyValueFormat` (value data: per-move score only) |
| `crates/datagen/src/thread.rs` | self-play loop |
| `crates/train-value/src/main.rs`, `input.rs` | bullet value trainer |
| `crates/train-policy/src/main.rs`, `model.rs`, `model/select_affine.rs`, `model/loss.rs` | bullet/acyclib policy trainer with custom CUDA kernels |

## 2. Value network

### 2.1 Architecture (`src/networks/value.rs`)

```
QA = 128, QB = 1024, L1 = 8192

struct ValueNetwork {
    pst: [Accumulator<f32, 3>; TOTAL],              // per-feature WDL skip weights
    l1:  Layer<i8,  TOTAL, L1>,                      // sparse input -> 8192, i8 weights + i8 biases
    l2:  TransposedLayer<i16, L1/2, 16>,             // 4096 -> 16, i16
    l3:  Layer<f32, 16, 128>,                        // f32
    l4:  Layer<f32, 128, 3>,                         // f32 -> WDL logits
}
```

`TOTAL = 80624` (see derived constants). The struct is `#[repr(C, align(64))]`
and the weight file is a raw dump of it (see section 6).

### 2.2 Input features (`src/networks/value/threats.rs: map_features`)

Input is `bbs: [u64; 8]` (two colour occupancy boards + six piece boards) and
`stm`. Steps, in order:

1. **Side-to-move perspective.** If `stm == BLACK`, swap the two colour boards and
   `swap_bytes()` every board (vertical flip). After this, index 0 is "us".
2. **Horizontal mirror by our king file.** `ksq = lsb(bbs[0] & bbs[KING])`; if
   `ksq % 8 > 3` every board is mirrored with `flip_horizontal`. So our king is
   always on files a–d. This is what halves the piece-square table.
3. Build `pieces[64]` with `pc = 6*side + piece - 2` (0..11, 13 = empty).
4. For each side and each piece on the board, emit two kinds of features:
   * **Piece-square feature**: `TOTAL_THREATS + [0, 384][side] + 64*(piece-2) + sq`.
     768 of these (2 sides × 6 types × 64 squares).
   * **Threat features**: for every square `dest` the piece attacks that is
     *occupied* (`threats = attacks(sq, occ) & occ`, so attacks on empty squares
     are not features), call `map_piece_threat(piece, sq, dest, pieces[dest],
     enemy)`; emit `side_offset + idx` where `side_offset = ValueOffsets::END * side`.
     So threats by "us" and threats by "them" are disjoint halves of the same
     table (`TOTAL_THREATS = 2 * END`).

Threat index construction (`map_*_threat` functions and `attacks.rs`):

* The target piece type is compressed per attacker type with `offset_mapping`,
  which lists which target types are kept. Pawn attacker keeps targets
  {P, N, R} (6 slots incl. colour), bishop/rook keep {P, N, B, R, K} (10), king
  keeps {P, N, B, R} (8), knight and queen keep all 12. Targets outside the list
  return `None` and produce no feature.
* Symmetric pairs are de-duplicated: e.g. knight-attacks-knight, bishop-attacks-
  bishop, rook-attacks-rook, queen-attacks-queen are only emitted when
  `dest < src`; pawn-attacks-enemy-pawn only when `dest < src`.
* For sliders/knights/kings the pair `(src, dest)` is compressed with
  `ValueIndices::X[src] + below(src, dest, &ValueAttacks::X)`, i.e. a prefix sum
  over the empty-board attack set of `src`, so each geometrically possible
  (src, dest) pair gets a dense index. Pawns use `(src/8 - 1) * 14 + 2*(src%8) +
  id - 1` with 84 pawn slots per target type.
* Offsets: `PAWN=0, KNIGHT=6*84, BISHOP=KNIGHT+12*336, ROOK=BISHOP+10*560,
  QUEEN=ROOK+10*896, KING=QUEEN+12*1456, END=KING+8*420 = 39928`.

Active feature count: 32 piece-square at most, plus threats; `eval()` reserves
`feats[160]` and the trainer declares `max_active() = 128`.

### 2.3 Forward pass (`ValueNetwork::eval`)

1. `pst` accumulator (f32×3) is summed over the active features while they are
   collected (the **PST skip connection**).
2. `l2` accumulator (i16 × 8192) starts from `l1.biases` widened to i16, then
   `add_multi_i8(&feats, &l1.weights)` adds each active feature's i8 row. This is
   a from-scratch sum every call; nothing is cached between positions.
   `add_multi_i8` processes 128 lanes per outer iteration ("8 registers of 16
   i16") to let the compiler keep partial sums in registers.
3. **Pairwise multiply activation**: `act[i] = clamp(l2[i], 0, QA) *
   clamp(l2[i + L1/2], 0, QA)` for `i < 4096`. Result fits in i16 (`128*128`),
   and represents the product of two `[0,1]` values scaled by `QA*QA`.
4. Dense `l2` (i16 weights, 4096 → 16): i32 dot products; then
   `l3_in[j] = (acc / (QA*QA) + bias_j) / QB`. So the i16 weights carry scale
   `QB = 1024`, biases are i16 in the same scale.
5. `l3` (16 → 128, f32) and `l4` (128 → 3, f32) with `SCReLU` activation on
   their *inputs* (`Layer::forward::<SCReLU>` activates the input vector:
   `clamp(x,0,1)^2`).
6. `out += pst`; softmax over the three logits, ordering `[loss, draw, win]`
   (`out.0[2]` is win). Returns `(win, draw, loss)`.

### 2.4 Post-processing before backup (`src/chess.rs`)

`ChessState::evaluate_material_wdl` (release build):

```
draw_adj = draw * sharpness_scale + draw^2 * sharpness_quadratic    // 2.4474, 0.8899
sum      = win + draw + draw_adj + loss
material = (win/sum, (draw + draw_adj)/sum, loss/sum)
```

i.e. the draw probability is **inflated** (the name "sharpness" is historical);
`cp = material.to_cp_i32()` with `cp = -400 ln(1/score - 1)`, `score = win +
0.5 draw`. In the datagen build this adjustment is skipped and the raw WDL is used.

`eval_with_contempt` then applies `EvalWdl::apply_contempt(contempt *
perspective)` where perspective is +1 when the node's side to move equals the
root side to move. Contempt is a logistic re-centering in "mu/s" space
(`apply_contempt`), off by default (`contempt = 0`).

The value backed up is `score = win + 0.5*draw` from the *node's* side to move
(`get_utility` in `iteration.rs`), together with `draw` which is tracked
separately in the node.

`calibrate_wdl` in `mcts.rs` is a 3×3 log-linear recalibration used **only for
UCI `wdl` display**, not in search.

## 3. Policy network

### 3.1 Architecture (`src/networks/policy.rs`)

```
QA = 128, QB = 128, FACTOR = 32
INPUT_SIZE = 3072
L1 = 40960 (release), 16384 (datagen)

struct PolicyNetwork {
    l1: Layer<i8, INPUT_SIZE, L1>,                        // 3072 -> 40960, i8
    l2: TransposedLayer<i8, L1/2, NUM_MOVES_INDICES>,     // one i8 row of 20480 per move bucket, i8 bias
}
```

No trunk is shared with the value net; the two files are independent.

### 3.2 Inputs (`src/networks/policy/inputs.rs: map_features`)

* `flip = stm == BLACK`; `hm = 7 if king_index % 8 > 3 else 0` (horizontal
  mirror by *our* king file, applied as `sq ^ hm`). Vertical flip via
  `swap_bytes` on the bitboards.
* `threats = pos.threats_by(!stm)` (squares attacked by the opponent),
  `defences = pos.threats_by(stm)` (squares we attack), both flipped.
* For each of our pieces: `feat = 64*(piece-2) + (sq ^ hm)`; for each of theirs
  `feat = 384 + 64*(piece-2) + (sq ^ hm)`. Then `feat += 768` if the square is
  attacked by the opponent, `feat += 1536` if defended by us. So the 3072
  inputs are `768 × {plain, attacked, defended, attacked+defended}`, and each
  piece produces exactly **one** feature (max 32 active; trainer uses
  `MAX_ACTIVE_BASE = 32`).

### 3.3 Hidden vector per node (`PolicyNetwork::hl`)

Identical structure to the value L1: bias widened to i16, `add_multi_i8` over
the active features (from scratch, no caching), then pairwise multiply with an
extra shift: `res[i] = (clamp(a,0,QA) * clamp(b,0,QA)) / (QA / FACTOR)` =
`a*b/4`, stored as i16 (max `128*128/4 = 4096`). Output has `L1/2 = 20480`
entries. `hl` is computed **once per expansion** (`ChessState::map_moves_with_
policies`) and reused for every legal move of that node.

### 3.4 One logit per move (`PolicyNetwork::get`)

```
idx = map_move_to_index(pos, mov)
res = sum_i l2.weights[idx][i] * hl[i]                       // i32
logit = (res / (QA*FACTOR) + l2.biases[idx]) / QB           // f32
```

`res / (QA*FACTOR)` undoes the `a*b/4` scaling (`QA*QA / (QA/FACTOR) = QA*FACTOR`).

### 3.5 Move bucket table (`src/networks/policy/outputs.rs: map_move_to_index`)

Moves are first normalised to the same frame as the inputs: `flip = hm ^ (56 if
stm == BLACK else 0)`, `src ^= flip`, `dst ^= flip`.

Bucket id (`idx`) in `[0, FROM_TO)`:

| Move kind | Index |
|---|---|
| Normal (non-promotion, non-castle, non-double-push) | `OFFSETS[pc][src] + popcount(DESTINATIONS[src][pc] & ((1<<dst)-1))` — dense (piece type, from, to) over every geometrically possible destination on an empty board (`DESTINATIONS` = pawn pushes+captures, knight, bishop, rook, queen, king tables) |
| Promotion | `OFFSETS[5][64] + 22*(promo_pc - KNIGHT) + (2*from_file + to_file)` — 4 promo pieces × 22 (from-file,to-file) combos = `PROMOS = 88` |
| Castling | `OFFSETS[5][64] + PROMOS + (is_ks ^ is_hm)` — 2 buckets, mirrored with the board |
| Double pawn push | `OFFSETS[5][64] + PROMOS + 2 + src_file` — 8 buckets |

`FROM_TO = OFFSETS[5][64] + PROMOS + 2 + 8 = 3822 + 88 + 10 = 3920`.

**SEE doubling**: `good_see = see::greater_or_equal_to(pos, &mov, -108)`;
final index `= FROM_TO * good_see + idx`, so `NUM_MOVES_INDICES = 2 * FROM_TO =
7840`. Every move (captures *and* quiet moves) is evaluated by SEE against the
threshold −108 (slightly more than a pawn, `SEE_VALS = [0,0,100,450,450,650,
1250,0]`), so "bad" quiet moves (e.g. hanging a piece) also get the second row.
The SEE in `see.rs` is unusually complete (pins, x-rays, check restrictions,
promotions, en-passant legality); the comment claims 99.997% legality on a
puzzle set.

### 3.6 Softmax, temperature and history bonus (`src/tree.rs: expand_node`)

At expansion (which happens on the node's **second** visit, see 4.1):

1. For each legal move: `adjusted = policy_logit + butterfly.policy_bonus(stm, mov)`.
   `policy_bonus = butterfly[side][from][to] / butterfly_policy_divisor` (17179).
2. `pst = SearchHelpers::get_pst(depth, node.q(), params)`:
   ```
   t        = max(q - winning_pst_threshold, 0) / (1 - winning_pst_threshold)   // 0.5655
   base_pst = 1 - base_pst_adjustment + (depth - root_pst_adjustment)^(-depth_pst_adjustment)
            = 0.904 + (depth - 0.3349)^(-1.5777)
   pst      = base_pst + (winning_pst_max - base_pst) * t                       // winning_pst_max 1.6260
   ```
   Depth 1 (root) gives `pst ≈ 0.904 + 1.90 = 2.80`, depth 2 ≈ 1.53, depth 5 ≈
   1.0, deep nodes → 0.904. A high Q (a winning node) raises the temperature
   toward 1.626 so the search spreads visits among winning alternatives.
3. `p_i = exp((logit_i - max) / pst)`, normalised; children sorted by descending
   policy (this matters for the top-p pruning in selection); each child stores
   `policy` as a u16 fraction. Gini impurity `1 - Σ p_i²` is stored on the
   parent (u8) and used for exploration scaling.

`relabel_policy` recomputes the root's and root-children's policies with the
root PST at the start of every search (tree reuse changes their depth).

## 4. Search

### 4.1 One iteration (`src/mcts/iteration.rs: perform_one`)

Recursive descent from the root with a cloned `ChessState`:

```
depth += 1
if node is terminal or node.visits == 0:
    if visits == 0: node.state = pos.game_state()          // terminal detection on first visit
    value = hash_table.probe(pos.hash) or get_utility()    // network eval only here
else:
    if node.is_not_expanded(): tree.expand_node(...)       // expansion on SECOND visit
    tree.fetch_children(...)                               // migrate children from old half if needed
    action = pick_action(...)                              // PUCT
    pos.make_move(child.move); child.inc_threads()
    lock = node.actions_mut() if child.visits == 0 else None   // serialise first visit of a child
    value = perform_one(child ...)
    child.dec_threads()
    if child ongoing: tree.update_butterfly(stm, mov, value.0)
    tree.propogate_proven_mates(node, child.state)
push hash (q stored from the side to move at that position)
value.0 = 1 - value.0                                       // flip perspective
tree.update_node_stats(node, value)
```

Key points:

* A node is evaluated by the value net on its **first** visit and only expanded
  (policy net) on its **second** visit. Leaf nodes that are visited once never
  pay for policy inference.
* `get_utility` returns `(score, draw)`; terminal states give `(0.5,1)`,
  `(0,0)`, `(1,0)`.
* The **hash table** (`tree/hash.rs`) caches `(q, d, visits)` per position hash
  (32-bit key check, 16-byte entries, size ≈ tree_bytes/16/4 entries). On a hit
  the network is skipped. Entries are replaced when the key differs or the new
  visit count is ≥ the stored one.
* Backup is `1 - value` at each level, so `node.q()` is "expected score for the
  player who *moved into* this node" — the same convention as JHBR3's
  `child.win / move_count`.

### 4.2 Selection (`pick_action`)

```
cpuct = get_cpuct(params, node, is_root)
fpu   = 1 - node.q()                                       // parent's Q flipped
expl  = cpuct * get_explore_scaling(params, node)
limit = top-p prefix of children (policy_top_p 0.7105), at least min_policy_actions (6),
        +2 children each time visits pass 2^visit_threshold_power (4), 8, 16, ...
score(child) = q_vl + expl * child.policy / (1 + child.visits)
```

where `q_vl` is `fpu` for unvisited children, else `child.q()` scaled by the
virtual-loss formula `q * v / (v + 1 + virtual_loss_weight*(threads-1))`
(`virtual_loss_weight 2.4747`) if other threads are currently inside this child.
Note the denominator `1 + visits`, not `sqrt`.

`get_cpuct` (`helpers.rs`):

```
cpuct  = root_cpuct (0.4119) if root else cpuct (0.2826)
cpuct *= 1 + ln((visits + 128*cpuct_visits_scale) / (128*cpuct_visits_scale))     // grows with visits, scale 37.63
if visits > 1:                                                                     // variance scaling
    frac  = sqrt(var(q)) / cpuct_var_scale (0.2710)
    frac += (1 - frac) / (1 + cpuct_var_warmup (0.4988) * visits)
    cpuct *= 1 + cpuct_var_weight (0.8462) * (frac - 1)
```

`get_explore_scaling`:

```
scale  = exp(expl_tau (0.648) * ln(max(visits,1)))            // = visits^0.648
scale *= min(gini_base (0.5129) - gini_ln_multiplier (1.4737) * ln(gini + 0.001), gini_min (2.2546))
```

So the exploration term is `cpuct * visits^0.648 * f(gini) * P / (1+n)` — a
policy-sharpness-aware variant of PUCT; the `visits^tau` factor replaces the
`sqrt(N)` of AlphaZero.

The root is expanded and evaluated immediately in `Searcher::search` when the
tree is empty; the root value is backed up once.

### 4.3 Proven results

`propogate_proven_mates` (`tree.rs`): a lost child makes the parent
`Won(n+1)`; a won child makes the parent `Lost` only if *all* children are won.
`GameState` is stored as u16 (`Ongoing`, `Draw`, `Lost(n)`, `Won(n)`), and
`get_utility` returns exact values for terminal nodes so they stay in the
tree and keep being backed up. Move selection (`get_best_child`) ranks
`Lost(n)` children (i.e. wins for us) above everything, then Q, then `Won(n)`
(losses) last.

### 4.4 Node layout and statistics (`src/tree/node.rs`)

`Node` is exactly 64 bytes (`const _: () = assert!(size_of::<Node>() == 64)`):
`actions` (CustomLock: u64 ptr + bool), `sum_q`, `sum_sq_q`, `draws` (u64 each,
values quantised by `QUANT = 65536`), `visits` and `nodes` as split u48
(u32 + u16), `state` u16, `threads` u16, `mov` u16, `policy` u16, `num_actions`
u8, `gini_impurity` u8. There is no per-node hash, no move list, and no
accumulator: children are a contiguous run `actions .. actions + num_actions`
in the tree half.

`Tree::new_mb` allocates `bytes / 66` nodes total, split into two halves, plus
a hash table of `nodes/16/4` entries.

### 4.5 Multithreading (`mcts.rs`, `tree.rs`, `tree/half.rs`, `tree/lock.rs`)

* `threads` worker threads run `perform_one` independently on the shared tree
  (tree parallelism, no batching, no leaf queue). Thread 0 is the "main"
  thread that checks limits and prints info.
* Node allocation: each `TreeHalf` has a bump pointer `used`; threads reserve
  1024-node chunks (`reserve_nodes_thread`) so allocation is mostly
  contention-free.
* The only lock is the per-node `CustomLock` on the `actions` pointer. It is
  taken as a write lock during `expand_node` (double-checked with
  `is_not_expanded`), and, importantly, held by the *parent* across the
  recursive call while a child with 0 visits is being visited (`lock` in
  `perform_one`). That serialises first visits of a child so two threads cannot
  both evaluate it. Statistics use relaxed atomics.
* Virtual loss is not a visit-count hack: `threads` (u16) counts threads in a
  subtree and only lowers the child's Q in PUCT (see 4.2).
* `RootAccumulator` (`tree.rs`): backups to the root and to very hot nodes
  (`visits ≥ NODE_BATCH_THRESHOLD = 16384`, up to 32 tracked nodes) are
  accumulated per thread and flushed every 32 visits to avoid atomic
  contention on the same cache line. This is an optimisation for high thread
  counts.

### 4.6 Tree halves and reuse

* Two halves (`TreeHalf`). When the current half is full, `Tree::flip` swaps
  halves, clears the new one, copies the root node across, and search
  continues; children are copied lazily into the new half on access
  (`fetch_children` / `copy_across`). `cross_links` remember which old-half
  nodes point into the new half so they can be cleared on the next flip.
* `set_root_position` (`tree.rs`): searches at most **two plies** below the
  old root for a node whose board equals the new root (`recurse_find`), then
  copies that node into slot 0 of the current half. Otherwise both halves are
  cleared. The hash table and butterfly table are kept across moves and only
  cleared by `ucinewgame` (`uci.rs`) or `Tree::clear`.

### 4.7 Butterfly history (`tree.rs: ButterflyTable`)

`[2 sides][64 from][64 to]` i16 entries. After every completed iteration that
passed through an ongoing child, `update(side, mov, score)` converts the
backed-up score into centipawns `cp = -400 ln(1/score - 1)` (clamped) and adds
`cp - current*|cp|/butterfly_reduction_factor (8358)` with a CAS loop — a
standard history "gravity" update, with the *value* as the bonus rather than a
depth bonus. Read at expansion as a logit bonus divided by 17179 (so a history
entry at its i16 limit adds about ±1.9 to a logit).

### 4.8 Time management (`helpers.rs`)

`get_time` gives (opt, max) from remaining time, increment and ply using the
`tm_*` parameters; `soft_time_cutoff` scales opt by falling-eval, best-move
instability and best-move visit share. Checked every 128 iterations (hard) and
every 4096 iterations (soft). Not relevant to network design; JHBR3's own
time manager will be kept.

## 5. Derived constants (recomputed)

| Quantity | Value | Where |
|---|---|---|
| Value piece-square inputs | 768 | `threats.rs` |
| `ValueOffsets::END` (threat slots per side) | 39,928 | `attacks.rs` |
| `threats::TOTAL` (value inputs) | **80,624** | `= 2*39928 + 768` |
| Value L1 width | 8,192 (pairwise → 4,096) | `value.rs` |
| Value L1 weight bytes (i8) | 660 MB | `80624 × 8192` |
| Bytes touched per value eval (≈110 active features) | ≈ 0.9 MB | 8 KB per feature row |
| Policy inputs | 3,072 = 768 × 4 | `inputs.rs` |
| Policy L1 width | 40,960 (pairwise → 20,480) | `policy.rs` |
| Policy L1 weight bytes | 126 MB; ≈ 1.3 MB touched per expansion (32 rows × 40 KB) | |
| `FROM_TO` (move buckets before SEE) | 3,920 | `outputs.rs` |
| `NUM_MOVES_INDICES` | **7,840** | `= 2 × 3920` |
| Policy L2 weight bytes | 161 MB (7840 rows × 20,480 i8); 20 KB read per legal move | |
| Node size | 64 B | `node.rs` |
| Hash entry | 16 B | `hash.rs` |

Take-away for JHBR5: Monty's inference is **memory-bandwidth bound**, not
compute bound. A value evaluation streams ~1 MB of i8 weights and an expansion
streams ~1.3 MB plus ~20 KB per legal move. That is affordable on a modern
desktop with large L2/L3 and fast DRAM, but the JHBR5 dev machine (i7-10870H,
dual-channel DDR4) will need smaller L1 widths at first; see DESIGN.md.

## 6. Weight file format (`src/lib.rs: read_into_struct_unchecked`)

There is no header, version or checksum. The file is a byte-exact dump of the
`#[repr(C)]` struct (`ValueNetwork` / `PolicyNetwork`), memory-mapped and cast.
The only check is `file_size == size_of::<T>()` and alignment. The networks
are identified by file name (`nn-<12 hex>.network`, the sha256 prefix) and
`build.rs` verifies that hash when downloading. JHBR5 will add a real header
(magic, version, arch hash, layer sizes, checksum) as the task requires.

## 7. Datagen and training loop

### 7.1 Self-play (`crates/datagen/src/thread.rs: run_game`)

* Start from the standard position or a random book line.
* Per move: `Tree::new_mb(8, 1)` (8 MB tree, one thread), `Limits { max_nodes:
  100000, max_depth: 64, kld_min_gain: 5e-6 }`. The KLD-gain stop (`check_limits`
  in `mcts.rs`, datagen only) ends the search early when the root visit
  distribution stops changing, so easy positions use far fewer than 100k nodes.
* Root Dirichlet noise: `alpha = 0.03`, `epsilon = 0.25` for value data, `0.05`
  for policy data (`Searcher::search`, datagen feature).
* Move choice: `get_best_child_temp(root, temp)` samples proportional to
  `visits^(1/temp)`; `temp` starts at 0.8 and is multiplied by 0.9 each move
  until it drops below 0.2, after which the most-visited move is played.
* Recorded per move: `best_move`, `score` (root Q from side to move, as `win +
  0.5 draw`), and for policy data the full root visit distribution.
* Game result from `game_state()` after each move; the tree is cleared after
  every move (no reuse in datagen).
* Datagen uses the datagen-build networks (`DatagenValueFileName`,
  `DatagenPolicyFileName`, policy L1 = 16384) and `cpuct = 0.157`, `root_cpuct
  = 1.0`, and an inverse-gini exploration formula (`helpers.rs`).

### 7.2 Data formats (`crates/montyformat`)

* **Policy format** (`format.rs: MontyFormat`): per game a 43-byte header
  (compressed board: 4 × u64 quad-bitboards, stm, ep, rights, halfm, fullm,
  rook files, result as u8 `2*result`), then per move: `u16 move`, `u16 score
  = score*65535`, `u8 num_moves`, and `num_moves` × `u8` visit counts scaled so
  the max is 255. Moves in the distribution are stored in canonical order
  (sorted by u16 move value) and re-derived from the legal move list on read,
  so the moves themselves are not stored. Terminated by a null move.
* **Value format** (`value.rs: MontyValueFormat`): same header, then per move
  `u16 move`, `i16 score` in centipawns from **White's** perspective (`push`
  flips for Black). No visit distribution. Converted to bullet's `ChessBoard`
  format before training (`train-value/src/bin/bulletformat.rs`).
* `interleave.rs`: interleaves many files with a seed for shuffling.

### 7.3 Value trainer (`crates/train-value/src/main.rs`)

bullet `ValueTrainerBuilder`, `.wdl_output()`, inputs `ThreatInputs`
(`input.rs` reuses `monty::networks::value::threats::map_features` — the
**engine's** feature mapping, no second implementation). Graph:

```
l0 = affine(TOTAL -> l1).crelu().pairwise_mul()
l1 = affine(l1/2 -> 16).screlu()
l2 = affine(16 -> 128).screlu()
l3 = affine(128 -> 3)
out = l3 + pst.matmul(inputs)            // pst: (3 × TOTAL) weights, init zero
loss = softmax_crossentropy(out, targets)
```

Optimiser AdamW (decay 0.01, betas 0.9/0.999, weights clipped to ±0.99 so they
quantise into i8/i16). LR 1e-3 → 1e-7 exponential over 4000 superbatches of
1526 × 65536 positions. `wdl_scheduler: ConstantWDL { 1.0 }` in the shown
config means the target is 100% game result (bullet's WDL blend λ; the shown
trainer file is older than the engine's 8192-wide net, which is why it says
`l1 = 3072`).

Export (`save_format`): `l0w`, `l0b` quantised i8 ×128 (rounded); `l1w`, `l1b`
i16 ×1024, `l1w` transposed; `pst`, `l2*`, `l3*` kept f32. That matches the
engine's `QA = 128`, `QB = 1024` and the `TransposedLayer` type of `l2`.

### 7.4 Policy trainer (`crates/train-policy`)

acyclib/bullet graph (`model.rs: make`):

```
inputs  : sparse (3072, max 32 active)
moves   : sparse (NUM_MOVES_INDICES, max 64 per position)   // bucket ids of the legal moves
targets : dense (64)                                         // normalised visit counts
hl      = affine(3072 -> hl).crelu().pairwise_mul()
logits  = SelectAffine(l1, hl, moves)                        // custom CUDA op: logit_k = W[moves_k] · hl + b[moves_k]
loss    = OptimisedSoftmaxCrossEntropy(logits, targets)      // softmax over the ≤64 legal moves only
```

`data/loader.rs: prepare` calls the engine's `map_features` and
`map_move_to_index` (again, the engine code is the single source of truth for
both mappings). AdamW as above; batch 16384, 6104 batches/superbatch, 800
superbatches, LR 1e-3 → 1e-5. Export (`save_quantised`): all four tensors
(`l0w`, `l0b`, `l1w`, `l1b`) as i8 × 128 with an exactness assert, matching
`QA = QB = 128`.

## 8. Things in Monty that JHBR5 will *not* copy, and why

* No header/checksum in weight files — the task requires one.
* Horizontal mirroring — shogi has no left/right symmetry (the board itself is
  symmetric but the initial setup and drops make positions asymmetric; more
  importantly, a mirrored position is not the same position because of
  rook/bishop placement conventions and it would halve nothing useful).
* Per-eval from-scratch L1 — analysed in DESIGN.md; the shogi feature count and
  the finny-table option make caching attractive.
* Contempt and the display-only WDL calibration — optional later.
* The two-halves tree with lazy cross-copying — JHBR3 already has a working
  pointer tree with reuse; replacing it is not required for CPU NNUE.
