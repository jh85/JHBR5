# YaneuraOu NNUE — technical reference notes

Scope: the "classic" NNUE evaluator in `YaneuraOu/source/eval/nnue/` (the code path
enabled by `EVAL_NNUE`), its SFNN extension (`SFNNwoPSQT`), the packed-sfen code in
`source/extra/sfen_packer.cpp`, and the training/teacher-data conventions in
`YaneuraOu-ScriptCollection/`. All paths below are relative to
`/home/ei/Downloads/jhbr5/` unless stated otherwise. Everything not explicitly marked
"(not from this tree)" was read from the source in this checkout.

Symbols used throughout (from `YaneuraOu/source/types.h` and `YaneuraOu/source/evaluate.h`):

| Symbol | Value | Where |
|---|---|---|
| `SQ_NB` | 81 | `types.h` |
| `FILE_NB` / `RANK_NB` | 9 / 9 | `types.h` |
| `Inv(sq)` | `(SQ_NB-1) - sq` (180° rotation) | `types.h:247` |
| `Mir(sq)` | `File(8 - file_of(sq)) \| rank_of(sq)` (left–right mirror) | `types.h:251` |
| `fe_hand_end` | 90 | `evaluate.h` enum `BonaPiece` |
| `fe_end` (= `fe_new_end` = `fe_old_end`, no `DISTINGUISH_GOLDS`) | 1548 = 90 + 18×81 | `evaluate.h` |
| `f_king` / `e_king` / `fe_end2` | 1548 / 1629 / 1710 | `evaluate.h` |
| `PIECE_NUMBER_KING` (= `PIECE_NUMBER_BKING`) | 38 | `types.h:666-671` |
| `PIECE_NUMBER_WKING` | 39 | `types.h` |
| `PIECE_NUMBER_NB` | 40 | `types.h` |
| `VALUE_MAX_EVAL` | `VALUE_SUPERIOR` = `VALUE_TB_WIN_IN_MAX_PLY - 1` | `types.h:492-495` |

---

## 1. BonaPiece encoding (the "P" of every feature)

`enum BonaPiece : int32_t` in `YaneuraOu/source/evaluate.h`:

| Range | Meaning |
|---|---|
| `0` (`BONA_PIECE_ZERO`) | empty / invalid slot |
| `1 .. 89` | hand pieces. Apery-WCSC26 layout: `f_hand_pawn=1, e_hand_pawn=20, f_hand_lance=39, e_hand_lance=44, f_hand_knight=49, e_hand_knight=54, f_hand_silver=59, e_hand_silver=64, f_hand_gold=69, e_hand_gold=74, f_hand_bishop=79, e_hand_bishop=82, f_hand_rook=85, e_hand_rook=88, fe_hand_end=90`. Each constant is the index of the *1st* copy; the n-th copy in hand is `base + (n-1)`. There is one spare slot per kind (the "0-th piece" gap noted in the comment). |
| `90 .. 1547` | board pieces, 81 squares per (piece-kind, colour): `f_pawn=90, e_pawn, f_lance, e_lance, f_knight, e_knight, f_silver, e_silver, f_gold, e_gold, f_bishop, e_bishop, f_horse, e_horse, f_rook, e_rook, f_dragon, e_dragon`. Promoted pawn/lance/knight/silver map onto `gold` (`kpp_board_index` in `eval/evaluate_bona_piece.cpp:67`) unless `DISTINGUISH_GOLDS`. |
| `1548 .. 1709` | kings: `f_king = fe_end`, `e_king = f_king + 81`, `fe_end2 = e_king + 81`. |

**Hand pieces with counts.** `Eval::EvalList::put_piece(piece_no, c, pt, i)` (`evaluate.h:~291`) sets
`fb = kpp_hand_index[c][pt].fb + i`, `fw = kpp_hand_index[c][pt].fw + i` where `i` is the
0-based copy index. So holding 3 pawns activates three distinct features
`f_hand_pawn+0, +1, +2` — count is encoded by *which slots* are active, not by a value.

**Perspective flip (`Inv`).** Every piece is stored twice in `Eval::EvalList`
(`evaluate.h:363-369`): `pieceListFb[40]` (from Black) and `pieceListFw[40]` (from White).
Board: `fb = kpp_board_index[pc].fb + sq`, `fw = kpp_board_index[pc].fw + Inv(sq)`;
`kpp_board_index` (`eval/evaluate_bona_piece.cpp:67-126`) swaps `f_*`/`e_*` for white pieces, so
`fw` is literally "the same position seen after a 180° rotation with colours swapped".
Hand: `kpp_hand_index[WHITE]` is the colour-swapped table (`evaluate_bona_piece.cpp:128-148`).
`ExtBonaPiece { BonaPiece fb, fw; }` / `from[2]` (`evaluate.h:236-246`) carries both.

**PieceNumber.** `enum PieceNumber : u8` (`types.h:666`): `PAWN=0..17, LANCE=18..21, KNIGHT=22..25,
SILVER=26..29, GOLD=30..33, BISHOP=34,35, ROOK=36,37, BKING=38, WKING=39`. Feature loops
`for i in [PIECE_NUMBER_ZERO, PIECE_NUMBER_KING)` visit the 38 non-king pieces;
`[.., PIECE_NUMBER_NB)` visits all 40.

---

## 2. Feature sets (`source/eval/nnue/features/`)

Common machinery:

* `features/features_common.h`: `enum class TriggerEvent { kNone, kFriendKingMoved, kEnemyKingMoved, kAnyKingMoved, kAnyPieceMoved }`, `enum class Side { kFriend, kEnemy }`.
* `features/feature_set.h`: `FeatureSet<Head, Tail...>` concatenates features (`kDimensions` = sum, `kMaxActiveDimensions` = sum, `kHashValue = Head ^ (Tail<<1) ^ (Tail>>31)`), builds the sorted, de-duplicated `kRefreshTriggers` list (`SortedTriggerSet`), and offsets Head indices by `Tail::kDimensions`.
  `FeatureSetBase::AppendChangedIndices` decides `reset[perspective]` per trigger:
  `kFriendKingMoved → dp.pieceNo[0] == PIECE_NUMBER_KING + perspective`,
  `kEnemyKingMoved → dp.pieceNo[0] == PIECE_NUMBER_KING + ~perspective`,
  `kAnyKingMoved → dp.pieceNo[0] >= PIECE_NUMBER_KING`, `kAnyPieceMoved → true`.
  Only `pieceNo[0]` is inspected (the moving piece is always slot 0 in `DirtyPiece`).
* `features/index_list.h`: `IndexList = ValueList<IndexType, RawFeatures::kMaxActiveDimensions>` — a fixed array, no heap.
* Every `Half*` feature has the same skeleton: `GetPieces()` picks `piece_list_fb()` or `piece_list_fw()` by perspective and computes the associated king square as `(pieces[PIECE_NUMBER_KING + (perspective or ~perspective)] - f_king) % SQ_NB` — i.e. the king square is *already in the perspective's frame* because it comes from the flipped list. `AppendActiveIndices` pushes `MakeIndex(sq_k, pieces[i])`; `AppendChangedIndices` pushes `MakeIndex(sq_k, old_piece.from[perspective])` to `removed` and `MakeIndex(sq_k, new_piece.from[perspective])` to `added`, skipping `pieceNo >= PIECE_NUMBER_KING` in the KP-family (king moves are handled by refresh).

| Feature (file) | `kDimensions` | `MakeIndex` | Max active / perspective | Refresh trigger | Hash |
|---|---|---|---|---|---|
| `HalfKP<Side>` (`half_kp.{h,cpp}`) | `SQ_NB*fe_end` = 125 388 | `fe_end*sq_k + p` | `PIECE_NUMBER_KING` = 38 | Friend/Enemy king moved | `0x5D69D5B9 ^ (Side==kFriend)` |
| `HalfKA1<Side>` (`half_ka1.*`) | `SQ_NB*fe_end2` = 138 510 | `fe_end2*sq_k + p` (both kings included, distinct planes) | `PIECE_NUMBER_NB` = 40 | same | `0x5f134cb9 ^ friend` |
| `HalfKA2<Side>` (`half_ka2.*`) | `SQ_NB*e_king` = 131 949 | `e_king*sq_k + (p >= e_king ? p - SQ_NB : p)` — enemy king folded onto own-king plane | 40 | same | `0x5f234cb9 ^ friend` |
| `HalfKA_hm1<Side>` (`half_ka_hm1.*`) | `5*FILE_NB*fe_end2` = 76 950 | as HalfKA1 but if `sq_k >= SQ_61` (king on files 6–9): `sq_k = Mir(sq_k)` and board pieces `p >= fe_hand_end` get `Mir` applied to their square; hand pieces untouched | 40 | same | `0x7f134cb9 ^ friend` |
| `HalfKA_hm2<Side>` (`half_ka_hm2.*`) | `5*FILE_NB*e_king` = 73 305 | HalfKA2 folding + hm mirroring | 40 | same | `0x7f234cb9 ^ friend` |
| `HalfKP_vm<Side>` (`half_kp_vm.*`) | `5*FILE_NB*fe_end` = 69 660 | HalfKP + hm mirroring (same rule) | 38 | same | `0x0B6B1D9B ^ friend` |
| `HalfKPE9<Side>` (`half_kpe9.*`) | `SQ_NB*fe_end*9` = 1 128 492 | `(fe_end*sq_k + p) + fe_end*SQ_NB*(e1*3 + e2)` where `e1,e2 = min(effect count of own/enemy on the piece's square, 2)`; hand pieces use 0/0 | 38 | same | `0x5D69D5B9 ^ friend` (identical to HalfKP) |
| `HalfRelativeKP<Side>` (`half_relative_kp.*`) | `kNumPieceKinds*17*17` with `kNumPieceKinds=(fe_end-fe_hand_end)/SQ_NB = 18` → 5 202 | relative file/rank of piece to king on a 17×17 virtual board: `H*W*piece_kind + H*rel_file + rel_rank`; hand pieces are *skipped* | 38 | same | `0xF9180919 ^ friend` |
| `K` (`k.*`) | `SQ_NB*2` = 162 | `pieces[PIECE_NUMBER_KING..NB) - fe_end` | 2 | `kNone` | `0xD3CEE169` |
| `P` (`p.*`) | `fe_end` = 1548 | `p` | 38 | `kNone` | `0x764CFB4B` |
| `PE9` (`pe9.*`) | `fe_end*9` = 13 932 | `p + fe_end*(e1*3+e2)` | 38 | `kNone` | `0x764CFB4B` (identical to P) |
| `A2` (`a2.*`) | `e_king` = 1629 | `bp >= e_king ? bp - SQ_NB : bp` (enemy king folded) | 40 | `kNone` | `0xA20DCB9B` |

"hm" = **half-mirror**: the header comments (`half_ka_hm1.h`) say "6筋～9筋に玉がいる場合、4筋～1筋に反転させる" —
when the associated king is on files 6–9 the whole board is mirrored so the king lands on files 1–5; hence
only `5*FILE_NB = 45` king squares are addressed. `SQ_61` is the first square of file 6 in YaneuraOu's
file-major `Square` numbering, so `sq_k >= SQ_61` ⇔ file ≥ 6.

`HalfKPE9` / `PE9` compile only with `LONG_EFFECT_LIBRARY && USE_BOARD_EFFECT_PREV`
(`half_kpe9.cpp:66`) and read `pos.board_effect[c].effect(sq)` / `board_effect_prev`; their
`AppendChangedIndices` additionally re-scans all non-dirty pieces to detect effect-count changes.

Composite sets used by shipped architectures (`architectures/*.h` and `nnue_arch_gen.py::FEATURE_INFO`):

| arch feature name | `RawFeatures` | input dims |
|---|---|---|
| `halfkp` | `FeatureSet<HalfKP<kFriend>>` | 125 388 |
| `kp` | `FeatureSet<K, P>` | 162 + 1548 = 1710 |
| `ka2` | `FeatureSet<K, A2>` | 162 + 1629 = 1791 |
| `halfkpe9`, `halfkpvm`, `halfka1`, `halfkahm1`, `halfka2`, `halfkahm2` | single `Half*<kFriend>` | see table above |

Note that shipped headers instantiate only `Side::kFriend`; the "enemy" perspective is obtained
not by `HalfKP<kEnemy>` but by evaluating the same Friend feature from the other colour's
`piece_list_fw` (both perspectives always get their *own* king as the associated king).

---

## 3. Accumulator and FeatureTransformer

### 3.1 `Accumulator` (`source/eval/nnue/nnue_accumulator.h`)

```cpp
struct alignas(64) Accumulator {
  std::int16_t accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  Value score = VALUE_ZERO;
  bool  computed_accumulation = false;
  bool  computed_score = false;
};
```

* Index 0 is the **perspective colour** (BLACK/WHITE, not stm), index 1 is the refresh-trigger
  group (`kRefreshTriggers.size()` — 1 for every shipped architecture, since each uses a single
  trigger kind; `FeatureSet<K,P>` is also 1 because both are `kNone`), index 2 is the FT output.
  A mixed set such as `FeatureSet<HalfKP<kFriend>, P>` would get 2 groups that `Transform()` sums.
* It lives inside `StateInfo` (`source/position.h:209`, `#if defined(EVAL_NNUE) Eval::NNUE::Accumulator accumulator;`), so every ply on the search stack owns one.
* Size: HalfKP-256 → 2×1×256×2 B = 1 KiB (+ flags); 1024 → 4 KiB; SFNN-1536 → 6 KiB.
* `DirtyPiece dirtyPiece` sits right after it in `StateInfo` (`position.h:214`).

### 3.2 `FeatureTransformer` (`source/eval/nnue/nnue_feature_transformer.h`)

| Member | Type / value |
|---|---|
| `BiasType`, `WeightType` | `std::int16_t` |
| `OutputType` | `TransformedFeatureType = std::uint8_t` (`nnue_common.h`) |
| `kInputDimensions` | `RawFeatures::kDimensions` |
| `kHalfDimensions` | `kTransformedFeatureDimensions` (256/512/1024/1536) |
| `kOutputDimensions` | `kHalfDimensions*2` (classic) or `kHalfDimensions` (`USE_ELEMENT_WISE_MULTIPLY`, i.e. SFNN) |
| storage | `alignas(64) int16 biases_[kHalf]; alignas(64) int16 weights_[kHalf * kInput]` — **feature-major**: row `index` is `weights_[kHalf*index .. +kHalf)` |
| `GetHashValue()` | `RawFeatures::kHashValue ^ kOutputDimensions`, or the fixed `0x5f134ab8` under `SFNNwoPSQT` |
| `GetStructureString()` | `"<FeatureName>[<in>-><half>x2]"` |

**Control flow (all in `nnue_feature_transformer.h`):**

* `UpdateAccumulatorIfPossible(pos)` (l.220): if `pos.state()->accumulator.computed_accumulation` return; else if `pos.state()->previous` exists **and** its accumulator is computed, call `update_accumulator(pos)`; else return `false`. It looks back **exactly one `StateInfo`** — there is no multi-ply walk like Stockfish's. The search keeps the chain intact by calling `Eval::evaluate_with_no_return(pos)` after every `do_move` (`engine/yaneuraou-engine/yaneuraou-search.cpp:2386,2397,2405,4549`), which is just `UpdateAccumulatorIfPossible`.
* `EnsureAccumulator(pos, refresh)` → `refresh_accumulator` when `refresh` or the incremental path is impossible.
* `Position::do_move` clears both flags (`position.cpp:1734-1735`). `Position::do_null_move` `memcpy`s the whole `StateInfo` (so the accumulator and `computed_accumulation=true` are inherited) and clears only `computed_score` (`position.cpp:2464`); `previous` is set to the pre-null state. `Position::set()` calls `Eval::compute_eval(pos)` (`position.cpp:735`) which is `ComputeScore(pos, /*refresh=*/true)`.
* `refresh_accumulator` (l.797): for each trigger group `i`, `RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active[2])`, then per colour `refresh_accumulator_from_scratch(acc[c][i], active[c], i)`: start from `biases_` when `i==0` else zero, add every active weight row. Sets `computed_accumulation = true`, `computed_score = false`.
* `update_accumulator` (l.828): `RawFeatures::AppendChangedIndices(pos, trigger, removed[2], added[2], reset[2])`; for a perspective with `reset` it rebuilds from bias + `added` (which then holds the *full* active list), otherwise copies `previous->accumulator` and applies `−removed`, `+added`.
* `DirtyPiece` (`evaluate.h:393`): `ChangedBonaPiece changed_piece[2]` (old/new `ExtBonaPiece`), `PieceNumber pieceNo[2]`, `int dirty_num` (0 for null move, 1 for a quiet move, 2 for a capture: mover in slot 0, captured piece in slot 1 with its new hand BonaPiece). Filled in `Position::do_move` (`position.cpp:1818-2021`).

**Vectorised add/sub (tiles).** With `VECTOR` defined (`USE_AVX512`/`USE_AVX2`/`USE_SSE2`/`USE_MMX`/`USE_NEON`):

| ISA | `vec_t` | `kNumRegs` | `kVectorHeight` (int16 lanes) |
|---|---|---|---|
| AVX-512 | `__m512i` | 8 | 32 |
| AVX2 | `__m256i` | 16 | 16 |
| SSE2 | `__m128i` | 16 (8 on 32-bit) | 8 |
| NEON | `int16x8_t` | 16 | 8 |

`kTileRegs` = largest divisor of `kNumVectorChunks = kHalf / kVectorHeight` not exceeding
`kNumRegs`; `kTileHeight = kTileRegs*kVectorHeight`. `update_accumulator_tiled(source, dest, fn)`
loads one tile of `source` (or bias, or zero) into registers, runs `fn` which calls
`sub_weight_from_tile` for each removed index then `add_weight_to_tile` for each added index
(`vec_sub_16`/`vec_add_16` on the weight row slice `&weights_[kHalf*index + tile_offset]`), and
stores the tile. No saturation is used (`add_epi16`, not `adds_epi16`).

### 3.3 `Transform()` — output activation and perspective ordering

`Transform(pos, output, refresh)` (l.241) first ensures the accumulator, then, with
`perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() }`, writes **side-to-move first**:

*Classic (no `USE_ELEMENT_WISE_MULTIPLY`)* — `offset = kHalfDimensions * p`; for each 16-bit
chunk: `sum = Σ_over_trigger_groups accumulation[persp][i]`, then
`_mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), 0)` + `_mm256_permute4x64_epi64(..., 0b11011000)`
(AVX2), `_mm512_permutexvar_epi64(setr(0,2,4,6,1,3,5,7), max_epi8(packs_epi16))` (AVX-512),
`_mm_max_epi8` or the SSE2 `subs/adds 0x80` trick, NEON `vmax_s8(vqmovn_s16)`. Scalar:
`clamp(sum, 0, 127)`. So the FT output is a plain **clipped ReLU into uint8 [0,127], no shift** —
FT weights are quantised so that 127 ≡ 1.0. Result: `output[0..kHalf)` = stm, `[kHalf..2kHalf)` = opponent.

*SFNN (`USE_ELEMENT_WISE_MULTIPLY`, defined from `SFNNwoPSQT`)* — pairwise product of the two halves
of **one** perspective (Stockfish-style "pairwise clipped ReLU"): `offset = (kHalf/2)*p`,
`in0 = acc[persp][0][0..kHalf/2)`, `in1 = acc[persp][0][kHalf/2..kHalf)`,
`One = 127*2`; `sum0 = slli16(max(min(in0, One), 0), shift)`, `sum1 = min(in1, One)`,
`out = packus16(mulhi16(sum0a, sum1a), mulhi16(sum0b, sum1b))`. `shift` is 7 when `USE_SSE2`
(which `config.h` defines transitively for every x86 target) and 6 otherwise (NEON uses
`vqdmulhq_s16`, which already doubles). Numerically: `out = a*b/512` for `a,b ∈ [0,254]`, max ≈ 126.
Scalar fallback: `product = (clamp(a,0,254) << shift) * clamp(b,0,254); value = product >> 16; clamp(0,255)`.
This is why `ReadParameters` calls `scale_weights(true)` (multiply weights and biases by 2 after
load) — on disk the FT still uses the 127 ≡ 1.0 convention. Output is `kHalf` bytes total:
`[0..kHalf/2)` = stm, `[kHalf/2..kHalf)` = opponent.

`ReadParameters` for SFNN also calls `permute_weights(inverse_order_packs)` (AVX2/AVX-512 only) so
that the lane interleaving produced by `packus_epi16` comes out in natural order without a
runtime shuffle (`order_packs`/`inverse_order_packs`, l.454-485).

---

## 4. Finny tables / accumulator caches

**Present in the tree, but compiled out by default.** `nnue_feature_transformer.h` l.22-26 and
l.575-793 implement them under `#if defined(USE_FINNY_TABLES)`. The comment at l.575 reads
"StockfishのAccumulatorCaches(Finny Tables)と同じ発想". I grepped the whole `YaneuraOu/`
checkout (Makefile, `config.h`, `.vcxproj`, docs): `USE_FINNY_TABLES` is **never defined**, so no
shipped build uses them unless you add `-DUSE_FINNY_TABLES`.

Implementation as written:

* `static constexpr bool kUseFinnyTables = kHalfDimensions <= 4096;`
* `struct alignas(64) FinnyEntry { int16 accumulation[kHalf]; Features::IndexList active_indices; bool initialized; }`
* `struct FinnyCache { const FeatureTransformer* owner; uint64 generation; std::array<std::array<std::array<FinnyEntry, SQ_NB>, COLOR_NB>, kRefreshTriggers.size()> entries; }` — one entry per (trigger, perspective, **king square**). Bucket square from `finny_bucket_square()`: own king for `kFriendKingMoved`/`kAnyKingMoved`, enemy king for `kEnemyKingMoved`, `SQ_ZERO` otherwise.
* Storage is `static thread_local std::unique_ptr<FinnyCache>` inside `refresh_accumulator_with_finny_cache`; it is reset when the owner pointer or `finny_generation_` (bumped in `ReadParameters`) changes.
* On refresh: compute the position's full active index list; if the entry is uninitialised, build it from scratch and write to both the entry and the current accumulator (`update_accumulator_tiled_to_two`); otherwise `make_index_diff(entry.active_indices, new_active, removed, added)` — element-wise compare when the sizes match (fast path, relies on the deterministic `PieceNumber` ordering of `AppendActiveIndices`), otherwise an O(n²) unmatched-element diff — then apply `−removed`/`+added` to the entry and copy into the current accumulator in one tiled pass. `entry.active_indices` is then overwritten.
* Memory per thread ≈ `triggers × 2 × 81 × (2·kHalf + 4·kMaxActive + 8)` B ≈ 0.4 MiB for kHalf=1024.

**Difference from Stockfish's `AccumulatorCaches` (not from this tree, from general knowledge of
Stockfish ≥ 16.1):** Stockfish keys the cache the same way (per king bucket × perspective) but
stores a *board snapshot* (`Bitboard byColorBB[2], byTypeBB[PIECE_TYPE_NB]`) in each entry and
diffs the current bitboards against it to derive removed/added features, and in newer versions
also stores PSQT accumulators. YaneuraOu instead stores the last **feature index list** and diffs
indices, which is simpler for a variable hand-piece encoding but costs O(n) storage per entry and
an O(n²) fallback when a capture changes the list length.

Separately, `source/search.h:577-585` declares `Eval::NNUE::AccumulatorStack accumulatorStack;
Eval::NNUE::AccumulatorCaches refreshTable;` under `#if defined(EVAL_SFNN)` and
`engine/yaneuraou-engine/yaneuraou-search.cpp:5181-5185` calls the Stockfish-signature
`Eval::evaluate(networks, pos, accumulatorStack, refreshTable, optimism)`. **No definition of
`AccumulatorStack` / `AccumulatorCaches` / `EVAL_SFNN` exists anywhere in this checkout** — this is
unfinished scaffolding for a future Stockfish-style port, not working code (`config.h:145`
only says `USE_CLASSIC_EVAL` must not be defined together with `EVAL_SFNN`).

---

## 5. Quantisation

| Item | Value | Source |
|---|---|---|
| FT weights / biases | `int16` | `FeatureTransformer::WeightType/BiasType` |
| FT activation | uint8, clipped ReLU 0..127 (classic) or pairwise product 0..~126 (SFNN) | `Transform()` |
| Affine weights | `int8` (`WeightType = std::int8_t`) | `layers/affine_transform*.h` |
| Affine biases / outputs | `int32` (`BiasType = OutputType = std::int32_t`) | same |
| `kWeightScaleBits` | 6 | `nnue_common.h` |
| ClippedReLU | `clamp(x >> 6, 0, 127)` → uint8 | `layers/clipped_relu.h:684-687` |
| SqrClippedReLU | `min(127, (x*x) >> (2*6+7))` = `>> 19` → uint8 | `layers/sqr_clipped_relu.h:795-798` |
| `FV_SCALE` | 16 by default, USI option `FV_SCALE` range 1..128 (comment: Suisho 5 works best with 24) | `evaluate_nnue.cpp:268,337` |
| Final score | `Value(output[0] / FV_SCALE)`, then `clamp(±VALUE_MAX_EVAL)` | `evaluate_nnue.cpp:924-928` |
| `kCacheLineSize` | 64 | `nnue_common.h` |
| `kSimdWidth` | 32 (AVX2+), 16 (SSE2/NEON/WASM), 8 (MMX) | `nnue_common.h` |
| `kMaxSimdWidth` | 32 — all padded dims are rounded to multiples of 32 (`CeilToMultiple`) | `nnue_common.h` |
| Padded input of an affine layer | `kPaddedInputDimensions = CeilToMultiple(kInput, kMaxSimdWidth)` | `affine_transform.h:182` |
| Layer output buffers | `kSelfBufferSize = CeilToMultiple(out * sizeof(OutputType), kCacheLineSize)`; per-eval scratch `alignas(64) char buffer[Network::kBufferSize]` on the stack | `affine_transform.h:186`, `evaluate_nnue.cpp:892` |
| Static check | `kTransformedFeatureDimensions % kMaxSimdWidth == 0`, `Network::kOutputDimensions == 1`, `OutputType == int32` | `nnue_architecture.h:70-72` |

So a hidden layer computes `Σ int8·uint8 + int32` exactly in int32 (per-4-byte `maddubs`→int16
then `madd`→int32; see §6), and the following ClippedReLU divides by 64 (= the weight scale) to
return to the 127 ≡ 1.0 activation domain. The last layer's int32 output is *not* shifted; it is
divided by `FV_SCALE`, so the network must be trained so that `output ≈ FV_SCALE × cp`. The
training-side scale constants (the trainer that produced the `nn.bin` files) are **not in this
tree** — see §8.

Saturation notes visible in code:
* `_mm256_maddubs_epi16` saturates int16 (two uint8×int8 products summed); the comment in
  `layers/simd.h` and the explicit dual/triple accumulator chains under `USE_NNUE_VNNI`/`USE_AVXVNNI` are about latency, not saturation, but the `madd_epi16(…, set1_epi16(1))` step immediately widens to int32.
* FT accumulation is plain `add_epi16`; the `int16` accumulator can wrap if weights are large — no run-time guard.
* `ComputeScore` clamps to `±VALUE_MAX_EVAL` with a long comment explaining aspiration-search fail-high loops otherwise.
* `evaluate_nnue.cpp:454-456`, `480-485`: hash mismatches on load are **warnings only**.

---

## 6. SIMD kernel structure

### 6.1 ISA selection

`source/config.h:648-694` builds a cascade: `USE_AVX512VNNI → USE_AVX512 → USE_AVX2 → USE_SSE42 → USE_SSE41 → USE_SSSE3 → USE_SSE2`. The Makefile (`source/Makefile:545-617`) adds per target:
`-DUSE_AVX2` (zen1/zen2/zen3, avx2, alderlake also `-DUSE_VNNI -DUSE_AVXVNNI`), `-DUSE_AVX512 [-DUSE_VNNI]`
(zen4, cascadelake, skylake-avx512), `-DUSE_SSE42/41/SSSE3/SSE2`, `-DUSE_NEON=8 [-DUSE_NEON_DOTPROD]`,
`-DUSE_WASM_SIMD -msimd128`. `layers/simd.h` picks the intrinsics header and defines
`USE_NNUE_VNNI` = `USE_VNNI && !NNUE_SFNN_HIDDEN1_7` (comment: measured VNNI slower than
`maddubs/madd` for the H1=7 SFNN shape; `NNUE_SFNN_HIDDEN1_7` is set by the Makefile when the
arch name contains `_7_`).

`layers/simd.h` helpers: `m512_hadd`, `m512_add_maddubs_epi32`, `m512_add_dpbusd_epi32`
(`_mm512_dpbusd_epi32` if VNNI, else `maddubs_epi16` + `madd_epi16(1)` + `add_epi32`), the `m256_*`
and `m128_*` equivalents, `dotprod_m128_add_dpbusd_epi32` (`vdotq_s32`), `neon_m128_add_dpbusd_epi32`
(`vmull_s8`/`vmull_high_s8`/`vpaddq_s16`/`vpadalq_s16`), `neon_m128_hadd`.

### 6.2 Dense affine (`layers/affine_transform.h`, `affine_transform_explicit.h`)

*Weight layout.* With `USE_SSSE3 || USE_NEON_DOTPROD` and `kOutputDimensions % 4 == 0`, weights are
stored scrambled at load: `GetWeightIndexScrambled(i) = (i/4) % (pad_in/4) * out*4 + i/pad_in*4 + i%4`
— i.e. grouped by 4 consecutive inputs, then all outputs, so one `__m256i` column holds 8 outputs ×
4 inputs of int8.

*Kernel (out > 1).* `kNumChunks = CeilToMultiple(in, 8)/4`; `acc[k] = bias vec k`; for each chunk
`in = set1_epi32(input32[i])` (broadcast 4 uint8 inputs) and `acc[k] += dpbusd(in, col[k])`
for `k < out/16` (AVX-512, needs `out%16==0`), `out/8` (AVX2, `out%8==0`), `out/4` (SSSE3),
`out/4` (NEON dotprod). Under VNNI the explicit variant splits into 2 (AVX2/AVX-512) independent
accumulator chains and merges at the end.

*Kernel (out == 1).* Row dot product: `sum0 += dpbusd(input_vec[j], row0[j])` over
`pad_in / (32|16)` chunks then `vec_hadd(sum0, bias)`. The comment notes AVX-512 is deliberately not
used here because the 32-input buffer is not padded to 64.

*Fallback `affine_transform_unaligned`* (used when the output count doesn't fit the mod
constraints, e.g. H1=8 on AVX-512 falls to the AVX2 branch first, and for SSE2-only / non-SIMD):
per output row, `unpacklo/hi_epi8` + `srai_epi16(8)` to sign-extend weights, zero-extend inputs,
`madd_epi16`, then horizontal add; scalar version iterates inputs and skips zeros.

### 6.3 Sparse-input affine (`affine_transform_sparse_input.h`, `..._explicit.h`)

Used for the **first hidden layer** in every classic architecture
(`HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, H1>>`) and as `fc_0` in SFNN
(`AffineTransformSparseInputExplicit<kInputDims, kHidden1OutputDims>`), because the FT output after
clipped ReLU is mostly zero. Requires `USE_SSSE3 || USE_NEON >= 8`.

* `find_nnz<N>(int32* input, uint16* out, count)`: views the uint8 input as int32 (4 inputs per
  word), `vec_nnz = cmpgt_epi32(x, 0)` + `movemask_ps` (or `_mm512_cmpgt_epi32_mask`), and expands
  each 8-bit mask through the 256×8 `lookup_indices` table (`pop_lsb` order) with a running base
  vector (`vec128_add(base, offsets)`, `base += 8`), `count += POPCNT32(lookup)`.
* Then the same broadcast-dpbusd loop as the dense kernel but only over the `count` nonzero chunks;
  `kChunkSize = 4` so column blocks are `weights_[i * out * 4]`. The explicit AVX-512 variant under
  VNNI uses **3** accumulator chains; AVX2 with `USE_AVXVNNI` uses 2.
* The SFNN-only `PropagateSfnnFromAccumulator<HalfDims>()` (AVX-512 + `SFNNwoPSQT`, `out == 8`)
  fuses the FT pairwise transform with `fc_0`: it computes each 64-byte transformed chunk in
  registers, `_mm512_cmpneq_epi32_mask` for nonzero dwords, and feeds only those into
  `m256_add_dpbusd_epi32` — never materialising the uint8 feature buffer. Enabled by
  `NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE`, which `nnue_arch_gen.py:540-552` currently emits only for the
  common+shard case (`enable_sparse_sfnn_accumulator_propagate` is hard-wired `False and …`).

### 6.4 Common+shard affine (`affine_transform_common_shard_input_explicit.h`)

`AffineTransformCommonShardInputExplicit<In, Out, Common, Shard, Groups>`: on disk identical to a
normal `fc_0` (`bias[Out], weight[Out][pad(In)]`); at load each output row keeps only the
`Common` columns plus its group's `Shard` slice (`kPaddedEffectiveInputDimensions`). Static
constraints: `Common + Shard*Groups == In`, `Out % Groups == 0`, `Common % 64 == 0`,
`Shard % 64 == 0`, `Groups > 1` (see `architectures/README.md` "SFNN Common+Shard fc_0").

### 6.5 ClippedReLU (`layers/clipped_relu.h`, `clipped_relu_explicit.h`)

AVX2 (`in % 32 == 0`): `words = srai_epi16(packs_epi32(in0,in1), 6)` twice, then
`permutevar8x32_epi32(max_epi8(packs_epi16(words0, words1), 0), {7,3,6,2,5,1,4,0})`.
AVX-512 (`in % 64 == 0`): same with `permutexvar_epi32({15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0})`.
SSE2: `packs`/`srai`/`packs` then `max_epi8` (SSE4.1) or the `adds/subs 0x80` trick. NEON:
`vqshrn_n_s32(…, 6)` + `vqmovn_s16` + `vmax_s8`. Tail scalar loop; padding to
`kPaddedOutputDimensions` is zero-filled.

`SqrClippedReLU::PropagatePair` (`sqr_clipped_relu.h:807`) computes both `sqr` and `clip` outputs
from one int32 input for SFNN: AVX-512 special-case `in==16` with `_mm512_cvtsepi32_epi16`, AVX2
`packs_epi32` + `permute4x64(0b11011000)`, then `sq = srli_epi16(mulhi_epi16(w,w), 2*6+7-16 = 3)` and
`cl = srli_epi16(max_epi16(w,0), 6)`.

### 6.6 SFNN packed tail (`sfnn_network.h`, `affine_transform_explicit.h:362-450`)

On AVX-512 when `H2 == 64`: `MakeHidden1InputPacked` packs the 14 activations (7 sqr + 7 clip for
H1=7) into 4 dwords, `fc_1.PropagateClippedReLUPackedToOutput` does `fc_1` (14→64) with the
broadcast loop, applies ClippedReLU in-register (`packs_epi32`/`srai`/`packs_epi16`/`max_epi8`),
and immediately dot-products with `fc_2`'s weights held in a lane-permuted copy
`weights_clipped_relu_packed_` (permutation `kPerm = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15}` built at
load) — never writing the 64-byte activation to memory.

---

## 7. Network file format (`nn.bin`)

### 7.1 Header and sections (`evaluate_nnue.cpp:518-565`, `evaluate_nnue.h:130-136`)

```
u32  version          = kVersion = 0x7AF32F16          (nnue_common.h)
u32  hash_value       = kHashValue
u32  size             = strlen(architecture)
char architecture[size]                                 (GetArchitectureString())
--- FeatureTransformer section ---
u32  FeatureTransformer::GetHashValue()
     biases  int16[kHalf]                 (LE; SFNN: sleb128 block)
     weights int16[kHalf * kInput]        (feature-major; SFNN: sleb128 block)
--- (SFNN with progressN only) Progress section ---
u32  0x6f50524f ("oPRO"); int32 bias_q16; int32 weights_q16[81][1548]
--- Network section, repeated kLayerStacks times ---
u32  Network::GetHashValue()
     layers, innermost first (see below)
EOF  (stream.peek() must be EOF, else FileCloseError)
```

* `ReadHeader` returns `FileMismatch` on a version mismatch; hash mismatches at every level are only
  `info string Warning` (`Detail::ReadParameters`, `LoadAndShare`).
* `GetArchitectureString()` = `"Features=" + FT::GetStructureString() + ",Network=" + Network::GetStructureString()`; for SFNN prefixed with `"ModelType=SFNNWithoutPsqt;"` and suffixed `"{LayerStack=N}"`.
* `kHashValue` = `FeatureTransformer::GetHashValue() ^ Network::GetHashValue()` (classic) or the constant `0x3c203b32` (`kSfnnBaseHashValue`, XOR `0x6f50524f` when progress buckets are used).
* Parameters are read into a temporary large-page buffer then published through
  `SystemWideSharedConstant<NnueNetworks> shared_networks` (`evaluate_nnue.cpp:429,472-515`,
  `source/shm.h`), keyed by a content hash so **multiple processes with the same net share one copy**.
  `NnueNetworks { FeatureTransformer feature_transformer; [Progress::Parameters progress;] Network network[kLayerStacks]; }` is `static_assert`ed trivially copyable.
* The net can be embedded in the binary with `incbin` (`INCBIN(EmbeddedNNUE, "nn.bin")`,
  `EvalDir = "<internal>"`) unless `NNUE_EMBEDDING_OFF`.

### 7.2 Per-layer serialisation

| Layer | Bytes written by `ReadParameters/WriteParameters` | Hash (`GetHashValue`) |
|---|---|---|
| `InputSlice<N, Offset>` | none | `0xEC42E90D ^ N ^ (Offset << 10)` |
| `AffineTransform[SparseInput]<Prev, Out>` | `int32 bias[Out]`, then `int8 weight[Out * pad32(In)]` row-major (in file order; scrambled only in memory) | `(0xCC03DAE4 + Out) ^ (prev>>1) ^ (prev<<31)` |
| `ClippedReLU<Prev>` / `SqrClippedReLU` | none | `0x538D24C7 + prev` |
| `Sum<...>` (`layers/sum.h`, unused by shipped archs) | recursive | `0xBCE400B4 ^ …` |
| `SfnnNetwork` | `fc_0` (bias int32[H1out], int8[H1out*pad(FT)]), `fc_1` (int32[H2], int8[H2*pad(2*H1)]), `fc_2` (int32[1], int8[pad(H2)]) | `0x6333718A` |

Classic layer stack (`architectures/halfkp_256x2-32-32.h`):

```
InputLayer   = InputSlice<kTransformedFeatureDimensions * 2>
HiddenLayer1 = ClippedReLU<AffineTransformSparseInput<InputLayer, 32>>
HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 32>>
OutputLayer  = AffineTransform<HiddenLayer2, 1>
Network      = OutputLayer
```

Because layers own their predecessor (`PreviousLayer previous_layer_`), `ReadParameters` recurses
innermost-first, so the file order is FT, fc1, fc2, fc3.

`read_leb_128` (`nnue_common.h`): magic `"COMPRESSED_LEB128"` (17 bytes, no NUL), `u32 byte_count`,
then signed LEB128 varints; asserted to end exactly at `byte_count`. Used only for the SFNN FT
(`USE_ELEMENT_WISE_MULTIPLY`). Classic FT is raw little-endian int16 (`read_little_endian`).

### 7.3 Shipped architecture headers (`nnue_architecture.h` selects by macro)

| Header | Macro | `RawFeatures` | FT (kHalf) | H1 | H2 | FT weight bytes |
|---|---|---|---|---|---|---|
| `halfkp_256x2-32-32.h` (default) | `EVAL_NNUE_HALFKP256` or none | HalfKP | 256 | 32 | 32 | 125 388×256×2 = 64.2 MB |
| `halfkp_512x2-16-32.h` | `YANEURAOU_ENGINE_NNUE_HALFKP_512X2_16_32` | HalfKP | 512 | 16 | 32 | 128 MB |
| `halfkp_1024x2-8-32.h` | `…_1024X2_8_32` | HalfKP | 1024 | 8 | 32 | 257 MB |
| `halfkp_1024x2-8-64.h` | `…_1024X2_8_64` | HalfKP | 1024 | 8 | 64 | 257 MB |
| `halfkpe9_256x2-32-32.h` | `EVAL_NNUE_HALFKPE9` | HalfKPE9 | 256 | 32 | 32 | 578 MB |
| `halfkpvm_256x2-32-32.h` | `EVAL_NNUE_HALFKP_VM_256X2_32_32` | HalfKP_vm | 256 | 32 | 32 | 35.7 MB |
| `kp_256x2-32-32.h` | `EVAL_NNUE_KP256` | `FeatureSet<K,P>` | 256 | 32 | 32 | 0.9 MB |
| `sfnn-1536.h` | `YANEURAOU_ENGINE_SFNN1536` (+`SFNNwoPSQT`) | HalfKA_hm2 | 1536 | 15 (+1 shortcut) | 32 | 73 305×1536×2 = 225 MB, `LayerStacks = 9` |
| generated `SFNN_*.h` / `NNUE_*.h` | `NNUE_ARCHITECTURE_HEADER` | any of §2 | free | free | free | `python3 nnue_arch_gen.py <name>` from the Makefile |

### 7.4 SFNN network (`architectures/sfnn_network.h`)

```
fc_0   : Fc0Layer (AffineTransformSparseInputExplicit<FT, H1 (+1 if shortcut)> or CommonShard)
ac_0   : ClippedReLUExplicit<H1>       ┐ both from fc_0_out[0..H1)
ac_sqr_0: SqrClippedReLU<H1>          ┘ concatenated as [sqr | clip] (2*H1 bytes, zero-padded to 32)
fc_1   : AffineTransformExplicit<2*H1, H2>
ac_1   : ClippedReLUExplicit<H2>
fc_2   : AffineTransformExplicit<H2, 1>
output += fc_0_out[H1]   if kUseShortcut (H1 % 8 == 7)
```

`SfnnNetwork<Fc0, In, H1, H2, UseShortcut = (H1 % 8 == 7)>`; static-asserts H1 is `8n` or `8n-1`.
Layer-stack index is chosen per evaluation by `stack_index_for_nnue(pos)` (`evaluate_nnue.cpp:840`):
`idx = hand_bucket; idx = idx*king_count + king_bucket; idx = idx*progress_count + progress_bucket`.
King buckets: `k3k3` (9: own/enemy king rank thirds, ranks normalised to stm view via `Inv`),
`k9k9` (81), `k9k9z`/`k13k13z` (81/169 zone schemes splitting the home ranks by file thirds),
`k21k21` (441), `k29k29` (841). Hand buckets: `hand4/16/64/64z/256/1024` from presence bits or a
hand-point score (`hand64z`: P=1, L/N=2, S/G=3, B/R=5, `min((score+3)/4, 7)`). Progress buckets
from `Progress::Parameters::Value0To255` (a KP-style int32 Q16 table thresholded through a logit
grid, cached in `StateInfo` only when progress buckets are compiled in). Full tables in
`architectures/README.md`. Small-FT SFNNs (`FT < 128`) define `NNUE_SMALL_SFNN_FT` and skip the
AVX2/512 pairwise path (`nnue_arch_gen.py:464-466`, `nnue_feature_transformer.h:190,247`).

### 7.5 Evaluate entry points (`evaluate_nnue.cpp:1057-1103`)

* `Eval::compute_eval(pos)` → `ComputeScore(pos, true)` (full refresh; called from `Position::set`).
* `Eval::evaluate(pos)` → returns cached `accumulator.score` if `computed_score`; optional
  `USE_EVAL_HASH` lookup (`ScoreKeyValue` 16-byte key/score, `EvaluateHashTable g_evalTable`);
  otherwise `ComputeScore(pos)`. Search calls it via `Search::YaneuraOuWorker::evaluate`
  (`yaneuraou-search.cpp:5179-5190`).
* `Eval::evaluate_with_no_return(pos)` → incremental accumulator update only.
* `ComputeScore` runs `Transform` → `network[bucket].Propagate(transformed_features, buffer)` →
  `output[0] / FV_SCALE` → clamp → cache in `accumulator.score`. Score is from the **side to move**.

---

## 8. Trainer conventions (`YaneuraOu-ScriptCollection/trainer/`)

**What it is.** `trainer.py` (1370 lines) is a *wrapper around dlshogi*
(`TadaoYamaoka/DeepLearningShogi`, cloned separately): it invokes `python -m dlshogi.train`
(legacy backend) or `python -m dlshogi.ptl` (PyTorch-Lightning backend) once per teacher file,
manages rounds/checkpoints/logs. It trains the **dlshogi policy+value CNN/Transformer network**
(default `--network exp___i20x256`), **not an NNUE**. There is no NNUE trainer, quantiser, or
`nn.bin` exporter anywhere in `YaneuraOu-ScriptCollection/` or in this `YaneuraOu/` checkout (no
`learn/` directory; the only `nn.bin` writer is `architectures/nnue_dummy_gen.py`, which emits
random/zero dummies for NPS measurement).

Defaults (from `trainer/readme.md` and `trainer.py::main`):

| Setting | Default | Notes |
|---|---|---|
| framework | PyTorch (dlshogi) / PyTorch Lightning 2.2 (`--backend ptl`) | `torch.compile` optional (`--use_compile`, `--compile_backend inductor`) |
| data | all `*.hcpe` / `*.hcpe3` in `train_dir`, sorted; one file = one "epoch"/checkpoint | `collect_teacher_files` |
| batchsize | 1024 (`--batches-per-update` for grad accumulation, train backend only) | |
| optimizer | `SGD(momentum=0.9, nesterov=True)`, `weight_decay=1e-4` | `trainer.py:1036-1039`, `837-845` |
| lr | 0.03 → `lr_min` 1e-5, `CosineLRScheduler(t_initial=files-1, cycle_limit=1)` stepped per file; or `ExponentialLR` | `cosine_scheduler_train_arg` |
| AMP | on, `bfloat16` (because exp_i has Transformer layers) | |
| SWA/EMA | on: `--use_swa --swa_freq 250 --swa_n_avr 10 --swa_start_epoch 1`; PTL `ema_decay = n_avr/(n_avr+1)` | |
| value-loss blend | `--val_lambda 1.0` (override per format: `--hcpe_val_lambda`, `--hcpe3_val_lambda`) | |
| `--use_average`, `--use_evalfix` | on (evalfix = fit an eval→winrate coefficient for HCPE3 data) | |
| gradient clip | 10.0 (PTL config) | `write_ptl_config` |
| logs parsed | `train loss avr = policy, result, value, total`; `test accuracy = policy, value` | `TRAIN_SUMMARY_RE`, `train_log_row_to_dict` |

**Loss.** From the log schema the loss has three terms: policy cross-entropy, a value-vs-game-result
term and a value-vs-eval term, mixed by `val_lambda`. *(Not from this tree, dlshogi knowledge:
`loss = policy_CE + (1-λ)·BCE(v, sigmoid(eval·0.0013226)) + λ·BCE(v, result)` with
`0.0013226 ≈ 1/756`; the `evalfix` option re-fits that coefficient from the data.)* Nothing in
this tree uses the classic NNUE "sigmoid(eval/600)" convention; if you need it for an NNUE
trainer it has to come from the nodchip/yaneuraou learner, which is absent here.

---

## 9. Packed position / teacher formats

### 9.1 `PackedSfen` — 32 bytes (`source/position.h:254`, `source/extra/sfen_packer.cpp`)

`struct PackedSfen { u8 data[32]; void flip(); PackedSfen flipped() const; Color color() const; }`
(`color()` = `data[0] & 1`). Written by `SfenPacker::pack` through a `BitStream` (LSB-first within
each byte: `data[cursor/8] |= 1 << (cursor & 7)`).

Bit layout (always exactly 256 bits, `ASSERT_LV3(stream.get_cursor() == 256)`):

| Bits | Content |
|---|---|
| 1 | side to move (0 = BLACK) |
| 7 + 7 | Black king square, White king square (`SQ_NB`=81 means "no king", for tsume boards) |
| ≤241 | every square `SQ_11..SQ_99` in `Square` order except the king squares: Huffman code + promote bit (not for gold) + colour bit |
| rest | hand pieces, Black then White, kind order `PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK` (`to_apery_pieces`), one entry per copy: board code with bit0 dropped, promote bit = 0 (not for gold), colour bit |
| tail | "piece box" (pieces missing from the board+hands, for handicap games): `huffman_table_piecebox` codes, encoded like a hand piece with promote bit = 1 (gold uses a special code that consumes the colour bit) — cshogi-compatible |

Huffman table (`huffman_table[]`, code LSB-first, then flags):

| Piece | board code (bits) | + promote | + colour | total |
|---|---|---|---|---|
| empty | `0` (1) | – | – | 1 |
| PAWN | `01` (2) | 1 | 1 | 4 |
| LANCE | `0011` (4) | 1 | 1 | 6 |
| KNIGHT | `1011` (4) | 1 | 1 | 6 |
| SILVER | `0111` (4) | 1 | 1 | 6 |
| GOLD | `01111` (5) | – | 1 | 6 |
| BISHOP | `011111` (6) | 1 | 1 | 8 |
| ROOK | `111111` (6) | 1 | 1 | 8 |

(codes as `{0x00,1},{0x01,2},{0x03,4},{0x0b,4},{0x07,4},{0x1f,6},{0x3f,6},{0x0f,5}` in the table;
piece-box table `{0x02,2},{0x09,4},{0x0d,4},{0x0b,4},{0x2f,6},{0x3f,6},{0x1b,5}`.) Hand pieces use
`code >> 1` with `bits-1`. The comment derives 41 empties + 72 + 24×4 + 16×2 = 241 bits for a full
board, and shows moving a piece to hand keeps the total constant. `Position::set_from_packed_sfen`
decodes directly into `Position` (with optional `mirror` via `Mir`) and fills `EvalList`
(piece numbers per kind via `piece_no_count[]`); `PackedSfen::flipped()` rotates 180° and swaps
colours at the packed level.

### 9.2 `PackedSfenValue` — 40 bytes

**Not defined in this YaneuraOu checkout** (grep for `PackedSfenValue` in `source/` returns nothing;
the historical `learn/` directory is absent). The ScriptCollection uses cshogi's definition
(`CommonLib/TeacherFormatLib.py`: `PSV = cshogi.PackedSfenValue`, asserted `PSV_SIZE == 40`). cshogi's
numpy dtype (queried from the installed package):

```
sfen        u1[32]   PackedSfen (cshogi Board.to_psfen / set_psfen)
score       <i2      eval from the side to move
move        <u2      YaneuraOu 16-bit move (cshogi.move16_to_psv / move16_from_psv)
gamePly     <u2
game_result i1       from side to move: +1 win, -1 loss, 0 draw   (TeacherFormatLib.game_result_for_side_to_move)
padding     u1
```

### 9.3 HCPE / HCPE3 (cshogi / dlshogi formats, as used in `CommonLib/TeacherFormatLib.py`)

`HCPE = cshogi.HuffmanCodedPosAndEval`, 38 bytes:

```
hcp        u1[32]   cshogi HuffmanCodedPos (Apery Huffman coding — a *different* 32-byte code from PackedSfen; cshogi exposes both to_hcp() and to_psfen())
eval       <i2
bestMove16 <i2
gameResult i1       absolute: 0 draw/unknown, 1 Black wins, 2 White wins
dummy      u1
```

HCPE3 (variable length per game), dtypes in `TeacherFormatLib.py`:

```
HCPE3_HEADER (36 B): hcp u1[32], moveNum <u2, result u1, gameInfo u1
  then moveNum × MOVE_INFO (6 B): selectedMove16 <i2, eval <i2, candidateNum <u2
       each followed by candidateNum × MOVE_VISITS (4 B): move16 <i2, visitNum <u2
```

`result` flags (`CommonLib/YaneShogiLib.py:665-670`): `HCPE3_DRAW=0, BLACK_WIN=1, WHITE_WIN=2,
RESULT_REPETITION=4, RESULT_NYUGYOKU=8, RESULT_MAX_MOVES=16` (low 2 bits = winner).
`GameDataDecoder`/`convert_pack_to_hcpe_file` (`YaneShogiLib.py:810`, `TeacherConvertLib.py:32`)
also read the GenSfen `.pack` stream: `u8 state` (1 = startpos; 0 → `hcp[32]` + `u16 ply`), then
repeated `{u16 move, i16 eval}` until a `move` whose two 7-bit square fields are equal, which
encodes the result, followed by one `u8`.

Conversions (`teacher/convert_teacher.py`): pack→hcpe, hcpe↔psv, hcpe3→hcpe, hcpe3→psv; `gamePly`
is lost going PSV→HCPE. `teacher/shuffle_split_teacher_external.py` buckets fixed-length records by
a hash of the 32-byte position.

---

## 10. Build/selection summary

| Macro | Effect |
|---|---|
| `EVAL_NNUE` | enable this evaluator (`config.h:428`, implies `USE_CLASSIC_EVAL`, `USE_EVAL_LIST`) |
| `EVAL_NNUE_HALFKP256`, `EVAL_NNUE_KP256`, `EVAL_NNUE_HALFKPE9`, `YANEURAOU_ENGINE_NNUE_HALFKP_{512X2_16_32,1024X2_8_32,1024X2_8_64}`, `EVAL_NNUE_HALFKP_VM_256X2_32_32`, `YANEURAOU_ENGINE_SFNN1536` | pick a fixed header in `nnue_architecture.h` |
| `NNUE_ARCHITECTURE_HEADER="architectures/<name>.h"` | Makefile-generated header via `nnue_arch_gen.py` for any `NNUE_*`/`SFNN_*` name |
| `SFNNwoPSQT` | SFNN: `USE_ELEMENT_WISE_MULTIPLY`, LEB128 FT, layer stacks, fixed FT hash |
| `NNUE_SFNN_HIDDEN1_7` | set when arch name contains `_7_`; disables `USE_NNUE_VNNI` |
| `NNUE_SMALL_SFNN_FT` | FT < 128: scalar pairwise path |
| `NNUE_HAS_SFNN_ACCUMULATOR_PROPAGATE` | fused accumulator→fc_0 (AVX-512, common+shard only at present) |
| `USE_FINNY_TABLES` | finny cache in `FeatureTransformer` — **not set by any build** |
| `USE_EVAL_HASH` | evaluation hash table |
| `NNUE_EMBEDDING_OFF` | do not `incbin` `nn.bin` |
| `USE_AVX512VNNI/USE_AVX512/USE_AVX2/USE_SSE42/41/SSSE3/SSE2/USE_NEON[=8]/USE_NEON_DOTPROD/USE_VNNI/USE_AVXVNNI/USE_WASM_SIMD` | kernel selection (§6.1) |

---

## 11. Ideas worth borrowing for a CPU MCTS shogi engine

1. **Finny tables are the right refresh strategy for MCTS.** Tree traversal is not a linear ply
   chain, so the one-ply `previous` link YaneuraOu relies on rarely helps; a per-thread
   `(perspective × king-square)` cache holding the last accumulator plus its active-index list
   (`FinnyEntry`) turns most "refreshes" into a handful of add/sub rows. Store the feature index
   list (YaneuraOu) or bitboards (Stockfish) — index lists are simpler with hand-piece counting.
2. Keep `Accumulator` inside the per-node/per-ply state (`StateInfo`) with two flags
   (`computed_accumulation`, `computed_score`) — enables lazy evaluation and score caching.
3. **Feature-major int16 weight rows + register-tiled add/sub** (`update_accumulator_tiled`,
   `kTileRegs`) so each changed feature touches memory once per tile.
4. Encode hand pieces as *count-indexed slots* (`f_hand_pawn + (n-1)`) — additive, incremental,
   and the enemy-perspective flip is a table swap (`kpp_hand_index[~c]`).
5. **Half-mirror (`_hm`) king buckets** halve FT parameters with negligible loss; the
   `A2`/`HalfKA2` trick (fold enemy king onto the own-king plane) shaves 81×kHalf more.
6. Pairwise-product FT activation (`Transform` under `USE_ELEMENT_WISE_MULTIPLY`) gives a cheaper
   and stronger nonlinearity than plain clipped ReLU and halves the first dense layer's input.
7. **Sparse first hidden layer** with `find_nnz` (cmpgt + movemask + 256×8 index LUT) — the FT
   output is ≥ 80 % zeros; only nonzero 4-byte chunks are multiplied.
8. Scramble int8 weights at load (`GetWeightIndexScrambled`) so the hot loop is
   `set1_epi32(4 inputs)` × `dpbusd/maddubs` with no shuffles; likewise `permute_weights` for the
   `packus` lane order and `weights_clipped_relu_packed_` for the fused tail.
9. Layer stacks selected by (hand bucket × king bucket × progress bucket) — cheap way to get
   phase-specific output heads; index composition order documented in `architectures/README.md`.
10. `SystemWideSharedConstant<NnueNetworks>`: put the network in content-hashed shared memory so
    several engine processes (or MCTS workers) share one 200 MB copy; `NnueNetworks` must be
    trivially copyable.
11. LEB128-compress the FT on disk (`COMPRESSED_LEB128`), keep raw int16 in memory.
12. Clamp the final score to `±VALUE_MAX_EVAL` and keep `FV_SCALE` a runtime option — trivial
    knob for matching a network's output scale to the search's cp domain.
13. `PackedSfen` (32 B, exactly 256 bits, deterministic Huffman) is a good key for transposition/
    dataset storage; cshogi implements the same coding (`to_psfen`) for Python tooling.
14. Optional `EvaluateHashTable` (`ScoreKeyValue`, 16 B atomic entries) in front of the net —
    cheap win when MCTS revisits leaves.
15. Everything is compile-time typed (`FeatureSet<…>`, layer templates, hashes in the file
    header) so an architecture mismatch is caught at load, not by silent garbage.

---

## 12. Licence note

YaneuraOu is distributed under **GPL-3.0** (`YaneuraOu/LICENSE`, "GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007"); YaneuraOu-ScriptCollection ships its own `LICENSE` file (not read in
detail here). Anything you copy from the files below into another engine inherits GPL-3.0
obligations; the *ideas* in §11 are described here in my own words.

Files read for this document (read-only, nothing modified):

* `YaneuraOu/source/eval/nnue/`: `evaluate_nnue.h`, `evaluate_nnue.cpp`, `nnue_common.h`,
  `nnue_accumulator.h`, `nnue_architecture.h`, `nnue_feature_transformer.h`, `wasm_simd.h`;
  `features/{features_common.h, feature_set.h, index_list.h, half_kp.{h,cpp}, half_ka1.{h,cpp},
  half_ka2.{h,cpp}, half_ka_hm1.{h,cpp}, half_ka_hm2.{h,cpp}, half_kpe9.{h,cpp}, half_kp_vm.{h,cpp},
  half_relative_kp.{h,cpp}, k.{h,cpp}, p.{h,cpp}, pe9.{h,cpp}, a2.{h,cpp}}`;
  `layers/{affine_transform.h, affine_transform_explicit.h, affine_transform_sparse_input.h,
  affine_transform_sparse_input_explicit.h, affine_transform_common_shard_input_explicit.h,
  clipped_relu.h, clipped_relu_explicit.h, sqr_clipped_relu.h, input_slice.h, sum.h, simd.h}`;
  `architectures/{README.md, halfkp_256x2-32-32.h, halfkp_512x2-16-32.h, halfkp_1024x2-8-32.h,
  halfkp_1024x2-8-64.h, halfkpe9_256x2-32-32.h, halfkpvm_256x2-32-32.h, kp_256x2-32-32.h,
  sfnn-1536.h, sfnn_network.h, nnue_arch_gen.py, nnue_dummy_gen.py}`.
* `YaneuraOu/source/`: `evaluate.h`, `eval/evaluate_bona_piece.cpp`, `eval/evaluate_mir_inv_tools.h`
  (grep), `extra/sfen_packer.cpp`, `position.h`, `position.cpp` (grep/excerpts), `search.h`,
  `engine/yaneuraou-engine/yaneuraou-search.cpp` (grep/excerpts), `types.h` (grep), `config.h`
  (excerpts), `Makefile` (excerpts), `LICENSE`.
* `YaneuraOu-ScriptCollection/`: `trainer/readme.md`, `trainer/trainer.py`,
  `CommonLib/TeacherFormatLib.py`, `CommonLib/TeacherConvertLib.py`, `CommonLib/YaneShogiLib.py`
  (excerpts), `teacher/README.md` (excerpts); the installed `cshogi` package was queried for its
  numpy dtypes.

Things I looked for and did **not** find in this tree: a `PackedSfenValue` struct, a `learn/`
directory or any NNUE trainer/exporter, a build flag enabling `USE_FINNY_TABLES`, and any
definition of `AccumulatorStack`/`AccumulatorCaches`/`EVAL_SFNN`.
