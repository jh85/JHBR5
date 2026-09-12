# NNUE v2 architecture — design spec (implementation contract)

Status: APPROVED BY USER (2026-09-12). Supersedes nothing — v1 remains fully
supported; v2 is an additional architecture selected at net-file load time by
`NetHeader.version == 2`.

Motivation: v1 saturated in training (val loss and SPRT both flat across width,
init, and data-mix probes — see runs/dist1/FINAL_REPORT.md §4). The ceiling is
feature expressiveness, not capacity or training length. v2 raises
expressiveness where it is cheapest in NPS:

1. King-bucketed king-relative features (3×3 = 9 buckets) replacing the
   exact-king-square (81-row) group A — strictly more expressive than v1's
   factorised 81×2344 table, 9× smaller, and the finny cache gets a 9× higher
   hit rate (king shuffles within a bucket need no refresh).
2. Full-width SCReLU at L1 instead of pairwise-mul halving (3·l1 activation
   instead of 3·l1/2); same int16 range (max 128² = 16384), negligible cost.
3. Residual, phase-conditioned head: L2 widened 16→32, L3 32→32 with residual
   add, L4 conditioned on 8 material-phase buckets.
4. Policy v2: king-bucketed piece features added alongside the existing
   absolute attack/defend-flag features; full-width SCReLU; bucket readout
   unchanged.

Non-goals (document, do not implement): shared value/policy trunk, horizontal
mirroring, QAT (fake-quant forward), group-B changes, MCTS-target machinery
(visit distributions already exist in .rec/datagen).

## Invariants (hard requirements)

- v1 nets load and evaluate bit-identically to before; v1 training pipeline
  unchanged (`--arch v1` stays the default everywhere).
- v1 `feature_set_id()` / `bucket_table_id()` values MUST NOT change. Record
  them (`PYTHONPATH=build python -c "import jhbr5; ..."`) BEFORE editing and
  assert equality after.
- Python never re-implements index arithmetic (DESIGN.md §9.1): all new
  mappings live in C++ and are exposed through pyext.
- Scalar and AVX2 kernels bit-identical (enforced by nnue_kernels tests);
  AVX-512 stub untouched.
- Little-endian, 64-byte tensor alignment, header/payload CRC-32C — unchanged.
- All ctest tests pass: existing ones unmodified in behavior, plus new v2
  coverage listed in §8.

## 1. Feature mapping v2 (C++, nnue/types.h + nnue/features.{h,cc})

Frames/perspective unchanged: `FrameSq(p, sq) = sq if p==BLACK else 80−sq`,
square = file*9+rank. Piece slots (`kNumSlots = 2344`: 76 thermometer hand
slots + 2268 board slots) unchanged. Owner is frame-relative as in v1.

New constants:
- `kKingBuckets = 9`. `KingBucket(frame_sq) = ((frame_sq/9)/3)*3 + (frame_sq%9)/3`
  (i.e. file/3 × 3 + rank/3; a 3×3 grid over the 9×9 board, in frame coords,
  so 180°-flip invariance is inherited).
- `kGroupA2Inputs = kKingBuckets * kNumSlots = 21096`.
- `kPolicy2Inputs = kGroupA2Inputs + kBoardSlots*4 + kHandSlots`
  = 21096 + 9072 + 76 = **30244**.
- `kPhaseBuckets = 8`, `kValue2L2 = 32` (L3 stays 32).

**Group A v2** (value net, both frames): for frame p with king frame-square k,
`index = KingBucket(k)*kNumSlots + slot_p(piece)` — v1 formula with bucket in
place of the exact king square. Kingless convention preserved (k=0 → bucket 0).
One shared table for both frames, exactly as v1. Active count identical to v1
(≤ `kMaxActiveA`).

**Group B v2 = group B v1, unchanged** (absolute slots + threat pairs,
`kGroupBInputs = 265640`, from scratch each eval, PST skip unchanged).

**Phase buckets** (value head conditioning): material units over board + hands,
both colors: `units = 1·P + 3·(L,N) + 5·(S,G,+P,+L,+N,+S) + 8·(B,+B) + 9·(R,+R)`
(promoted pieces count as their base type's class; gold-class = 5). Max units
= 104. `phase = min(units*8/105, 7)` ∈ [0,8). Implemented ONCE in C++
(`PhaseBucket(const ShogiBoard&)`), used by engine eval, BatchReader, and
exposed to Python for tests.

**Policy v2 inputs**: three concatenated ranges —
1. `[0, 21096)`: bucketed slots, king-relative, INCLUDING king pieces
   (tid 0 live, unlike group A): `KingBucket(KingFrameSq(board,stm))*2344 + slot`.
2. `[21096, 30168)`: v1 absolute flag features verbatim:
   `21096 + ((owner*14+tid)*81 + FrameSq(stm,sq)) + 2268*(attacked_by_them(sq)
   + 2*defended_by_us(sq))` (same AttackedSquares bitboards as v1).
3. `[30168, 30244)`: v1 absolute hand slots verbatim: `30168 + HandSlot(...)`.

**feature_set_id v2**: FNV-1a over `"JHBR5-FS2"`, the v2 dims
(21096 / 265640 / 30244 / 9 / 8 …) and the same offset tables as v1. Any
change to the mapping must change the hash. `bucket_table_id` unchanged (move
buckets untouched).

## 2. Value net v2 (engine)

Finny cache: `acc_[2][kKingBuckets][l1]` int16 + `FrameState state_[2][9]`
(replaces [2][81] on the v2 path only). Refresh keyed by
`(frame p, KingBucket(KingFrameSq(board,p)))`; same XOR-diff/thermometer logic
(`GroupA2Diff`), same `simd::UpdateRows`. Worst-case diff on bucket change = all
38 pieces — within `kMaxActiveA = 128`.

Evaluate v2:
1. `acc_us/acc_them` from finny cache (scale ×128), `acc_b` from scratch (v1).
2. **ScreluFull**: `act_g[i] = clamp(acc_g[i],0,128)²` for the FULL l1 (no
   halving, no shift), int16 (max 16384). Concatenated act width **3·l1**.
   New simd kernel `simd::ScreluFull` (scalar + AVX2, bit-identical).
3. L2 (32): `s_j = l2_w[j]·act` (i16×i16 int32 dot), `x[j] = (s_j/kQA² + l2_b[j])/qb`
   with calibrated power-of-two qb (same calibration machinery as v1).
4. L3 (32→32, f32) with residual: `y1 = screlu(x)`, `h = y1 + screlu(l3(y1))`.
5. L4 per-phase: `z = l4_b[ph] + l4_w[ph] @ h`, ph = PhaseBucket(board) — 3 logits.
6. PST skip from group-B features (unchanged). Softmax → WDL. cp conversion
   unchanged.

Tensors (value v2): `a_w` i8 [21096, l1], `a_b` i16 [l1], `b_w` i8
[265640, l1], `b_b` i16 [l1], `l2_w` i16 [32, 3·l1], `l2_b` i16 [32],
`l3_w` f32 [32,32], `l3_b` f32 [32], `l4_w` f32 [8,3,32] (rank-3 OK),
`l4_b` f32 [8,3], `pst` i16 [265640, 3]. Defaults l1 = 1024 → file ≈ 300 MB.

## 3. Policy net v2 (engine)

Hidden: from-scratch `acc = l1_b + Σ l1_w[idx]` over the 30244-input v2 mapping
(typical active ≈ v1's 55 + king-bucketed extras; raise kMaxActivePolicy if
needed — measure in tests). Activation: full-width SCReLU with the SAME output
scale as v1's pairwise path: `hl[i] = clamp(acc[i],0,kQA)² >> kPolicyShift`
(width l1 = 4096 instead of l1/2). `Logit`: `(dot_i16_i8(hl, out_w[bucket]) /
(kQA·kPolicyFactor) + out_b[bucket]) / kPolicyQB` — constants unchanged;
`out_w` rows now l1-wide.

Tensors (policy v2): `l1_w` i8 [30244, l1], `l1_b` i16 [l1], `out_w` i8
[rows, l1], `out_b` i16 [rows]. l1 = 4096 → ≈ 124 MB + 82 MB readout.

## 4. File format (nnue/net_format.{h,cc})

- `version = 2` for v2 files; `kind` semantics unchanged (1 value / 2 policy).
- Add `uint16_t n_phase` to the header (8 for value v2; 0 for policy v2 and
  all v1 files). Place inside existing reserved/padding space of the 256-byte
  header if available; otherwise extend the struct (still 256 B total,
  header CRC covers it as before).
- Loader dispatch on `version`: v1 → current checks/paths verbatim; v2 value →
  `l1%256==0`, `l2==32`, `l3==32`, `qa==128`, `q_pst==256`, `n_phase==8`;
  v2 policy → `l1%256==0`, `qa==128`, `qb==128`, `n_phase==0`.
- `NetWriter` extended with `version` + `n_phase`; `jhbr5.write_net` gains
  matching kwargs (default version=1 → byte-identical v1 output).

## 5. pyext (pyext/jhbr5_py.cc)

New constants: `KING_BUCKETS`, `PHASE_BUCKETS`, `GROUP_A2_INPUTS`,
`POLICY2_INPUTS`, `VALUE2_L2`. New functions: `king_bucket(sq_array)`,
`phase_bucket(sfen)`, `feature_set_id(arch=1|2)` (keyword arg, default 1),
`value_features(sfen, arch=1|2)` (arch=2 → v2 indices + also return phase),
`policy_features(sfen, arch=1|2)`. `write_net(..., version=1, n_phase=0)`.
`BatchReader(..., arch=1)`: arch=2 → v2 group-A / policy indices, extra batch
field `phase` (int32 [n]), skip `a_*_kp` computation entirely (v2 has no
factoriser). `RecordWriter` unchanged.

## 6. Training (train/)

- `model.py`: add `ValueNetV2(l1=1024)` — `a = SparseGroup(GROUP_A2_INPUTS,
  l1, 38)`, `b` identical to v1, no factorisers, `l2 = Linear(3*l1, 32)`,
  `l3 = Linear(32, 32)`, `l4_w/l4_b` phase-indexed (embedding or parameter
  gather by `batch['phase']`), `pst` as v1. Forward mirrors the int pipeline in
  float the same way v1 does (`screlu_full(x) = clamp(x,0,1)²`), residual add,
  phase-gathered L4, PST skip. Add `PolicyNetV2(l1=4096)` — hidden
  `SparseGroup(POLICY2_INPUTS, l1, ~95)`, full-width screlu, `out_w`
  Embedding(rows, l1). Keep v1 classes untouched.
- `train.py`: `--arch {v1,v2}` (default v1) selecting net class + BatchReader
  arch + export path. Losses, schedules, `--export` flow unchanged.
- `quantized.py`: v2 reference forwards (`value_forward_q_v2`,
  `policy_forward_q_v2`) for the round-trip test; `calibrate_qb` reused for
  v2's wider L2 input (verify headroom analysis: input width 3·l1 vs 3·l1/2).
- `export.py`: `export_value_v2` / `export_policy_v2` → `write_net(version=2,
  n_phase=8|0)`. No factoriser folding for v2 (a_w written directly).

## 7. Engine plumbing

- `ValueNet::Load` / `PolicyNet::Load` dispatch on header version; v1 and v2
  objects coexist (separate scratch types or templated paths — implementer's
  choice, v1 path must remain bit-identical).
- `make_random_net` gains `--arch v2` (used by tests).
- `dump_nnue_features` gains `--arch 2` (python-reference test input).
- No USI option changes; `ValueNet`/`PolicyNet` paths just work with either
  version. `EnsureNetworks`, datagen `--value/--policy` untouched.

## 8. Tests (all must pass; extend, don't weaken)

- `test_nnue_features.cc`: KingBucket range/uniqueness/flip-invariance;
  GroupA2 range/uniqueness/active-count; GroupA2Diff == from-scratch and
  idempotent; Policy2 range/uniqueness; PhaseBucket bounds + flip-invariance
  (units are color-symmetric).
- `test_nnue_kernels.cc`: ScreluFull scalar vs AVX2 bit-identical.
- `test_nnue_net.cc`: v2 — scalar vs AVX2 bit-identical WDL/logits; finny ==
  fresh along random walks (incl. king moves within/across buckets);
  determinism. Keep v1 assertions intact.
- `test_net_roundtrip.py`: v2 case — PyTorch → export → engine vs
  quantized.py reference (same tolerances as v1).
- `test_nnue_features_ref.py`: extend pure-Python reference with v2 mapping
  (bucket formula only — slots/threats/buckets reused).
- `test_train_smoke.py`: v2 case — synthetic shard → few steps → export →
  engine loads.
- Record v1 feature/bucket ids before and after; assert unchanged.

## 9. Docs to update in the same branch

- `docs/NNUE_FORMAT.md`: v2 header fields + v2 tensor tables.
- `docs/DESIGN.md`: new §"NNUE v2" summarizing this file + rationale +
  measured NPS/size deltas.
- `docs/CHANGELOG.md`: entry.
- `nets/README.md`: note v1/v2 coexistence.
