# JHBR5 network file format (`.nn`)

Normative source: `nnue/net_format.h`. This page is for the Python exporter
and for reviewers.

## Layout

```
NetHeader        256 bytes
TensorDesc[n]    64 bytes each (n = header.n_tensors)
payload          header.payload_bytes; each tensor starts 64-byte aligned
```

All integers little endian.

### NetHeader

| Offset | Type | Field | Value |
|---|---|---|---|
| 0 | char[8] | magic | `"JHBR5NN\0"` |
| 8 | u32 | version | 1 |
| 12 | u32 | kind | 1 value, 2 policy |
| 16 | u32 | feature_set_id | `nnue::FeatureSetId()` (FNV-1a over the mapping tables) |
| 20 | u32 | bucket_table_id | policy: `nnue::BucketTableId(see)`; value: 0 |
| 24 | u32[4] | l1 | value: `[L1, L1, L1, 0]`; policy: `[L1p, 0, 0, 0]` |
| 40 | u32 | l2 | value: 16 |
| 44 | u32 | l3 | value: 32 |
| 48 | u32 | qa | 128 |
| 52 | u32 | qb | value: calibrated (power of two); policy: 128 |
| 56 | u32 | q_pst | 256 |
| 60 | u32 | n_tensors | |
| 64 | u64 | payload_bytes | |
| 72 | u32 | payload_crc32c | CRC-32C (Castagnoli) of the payload bytes incl. padding |
| 76 | u32 | header_crc32c | CRC-32C of the 256-byte header with this field zeroed |
| 80 | u8[176] | reserved | zero |

### TensorDesc

| Offset | Type | Field |
|---|---|---|
| 0 | char[16] | name, NUL padded |
| 16 | u32 | dtype: 1 i8, 2 i16, 3 i32, 4 f32 |
| 20 | u32 | rank (1..3) |
| 24 | u64[3] | shape, row-major |
| 48 | u64 | offset from payload start (multiple of 64) |
| 56 | u8[8] | reserved |

## Value network tensors

| Name | dtype | shape | scale |
|---|---|---|---|
| `a_w` | i8 | [189864, L1] | ×128 (group A, shared by both frames) |
| `a_b` | i16 | [L1] | ×128 |
| `b_w` | i8 | [265640, L1] | ×128 (group B) |
| `b_b` | i16 | [L1] | ×128 |
| `l2_w` | i16 | [16, 3·L1/2] | ×qb |
| `l2_b` | i16 | [16] | ×qb |
| `l3_w`, `l3_b` | f32 | [32, 16], [32] | |
| `l4_w`, `l4_b` | f32 | [3, 32], [3] | output order W, D, L |
| `pst` | i16 | [265640, 3] | ×256, added to the W/D/L logits for each active group-B feature |

Forward pass: `nnue/value_net.cc: ValueScratch::Evaluate`.

## Policy network tensors

| Name | dtype | shape | scale |
|---|---|---|---|
| `l1_w` | i8 | [9148, L1p] | ×128 |
| `l1_b` | i16 | [L1p] | ×128 |
| `out_w` | i8 | [rows, L1p/2] | ×128, rows = 20086 (SEE doubling) or 10043 |
| `out_b` | i16 | [rows] | ×128 |

Forward pass: `nnue/policy_net.cc`. Hidden `hl = (clamp(a,0,128)·clamp(b,0,128)) >> 2`;
`logit = (dot(hl, out_w[row]) / (128·32) + out_b[row]) / 128`.

## Version 2

The v2 architecture (`docs/NNUE_V2_DESIGN.md`) is selected per file by
`header.version == 2`; `kind` semantics are unchanged (1 value, 2 policy), as
are the layout, alignment and both CRCs. v1 files and loaders are untouched.

### NetHeader differences

| Offset | Type | Field | Value |
|---|---|---|---|
| 8 | u32 | version | **2** |
| 16 | u32 | feature_set_id | `nnue::FeatureSetIdV2()` = 2746015459 (FNV-1a over `"JHBR5-FS2"`, the v2 dims 21096/265640/30244/9/8 and the same threat-pair offset table as v1) |
| 24 | u32[4] | l1 | same convention as v1: value `[L1, L1, L1, 0]`, policy `[L1p, 0, 0, 0]` |
| 40 | u32 | l2 | value: **32** |
| 80 | u16 | n_phase | value v2: **8**; policy v2 and all v1 files: 0 (carved out of `reserved`, which shrinks 176 → 174 bytes; files written before this field existed read 0 because the reserved space is zero) |

`bucket_table_id`, `l3`, `qa`, `qb`, `q_pst` keep their v1 meanings.

### Loader validation (v2)

`version == 2` dispatches to `LoadV2`; anything else takes the v1 path
verbatim. The v2 value loader (`nnue/value_net.cc: ValueNet::LoadV2`) requires:
`kind == 1`; `feature_set_id == FeatureSetIdV2()`; `l1[0] > 0`,
`l1[0] % 256 == 0`, `l1[1] == l1[0]`, `l1[2] == l1[0]`; `l2 == 32`;
`l3 == 32`; `qa == 128`; `q_pst == 256`; `n_phase == 8`; `qb > 0`
(calibrated power of two, as v1). The v2 policy loader (`nnue/policy_net.cc:
PolicyNet::LoadV2`) requires: `kind == 2`; `bucket_table_id` equal to
`BucketTableId(see)` for `see` true/false (unchanged, selects `rows`); the v2
feature set id; `l1[0] > 0`, `l1[0] % 256 == 0`; `qa == 128`; `qb == 128`;
`n_phase == 0`. Every tensor must then match its name, dtype and shape below
exactly (shape entries of 0 are not used by these loaders).

### Value network tensors (v2)

| Name | dtype | shape | scale |
|---|---|---|---|
| `a_w` | i8 | [21096, L1] | ×128 (group A v2: 9 king buckets × 2344 slots, shared by both frames) |
| `a_b` | i16 | [L1] | ×128 |
| `b_w` | i8 | [265640, L1] | ×128 (group B, unchanged from v1) |
| `b_b` | i16 | [L1] | ×128 |
| `l2_w` | i16 | [32, 3·L1] | ×qb |
| `l2_b` | i16 | [32] | ×qb |
| `l3_w`, `l3_b` | f32 | [32, 32], [32] | |
| `l4_w`, `l4_b` | f32 | [8, 3, 32], [8, 3] | per-phase, output order W, D, L |
| `pst` | i16 | [265640, 3] | ×256, added to the W/D/L logits for each active group-B feature |

Forward pass (`nnue/value_net.cc: ValueScratch::EvaluateV2`): full-width
SCReLU `act_g[i] = clamp(acc_g[i],0,128)²` (i16, ≤ 16384, no halving, no
shift), concatenated `h` of width 3·L1; L2
`x[j] = (l2_w[j]·h / 128² + l2_b[j]) / qb` (32 outputs); `y1 = screlu(x)`;
residual L3 `h2 = y1 + screlu(l3_w·y1 + l3_b)`; per-phase L4
`z = l4_b[ph] + l4_w[ph]·h2` with `ph = PhaseBucket(board)` (8 material-phase
buckets); PST skip unchanged; softmax → (W, D, L).

### Policy network tensors (v2)

| Name | dtype | shape | scale |
|---|---|---|---|
| `l1_w` | i8 | [30244, L1p] | ×128 (21096 king-bucketed slots + 9072 absolute attack/defend flags + 76 hands) |
| `l1_b` | i16 | [L1p] | ×128 |
| `out_w` | i8 | [rows, L1p] | ×128, rows = 20086 (SEE doubling) or 10043 |
| `out_b` | i16 | [rows] | ×128 |

Hidden: full-width SCReLU with the v1 output scale,
`hl[i] = clamp(acc[i],0,128)² >> 2`, width L1p (not L1p/2). Logit formula
unchanged: `logit = (dot(hl, out_w[row]) / (128·32) + out_b[row]) / 128`.
