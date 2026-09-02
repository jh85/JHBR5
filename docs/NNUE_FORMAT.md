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
