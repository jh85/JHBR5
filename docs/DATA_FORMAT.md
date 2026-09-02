# JHBR5 training record format

Normative source: `nnue/record.h`. Little endian throughout.

## File

```
FileHeader   16 bytes:  char magic[8] = "JHBR5RC\0", u32 version = 1, u32 reserved = 0
Record*      until end of file
```

Files are called *shards*; the trainer takes any number of them. There is no
index; records are variable length and read sequentially.

## Record

```
RecordHead   44 bytes
DistEntry    4 bytes × n_dist
```

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | u8[32] | sfen | YaneuraOu packed sfen (`shogi/packed_sfen.cc`) |
| 32 | i16 | score | search score in centipawns, **side to move** perspective; 0 if unknown |
| 34 | u16 | move | played or best move as JHBR3 `Move16` (`shogi/types.h`); 0 if none |
| 36 | u16 | game_ply | ply of the position in its game (1-based) |
| 38 | i8 | result | game result for the side to move: +1 win, 0 draw, −1 loss |
| 39 | u8 | flags | bit 0 `kHasDist`, bit 1 `kTeacherJhbr3`, bit 2 `kSelfPlay`, bit 3 `kResignAdjudicated`, bit 4 `kImportedPsv` |
| 40 | u16 | n_dist | number of distribution entries that follow |
| 42 | u16 | reserved | 0 |

The first 40 bytes are byte-identical to a YaneuraOu `PackedSfenValue`
(`sfen, score, move, gamePly, game_result, padding`), so a PSV file is
converted by prefixing the file header and appending `n_dist = 0,
reserved = 0` to each record. YaneuraOu's `move` is its own `Move16`; the
importer re-encodes it as JHBR3 `Move16` (same bit layout by construction:
to | from << 7 | drop << 14 | promote << 15).

`DistEntry { u16 move; u16 visits; }`: the root visit distribution of the
search that produced the record. `visits` are scaled so the most visited move
has 65535; moves with zero visits may be omitted. Moves are stored as
`Move16`, **not** as policy bucket ids: bucket tables are versioned with the
network (`bucket_table_id`), moves are not. The trainer maps moves to buckets
with the engine's own code (`jhbr5.move_bucket`).

## Conventions

* `score` and `result` are always from the side to move, as in PSV.
* A record with `n_dist = 0` is value-only; the policy trainer skips it.
* The value trainer uses `score` and `result` with a λ blend
  (`docs/DESIGN.md` §9.4); records with `score = 0` and `result = 0` are
  still valid (draw with unknown score).
