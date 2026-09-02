# Third-party code and ideas

JHBR5 is a fork of JHBR3, which is itself derived from Leela Chess Zero / Leela
Shogi Zero code (GPL-3.0). JHBR5 is therefore distributed under the **GNU GPL
v3** (see the licence headers in `shogi/*.h` and `mate/*.h`).

## Policy

* Ideas from Monty and YaneuraOu are re-implemented from the study notes in
  `docs/MONTY_NOTES.md` and `docs/YANEURAOU_NNUE_NOTES.md`; code is not pasted.
* Any exception (a verbatim or lightly edited copy) must be listed in the table
  below with the source file, the destination file, and the licence.
* Monty is **AGPL-3.0**. AGPL code combined into a GPL-3.0 program is permitted
  by GPL-3.0 §13, but the combined work then carries the AGPL network-use
  obligation. To keep JHBR5 plain GPL-3.0, **no Monty code is copied**; only
  designs, constants and formulas are reused, and those are documented in
  `docs/MONTY_NOTES.md` with file/function citations.
* YaneuraOu is **GPL-3.0**, licence-compatible with JHBR5. Re-implementation is
  still preferred so that JHBR5's board API stays self-consistent; where a
  routine is ported closely (e.g. SEE, packed-sfen codec) it is listed below.

## Inventory

| JHBR5 file | Origin | Licence | Nature |
|---|---|---|---|
| `shogi/packed_sfen.cc` (inherited from JHBR3) | YaneuraOu `source/extra/sfen_packer.cpp` | GPL-3.0 | re-implementation of the 32-byte Huffman packed sfen |
| `mate/dfpn.cc`, `mate/bns.cc`, `shogi/mate1ply.cc` (inherited from JHBR3) | YaneuraOu `source/mate/*`, dlshogi | GPL-3.0 | ports, see headers |
| `mate/shallow_mate.h` (inherited from JHBR3) | dlshogi `usi/mate.h` | GPL-3.0 | port, see header |
| `mcts/*` (inherited from JHBR3) | dlshogi `usi/UctSearch.cpp`, lc0 | GPL-3.0 | port/adaptation |

Entries for Phase 1+ (NNUE kernels, SEE, feature sets, trainer) will be added
as they are written. The intent is that every Phase 1+ entry reads
"re-implemented from notes".

## References (no code taken)

* Monty — https://github.com/official-monty/Monty — AGPL-3.0. Studied: value
  and policy network layout, move-bucket table, PUCT variant, butterfly history,
  policy softmax temperature, datagen and trainer conventions.
* YaneuraOu — https://github.com/yaneurao/YaneuraOu — GPL-3.0. Studied: NNUE
  feature sets, accumulator refresh/update, quantisation, SIMD kernels, network
  file format, packed sfen, learner conventions.
* Stockfish — GPL-3.0. Studied (from memory, not from a checkout): the
  "finny table" accumulator cache.
* bullet (jw1912/bullet) — MIT. Monty's trainers are built on it; JHBR5 uses
  PyTorch instead.
