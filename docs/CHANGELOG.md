# JHBR5 changelog

## Unreleased — Phase 0 (study and design)

* Forked from JHBR3 `2835d45` ("Update training pipeline docs for v2.1 defaults
  and Attention Residuals"). The JHBR3 remote is kept as `jhbr3`.
* Added `docs/MONTY_NOTES.md`: how Monty's value net, policy net, move buckets,
  SEE doubling, PUCT variant, butterfly history, tree, datagen and trainers work,
  with recomputed constants (80,624 value inputs, 7,840 move buckets).
* Added `docs/YANEURAOU_NNUE_NOTES.md`: YaneuraOu NNUE feature sets,
  accumulator update, quantisation, SIMD kernels, file format, trainer and
  packed-sfen conventions.
* Added `docs/DESIGN.md`: proposed JHBR5 architecture (feature sets, move
  buckets, accumulator strategy, search changes, threading, training, data
  plan, open questions). Awaiting review before Phase 1.
* Added `THIRD_PARTY.md` (licence policy: GPL-3.0, no Monty code copied).
