// JHBR5 NNUE — sparse feature extraction for the value and policy networks.
//
// Frames: see docs/DESIGN.md §2. All indices are computed without physically
// rotating the board: squares are mapped with FrameSq()/OwnSq() as they are
// popped from bitboards.
//
//   Group A (value, king-relative, one shared table, two frames):
//     A[ksq_p * kNumSlots + slot_p(piece)]           38 active per frame
//   Group B (value, frame = side to move):
//     absolute slots [0, kNumSlots) and threat pairs
//     kNumSlots + ((owner * kPairsPerOwner + pair(tid, own_from, own_to)) * 16 + class(target))
//   Policy (frame = side to move):
//     ((owner * 14 + tid) * 81 + sq) + kBoardSlots * (attacked_by_them + 2 * defended_by_us)
//     kPolicyBoardInputs + hand slot

#pragma once

#include <cstdint>

#include "nnue/types.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

namespace jhbr5::nnue {

using lczero::Bitboard;
using lczero::ShogiBoard;

// Board state a finny-table entry was built from. Bitboards are kept in REAL
// coordinates (never rotated); only the owner index is frame-relative.
struct FrameState {
  Bitboard pieces[2][kTypeIds];  // [owner][tid], king rows unused
  uint8_t hand[2][kHandKinds];
  bool initialized = false;
};

// King square of frame p, in frame p coordinates.
int KingFrameSq(const ShogiBoard& board, Color p);

// Group A, from scratch.
void GroupAFeatures(const ShogiBoard& board, Color p,
                    FeatureList<kMaxActiveA>* out);

// Group A as a diff against `st` (which is updated to the current board).
// Row indices already include the king-square base of the current king square
// (the caller guarantees `st` belongs to that king square).
void GroupADiff(const ShogiBoard& board, Color p, FrameState* st,
                FeatureList<128>* adds, FeatureList<128>* subs);

// Group B, from scratch (frame = side to move).
void GroupBFeatures(const ShogiBoard& board, FeatureList<kMaxActiveB>* out);

// Policy inputs (frame = side to move).
void PolicyFeatures(const ShogiBoard& board,
                    FeatureList<kMaxActivePolicy>* out);

// Squares attacked by at least one piece of colour c (current occupancy).
Bitboard AttackedSquares(const ShogiBoard& board, Color c);

// Dense index of the geometric pair (type, from, to) in the piece's own frame.
int PairIndex(int tid, int own_from, int own_to);

// Squares with index strictly below `idx` (for popcount ranking).
Bitboard BelowMask(int idx);

// Tables (exposed for tests and the feature dump tool).
struct PairTables {
  Bitboard dest[kTypeIds][81];  // empty-board reach of a BLACK piece of type tid
  int off[kTypeIds][81];        // prefix sums, tid-major
  int total = 0;
};
const PairTables& Pairs();

}  // namespace jhbr5::nnue
