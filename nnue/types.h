// JHBR5 NNUE — shared constants and small types.
//
// Everything that defines the *meaning* of a feature index or a move bucket
// lives here and in features.cc / move_buckets.cc. The Python trainer must
// never re-implement any of it; it calls this code through pybind11.
//
// See docs/DESIGN.md sections 3–4 for the rationale.

#pragma once

#include <cstdint>

#include "shogi/types.h"

namespace jhbr5::nnue {

using lczero::BLACK;
using lczero::Color;
using lczero::PieceType;
using lczero::Square;
using lczero::WHITE;

// ---------------------------------------------------------------------------
// Piece type ids used by every table: 0 = king, 1..7 = JHBR3 PieceType idx
// (P L N S B R G), 8..13 = promoted (+P +L +N +S +B +R, JHBR3 idx 9..14 - 1).
// ---------------------------------------------------------------------------
constexpr int kTypeIds = 14;

constexpr int TypeId(PieceType pt) {
  return pt.idx == 8 ? 0 : (pt.idx < 8 ? pt.idx : pt.idx - 1);
}

constexpr PieceType TypeFromId(int tid) {
  return PieceType::FromIdx(static_cast<uint8_t>(
      tid == 0 ? 8 : (tid < 8 ? tid : tid + 1)));
}

// Hand piece kinds are indexed by JHBR3 PieceType idx - 1: 0 P, 1 L, 2 N,
// 3 S, 4 B, 5 R, 6 G.
constexpr int kHandKinds = 7;
constexpr int kHandMax[kHandKinds] = {18, 4, 4, 4, 2, 2, 4};
constexpr int kHandOff[kHandKinds] = {0, 18, 22, 26, 30, 32, 34};
constexpr int kHandSlotsPerOwner = 38;

// ---------------------------------------------------------------------------
// Piece slots ("BonaPiece"): every non-king piece occupies exactly one slot in
// a frame. Hands use thermometer encoding (k pieces -> slots 1..k).
// ---------------------------------------------------------------------------
constexpr int kHandSlots = 2 * kHandSlotsPerOwner;              // 76
constexpr int kBoardSlots = 2 * kTypeIds * 81;                  // 2268
constexpr int kNumSlots = kHandSlots + kBoardSlots;             // 2344

// owner: 0 = the frame's side, 1 = its opponent. sq is already in the frame.
constexpr int BoardSlot(int owner, int tid, int frame_sq) {
  return kHandSlots + (owner * kTypeIds + tid) * 81 + frame_sq;
}
constexpr int HandSlot(int owner, int hand_kind, int count_index /*1-based*/) {
  return owner * kHandSlotsPerOwner + kHandOff[hand_kind] + count_index - 1;
}

// ---------------------------------------------------------------------------
// Value net input groups.
// ---------------------------------------------------------------------------
// Group A: king-relative slots, one table shared by the two frames.
constexpr int kGroupAInputs = 81 * kNumSlots;                   // 189,864
constexpr int kMaxActiveA = 128;   // 38 in legal positions; test/tsume sfens may exceed

// Training-only virtual feature for group A (docs/DESIGN.md §9.3): board
// slot type/owner x king-relative offset (17x17); hand slots map to a zero
// padding row. Folded into the group-A table at export; the engine never
// uses it, but the index lives here so the trainer has no mapping code.
constexpr int kKPrelOffsets = 17 * 17;
constexpr int kKPrelInputs = 2 * kTypeIds * kKPrelOffsets + 1;   // 8093, last = padding
constexpr int KPrelIndex(int group_a_index) {
  const int k = group_a_index / kNumSlots;
  const int slot = group_a_index % kNumSlots;
  if (slot < kHandSlots) return kKPrelInputs - 1;
  const int b = slot - kHandSlots;
  const int owner_tid = b / 81;   // owner * kTypeIds + tid
  const int sq = b % 81;
  const int dx = sq / 9 - k / 9 + 8;
  const int dy = sq % 9 - k % 9 + 8;
  return owner_tid * kKPrelOffsets + dx * 17 + dy;
}

// Group B: absolute slots (stm frame) + attacker->target threat pairs.
constexpr int kPairsPerOwner = 8228;    // verified at Init()
constexpr int kThreatClasses = 16;      // target owner (2) x type class (8)
constexpr int kThreatInputs = 2 * kPairsPerOwner * kThreatClasses;  // 263,296
constexpr int kGroupBInputs = kNumSlots + kThreatInputs;            // 265,640
constexpr int kMaxActiveB = 512;        // 38 slots + threat pairs (bounded well below)

// Threat target class: P 0, L 1, N 2, S 3, gold-like 4, bishop-like 5,
// rook-like 6, king 7.
constexpr int ThreatClass(int tid) {
  switch (tid) {
    case 0: return 7;
    case 1: return 0;
    case 2: return 1;
    case 3: return 2;
    case 4: return 3;
    case 5: return 5;
    case 6: return 6;
    case 7: return 4;
    case 8: case 9: case 10: case 11: return 4;
    case 12: return 5;
    default: return 6;  // 13
  }
}

// ---------------------------------------------------------------------------
// Policy net inputs (stm frame): piece-square (king included) x {plain,
// attacked by them, defended by us, both} + hand slots.
// ---------------------------------------------------------------------------
constexpr int kPolicyBoardInputs = kBoardSlots * 4;             // 9072
constexpr int kPolicyInputs = kPolicyBoardInputs + kHandSlots;  // 9148
constexpr int kMaxActivePolicy = 40 + 2 * kHandSlotsPerOwner;   // 116 (hard bound)

// ---------------------------------------------------------------------------
// Architecture v2 (docs/NNUE_V2_DESIGN.md): king-bucketed group A, full-width
// SCReLU, phase-conditioned value head, bucketed+absolute policy inputs.
// Everything above is v1 and MUST NOT change (v1 nets stay bit-identical).
// ---------------------------------------------------------------------------
// 3x3 grid over the 9x9 board in frame coordinates (file/3 x 3 + rank/3).
constexpr int kKingBuckets = 9;
constexpr int KingBucket(int frame_sq) {
  return ((frame_sq / 9) / 3) * 3 + (frame_sq % 9) / 3;
}
static_assert(KingBucket(0) == 0 && KingBucket(80) == 8);

// Group A v2: v1 formula with the king bucket in place of the exact square.
constexpr int kGroupA2Inputs = kKingBuckets * kNumSlots;        // 21,096
static_assert(kGroupA2Inputs == 21096);

// Policy v2: bucketed slots (kings included) + v1 absolute flags + v1 hands.
constexpr int kPolicy2Inputs =
    kGroupA2Inputs + kPolicyBoardInputs + kHandSlots;           // 30,244
static_assert(kPolicy2Inputs == 30244);
constexpr int kMaxActivePolicy2 = 2 * kMaxActivePolicy;         // 232 (hard bound)

// Value head v2: material phase buckets and the widened L2.
constexpr int kPhaseBuckets = 8;
constexpr int kValue2L2 = 32;
// Material units per piece type id (PhaseBucket): king 0, P 1, L/N 3,
// S/G and promoted P/L/N/S (gold class) 5, B/+B 8, R/+R 9.
constexpr int kPhaseUnits[kTypeIds] = {0, 1, 3, 3, 5, 8, 9, 5, 5, 5, 5, 5, 8, 9};

// ---------------------------------------------------------------------------
// Move buckets (stm frame), see move_buckets.cc.
// ---------------------------------------------------------------------------
constexpr int kPlainBuckets = 8115;
constexpr int kPromoBuckets = 1397;
constexpr int kDropBuckets = 531;
constexpr int kNumBuckets = kPlainBuckets + kPromoBuckets + kDropBuckets;  // 10,043
constexpr int kNumBucketsSee = 2 * kNumBuckets;                            // 20,086
constexpr int kSeeThreshold = -90;

// ---------------------------------------------------------------------------
// Quantisation constants.
// ---------------------------------------------------------------------------
constexpr int kQA = 128;          // L1 weight scale and activation clamp
constexpr int kPolicyShift = 2;   // pairwise product >> 2 in the policy net (Monty FACTOR = 32)
constexpr int kPolicyFactor = 32;
constexpr int kPolicyQB = 128;    // policy readout weight scale
constexpr int kQPst = 256;        // PST skip weight scale
constexpr int kValueL2 = 16;
constexpr int kValueL3 = 32;

struct Wdl {
  float win = 0.0f;
  float draw = 0.0f;
  float loss = 0.0f;
  float Score() const { return win + 0.5f * draw; }
};

// Perspective helpers. FrameSq maps a real square into frame p (180° rotation
// for WHITE). OwnSq maps into a piece's own frame (the frame of its colour).
constexpr int FrameSq(Color p, Square sq) {
  return p == BLACK ? sq.as_idx() : 80 - sq.as_idx();
}
constexpr int OwnSq(Color c, Square sq) { return FrameSq(c, sq); }

// A fixed-capacity list of active feature indices.
template <int N>
struct FeatureList {
  int n = 0;
  int idx[N];
  void Push(int i) { idx[n++] = i; }
  void Clear() { n = 0; }
};

// One-time table construction (after lczero::ShogiTables::Init()).
void Init();

// Identifiers of the compiled mappings, stored in weight-file headers.
// FeatureSetId() is the v1 id; FeatureSetIdV2() the v2 id (docs/NNUE_V2_DESIGN.md).
uint32_t FeatureSetId();
uint32_t FeatureSetIdV2();
uint32_t BucketTableId(bool see_doubling);

}  // namespace jhbr5::nnue
