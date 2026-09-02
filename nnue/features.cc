#include "nnue/features.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "nnue/move_buckets.h"

namespace jhbr5::nnue {

using lczero::Piece;
using lczero::ShogiTables::BishopEffect;
using lczero::ShogiTables::LanceEffect;
using lczero::ShogiTables::RookEffect;

namespace {

PairTables g_pairs;
uint32_t g_feature_set_id = 0;
uint32_t g_bucket_table_id[2] = {0, 0};
bool g_initialized = false;

uint32_t Fnv1a(uint32_t h, const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

uint32_t FnvInt(uint32_t h, int v) { return Fnv1a(h, &v, sizeof(v)); }

// Empty-board reach of a BLACK piece of type `tid` on `sq`.
Bitboard EmptyBoardReach(int tid, Square sq) {
  const PieceType pt = TypeFromId(tid);
  const int i = sq.as_idx();
  const Bitboard zero = Bitboard::Zero();
  using namespace lczero::ShogiTables;
  switch (pt.idx) {
    case lczero::kPawn.idx: return PawnEffectBB[i][BLACK];
    case lczero::kLance.idx: return LanceEffect(BLACK, sq, zero);
    case lczero::kKnight.idx: return KnightEffectBB[i][BLACK];
    case lczero::kSilver.idx: return SilverEffectBB[i][BLACK];
    case lczero::kBishop.idx: return BishopEffect(sq, zero);
    case lczero::kRook.idx: return RookEffect(sq, zero);
    case lczero::kGold.idx:
    case lczero::kProPawn.idx:
    case lczero::kProLance.idx:
    case lczero::kProKnight.idx:
    case lczero::kProSilver.idx: return GoldEffectBB[i][BLACK];
    case lczero::kKing.idx: return KingEffectBB[i];
    case lczero::kHorse.idx: return BishopEffect(sq, zero) | HorseStepBB[i];
    case lczero::kDragon.idx: return RookEffect(sq, zero) | DragonStepBB[i];
    default: return zero;
  }
}

void InitPairs() {
  int running = 0;
  for (int tid = 0; tid < kTypeIds; ++tid) {
    for (int s = 0; s < 81; ++s) {
      g_pairs.dest[tid][s] = EmptyBoardReach(tid, Square::FromIdx(s));
      g_pairs.off[tid][s] = running;
      running += g_pairs.dest[tid][s].PopCount();
    }
  }
  g_pairs.total = running;
  if (running != kPairsPerOwner) {
    std::fprintf(stderr, "nnue: pair table has %d entries, expected %d\n",
                 running, kPairsPerOwner);
    std::abort();
  }
}

}  // namespace

void Init() {
  if (g_initialized) return;
  InitPairs();
  InitMoveBuckets();

  uint32_t h = 2166136261u;
  h = Fnv1a(h, "JHBR5-FS1", 9);
  h = FnvInt(h, kNumSlots);
  h = FnvInt(h, kGroupAInputs);
  h = FnvInt(h, kGroupBInputs);
  h = FnvInt(h, kPolicyInputs);
  h = FnvInt(h, kThreatClasses);
  h = Fnv1a(h, g_pairs.off, sizeof(g_pairs.off));
  g_feature_set_id = h;

  for (int see = 0; see < 2; ++see) {
    uint32_t b = 2166136261u;
    b = Fnv1a(b, "JHBR5-MB1", 9);
    b = FnvInt(b, kNumBuckets);
    b = FnvInt(b, see);
    b = FnvInt(b, kSeeThreshold);
    b = Fnv1a(b, BucketOffsetsForHash(), BucketOffsetsBytes());
    g_bucket_table_id[see] = b;
  }
  g_initialized = true;
}

uint32_t FeatureSetId() {
  assert(g_initialized);
  return g_feature_set_id;
}

uint32_t BucketTableId(bool see_doubling) {
  assert(g_initialized);
  return g_bucket_table_id[see_doubling ? 1 : 0];
}

const PairTables& Pairs() { return g_pairs; }

Bitboard BelowMask(int idx) {
  if (idx < lczero::kBBSplit) {
    return Bitboard::FromRaw((UINT64_C(1) << idx) - 1, 0);
  }
  return Bitboard::FromRaw(lczero::kBBMask0,
                           (UINT64_C(1) << (idx - lczero::kBBSplit)) - 1);
}

int PairIndex(int tid, int own_from, int own_to) {
  const Bitboard& dest = g_pairs.dest[tid][own_from];
  assert(dest.Test(Square::FromIdx(static_cast<uint8_t>(own_to))));
  return g_pairs.off[tid][own_from] + (dest & BelowMask(own_to)).PopCount();
}

int KingFrameSq(const ShogiBoard& board, Color p) {
  // Positions without a king (tsume problems, tests) use square 0 so that the
  // mapping stays total; real games always have both kings.
  const Square k = board.king_square(p);
  return k.IsValid() ? FrameSq(p, k) : 0;
}

namespace {

inline PieceType HandKindType(int h) {
  return PieceType::FromIdx(static_cast<uint8_t>(h + 1));
}

}  // namespace

void GroupAFeatures(const ShogiBoard& board, Color p,
                    FeatureList<kMaxActiveA>* out) {
  out->Clear();
  const int base = KingFrameSq(board, p) * kNumSlots;
  for (Color c : {BLACK, WHITE}) {
    const int owner = c != p;
    for (int tid = 1; tid < kTypeIds; ++tid) {
      Bitboard bb = board.pieces(c, TypeFromId(tid));
      while (bb.Any()) {
        const Square sq = bb.Pop();
        out->Push(base + BoardSlot(owner, tid, FrameSq(p, sq)));
      }
    }
    const lczero::Hand hand = board.hand(c);
    for (int h = 0; h < kHandKinds; ++h) {
      const int cnt = hand.Count(HandKindType(h));
      for (int k = 1; k <= cnt; ++k) out->Push(base + HandSlot(owner, h, k));
    }
  }
}

void GroupADiff(const ShogiBoard& board, Color p, FrameState* st,
                FeatureList<128>* adds, FeatureList<128>* subs) {
  adds->Clear();
  subs->Clear();
  const int base = KingFrameSq(board, p) * kNumSlots;
  for (Color c : {BLACK, WHITE}) {
    const int owner = c != p;
    for (int tid = 1; tid < kTypeIds; ++tid) {
      const Bitboard cur = board.pieces(c, TypeFromId(tid));
      const Bitboard old = st->pieces[owner][tid];
      Bitboard diff = cur ^ old;
      if (diff.Empty()) continue;
      Bitboard removed = diff & old;
      Bitboard added = diff & cur;
      while (removed.Any()) {
        const Square sq = removed.Pop();
        subs->Push(base + BoardSlot(owner, tid, FrameSq(p, sq)));
      }
      while (added.Any()) {
        const Square sq = added.Pop();
        adds->Push(base + BoardSlot(owner, tid, FrameSq(p, sq)));
      }
      st->pieces[owner][tid] = cur;
    }
    const lczero::Hand hand = board.hand(c);
    for (int h = 0; h < kHandKinds; ++h) {
      const int nc = hand.Count(HandKindType(h));
      const int oc = st->hand[owner][h];
      for (int k = oc + 1; k <= nc; ++k) adds->Push(base + HandSlot(owner, h, k));
      for (int k = nc + 1; k <= oc; ++k) subs->Push(base + HandSlot(owner, h, k));
      st->hand[owner][h] = static_cast<uint8_t>(nc);
    }
  }
  st->initialized = true;
}

void GroupBFeatures(const ShogiBoard& board, FeatureList<kMaxActiveB>* out) {
  out->Clear();
  const Color stm = board.side_to_move();
  const Bitboard occ = board.occupied();

  for (Color c : {BLACK, WHITE}) {
    const int owner = c != stm;
    // Absolute slots.
    for (int tid = 1; tid < kTypeIds; ++tid) {
      Bitboard bb = board.pieces(c, TypeFromId(tid));
      while (bb.Any()) {
        const Square sq = bb.Pop();
        out->Push(BoardSlot(owner, tid, FrameSq(stm, sq)));
      }
    }
    const lczero::Hand hand = board.hand(c);
    for (int h = 0; h < kHandKinds; ++h) {
      const int cnt = hand.Count(HandKindType(h));
      for (int k = 1; k <= cnt; ++k) out->Push(HandSlot(owner, h, k));
    }
    // Threat pairs (attacks on occupied squares, own pieces included).
    for (int tid = 0; tid < kTypeIds; ++tid) {
      const PieceType pt = TypeFromId(tid);
      Bitboard bb = board.pieces(c, pt);
      while (bb.Any()) {
        const Square from = bb.Pop();
        Bitboard att = board.PieceAttacks(pt, c, from, occ) & occ;
        const int own_from = OwnSq(c, from);
        while (att.Any()) {
          const Square to = att.Pop();
          const Piece target = board.piece_on(to);
          const int cls = (target.GetColor() != stm ? 8 : 0) +
                          ThreatClass(TypeId(target.GetType()));
          const int pair = PairIndex(tid, own_from, OwnSq(c, to));
          out->Push(kNumSlots +
                    ((owner * kPairsPerOwner + pair) * kThreatClasses + cls));
        }
      }
    }
  }
  assert(out->n <= kMaxActiveB);
}

Bitboard AttackedSquares(const ShogiBoard& board, Color c) {
  const Bitboard occ = board.occupied();
  Bitboard result = Bitboard::Zero();
  for (int tid = 0; tid < kTypeIds; ++tid) {
    const PieceType pt = TypeFromId(tid);
    Bitboard bb = board.pieces(c, pt);
    while (bb.Any()) {
      const Square from = bb.Pop();
      result |= board.PieceAttacks(pt, c, from, occ);
    }
  }
  return result;
}

void PolicyFeatures(const ShogiBoard& board,
                    FeatureList<kMaxActivePolicy>* out) {
  out->Clear();
  const Color stm = board.side_to_move();
  const Bitboard attacked = AttackedSquares(board, ~stm);
  const Bitboard defended = AttackedSquares(board, stm);
  for (Color c : {BLACK, WHITE}) {
    const int owner = c != stm;
    for (int tid = 0; tid < kTypeIds; ++tid) {
      Bitboard bb = board.pieces(c, TypeFromId(tid));
      while (bb.Any()) {
        const Square sq = bb.Pop();
        const int flags = (attacked.Test(sq) ? 1 : 0) + (defended.Test(sq) ? 2 : 0);
        out->Push((owner * kTypeIds + tid) * 81 + FrameSq(stm, sq) +
                  kBoardSlots * flags);
      }
    }
    const lczero::Hand hand = board.hand(c);
    for (int h = 0; h < kHandKinds; ++h) {
      const int cnt = hand.Count(HandKindType(h));
      for (int k = 1; k <= cnt; ++k) {
        out->Push(kPolicyBoardInputs + HandSlot(owner, h, k));
      }
    }
  }
}

}  // namespace jhbr5::nnue
