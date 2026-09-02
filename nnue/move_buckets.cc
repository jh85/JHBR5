#include "nnue/move_buckets.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "nnue/features.h"

namespace jhbr5::nnue {

using lczero::Bitboard;
using lczero::Square;

namespace {

BucketTables g_b;
// Flat copy of every offset, in a fixed order, for the table hash.
int g_offsets_flat[kTypeIds * 81 * 2 + kHandKinds];

Bitboard RankMask(int rank) { return lczero::ShogiTables::RankBB[rank]; }

}  // namespace

void InitMoveBuckets() {
  const PairTables& pairs = Pairs();
  const Bitboard zone = lczero::ShogiTables::PromotionZoneBB[BLACK];  // ranks 0..2
  const Bitboard rank0 = RankMask(0);
  const Bitboard rank01 = RankMask(0) | RankMask(1);

  int running = 0;
  for (int tid = 0; tid < kTypeIds; ++tid) {
    for (int s = 0; s < 81; ++s) {
      Bitboard dest = pairs.dest[tid][s];
      // Squares where the piece could not stand unpromoted.
      if (tid == 1 || tid == 2) dest = dest & ~rank0;        // P, L
      if (tid == 3) dest = dest & ~rank01;                    // N
      g_b.plain_dest[tid][s] = dest;
      g_b.plain_off[tid][s] = running;
      running += dest.PopCount();
    }
  }
  if (running != kPlainBuckets) {
    std::fprintf(stderr, "nnue: plain buckets %d != %d\n", running, kPlainBuckets);
    std::abort();
  }
  for (int tid = 0; tid < kTypeIds; ++tid) {
    for (int s = 0; s < 81; ++s) {
      Bitboard dest = Bitboard::Zero();
      if (tid >= 1 && tid <= 6) {  // P L N S B R can promote
        const Square from = Square::FromIdx(static_cast<uint8_t>(s));
        dest = zone.Test(from) ? pairs.dest[tid][s] : (pairs.dest[tid][s] & zone);
      }
      g_b.promo_dest[tid][s] = dest;
      g_b.promo_off[tid][s] = running;
      running += dest.PopCount();
    }
  }
  if (running != kPlainBuckets + kPromoBuckets) {
    std::fprintf(stderr, "nnue: promo buckets %d != %d\n",
                 running - kPlainBuckets, kPromoBuckets);
    std::abort();
  }
  for (int h = 0; h < kHandKinds; ++h) {
    Bitboard dest = Bitboard::All();
    if (h == 0 || h == 1) dest = dest & ~rank0;   // P, L
    if (h == 2) dest = dest & ~rank01;            // N
    g_b.drop_dest[h] = dest;
    g_b.drop_off[h] = running;
    running += dest.PopCount();
  }
  if (running != kNumBuckets) {
    std::fprintf(stderr, "nnue: buckets %d != %d\n", running, kNumBuckets);
    std::abort();
  }

  int k = 0;
  for (int tid = 0; tid < kTypeIds; ++tid)
    for (int s = 0; s < 81; ++s) g_offsets_flat[k++] = g_b.plain_off[tid][s];
  for (int tid = 0; tid < kTypeIds; ++tid)
    for (int s = 0; s < 81; ++s) g_offsets_flat[k++] = g_b.promo_off[tid][s];
  for (int h = 0; h < kHandKinds; ++h) g_offsets_flat[k++] = g_b.drop_off[h];
}

const void* BucketOffsetsForHash() { return g_offsets_flat; }
size_t BucketOffsetsBytes() { return sizeof(g_offsets_flat); }
const BucketTables& Buckets() { return g_b; }

int MoveBucket(const ShogiBoard& board, Move m) {
  const Color stm = board.side_to_move();
  const int to = OwnSq(stm, m.to());
  if (m.is_drop()) {
    const int h = m.drop_piece().idx - 1;
    assert(g_b.drop_dest[h].Test(Square::FromIdx(static_cast<uint8_t>(to))));
    return g_b.drop_off[h] + (g_b.drop_dest[h] & BelowMask(to)).PopCount();
  }
  const int tid = TypeId(board.piece_on(m.from()).GetType());
  const int from = OwnSq(stm, m.from());
  const Bitboard& dest = m.is_promotion() ? g_b.promo_dest[tid][from]
                                          : g_b.plain_dest[tid][from];
  const int off = m.is_promotion() ? g_b.promo_off[tid][from]
                                   : g_b.plain_off[tid][from];
  assert(dest.Test(Square::FromIdx(static_cast<uint8_t>(to))));
  return off + (dest & BelowMask(to)).PopCount();
}

bool MoveSeeGood(const ShogiBoard& board, Move m) {
  return board.SeeGe(m, kSeeThreshold);
}

int MoveBucketSee(const ShogiBoard& board, Move m) {
  return MoveBucket(board, m) + (MoveSeeGood(board, m) ? kNumBuckets : 0);
}

}  // namespace jhbr5::nnue
