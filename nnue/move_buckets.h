// JHBR5 NNUE — policy move buckets (docs/DESIGN.md §4.3).
//
// bucket(m) in [0, kNumBuckets): dense (type, from, to) for board moves
// without promotion, then with promotion, then (hand kind, to) for drops, all
// in the side-to-move frame. With SEE doubling the row is
// kNumBuckets * see_good + bucket.

#pragma once

#include <cstddef>

#include "nnue/types.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

namespace jhbr5::nnue {

using lczero::Move;
using lczero::ShogiBoard;

void InitMoveBuckets();

int MoveBucket(const ShogiBoard& board, Move m);
bool MoveSeeGood(const ShogiBoard& board, Move m);
int MoveBucketSee(const ShogiBoard& board, Move m);

// Raw offset tables for hashing (see features.cc Init()).
const void* BucketOffsetsForHash();
size_t BucketOffsetsBytes();

struct BucketTables {
  lczero::Bitboard plain_dest[kTypeIds][81];
  int plain_off[kTypeIds][81];
  lczero::Bitboard promo_dest[kTypeIds][81];  // only tids 1..6 populated
  int promo_off[kTypeIds][81];
  lczero::Bitboard drop_dest[kHandKinds];
  int drop_off[kHandKinds];
};
const BucketTables& Buckets();

}  // namespace jhbr5::nnue
