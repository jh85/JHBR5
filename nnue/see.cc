// Static exchange evaluation for shogi (ShogiBoard::SeeGe).
//
// Re-implemented from the well-known swap-list algorithm (Stockfish
// `Position::see_ge`, YaneuraOu `Position::see_ge`, both GPL-3.0) on JHBR3's
// board API; no code copied. Differences from the chess version: drops
// capture nothing and put the dropped piece at risk; promotion adds the
// promotion gain and the promoted piece is the one at risk; pieces recapture
// at their board value (piece values follow YaneuraOu's scale, pawn = 90).
// Pins are ignored, as in YaneuraOu. Attackers are recomputed from the
// current occupancy every iteration, which handles x-rays without bookkeeping.

#include "shogi/board.h"

namespace lczero {

namespace {

// Indexed by PieceType::idx (0 = none).
constexpr int kSeeValue[16] = {
    0,     90,  315, 405, 495, 855, 990,  540,   // -, P, L, N, S, B, R, G
    15000, 540, 540, 540, 540, 945, 1395, 0,     // K, +P, +L, +N, +S, +B, +R
};

constexpr PieceType kAttackerOrder[] = {
    kPawn,   kLance,  kKnight,    kSilver, kGold,   kProPawn, kProLance,
    kProKnight, kProSilver, kBishop, kRook, kHorse, kDragon, kKing,
};

}  // namespace

bool ShogiBoard::SeeGe(Move m, int threshold) const {
  const Square to = m.to();
  Bitboard occ = occupied();
  int swap;
  PieceType landing = kKing;

  if (m.is_drop()) {
    swap = -threshold;
    landing = m.drop_piece();
  } else {
    const PieceType moved = piece_on(m.from()).GetType();
    swap = kSeeValue[piece_on(to).GetType().idx] - threshold;
    landing = moved;
    if (m.is_promotion()) {
      landing = moved.Promote();
      swap += kSeeValue[landing.idx] - kSeeValue[moved.idx];
    }
    occ.Clear(m.from());
  }
  if (swap < 0) return false;

  swap = kSeeValue[landing.idx] - swap;
  if (swap <= 0) return true;

  occ.Clear(to);
  Color stm = side_to_move_;
  int res = 1;

  while (true) {
    stm = ~stm;
    const Bitboard attackers = AttackersTo(to, occ, stm) & occ;
    if (attackers.Empty()) break;
    res ^= 1;

    PieceType chosen = kKing;
    Bitboard bb;
    for (PieceType pt : kAttackerOrder) {
      bb = attackers & pieces(stm, pt);
      if (bb.Any()) {
        chosen = pt;
        break;
      }
    }
    if (chosen == kKing) {
      // Capturing with the king is only possible if nothing recaptures.
      const Bitboard opp = AttackersTo(to, occ, ~stm) & occ;
      return opp.Any() ? (res ^ 1) : res;
    }
    swap = kSeeValue[chosen.idx] - swap;
    if (swap < res) break;
    occ.Clear(bb.Peek());
  }
  return res != 0;
}

}  // namespace lczero
