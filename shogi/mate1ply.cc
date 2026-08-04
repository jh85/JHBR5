/*
  Fast approximate mate-in-1 probe.

  Faithful port of Apery/cshogi's Position::mateMoveIn1Ply<US, false>
  (cshogi src/position.cpp, GPLv3) onto JHBR3's ShogiBoard. Sound but
  deliberately incomplete, exactly like the original: only drops and
  board moves landing next to the defending king (plus knight checks)
  are considered, which removes interposition handling entirely and
  keeps the candidate set tiny. Distant slider mates are not found here;
  a caller's deeper search discovers them one expansion later.

  Every returned move is a legal checkmate. The routine requires the
  side to move NOT to be in check (returns null otherwise).
*/

#include <cassert>
#include "shogi/bitboard.h"
#include "shogi/board.h"

namespace lczero {

namespace {

using namespace ShogiTables;

// Ranks strictly in front of `rank` from `c`'s point of view.
// BLACK looks toward rank index 0.
struct FrontTables {
  Bitboard f[COLOR_NB][kBoardSize];
  FrontTables() {
    for (int r = 0; r < 9; r++) {
      Bitboard b = Bitboard::Zero();
      for (int r2 = 0; r2 < r; r2++) b |= RankBB[r2];
      f[BLACK][r] = b;
      b = Bitboard::Zero();
      for (int r2 = r + 1; r2 < 9; r2++) b |= RankBB[r2];
      f[WHITE][r] = b;
    }
  }
};

inline Bitboard InFrontOf(Color c, int rank_idx) {
  static const FrontTables t;
  return t.f[c][rank_idx];
}

template <Color Us>
inline Bitboard PromoZone() {
  return Us == BLACK ? RankBB[0] | RankBB[1] | RankBB[2]
                     : RankBB[6] | RankBB[7] | RankBB[8];
}

}  // namespace

// them's king cannot capture `sq` (guarded) and cannot step to any
// square of `mask`; can it escape at all? `occ_now` excludes the moved
// piece; `sq` is the checker's square.
bool ShogiBoard::MateApproxKingCanEscape(Color us, Square sq,
                                         const Bitboard& mask,
                                         const Bitboard& occ_now) const {
  const Color them = ~us;
  const Square ksq = king_sq_[them];
  Bitboard king_moves = mask.AndNot(
      pieces(them).AndNot(ShogiTables::KingEffectBB[ksq.as_idx()]));
  king_moves.Clear(sq);
  if (!king_moves.Any()) return false;

  Bitboard temp_occ = occ_now;
  temp_occ.Set(sq);
  temp_occ.Clear(ksq);
  while (king_moves.Any()) {
    const Square to = king_moves.Pop();
    if (!IsSquareAttacked(to, temp_occ, us)) return true;
  }
  return false;
}

// Can a non-king defender piece legally capture on `sq`? `dc_them` are
// the defender pieces pinned by our sliders (capturing off the pin line
// is illegal for them). `occ_now` excludes our moved piece.
bool ShogiBoard::MateApproxPieceCanCapture(Color them, Square sq,
                                           const Bitboard& dc_them,
                                           const Bitboard& occ_now) const {
  Bitboard from_bb = AttackersTo(sq, occ_now) & pieces(them);
  from_bb.Clear(king_sq_[them]);
  const int ksq = king_sq_[them].as_idx();
  while (from_bb.Any()) {
    const Square from = from_bb.Pop();
    if (!dc_them.Test(from) ||
        ShogiTables::LineBB[from.as_idx()][ksq].Test(sq)) {
      return true;
    }
  }
  return false;
}

template <Color Us>
Move ShogiBoard::FindMateInOneApproxImpl() {
  constexpr Color Them = Us == BLACK ? WHITE : BLACK;
  const Square ksq = king_sq_[Them];
  const int k = ksq.as_idx();
  const int krank = ksq.rank().idx;
  const Square our_ksq = king_sq_[Us];
  const Bitboard occ = occupied();
  const Bitboard them_bb = pieces(Them);
  const Bitboard us_bb = pieces(Us);
  const Bitboard empty = ~occ;
  const Hand our_hand = hand_[Us];
  // BLACK attacks a king from the south (rank+1); WHITE from the north.
  const int t_south = Us == BLACK ? +1 : -1;  // rank delta toward us

  // Defender pieces pinned by our sliders, before our move. Drops never
  // change it.
  const Bitboard dc_them = ComputeBlockersForKing(Them) & them_bb;

  auto king_escapes = [&](Square sq, const Bitboard& mask,
                          const Bitboard& occ_now) {
    return MateApproxKingCanEscape(Us, sq, mask, occ_now);
  };
  auto piece_captures = [&](Square sq, const Bitboard& dc,
                            const Bitboard& occ_now) {
    return MateApproxPieceCanCapture(Them, sq, dc, occ_now);
  };
  auto supported = [&](Square to, const Bitboard& occ_now) {
    return IsSquareAttacked(to, occ_now, Us);
  };

  // ============================================================
  // Drops (patterns adjacent to the king; distant checks skipped)
  // ============================================================

  if (our_hand.Has(kRook)) {
    Bitboard to_bb =
        empty & ShogiTables::KingEffectBB[k] & ShogiTables::RookEffectBB[k];
    while (to_bb.Any()) {
      const Square to = to_bb.Pop();
      if (!supported(to, occ)) continue;
      if (!king_escapes(to, ShogiTables::RookEffectBB[to.as_idx()], occ) &&
          !piece_captures(to, dc_them, occ)) {
        return Move::Drop(kRook, to);
      }
    }
  } else if (our_hand.Has(kLance) && (Us == BLACK ? krank <= 7 : krank >= 1)) {
    // Only the square directly in front of the king (from our side).
    const Square to = Square(ksq.file(), Rank::FromIdx(krank + t_south));
    if (empty.Test(to) && supported(to, occ)) {
      if (!king_escapes(to, ShogiTables::LanceMaskBB[to.as_idx()][Us], occ) &&
          !piece_captures(to, dc_them, occ)) {
        return Move::Drop(kLance, to);
      }
    }
  }

  if (our_hand.Has(kBishop)) {
    Bitboard to_bb =
        empty & ShogiTables::KingEffectBB[k] & ShogiTables::BishopEffectBB[k];
    while (to_bb.Any()) {
      const Square to = to_bb.Pop();
      if (!supported(to, occ)) continue;
      if (!king_escapes(to, ShogiTables::BishopEffectBB[to.as_idx()], occ) &&
          !piece_captures(to, dc_them, occ)) {
        return Move::Drop(kBishop, to);
      }
    }
  }

  if (our_hand.Has(kGold)) {
    Bitboard to_bb = empty & ShogiTables::GoldEffectBB[k][Them];
    if (our_hand.Has(kRook)) {
      // The straight-behind gold was already covered by the rook drop.
      to_bb = ShogiTables::PawnEffectBB[k][Us].AndNot(to_bb);
    }
    while (to_bb.Any()) {
      const Square to = to_bb.Pop();
      if (!supported(to, occ)) continue;
      if (!king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us], occ) &&
          !piece_captures(to, dc_them, occ)) {
        return Move::Drop(kGold, to);
      }
    }
  }

  if (our_hand.Has(kSilver)) {
    Bitboard to_bb;
    bool consider = true;
    if (our_hand.Has(kGold)) {
      if (our_hand.Has(kBishop)) {
        consider = false;  // gold covered front, bishop covered diagonals
      } else {
        // Only the diagonally-behind squares remain.
        to_bb = empty & ShogiTables::SilverEffectBB[k][Them] &
                InFrontOf(Us, krank);
      }
    } else if (our_hand.Has(kBishop)) {
      // Diagonals covered by the bishop drop: front squares remain.
      to_bb = empty & ShogiTables::GoldEffectBB[k][Them] &
              ShogiTables::SilverEffectBB[k][Them];
    } else {
      to_bb = empty & ShogiTables::SilverEffectBB[k][Them];
    }
    if (consider) {
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ)) continue;
        if (!king_escapes(to, ShogiTables::SilverEffectBB[to.as_idx()][Us],
                          occ) &&
            !piece_captures(to, dc_them, occ)) {
          return Move::Drop(kSilver, to);
        }
      }
    }
  }

  if (our_hand.Has(kKnight)) {
    Bitboard to_bb = empty & ShogiTables::KnightEffectBB[k][Them];
    while (to_bb.Any()) {
      const Square to = to_bb.Pop();
      // A knight needs no support: the king cannot capture its checker.
      if (!king_escapes(to, Bitboard::Zero(), occ) &&
          !piece_captures(to, dc_them, occ)) {
        return Move::Drop(kKnight, to);
      }
    }
  }

  // Pawn-drop mate is illegal: never tried.

  // ============================================================
  // Board moves (destinations adjacent to the king, plus knights)
  // ============================================================

  const Bitboard move_target = us_bb.AndNot(ShogiTables::KingEffectBB[k]);
  const Bitboard pinned_us = ComputeBlockersForKing(Us) & us_bb;
  const Bitboard dc_us = ComputeBlockersForKing(Them) & us_bb;
  const Bitboard zone = PromoZone<Us>();

  auto pinned_illegal = [&](Square from, Square to, bool knight) {
    if (!pinned_us.Test(from)) return false;
    if (knight) return true;
    return !ShogiTables::LineBB[from.as_idx()][our_ksq.as_idx()].Test(to);
  };
  auto discovered = [&](Square from, Square to, bool knight) {
    if (!dc_us.Test(from)) return false;
    if (knight) return true;
    return !ShogiTables::LineBB[from.as_idx()][k].Test(to);
  };

  // Runs `body(to, promote)` for each candidate destination of `from`
  // after temporarily removing the piece; returns the mating move.
  // dc_them_after is recomputed once per source.

  // --- Dragon ---
  {
    Bitboard from_bb = pieces(Us, kDragon);
    while (from_bb.Any()) {
      const Square from = from_bb.Pop();
      Bitboard to_bb =
          move_target & (ShogiTables::RookEffect(from, occ) |
                         ShogiTables::KingEffectBB[from.as_idx()]);
      if (!to_bb.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kDragon, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ_now)) continue;
        Bitboard occ_nk = occ_now;
        occ_nk.Clear(ksq);
        const Bitboard esc = ShogiTables::RookEffect(to, occ_nk) |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Rook ---
  {
    Bitboard all_from = pieces(Us, kRook);
    Bitboard from_zone = all_from & zone;
    Bitboard from_rest = zone.AndNot(all_from);
    // From inside the zone: every adjacent destination, always promote.
    while (from_zone.Any()) {
      const Square from = from_zone.Pop();
      Bitboard to_bb = move_target & ShogiTables::RookEffect(from, occ);
      if (!to_bb.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kRook, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ_now)) continue;
        Bitboard occ_nk = occ_now;
        occ_nk.Clear(ksq);
        const Bitboard esc = ShogiTables::RookEffect(to, occ_nk) |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Promotion(from, to);
        }
      }
    }
    // From outside: promote when landing in the zone, else orthogonal
    // neighbors of the king only.
    while (from_rest.Any()) {
      const Square from = from_rest.Pop();
      Bitboard to_all = move_target & ShogiTables::RookEffect(from, occ) &
                        ((ShogiTables::KingEffectBB[k] &
                          ShogiTables::RookEffectBB[k]) |
                         zone);
      if (!to_all.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kRook, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      Bitboard to_promo = to_all & zone;
      Bitboard to_plain = zone.AndNot(to_all);
      while (to_promo.Any()) {
        const Square to = to_promo.Pop();
        if (!supported(to, occ_now)) continue;
        Bitboard occ_nk = occ_now;
        occ_nk.Clear(ksq);
        const Bitboard esc = ShogiTables::RookEffect(to, occ_nk) |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Promotion(from, to);
        }
      }
      while (to_plain.Any()) {
        const Square to = to_plain.Pop();
        if (!supported(to, occ_now)) continue;
        if (!king_escapes(to, ShogiTables::RookEffectBB[to.as_idx()],
                          occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Horse ---
  {
    Bitboard from_bb = pieces(Us, kHorse);
    while (from_bb.Any()) {
      const Square from = from_bb.Pop();
      Bitboard to_bb =
          move_target & (ShogiTables::BishopEffect(from, occ) |
                         ShogiTables::KingEffectBB[from.as_idx()]);
      if (!to_bb.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kHorse, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ_now)) continue;
        const Bitboard esc = ShogiTables::BishopEffectBB[to.as_idx()] |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Bishop ---
  {
    Bitboard all_from = pieces(Us, kBishop);
    Bitboard from_zone = all_from & zone;
    Bitboard from_rest = zone.AndNot(all_from);
    while (from_zone.Any()) {
      const Square from = from_zone.Pop();
      Bitboard to_bb = move_target & ShogiTables::BishopEffect(from, occ);
      if (!to_bb.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kBishop, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ_now)) continue;
        const Bitboard esc = ShogiTables::BishopEffectBB[to.as_idx()] |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Promotion(from, to);
        }
      }
    }
    while (from_rest.Any()) {
      const Square from = from_rest.Pop();
      Bitboard to_all = move_target & ShogiTables::BishopEffect(from, occ) &
                        ((ShogiTables::KingEffectBB[k] &
                          ShogiTables::BishopEffectBB[k]) |
                         zone);
      if (!to_all.Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kBishop, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      Bitboard to_promo = to_all & zone;
      Bitboard to_plain = zone.AndNot(to_all);
      while (to_promo.Any()) {
        const Square to = to_promo.Pop();
        if (!supported(to, occ_now)) continue;
        const Bitboard esc = ShogiTables::BishopEffectBB[to.as_idx()] |
                             ShogiTables::KingEffectBB[to.as_idx()];
        if (!king_escapes(to, esc, occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Promotion(from, to);
        }
      }
      while (to_plain.Any()) {
        const Square to = to_plain.Pop();
        if (!supported(to, occ_now)) continue;
        if (!king_escapes(to, ShogiTables::BishopEffectBB[to.as_idx()],
                          occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Gold movers ---
  {
    Bitboard from_bb = (pieces(Us, kGold) | pieces(Us, kProPawn) |
                        pieces(Us, kProLance) | pieces(Us, kProKnight) |
                        pieces(Us, kProSilver)) &
                       ShogiTables::GoldMoveCheckBB[k][Us];
    while (from_bb.Any()) {
      const Square from = from_bb.Pop();
      Bitboard to_bb = move_target &
                       ShogiTables::GoldEffectBB[from.as_idx()][Us] &
                       ShogiTables::GoldEffectBB[k][Them];
      if (!to_bb.Any()) continue;
      const PieceType pt = piece_on(from).GetType();
      MateProbeSourceRemoval rm(*this, Us, pt, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_bb.Any()) {
        const Square to = to_bb.Pop();
        if (!supported(to, occ_now)) continue;
        if (!king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us],
                          occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Silver ---
  {
    Bitboard all_from =
        pieces(Us, kSilver) & ShogiTables::SilverMoveCheckBB[k][Us];
    if (all_from.Any()) {
      const Bitboard chk = ShogiTables::SilverEffectBB[k][Them];
      const Bitboard chk_promo = ShogiTables::GoldEffectBB[k][Them];
      const Bitboard rank4plus =
          Us == BLACK ? InFrontOf(WHITE, 3) : InFrontOf(BLACK, 5);

      auto run_silver = [&](Square from, Bitboard to_promo,
                            Bitboard to_plain) -> Move {
        if (!(to_promo | to_plain).Any()) return Move();
        MateProbeSourceRemoval rm(*this, Us, kSilver, from);
        const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
        const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
        while (to_promo.Any()) {
          const Square to = to_promo.Pop();
          if (!supported(to, occ_now)) continue;
          if (!king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us],
                            occ_now) &&
              (discovered(from, to, false) ||
               !piece_captures(to, dc_after, occ_now)) &&
              !pinned_illegal(from, to, false)) {
            return Move::Promotion(from, to);
          }
        }
        while (to_plain.Any()) {
          const Square to = to_plain.Pop();
          if (!supported(to, occ_now)) continue;
          if (!king_escapes(to, ShogiTables::SilverEffectBB[to.as_idx()][Us],
                            occ_now) &&
              (discovered(from, to, false) ||
               !piece_captures(to, dc_after, occ_now)) &&
              !pinned_illegal(from, to, false)) {
            return Move::Normal(from, to);
          }
        }
        return Move();
      };

      Bitboard from_zone = all_from & zone;
      Bitboard rest = zone.AndNot(all_from);
      while (from_zone.Any()) {
        const Square from = from_zone.Pop();
        Bitboard to_bb =
            move_target & ShogiTables::SilverEffectBB[from.as_idx()][Us];
        Bitboard to_promo = to_bb & chk_promo;
        Bitboard to_plain = to_bb & chk;
        // If promotion fails to mate on a forward square, non-promotion
        // fails too (sakurapyon's pruning, as in the original).
        to_plain = InFrontOf(Them, krank).AndNot(to_plain);
        Move m = run_silver(from, to_promo, to_plain);
        if (!m.is_null()) return m;
      }
      Bitboard from_back = rest & rank4plus;
      rest = rank4plus.AndNot(rest);
      while (from_back.Any()) {
        const Square from = from_back.Pop();
        Bitboard to_plain = move_target &
                            ShogiTables::SilverEffectBB[from.as_idx()][Us] &
                            chk;
        Move m = run_silver(from, Bitboard::Zero(), to_plain);
        if (!m.is_null()) return m;
      }
      while (rest.Any()) {  // exactly rank 4 (our side): promote forward only
        const Square from = rest.Pop();
        Bitboard to_bb =
            move_target & ShogiTables::SilverEffectBB[from.as_idx()][Us];
        Bitboard to_promo = to_bb & zone & chk_promo;
        Bitboard to_plain = to_bb & chk;
        Move m = run_silver(from, to_promo, to_plain);
        if (!m.is_null()) return m;
      }
    }
  }

  // --- Knight ---
  {
    Bitboard from_bb =
        pieces(Us, kKnight) & ShogiTables::KnightMoveCheckBB[k][Us];
    const Bitboard chk_promo = ShogiTables::GoldEffectBB[k][Them] & zone;
    const Bitboard chk = ShogiTables::KnightEffectBB[k][Them];
    while (from_bb.Any()) {
      const Square from = from_bb.Pop();
      Bitboard to_bb = us_bb.AndNot(
          ShogiTables::KnightEffectBB[from.as_idx()][Us]);
      Bitboard to_promo = to_bb & chk_promo;
      Bitboard to_plain = to_bb & chk;
      if (!(to_promo | to_plain).Any()) continue;
      MateProbeSourceRemoval rm(*this, Us, kKnight, from);
      const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
      const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
      while (to_promo.Any()) {
        const Square to = to_promo.Pop();
        if (!supported(to, occ_now)) continue;
        if (!king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us],
                          occ_now) &&
            (discovered(from, to, true) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, true)) {
          return Move::Promotion(from, to);
        }
      }
      while (to_plain.Any()) {
        const Square to = to_plain.Pop();
        // Knight checks need no support.
        if (!king_escapes(to, Bitboard::Zero(), occ_now) &&
            (discovered(from, to, true) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, true)) {
          return Move::Normal(from, to);
        }
      }
    }
  }

  // --- Lance ---
  {
    Bitboard from_bb =
        pieces(Us, kLance) & ShogiTables::LanceMoveCheckBB[k][Us];
    if (from_bb.Any()) {
      const Bitboard chk_promo = ShogiTables::GoldEffectBB[k][Them] & zone;
      // The single square in front of the king; skipped when the king
      // stands on our first rank (the promotion case covers it).
      const bool plain_ok = Us == BLACK ? krank >= 1 : krank <= 7;
      while (from_bb.Any()) {
        const Square from = from_bb.Pop();
        Bitboard to_bb =
            move_target & ShogiTables::LanceEffect(Us, from, occ);
        Bitboard to_promo = to_bb & chk_promo;
        Bitboard to_plain =
            plain_ok ? to_bb & ShogiTables::PawnEffectBB[k][Them]
                     : Bitboard::Zero();
        if (!(to_promo | to_plain).Any()) continue;
        MateProbeSourceRemoval rm(*this, Us, kLance, from);
        const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
        const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
        while (to_promo.Any()) {
          const Square to = to_promo.Pop();
          if (!supported(to, occ_now)) continue;
          if (!king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us],
                            occ_now) &&
              (discovered(from, to, false) ||
               !piece_captures(to, dc_after, occ_now)) &&
              !pinned_illegal(from, to, false)) {
            return Move::Promotion(from, to);
          }
        }
        while (to_plain.Any()) {
          const Square to = to_plain.Pop();
          if (!supported(to, occ_now)) continue;
          if (!king_escapes(to, ShogiTables::LanceMaskBB[to.as_idx()][Us],
                            occ_now) &&
              (discovered(from, to, false) ||
               !piece_captures(to, dc_after, occ_now)) &&
              !pinned_illegal(from, to, false)) {
            return Move::Normal(from, to);
          }
        }
      }
    }
  }

  // --- Pawn ---
  {
    // A pawn move checks only when the king is on ranks 1..7 (our view).
    const bool any_pawn_check = Us == BLACK ? krank <= 6 : krank >= 2;
    if (any_pawn_check) {
      // Promotions: king inside (or adjacent to) the zone.
      Bitboard to_promo = move_target & ShogiTables::GoldEffectBB[k][Them] &
                          zone;
      while (to_promo.Any()) {
        const Square to = to_promo.Pop();
        const int from_rank = to.rank().idx + t_south;
        if (from_rank < 0 || from_rank > 8) continue;
        const Square from = Square(to.file(), Rank::FromIdx(from_rank));
        if (!pieces(Us, kPawn).Test(from)) continue;
        MateProbeSourceRemoval rm(*this, Us, kPawn, from);
        const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
        const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
        if (supported(to, occ_now) &&
            !king_escapes(to, ShogiTables::GoldEffectBB[to.as_idx()][Us],
                          occ_now) &&
            (discovered(from, to, false) ||
             !piece_captures(to, dc_after, occ_now)) &&
            !pinned_illegal(from, to, false)) {
          return Move::Promotion(from, to);
        }
      }
      // Plain push: only when promotion is not forced (king past rank 2).
      const bool plain_ok = Us == BLACK ? krank <= 6 && krank >= 2
                                        : krank >= 2 && krank <= 6;
      if (plain_ok) {
        const int to_rank = krank + t_south;
        const int from_rank = to_rank + t_south;
        if (from_rank >= 0 && from_rank <= 8) {
          const Square to = Square(ksq.file(), Rank::FromIdx(to_rank));
          const Square from = Square(ksq.file(), Rank::FromIdx(from_rank));
          if (pieces(Us, kPawn).Test(from) && !us_bb.Test(to)) {
            MateProbeSourceRemoval rm(*this, Us, kPawn, from);
            const Bitboard occ_now = occ ^ Bitboard::FromSquare(from);
            const Bitboard dc_after = ComputeBlockersForKing(Them) & them_bb;
            if (supported(to, occ_now) &&
                !king_escapes(to, Bitboard::Zero(), occ_now) &&
                (discovered(from, to, false) ||
                 !piece_captures(to, dc_after, occ_now)) &&
                !pinned_illegal(from, to, false)) {
              return Move::Normal(from, to);
            }
          }
        }
      }
    }
  }

  return Move();
}

Move ShogiBoard::FindMateInOneApprox() {
  if (!king_sq_[BLACK].IsValid() || !king_sq_[WHITE].IsValid()) return Move();
  if (InCheck()) return Move();
  return side_to_move_ == BLACK ? FindMateInOneApproxImpl<BLACK>()
                                : FindMateInOneApproxImpl<WHITE>();
}

template Move ShogiBoard::FindMateInOneApproxImpl<BLACK>();
template Move ShogiBoard::FindMateInOneApproxImpl<WHITE>();

}  // namespace lczero
