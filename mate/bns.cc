/*
  JHBR3 Shogi Engine — BNS (Branch Number Search) Checkmate Solver

  See mate/bns.h for the algorithm description and references.

  Search shape (recursive threshold iteration over a transposition
  table) follows cshogi/dlshogi's dfpn_inner (src/dfpn.cpp, GPLv3);
  repetition semantics follow mate/dfpn.cc. The branch-number arithmetic
  itself implements Okabe's paper directly.
*/

#include "mate/bns.h"

#include <algorithm>
#include <cassert>

#include "mate/shallow_mate.h"

namespace jhbr2 {

using namespace lczero;
using RepetitionResult = ShogiBoard::RepetitionResult;

// =====================================================================
// Entry points
// =====================================================================

Move MateBnsSolver::search(ShogiBoard board, size_t nodes_limit) {
  return search(std::move(board), nodes_limit, Deadline::max());
}

Move MateBnsSolver::search(ShogiBoard board, size_t nodes_limit,
                           Deadline deadline) {
  deadline_ = deadline;
  deadline_passed_ = false;
  nodes_limit_ = nodes_limit;
  stats_ = Stats();
  mate_ply_ = 0;
  pv_.clear();
  path_hashes_.clear();
  overrides_.clear();
  limit_hit_ = false;
  resource_taint_seen_ = false;
  stop_check_counter_ = 0;

  tt_.Resize(tt_mb_);
  tt_.NewSearch();
  if (move_cache_mb_ > 0) {
    size_t want = (move_cache_mb_ << 20) / sizeof(MoveCacheSlot);
    size_t n = 1024;
    while (n * 2 <= want) n *= 2;
    if (move_cache_mask_ != n - 1) {
      move_cache_ = std::make_unique<MoveCacheSlot[]>(n);
      move_cache_mask_ = n - 1;
      move_cache_gen_ = 0;
    }
    if (++move_cache_gen_ == 0) {
      for (size_t i = 0; i <= move_cache_mask_; i++) move_cache_[i].gen = 0;
      move_cache_gen_ = 1;
    }
  } else {
    move_cache_.reset();
    move_cache_mask_ = 0;
  }
  mcache_hits_ = mcache_probes_ = 0;
  if (use_dominance_) {
    finals_.Resize(std::max<size_t>(tt_mb_ / 4, 1));
    finals_.NewSearch();
  }
  if (frames_size_ != static_cast<size_t>(max_ply_) + 2) {
    frames_size_ = static_cast<size_t>(max_ply_) + 2;
    frames_ = std::make_unique<Frame[]>(frames_size_);
  }

  // Sticky stop, as in MateDfpnSolver: a cancellation issued before the
  // worker enters search() must not be lost.
  if (ShouldStop()) return Move();

  ShogiBoard& pos = board;
  const uint64_t root_hash = pos.Hash();
  path_hashes_.push_back(root_hash);

  if (arith_ == Arith::kBns) {
    SearchImpl<true, false>(pos, bns::kInf, bns::kInf, 0);
  } else {
    SearchImpl<true, true>(pos, bns::kInf, bns::kInf, 0);
  }

  path_hashes_.pop_back();

  stats_.mcache_probes = mcache_probes_;
  stats_.mcache_hits = mcache_hits_;
  stats_.tt_probes = tt_.probes();
  stats_.tt_hits = tt_.hits();
  stats_.tt_stores = tt_.stores();
  stats_.tt_evictions = tt_.evictions();

  if (stop_.load(std::memory_order_acquire)) return Move();

  // Read the root verdict from the table only. Derived root verdicts
  // always land there: path dependencies anchored at the root dissolve
  // (see the loop-head rule in SearchImpl), and resource-tainted
  // verdicts stay out of the table, leaving the root entry non-final —
  // correctly UNSOLVED. Any leftover override under the root hash
  // describes a deep revisit of the root position inside the tree, not
  // the root's own value, and must be ignored here.
  bns::ChildView root{1, 1};
  root_channel_ = RootChannel::kNone;
  if (BnsTTEntry* e = tt_.Probe(root_hash, 0)) {
    root = {e->abn, e->obn};
    root_channel_ = RootChannel::kTT;
  }
  root_view_ = root;

  if (root.abn == 0) {
    // Mate proved. Extract the principal variation from the table.
    ExtractPV(pos);
    if (!pv_.empty()) return pv_[0];
    return Move();  // Proof exists but the line could not be reassembled.
  }
  if (root.obn == 0) {
    return NoMateMove();
  }
  return Move();  // Unsolved (limits, stop, or resource-tainted disproof).
}

bool MateBnsSolver::ShouldStop() {
  if (stop_.load(std::memory_order_acquire)) return true;
  if (deadline_passed_) return true;
  if (deadline_ != Deadline::max() &&
      (++stop_check_counter_ & 1023) == 0 &&
      Clock::now() >= deadline_) {
    deadline_passed_ = true;
    return true;
  }
  return false;
}

// =====================================================================
// Route-dependent verdicts
// =====================================================================

void MateBnsSolver::RecordVerdict(uint64_t hash, int ply, bns::ChildView v,
                                  bool tainted, int anchor_ply, TaintKind kind,
                                  uint16_t best_move, uint64_t board_key,
                                  uint32_t attacker_hand) {
  if (tainted) {
    if (kind == TaintKind::kResource) resource_taint_seen_ = true;
    overrides_.push_back({hash, v, anchor_ply, kind});
    override_version_++;
    return;
  }
  BnsTTEntry* e = tt_.Store(hash, ply);
  e->abn = v.abn;
  e->obn = v.obn;
  if (best_move) e->best_move = best_move;
  if (use_dominance_ && board_key && (v.abn == 0 || v.obn == 0)) {
    finals_.Store(board_key, lczero::Hand(attacker_hand), v.abn == 0);
  }
}

void MateBnsSolver::DropOverridesAtReturn(int ply) {
  if (overrides_.empty()) return;
  const size_t erased = std::erase_if(overrides_, [ply](const PathOverride& o) {
    return o.anchor_ply >= ply;
  });
  if (erased) override_version_++;
}

// =====================================================================
// Core search
// =====================================================================

template <bool kOrNode, bool kPnDn>
void MateBnsSolver::SearchImpl(ShogiBoard& board, uint32_t abn_th,
                               uint32_t obn_th, int ply) {
  stats_.node_entries++;
  if (ply > stats_.max_ply) stats_.max_ply = ply;

  const uint64_t hash = board.Hash();
  const int kply = paper_ghi_mode_ ? 0 : ply;
  const bool first_visit = tt_.Probe(hash, kply) == nullptr;
  if (first_visit) stats_.first_visits++;
  const uint64_t bkey = use_dominance_ ? board.BoardKey() : 0;
  const uint32_t ahand =
      board.hand(kOrNode ? board.side_to_move() : ~board.side_to_move()).raw();

  Frame& f = frames_[ply];

  MoveCacheSlot* mslot = nullptr;
  if (move_cache_mask_) {
    mslot = &move_cache_[(hash + static_cast<uint64_t>(kply) *
                                     0x9E3779B97F4A7C15ull) &
                         move_cache_mask_];
    mcache_probes_++;
    if (mslot->gen == move_cache_gen_ && mslot->hash == hash &&
        mslot->ply == static_cast<uint16_t>(kply)) {
      mcache_hits_++;
      const int n = mslot->n;
      f.moves.assign(mslot->moves, n);
      for (int i = 0; i < n; i++) {
        f.child_hash[i] = mslot->child_hash[i];
        tt_.Prefetch(f.child_hash[i], paper_ghi_mode_ ? 0 : ply + 1);
      }
      if (use_dominance_) {
        const lczero::Hand own_hand2 = board.hand(board.side_to_move());
        for (int i = 0; i < n; i++) {
          const Move m = f.moves[i];
          f.child_bkey[i] = board.BoardKeyAfter(m);
          if (kOrNode) {
            lczero::Hand h2 = own_hand2;
            if (m.is_drop()) {
              h2.Sub(m.drop_piece());
            } else {
              lczero::Piece cap = board.piece_on(m.to());
              if (!cap.IsNone()) h2.Add(cap.GetType().Unpromote());
            }
            f.child_ahand[i] = h2.raw();
          } else {
            f.child_ahand[i] = ahand;
          }
        }
      } else {
        for (int i = 0; i < n; i++) {
          f.child_bkey[i] = 0;
          f.child_ahand[i] = 0;
        }
      }
      goto have_moves;
    }
  }

  if constexpr (kOrNode) {
    if (first_visit) {
      // Fast approximate probe when not in check; the complete routine
      // for the rare in-check (countercheck) nodes.
      Move m1 = board.InCheck() ? board.FindMateInOne()
                                : board.FindMateInOneApprox();
      if (!m1.is_null()) {
        stats_.mate1_hits++;
        RecordVerdict(hash, kply, bns::MateView(), false, 0, TaintKind::kPath,
                      m1.raw(), bkey, ahand);
        return;
      }
      // dlshogi-style fixed-depth probe at fresh nodes: resolving the
      // 3-ply frontier here avoids two levels of threshold iteration.
      // (Extraction rebuilds the line via Extract3PlyLine.)
      const bool m3 =
          use_mate3_probe_ &&
          (board.InCheck() ? shallow_mate::MateIn3Ply<true>(board)
                           : shallow_mate::MateIn3Ply<false>(board));
      if (m3) {
        stats_.mate3_hits++;
        RecordVerdict(hash, kply, bns::MateView(), false, 0, TaintKind::kPath,
                      0, bkey, ahand);
        return;
      }
    }
    f.moves = board.GenerateCheckingMoves();
    if (move_ordering_ && f.moves.size() > 2) {
      // Stable three-tier partition: captures, quiet board moves, drops.
      Move tmp[lczero::kMaxLegalMoves];
      const int nm = f.moves.size();
      int w = 0;
      for (int tier = 0; tier < 3 && w < nm; tier++) {
        for (int i = 0; i < nm; i++) {
          const Move m = f.moves[i];
          const int t = m.is_drop() ? 2
                        : !board.piece_on(m.to()).IsNone() ? 0
                                                           : 1;
          if (t == tier) tmp[w++] = m;
        }
      }
      for (int i = 0; i < nm; i++) f.moves[i] = tmp[i];
    }
    if (f.moves.empty()) {
      RecordVerdict(hash, kply, bns::NoMateView(), false, 0, TaintKind::kPath,
                    0, bkey, ahand);
      return;
    }
  } else {
    assert(board.InCheck());
    f.moves = board.GenerateEvasionMoves();
    if (f.moves.empty()) {
      // No legal evasion: checkmate. Illegal pawn-drop mates cannot
      // reach here — GenerateCheckingMoves() never emits them.
      RecordVerdict(hash, kply, bns::MateView(), false, 0, TaintKind::kPath,
                    0, bkey, ahand);
      return;
    }
  }

  {
  const int n = f.moves.size();
  const lczero::Hand own_hand = board.hand(board.side_to_move());
  for (int i = 0; i < n; i++) {
    const Move m = f.moves[i];
    f.child_hash[i] = board.HashAfter(m);
    tt_.Prefetch(f.child_hash[i], paper_ghi_mode_ ? 0 : ply + 1);
    if (use_dominance_) {
      f.child_bkey[i] = board.BoardKeyAfter(m);
      if (kOrNode) {
        lczero::Hand h = own_hand;
        if (m.is_drop()) {
          h.Sub(m.drop_piece());
        } else {
          lczero::Piece cap = board.piece_on(m.to());
          if (!cap.IsNone()) h.Add(cap.GetType().Unpromote());
        }
        f.child_ahand[i] = h.raw();
      } else {
        f.child_ahand[i] = ahand;
      }
    } else {
      f.child_bkey[i] = 0;
      f.child_ahand[i] = 0;
    }
  }
  if (mslot && n <= kMoveCacheMaxMoves) {
    mslot->hash = hash;
    mslot->gen = move_cache_gen_;
    mslot->ply = static_cast<uint16_t>(kply);
    mslot->n = static_cast<uint16_t>(n);
    for (int i = 0; i < n; i++) {
      mslot->moves[i] = f.moves[i];
      mslot->child_hash[i] = f.child_hash[i];
    }
  }
  }

have_moves:;
  const int n = f.moves.size();

  if (kOrNode && first_visit && use_new_node_block_) {
    // Every fact written here is position-intrinsic (movegen counts and
    // mate-in-1 proofs), so plain untainted stores are sound.
    const int cply = paper_ghi_mode_ ? 0 : ply + 1;
    const int gply = paper_ghi_mode_ ? 0 : ply + 2;
    for (int i = 0; i < n; i++) {
      const Move m = f.moves[i];
      UndoInfo u1 = board.DoMove(m, /*gives_check=*/true);
      MoveList ev = board.GenerateEvasionMoves();
      if (ev.empty()) {
        board.UndoMove(m, u1);
        RecordVerdict(f.child_hash[i], cply, bns::MateView(), false, 0,
                      TaintKind::kPath, 0, f.child_bkey[i], f.child_ahand[i]);
        RecordVerdict(hash, kply, bns::MateView(), false, 0, TaintKind::kPath,
                      m.raw(), bkey, ahand);
        stats_.mate3_hits++;
        return;
      }
      bool child_all_mated = true;
      for (const Move& m2 : ev) {
        UndoInfo u2 = board.DoMove(m2);
        Move m3 = board.InCheck() ? Move() : board.FindMateInOneApprox();
        if (!m3.is_null()) {
          RecordVerdict(board.Hash(), gply, bns::MateView(), false, 0,
                        TaintKind::kPath, m3.raw(), 0, 0);
          board.UndoMove(m2, u2);
        } else {
          board.UndoMove(m2, u2);
          child_all_mated = false;
          break;
        }
      }
      board.UndoMove(m, u1);
      if (child_all_mated) {
        RecordVerdict(f.child_hash[i], cply, bns::MateView(), false, 0,
                      TaintKind::kPath, 0, f.child_bkey[i], f.child_ahand[i]);
        RecordVerdict(hash, kply, bns::MateView(), false, 0, TaintKind::kPath,
                      m.raw(), bkey, ahand);
        stats_.mate3_hits++;
        return;
      }
      // Initialize the fresh child AND entry from its defense count: a
      // safe finite estimate (real summaries overwrite it on visit).
      BnsTTEntry* ce = tt_.Store(f.child_hash[i], cply);
      if (ce->effort == 0 && ce->abn == 1 && ce->obn == 1) {
        ce->abn = static_cast<uint32_t>(ev.size());
        ce->obn = 1;
      }
    }
  }

  // With depth-keyed entries, a child's subtree can only write entries
  // strictly deeper than the sibling level, so sibling views are frozen
  // while the search is inside one child: after the initial pass, only
  // the child just returned needs re-probing — unless the override list
  // changed (overrides are hash-keyed and can mark siblings from below),
  // or in paper mode where cross-depth writes are the point.
  int refresh_only = -1;
  uint64_t seen_override_version = ~uint64_t{0};
  for (;;) {
    // ---- Summarize current child views ----
    stats_.summaries++;
    const bool full_refresh = refresh_only < 0 || paper_ghi_mode_ ||
                              seen_override_version != override_version_;
    const int lo = full_refresh ? 0 : refresh_only;
    const int hi = full_refresh ? n : refresh_only + 1;
    for (int i = lo; i < hi; i++) {
      SourcedView sv = LookupChild(f.child_hash[i], paper_ghi_mode_ ? 0 : ply + 1,
                                   f.child_bkey[i], f.child_ahand[i]);
      f.views[i] = sv.view;
      f.tainted[i] = sv.tainted;
      f.anchor[i] = static_cast<int16_t>(sv.anchor_ply);
      f.kind[i] = static_cast<uint8_t>(sv.kind);
    }
    seen_override_version = override_version_;
    const bns::Summary s = bns::Summarize<kOrNode, kPnDn>(f.views, n);

    if (s.terminal()) {
      // Taint bookkeeping: a verdict determined by route-dependent child
      // verdicts is itself route-dependent and must stay out of the TT —
      // EXCEPT dependencies anchored at this very node. A cycle back to
      // this position exists on every route through it, so at the
      // anchoring node the path-dependence dissolves and the verdict is
      // unconditional (the loop head owns its loops). Resource
      // dependencies (depth cap) never dissolve.
      bool tainted = false;
      int anchor = 0;
      bool resource = false;
      uint16_t best_move = 0;
      auto absorb = [&](int i) {
        if (!f.tainted[i]) return;
        const TaintKind k = static_cast<TaintKind>(f.kind[i]);
        const int a = f.anchor[i];
        if (k == TaintKind::kResource) {
          resource = true;
          tainted = true;
          anchor = std::max(anchor, a);
        } else if (a < ply) {
          tainted = true;
          anchor = std::max(anchor, a);
        }
        // k == kPath && a == ply: self-anchored — intrinsic, no taint.
      };
      if (s.best >= 0) {
        // Decided by a single child (OR: proving child, AND: escaping
        // child).
        absorb(s.best);
        best_move = f.moves[s.best].raw();
      } else {
        // Decided by all children jointly. The combined verdict dies as
        // soon as any contributing override dies: anchor = max.
        for (int i = 0; i < n; i++) absorb(i);
      }
      RecordVerdict(hash, kply, {s.abn, s.obn}, tainted, anchor,
                    resource ? TaintKind::kResource : TaintKind::kPath,
                    best_move, bkey, ahand);
      return;
    }

    // ---- Store unresolved node values ----
    {
      BnsTTEntry* e = tt_.Store(hash, kply);
      e->abn = s.abn;
      e->obn = s.obn;
      e->best_move = f.moves[s.best].raw();
      if (e->effort != UINT32_MAX) e->effort++;
    }

    if (abn_th <= s.abn || obn_th <= s.obn) return;

    if (limit_hit_ || stats_.node_entries >= nodes_limit_ || ShouldStop()) {
      limit_hit_ = true;
      return;
    }

    // ---- Descend into the selected child ----
    uint32_t child_abn_th, child_obn_th;
    bns::ChildThresholds<kOrNode>(s, f.views[s.best], abn_th, obn_th,
                                  &child_abn_th, &child_obn_th);

    if (paper_ghi_mode_) {
      // Figure-4-style escape: a hammered entry signals an
      // expansion-history loop; widen the child window to the parent's
      // (sequential expansion) so the loop cannot re-trigger.
      if (BnsTTEntry* e = tt_.Probe(hash, 0)) {
        if (e->effort > ghi_escape_effort_) {
          child_abn_th = abn_th;
          child_obn_th = obn_th;
        }
        // Figure-3 marking: while this node is on the path, revisits
        // from below see (abn = child's abn threshold, obn = 1), which
        // makes the attacker avoid the cycle and the defender seek it.
        e->abn = child_abn_th;
        e->obn = 1;
      }
    }

    const Move bm = f.moves[s.best];
    const uint64_t chash = f.child_hash[s.best];
    UndoInfo undo;
    if constexpr (kOrNode) {
      undo = board.DoMove(bm, /*gives_check=*/true);
    } else {
      undo = board.DoMove(bm);
    }
    stats_.do_moves++;
    assert(board.Hash() == chash);

    // Route-terminal checks for the child: depth cap, rule-aware
    // repetition, then a full-path cycle scan beyond the rule window.
    bool handled = false;

    if (ply + 1 >= max_ply_) {
      stats_.depth_hits++;
      resource_taint_seen_ = true;
      RecordVerdict(chash, paper_ghi_mode_ ? 0 : ply + 1, bns::NoMateView(),
                    true, ply, TaintKind::kResource, 0, 0, 0);
      handled = true;
    }

    if (!handled && !disable_rule_repetition_) {
      int dist = 0;
      const auto rep =
          board.CheckRepetition(kRepetitionLookbackPly, &dist);
      if (rep != RepetitionResult::kNone) {
        stats_.rep_hits++;
        // The child's side to move is the defender under kOrNode and the
        // attacker otherwise; same mapping as MateDfpnSolver.
        const bool attacker_wins = kOrNode ? rep == RepetitionResult::kLoss
                                           : rep == RepetitionResult::kWin;
        const auto v = attacker_wins ? bns::MateView() : bns::NoMateView();
        const int anchor = (ply + 1) - dist;
        if (anchor < 0) {
          // Matched pre-root game history: unconditional for this
          // problem instance.
          RecordVerdict(chash, paper_ghi_mode_ ? 0 : ply + 1, v, false, 0,
                        TaintKind::kPath, 0, 0, 0);
        } else {
          RecordVerdict(chash, paper_ghi_mode_ ? 0 : ply + 1, v, true, anchor,
                        TaintKind::kPath, 0, 0, 0);
        }
        handled = true;
      }
    }

    if (!handled && !disable_path_scan_) {
      for (int idx = static_cast<int>(path_hashes_.size()) - 1; idx >= 0;
           idx--) {
        if (path_hashes_[idx] == chash) {
          // A cycle in a checking/evasion tree cannot establish mate;
          // the conservative verdict for this route is no-mate (as in
          // MateDfpnSolver's path guard).
          stats_.rep_hits++;
          RecordVerdict(chash, paper_ghi_mode_ ? 0 : ply + 1,
                        bns::NoMateView(), true, idx, TaintKind::kPath, 0, 0,
                        0);
          handled = true;
          break;
        }
      }
    }

    if (handled) {
      refresh_only = s.best;
      board.UndoMove(bm, undo);
      continue;
    }

    path_hashes_.push_back(chash);
    SearchImpl<!kOrNode, kPnDn>(board, child_abn_th, child_obn_th, ply + 1);
    path_hashes_.pop_back();
    board.UndoMove(bm, undo);
    DropOverridesAtReturn(ply + 1);
    refresh_only = s.best;

    if (limit_hit_ || stop_.load(std::memory_order_relaxed)) return;
  }
}

// =====================================================================
// PV extraction
// =====================================================================

// Rebuild the mating line of a node proved by the 3-ply probe (whose
// children were never stored). Appends 1 or 3 plies; returns the number
// of plies appended, or 0 if no 3-ply mate exists here.
static int Extract3PlyLine(ShogiBoard& board, std::vector<Move>* pv) {
  MoveList checks = board.GenerateCheckingMoves();
  for (const Move& m1 : checks) {
    UndoInfo u1 = board.DoMove(m1, true);
    MoveList ev = board.GenerateEvasionMoves();
    if (ev.empty()) {
      board.UndoMove(m1, u1);
      pv->push_back(m1);
      return 1;
    }
    Move em2, em3;
    bool all_mated = true;
    for (const Move& m2 : ev) {
      UndoInfo u2 = board.DoMove(m2);
      Move m3 = board.FindMateInOne();
      if (m3.is_null()) {
        all_mated = false;
      } else {
        em2 = m2;
        em3 = m3;
      }
      board.UndoMove(m2, u2);
      if (!all_mated) break;
    }
    board.UndoMove(m1, u1);
    if (all_mated) {
      pv->push_back(m1);
      pv->push_back(em2);
      pv->push_back(em3);
      return 3;
    }
  }
  return 0;
}
// TT walk in the cshogi get_pv style: at OR nodes follow any proved
// child (re-proving on demand if the proof was evicted); at AND nodes
// follow the longest proved defense. Returns the distance to mate, or
// -1 on failure.

template <bool kOrNode>
int MateBnsSolver::ExtractPvInner(ShogiBoard& board, std::vector<Move>* pv,
                                  int depth_left) {
  if (depth_left <= 0) return -1;
  if (extraction_steps_ >= kExtractionBudget) return -1;

  if constexpr (kOrNode) {
    Move m1 = board.FindMateInOne();
    if (!m1.is_null()) {
      pv->push_back(m1);
      return 1;
    }
    if (int d3 = Extract3PlyLine(board, pv)) return d3;
    for (int attempt = 0; attempt < 2; attempt++) {
      MoveList moves = board.GenerateCheckingMoves();
      int best_depth = -1;
      std::vector<Move> best_line;
      for (const Move& m : moves) {
        extraction_steps_++;
        const uint64_t chash = board.HashAfter(m);
        BnsTTEntry* e = tt_.Probe(
            chash,
            paper_ghi_mode_ ? 0 : static_cast<int>(path_hashes_.size()));
        bool child_proved = e && e->abn == 0;
        if (!child_proved && use_dominance_) {
          lczero::Hand h = board.hand(board.side_to_move());
          if (m.is_drop()) {
            h.Sub(m.drop_piece());
          } else {
            lczero::Piece cap = board.piece_on(m.to());
            if (!cap.IsNone()) h.Add(cap.GetType().Unpromote());
          }
          child_proved = finals_.Probe(board.BoardKeyAfter(m), h) > 0;
        }
        if (!child_proved) continue;
        // Avoid walking into a cycle.
        bool on_path = false;
        for (uint64_t h : path_hashes_) {
          if (h == chash) {
            on_path = true;
            break;
          }
        }
        if (on_path) continue;

        UndoInfo undo = board.DoMove(m, true);
        if (board.CheckRepetition(kRepetitionLookbackPly) !=
            RepetitionResult::kNone) {
          board.UndoMove(m, undo);
          continue;
        }
        path_hashes_.push_back(chash);
        std::vector<Move> line{m};
        const int d = ExtractPvInner<false>(board, &line, depth_left - 1);
        path_hashes_.pop_back();
        board.UndoMove(m, undo);
        // The attacker prefers the shortest proven line.
        if (d >= 0 && (best_depth < 0 || d + 1 < best_depth)) {
          best_depth = d + 1;
          best_line = std::move(line);
          if (best_depth <= 2) break;  // cannot beat a mate-in-1 reply
        }
      }
      if (best_depth >= 0) {
        pv->insert(pv->end(), best_line.begin(), best_line.end());
        return best_depth;
      }
      if (attempt == 0) {
        // The proof chain was evicted from the table; re-prove this
        // position (it is known mate) and retry once.
        const int ply = static_cast<int>(path_hashes_.size()) - 1;
        if (ply + 1 >= max_ply_) break;
        // Re-prove without dominance short-circuits so the line's
        // concrete children materialize in the main table.
        in_extraction_research_ = true;
        if (arith_ == Arith::kBns) {
          SearchImpl<true, false>(board, bns::kInf, bns::kInf, ply);
        } else {
          SearchImpl<true, true>(board, bns::kInf, bns::kInf, ply);
        }
        in_extraction_research_ = false;
      }
    }
    return -1;
  } else {
    MoveList moves = board.GenerateEvasionMoves();
    if (moves.empty()) return 0;

    int best_depth = -1;
    std::vector<Move> best_line;
    for (const Move& m : moves) {
      extraction_steps_++;
      const uint64_t chash = board.HashAfter(m);
      UndoInfo undo = board.DoMove(m);
      path_hashes_.push_back(chash);
      std::vector<Move> line{m};
      const int d = ExtractPvInner<true>(board, &line, depth_left - 1);
      path_hashes_.pop_back();
      board.UndoMove(m, undo);
      if (d >= 0 && d > best_depth) {
        best_depth = d;
        best_line = std::move(line);
      }
    }
    if (best_depth < 0) return -1;
    pv->insert(pv->end(), best_line.begin(), best_line.end());
    return best_depth + 1;
  }
}

void MateBnsSolver::ExtractPV(ShogiBoard& board) {
  pv_.clear();
  extraction_steps_ = 0;
  path_hashes_.clear();
  path_hashes_.push_back(board.Hash());
  const int d = ExtractPvInner<true>(board, &pv_, max_ply_);
  path_hashes_.clear();
  if (d < 0) {
    pv_.clear();
    mate_ply_ = 0;
    return;
  }
  mate_ply_ = static_cast<int>(pv_.size());
}

// Explicit instantiations.
template void MateBnsSolver::SearchImpl<true, false>(ShogiBoard&, uint32_t,
                                                     uint32_t, int);
template void MateBnsSolver::SearchImpl<false, false>(ShogiBoard&, uint32_t,
                                                      uint32_t, int);
template void MateBnsSolver::SearchImpl<true, true>(ShogiBoard&, uint32_t,
                                                    uint32_t, int);
template void MateBnsSolver::SearchImpl<false, true>(ShogiBoard&, uint32_t,
                                                     uint32_t, int);

}  // namespace jhbr2
