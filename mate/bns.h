/*
  JHBR3 Shogi Engine — BNS (Branch Number Search) Checkmate Solver

  Implements the route-branch-number search of:
    岡部文洋「経路分枝数を用いた詰め将棋解図について」
    (Fumihiro Okabe, "Application for solving tsume shogi problem by
     route branch number")

  Instead of df-pn's proof/disproof numbers (min/sum over children), BNS
  tracks per-node AND/OR *branch numbers*: the number of unresolved
  branches attached to the currently selected route.  Sums over siblings
  are replaced by counts of unresolved siblings, which makes the numbers
  immune to the DAG double-counting that inflates proof numbers.

    Unresolved leaf:  abn = 1,   obn = 1
    Proved mate:      abn = 0,   obn = INF
    Proved no-mate:   abn = INF, obn = 0
    OR node  (attacker; k active children, best = min abn):
        abn = best.abn,           obn = best.obn + (k - 1)
    AND node (defender; k active children, best = min obn):
        obn = best.obn,           abn = best.abn + (k - 1)
    Child thresholds (c2 = second-smallest relevant number):
        OR:  ABN' = min(abn(c2) + 1, ABN),   OBN' = OBN - k + 1
        AND: ABN' = ABN - k + 1,             OBN' = min(obn(c2) + 1, OBN)
    Iterate while ABN > abn && OBN > obn.

  The solver runs the search over a transposition table (BNS is designed
  for merged game graphs), with route-dependent results (repetitions,
  depth caps) kept out of the table via a path-scoped override list.

  Structure references (all GPLv3, like this project):
    - cshogi/src/dfpn.cpp        (dlshogi df-pn: TT-based recursion shape)
    - YaneuraOu source/engine/tanuki-mate-engine (clustered fixed-size TT)
    - mate/dfpn.{h,cc}           (host interfaces, repetition semantics)
*/

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;

// =====================================================================
// Pure branch-number arithmetic (unit-testable without a board)
// =====================================================================

namespace bns {

// Saturated infinity. Real branch numbers stay far below this: they only
// grow by +1 per unresolved sibling along a route, so anything at or
// above kInf means "infinite" (proved/disproved side), never a count.
constexpr uint32_t kInf = 1u << 30;

inline uint32_t SatAdd(uint32_t a, uint32_t b) {
  uint64_t s = static_cast<uint64_t>(a) + b;
  return s >= kInf ? kInf : static_cast<uint32_t>(s);
}

// A child's numbers as seen from its parent.
struct ChildView {
  uint32_t abn = 1;
  uint32_t obn = 1;
};

// Node summary computed from children.
struct Summary {
  uint32_t abn = 1;
  uint32_t obn = 1;
  int best = -1;          // selected child index (-1 when terminal)
  uint32_t second = kInf; // second-smallest relevant number (multiset)
  uint32_t k = 0;         // number of active (unresolved) children
  bool proved = false;    // mate proved at this node
  bool disproved = false; // no-mate proved at this node

  bool terminal() const { return proved || disproved; }
};

// Summarize children into node values.
//
// kOrNode: attacker to move. kPnDn: use proof/disproof-number arithmetic
// (sum over siblings) instead of branch numbers (count of unresolved
// siblings) — the control mode for algorithm comparisons; everything
// else in the solver is shared.
//
// The "relevant" number is abn at OR nodes and obn at AND nodes; the
// selected child minimizes it (ties broken toward the smaller opposite
// number, following the paper's section 5.1).
template <bool kOrNode, bool kPnDn>
inline Summary Summarize(const ChildView* c, int n) {
  Summary s;
  uint32_t best_rel = kInf, best_opp = kInf;
  uint32_t second = kInf;
  uint32_t k = 0;
  uint64_t opp_sum = 0;  // kPnDn only
  int best = -1;

  for (int i = 0; i < n; i++) {
    const uint32_t rel = kOrNode ? c[i].abn : c[i].obn;
    const uint32_t opp = kOrNode ? c[i].obn : c[i].abn;

    if (rel == 0) {
      // OR: a proved child proves the node. AND: a disproved child
      // disproves the node.
      s.best = i;
      s.proved = kOrNode;
      s.disproved = !kOrNode;
      if (kOrNode) {
        s.abn = 0;
        s.obn = kInf;
      } else {
        s.abn = kInf;
        s.obn = 0;
      }
      return s;
    }
    if (kPnDn) opp_sum += opp;
    if (rel >= kInf) continue;

    k++;
    if (rel < best_rel || (rel == best_rel && opp < best_opp)) {
      second = best_rel;
      best_rel = rel;
      best_opp = opp;
      best = i;
    } else if (rel < second) {
      second = rel;
    }
  }

  const bool exhausted = kPnDn ? opp_sum == 0 : k == 0;
  if (exhausted) {
    // OR: every child disproved -> no mate. AND: every child proved
    // (every defense mated) -> mate.
    s.proved = !kOrNode;
    s.disproved = kOrNode;
    if (kOrNode) {
      s.abn = kInf;
      s.obn = 0;
    } else {
      s.abn = 0;
      s.obn = kInf;
    }
    return s;
  }

  const uint32_t node_rel = best_rel;
  const uint32_t node_opp =
      kPnDn ? (opp_sum >= kInf ? kInf : static_cast<uint32_t>(opp_sum))
            : SatAdd(best_opp, k - 1);

  s.abn = kOrNode ? node_rel : node_opp;
  s.obn = kOrNode ? node_opp : node_rel;
  s.best = best;
  s.second = second;
  s.k = k;
  return s;
}

// Thresholds to pass to the selected child.
//
//   OR:  ABN' = min(second + 1, ABN)
//        OBN' = OBN - (node.obn - best.obn)   [= OBN - k + 1 for BNS]
//   AND: symmetric.
//
// The subtraction form is shared by both arithmetics: for BNS the
// difference (node.opp - best.opp) is exactly k - 1; for pn/dn it is the
// sum of the other children's opposite numbers (cshogi's formula).
// Caller guarantees ABN > node.abn and OBN > node.obn (else it returns).
template <bool kOrNode>
inline void ChildThresholds(const Summary& s, const ChildView& best,
                            uint32_t abn_th, uint32_t obn_th,
                            uint32_t* child_abn_th, uint32_t* child_obn_th) {
  const uint32_t rel_th = kOrNode ? abn_th : obn_th;
  const uint32_t opp_th = kOrNode ? obn_th : abn_th;
  const uint32_t node_opp = kOrNode ? s.obn : s.abn;
  const uint32_t best_opp = kOrNode ? best.obn : best.abn;

  const uint32_t rel_out =
      rel_th < SatAdd(s.second, 1) ? rel_th : SatAdd(s.second, 1);
  const uint32_t opp_out =
      opp_th >= kInf ? kInf : opp_th - (node_opp - best_opp);

  *child_abn_th = kOrNode ? rel_out : opp_out;
  *child_obn_th = kOrNode ? opp_out : rel_out;
}

constexpr ChildView MateView() { return ChildView{0, kInf}; }
constexpr ChildView NoMateView() { return ChildView{kInf, 0}; }

}  // namespace bns

// =====================================================================
// Transposition table
// =====================================================================

// Fixed-size, 4-way clustered table keyed by (position hash, ply from
// root). Generation-based lazy clearing.
//
// The ply is part of the key (as in cshogi/dlshogi): it layers the
// search graph so that a position's entry can never feed back into its
// own value through a transposition at a different depth — without
// this, cross-depth sharing creates circular value dependencies and the
// threshold iteration can enter a limit cycle instead of converging.
// The price is that transpositions are only shared at equal depth.
// Route-dependent results are never stored here (see PathOverride).
struct BnsTTEntry {
  uint32_t key32 = 0;   // upper 32 bits of the position hash
  uint32_t abn = 1;
  uint32_t obn = 1;
  uint32_t effort = 0;  // node entries below this position (GC / victim)
  uint16_t best_move = 0;
  uint16_t ply = 0;     // distance from the search root (part of the key)
  uint16_t gen = 0;     // 0 = never used
  uint16_t key16 = 0;   // hash bits 16..31 (extra verification)

  bool final_result() const { return abn == 0 || obn == 0; }
};
static_assert(sizeof(BnsTTEntry) == 24, "BnsTTEntry should stay compact");

class BnsTT {
 public:
  static constexpr int kClusterSize = 4;

  // Entry-retention policy under pressure (adapted from cshogi/OSL's
  // Small-Tree GC: in a fixed clustered table the eviction decision is
  // made at replacement time instead of by a sweep).
  enum class VictimPolicy {
    kEffortPriority,  // keep final results and high-effort subtrees
    kAlwaysFirst,     // naive: overwrite the first live slot
  };
  void set_victim_policy(VictimPolicy p) { victim_policy_ = p; }

  void Resize(size_t mb) {
    size_t bytes = mb << 20;
    size_t want = bytes / sizeof(BnsTTEntry);
    size_t n = size_t{1} << 12;
    while (n * 2 <= want) n *= 2;
    if (n != num_entries_) {
      table_ = std::make_unique<BnsTTEntry[]>(n);
      num_entries_ = n;
      mask_ = (n - 1) & ~static_cast<size_t>(kClusterSize - 1);
      gen_ = 0;
    }
  }

  void NewSearch() {
    if (!table_) Resize(256);
    if (++gen_ == 0) {  // uint16 wrap: hard-clear once every 65535 searches
      std::memset(table_.get(), 0, num_entries_ * sizeof(BnsTTEntry));
      gen_ = 1;
    }
    probes_ = hits_ = stores_ = evictions_ = 0;
  }

  // Returns the matching live entry, or nullptr.
  BnsTTEntry* Probe(uint64_t hash, int ply) {
    probes_++;
    BnsTTEntry* c = &table_[IndexOf(hash, ply)];
    const uint32_t key = static_cast<uint32_t>(hash >> 32);
    const uint16_t keyb = static_cast<uint16_t>(hash >> 16);
    for (int i = 0; i < kClusterSize; i++) {
      if (c[i].gen == gen_ && c[i].key32 == key && c[i].ply == ply &&
          c[i].key16 == keyb) {
        hits_++;
        return &c[i];
      }
    }
    return nullptr;
  }

  void Prefetch(uint64_t hash, int ply) const {
    __builtin_prefetch(&table_[IndexOf(hash, ply)]);
  }

  // Returns an entry to write for this key: the live match if present,
  // else a victim (empty/stale first, then lowest effort preferring to
  // keep final results).
  BnsTTEntry* Store(uint64_t hash, int ply) {
    stores_++;
    BnsTTEntry* c = &table_[IndexOf(hash, ply)];
    const uint32_t key = static_cast<uint32_t>(hash >> 32);
    const uint16_t keyb = static_cast<uint16_t>(hash >> 16);
    BnsTTEntry* victim = nullptr;
    for (int i = 0; i < kClusterSize; i++) {
      if (c[i].gen == gen_ && c[i].key32 == key && c[i].ply == ply &&
          c[i].key16 == keyb)
        return &c[i];
      if (c[i].gen != gen_) {
        if (!victim || victim->gen == gen_) victim = &c[i];
        continue;
      }
      if (!victim || victim->gen != gen_) continue;
      if (victim_policy_ == VictimPolicy::kAlwaysFirst) continue;
      // Prefer evicting non-final, low-effort entries.
      const bool v_better = (victim->final_result() && !c[i].final_result()) ||
                            (victim->final_result() == c[i].final_result() &&
                             c[i].effort < victim->effort);
      if (v_better) victim = &c[i];
    }
    if (!victim) victim = c;
    if (victim->gen == gen_) evictions_++;
    victim->key32 = key;
    victim->key16 = keyb;
    victim->abn = 1;
    victim->obn = 1;
    victim->effort = 0;
    victim->best_move = 0;
    victim->ply = static_cast<uint16_t>(ply);
    victim->gen = gen_;
    return victim;
  }

  size_t num_entries() const { return num_entries_; }
  uint64_t probes() const { return probes_; }
  uint64_t hits() const { return hits_; }
  uint64_t stores() const { return stores_; }
  uint64_t evictions() const { return evictions_; }

 private:
  size_t IndexOf(uint64_t hash, int ply) const {
    // Mix the ply into the cluster index so equal positions at different
    // depths spread across the table instead of fighting one cluster.
    return (hash + static_cast<uint64_t>(ply) * 0x9E3779B97F4A7C15ull) & mask_;
  }

  std::unique_ptr<BnsTTEntry[]> table_;
  size_t num_entries_ = 0;
  size_t mask_ = 0;
  uint16_t gen_ = 0;
  VictimPolicy victim_policy_ = VictimPolicy::kEffortPriority;
  uint64_t probes_ = 0, hits_ = 0, stores_ = 0, evictions_ = 0;
};

// =====================================================================
// Finals table (hand dominance)
// =====================================================================

// Secondary fixed-size table holding only route-independent final
// verdicts, keyed by the board-only key and scanned with the hand
// dominance rule (cshogi/dlshogi's 優越関係): a mate proved with
// attacker hand H holds for any hand dominating H (the defender then
// holds a subset), and a no-mate proved with H holds for any hand
// dominated by H. Verdicts here are depth-independent, so this table
// recovers the cross-depth and cross-hand sharing that the depth-keyed
// main table gives up.
struct BnsFinalEntry {
  uint32_t key32 = 0;  // board key upper 32 bits
  uint16_t key16 = 0;  // board key bits 16..31
  uint16_t gen = 0;    // 0 = empty
  uint32_t hand = 0;   // attacker hand (raw)
  uint16_t proved = 0; // 1 = mate, 0 = no-mate
  uint16_t pad = 0;
};
static_assert(sizeof(BnsFinalEntry) == 16, "finals entry should stay compact");

class BnsFinalsTable {
 public:
  static constexpr int kClusterSize = 4;

  void Resize(size_t mb) {
    size_t want = (mb << 20) / sizeof(BnsFinalEntry);
    size_t n = size_t{1} << 12;
    while (n * 2 <= want) n *= 2;
    if (n != num_entries_) {
      table_ = std::make_unique<BnsFinalEntry[]>(n);
      num_entries_ = n;
      mask_ = (n - 1) & ~static_cast<size_t>(kClusterSize - 1);
      gen_ = 0;
    }
  }

  void NewSearch() {
    if (!table_) return;
    if (++gen_ == 0) {
      std::memset(table_.get(), 0, num_entries_ * sizeof(BnsFinalEntry));
      gen_ = 1;
    }
    hits_ = 0;
  }

  // +1 mate, -1 no-mate, 0 unknown.
  int Probe(uint64_t board_key, lczero::Hand attacker_hand) {
    if (!table_) return 0;
    const BnsFinalEntry* c = &table_[board_key & mask_];
    const uint32_t key = static_cast<uint32_t>(board_key >> 32);
    const uint16_t keyb = static_cast<uint16_t>(board_key >> 16);
    for (int i = 0; i < kClusterSize; i++) {
      if (c[i].gen != gen_ || c[i].key32 != key || c[i].key16 != keyb)
        continue;
      const lczero::Hand h(c[i].hand);
      if (c[i].proved) {
        if (attacker_hand.Dominates(h)) {
          hits_++;
          return 1;
        }
      } else {
        if (h.Dominates(attacker_hand)) {
          hits_++;
          return -1;
        }
      }
    }
    return 0;
  }

  void Store(uint64_t board_key, lczero::Hand attacker_hand, bool proved) {
    if (!table_) return;
    BnsFinalEntry* c = &table_[board_key & mask_];
    const uint32_t key = static_cast<uint32_t>(board_key >> 32);
    const uint16_t keyb = static_cast<uint16_t>(board_key >> 16);
    BnsFinalEntry* victim = &c[0];
    for (int i = 0; i < kClusterSize; i++) {
      if (c[i].gen != gen_) {
        victim = &c[i];
        continue;
      }
      if (c[i].key32 == key && c[i].key16 == keyb &&
          (c[i].proved != 0) == proved) {
        const lczero::Hand h(c[i].hand);
        // Keep only the stronger fact of the pair.
        if (proved ? h.Dominates(attacker_hand)
                   : attacker_hand.Dominates(h)) {
          victim = &c[i];  // new verdict subsumes the old one
          break;
        }
        if (proved ? attacker_hand.Dominates(h)
                   : h.Dominates(attacker_hand)) {
          return;  // old verdict already subsumes the new one
        }
      }
    }
    victim->key32 = key;
    victim->key16 = keyb;
    victim->gen = gen_;
    victim->hand = attacker_hand.raw();
    victim->proved = proved ? 1 : 0;
  }

  uint64_t hits() const { return hits_; }

 private:
  std::unique_ptr<BnsFinalEntry[]> table_;
  size_t num_entries_ = 0;
  size_t mask_ = 0;
  uint16_t gen_ = 0;
  uint64_t hits_ = 0;
};

// =====================================================================
// MateBnsSolver
// =====================================================================

class MateBnsSolver {
 public:
  using Clock = std::chrono::steady_clock;
  using Deadline = Clock::time_point;

  // Same bounded rule-aware repetition window as MateDfpnSolver; cycles
  // beyond the window are caught by the full-path hash scan.
  static constexpr int kRepetitionLookbackPly = 16;

  enum class Arith {
    kBns,   // branch numbers (the paper's algorithm)
    kPnDn,  // proof/disproof numbers in the identical engine
  };

  struct Stats {
    uint64_t node_entries = 0;   // SearchImpl invocations
    uint64_t first_visits = 0;   // node entries that created the TT entry
    uint64_t do_moves = 0;       // moves made by the search
    uint64_t summaries = 0;      // summarize passes
    uint64_t mate1_hits = 0;
    uint64_t mate3_hits = 0;
    uint64_t rep_hits = 0;       // repetition terminals
    uint64_t depth_hits = 0;     // depth-cap terminals
    int max_ply = 0;
    uint64_t tt_probes = 0, tt_hits = 0, tt_stores = 0, tt_evictions = 0;
    uint64_t dom_hits = 0;
    uint64_t mcache_probes = 0, mcache_hits = 0;
  };

  explicit MateBnsSolver(size_t tt_mb = 256, size_t default_nodes_limit = 100000)
      : default_nodes_limit_(default_nodes_limit), tt_mb_(tt_mb) {}

  // Search for a forced checkmate; side to move is the attacker.
  // Returns the first mating move, NoMateMove() if no mate is proved, or
  // a null Move when unresolved (limits / stop). Matches MateDfpnSolver.
  Move search(ShogiBoard board, size_t nodes_limit);
  Move search(ShogiBoard board, size_t nodes_limit, Deadline deadline);
  Move search(ShogiBoard board) { return search(board, default_nodes_limit_); }

  static Move NoMateMove() {
    return Move::Normal(lczero::Square::FromIdx(127),
                        lczero::Square::FromIdx(127));
  }
  static bool IsNoMate(Move m) { return m.raw() == NoMateMove().raw(); }

  int get_mate_ply() const { return mate_ply_; }
  std::vector<Move> get_pv() const { return pv_; }
  size_t get_nodes_searched() const { return stats_.node_entries; }
  const Stats& stats() const { return stats_; }

  void set_arith(Arith a) { arith_ = a; }
  void set_max_ply(int p) { max_ply_ = p; }

  // Probe fresh OR nodes with the shallow 3-ply mate routine.
  void set_use_mate3_probe(bool v) { use_mate3_probe_ = v; }

  // Order checking moves at OR nodes: captures, board moves, drops.
  void set_move_ordering(bool v) { move_ordering_ = v; }

  // Move cache: memoize (moves, child hashes) per (position, ply) so
  // revisits skip move generation and hash recomputation. 0 disables.
  void set_move_cache_mb(size_t mb) { move_cache_mb_ = mb; }

  // cshogi/dlshogi-style new-node block: at fresh OR nodes, walk each
  // check once; mate immediately if a defense set collapses, store
  // grandchild mate-1 proofs, and initialize child AND entries from
  // their evasion counts. Unlike the plain 3-ply probe, every byproduct
  // persists in the table as position-intrinsic information.
  void set_use_new_node_block(bool v) { use_new_node_block_ = v; }

  // Experimental mode following the paper's section 5.2 directly:
  // hash-only table keying (maximum transposition sharing across
  // depths), Figure-3 on-path threshold marking, and a Figure-4-style
  // escape that widens thresholds for nodes whose entries are being
  // hammered (effort above the bound) to break expansion-history loops.
  // The rule-aware repetition handling and path overrides stay active.
  void set_paper_ghi_mode(bool v) { paper_ghi_mode_ = v; }
  void set_ghi_escape_effort(uint32_t v) { ghi_escape_effort_ = v; }

  void set_victim_policy(BnsTT::VictimPolicy p) { tt_.set_victim_policy(p); }

  // Hand-dominance finals table (cshogi-style 優越関係 reuse).
  void set_use_dominance(bool v) { use_dominance_ = v; }

  // Diagnostic knobs (tests only).
  void set_disable_path_scan(bool v) { disable_path_scan_ = v; }
  void set_disable_rule_repetition(bool v) { disable_rule_repetition_ = v; }
  // After search(): which channel produced the root verdict.
  enum class RootChannel { kNone, kTT, kOverride };
  RootChannel root_channel() const { return root_channel_; }
  // After search(): the root's final branch numbers.
  bns::ChildView root_view() const { return root_view_; }

  // Sticky cancellation, as in MateDfpnSolver.
  void stop() { stop_.store(true, std::memory_order_release); }

 private:
  // Route-dependent verdicts (repetition, depth cap) valid only while
  // the anchoring ancestor stays on the DFS path. anchor_ply is the ply
  // of that ancestor; entries are dropped when the search unwinds past
  // it. kind distinguishes path-dependence from resource limits (a
  // resource-limited "no mate" must never be reported as no-mate).
  enum class TaintKind : uint8_t { kPath, kResource };

  struct PathOverride {
    uint64_t hash;
    bns::ChildView view;
    int anchor_ply;
    TaintKind kind;
  };

  // A child's numbers plus where they came from.
  struct SourcedView {
    bns::ChildView view;
    bool tainted = false;
    int anchor_ply = 0;
    TaintKind kind = TaintKind::kPath;
  };

  // Revisit memoization: movegen output is a pure function of the
  // position, so (moves, child hashes) can be reused across visits.
  // Direct-mapped, generation-stamped; nodes with more moves than a
  // slot holds simply bypass the cache.
  static constexpr int kMoveCacheMaxMoves = 24;
  struct MoveCacheSlot {
    uint64_t hash = 0;
    uint16_t gen = 0;
    uint16_t ply = 0;
    uint16_t n = 0;
    uint16_t pad = 0;
    Move moves[kMoveCacheMaxMoves];
    uint64_t child_hash[kMoveCacheMaxMoves];
  };

  // Per-ply scratch space (heap-allocated once; keeps recursion frames
  // small). Sized for the movegen worst case.
  struct Frame {
    MoveList moves;
    uint64_t child_hash[lczero::kMaxLegalMoves];
    uint64_t child_bkey[lczero::kMaxLegalMoves];
    uint32_t child_ahand[lczero::kMaxLegalMoves];
    bns::ChildView views[lczero::kMaxLegalMoves];
    int16_t anchor[lczero::kMaxLegalMoves];
    uint8_t tainted[lczero::kMaxLegalMoves];
    uint8_t kind[lczero::kMaxLegalMoves];
  };

  bool ShouldStop();

  template <bool kOrNode, bool kPnDn>
  void SearchImpl(ShogiBoard& board, uint32_t abn_th, uint32_t obn_th,
                  int ply);

  SourcedView LookupChild(uint64_t hash, int ply, uint64_t board_key,
                          uint32_t attacker_hand) {
    SourcedView sv;
    if (!overrides_.empty()) {
      for (auto it = overrides_.rbegin(); it != overrides_.rend(); ++it) {
        if (it->hash == hash) {
          sv.view = it->view;
          sv.tainted = true;
          sv.anchor_ply = it->anchor_ply;
          sv.kind = it->kind;
          return sv;
        }
      }
    }
    if (BnsTTEntry* e = tt_.Probe(hash, ply)) {
      sv.view = {e->abn, e->obn};
      if (sv.view.abn == 0 || sv.view.obn == 0) return sv;
    }
    if (use_dominance_ && !in_extraction_research_) {
      const int dom = finals_.Probe(board_key, lczero::Hand(attacker_hand));
      if (dom > 0) {
        stats_.dom_hits++;
        sv.view = bns::MateView();
      } else if (dom < 0) {
        stats_.dom_hits++;
        sv.view = bns::NoMateView();
      }
    }
    return sv;
  }

  void RecordVerdict(uint64_t hash, int ply, bns::ChildView v, bool tainted,
                     int anchor_ply, TaintKind kind, uint16_t best_move,
                     uint64_t board_key, uint32_t attacker_hand);
  void DropOverridesAtReturn(int ply);

  // PV extraction by TT walk (attacker: any proved child; defender:
  // longest proved defense).
  template <bool kOrNode>
  int ExtractPvInner(ShogiBoard& board, std::vector<Move>* pv, int depth_left);
  void ExtractPV(ShogiBoard& board);

  // --- Members ---
  size_t default_nodes_limit_;
  size_t tt_mb_;
  size_t nodes_limit_ = 0;
  BnsTT tt_;
  // pn/dn is the measured-faster arithmetic on short generated mate
  // sets (the tail problems resolve sooner); kBns is the paper's mode
  // and remains fully supported via set_arith().
  Arith arith_ = Arith::kPnDn;
  int max_ply_ = 129;
  bool use_mate3_probe_ = false;
  bool move_ordering_ = false;
  size_t move_cache_mb_ = 64;
  std::unique_ptr<MoveCacheSlot[]> move_cache_;
  size_t move_cache_mask_ = 0;
  uint16_t move_cache_gen_ = 0;
  uint64_t mcache_hits_ = 0, mcache_probes_ = 0;
  bool use_new_node_block_ = false;
  bool use_dominance_ = false;
  bool in_extraction_research_ = false;
  BnsFinalsTable finals_;
  bool paper_ghi_mode_ = false;
  uint32_t ghi_escape_effort_ = 10000;
  bool disable_path_scan_ = false;
  bool disable_rule_repetition_ = false;
  RootChannel root_channel_ = RootChannel::kNone;
  bns::ChildView root_view_{1, 1};

  Stats stats_;
  int mate_ply_ = 0;
  std::vector<Move> pv_;
  std::atomic<bool> stop_{false};
  Deadline deadline_ = Deadline::max();
  bool deadline_passed_ = false;
  uint64_t stop_check_counter_ = 0;
  bool limit_hit_ = false;
  bool resource_taint_seen_ = false;

  std::unique_ptr<Frame[]> frames_;
  size_t frames_size_ = 0;

  static constexpr uint64_t kExtractionBudget = 4000000;
  uint64_t extraction_steps_ = 0;

  std::vector<uint64_t> path_hashes_;
  std::vector<PathOverride> overrides_;
  uint64_t override_version_ = 0;
};

}  // namespace jhbr2
