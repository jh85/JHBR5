/*
  JHBR3 — BNS solver tests

  1. Pure branch-number arithmetic on hand-constructed AND/OR child
     sets (the "small manually constructed AND/OR graphs" check).
  2. HashAfter() property test against DoMove() on generated positions.
  3. Fixed tsume positions: BNS verdict + PV legality, cross-checked
     against MateDfpnSolver.
*/

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "mate/bns.h"
#include "mate/dfpn.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace lczero;
using namespace jhbr2;

static int g_failures = 0;

#define CHECK(cond, ...)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      g_failures++;                                             \
      printf("FAIL %s:%d: %s : ", __FILE__, __LINE__, #cond);   \
      printf(__VA_ARGS__);                                      \
      printf("\n");                                             \
    }                                                           \
  } while (0)

// =====================================================================
// 1. Arithmetic on synthetic AND/OR graphs
// =====================================================================

static void TestSummarizeOr() {
  using namespace bns;
  {
    // Fresh OR node: three unresolved leaves.
    ChildView c[3] = {{1, 1}, {1, 1}, {1, 1}};
    Summary s = Summarize<true, false>(c, 3);
    CHECK(!s.terminal(), "fresh OR terminal");
    CHECK(s.abn == 1, "abn=%u", s.abn);
    CHECK(s.obn == 3, "obn=%u (1 + k-1 = 3)", s.obn);
    CHECK(s.k == 3, "k=%u", s.k);
    CHECK(s.second == 1, "second=%u", s.second);
  }
  {
    // Distinct values; best = min abn, obn = best.obn + (k-1).
    ChildView c[3] = {{5, 9}, {2, 7}, {4, 1}};
    Summary s = Summarize<true, false>(c, 3);
    CHECK(s.best == 1, "best=%d", s.best);
    CHECK(s.abn == 2 && s.obn == 7 + 2, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.second == 4, "second=%u", s.second);
  }
  {
    // Tie on abn: preserve generator order; second equals the tied value
    // (multiset semantics).
    ChildView c[3] = {{2, 9}, {2, 3}, {6, 1}};
    Summary s = Summarize<true, false>(c, 3);
    CHECK(s.best == 0, "tie best=%d", s.best);
    CHECK(s.second == 2, "tie second=%u", s.second);
    CHECK(s.obn == 9 + 2, "tie obn=%u", s.obn);
  }
  {
    // A proved child proves the OR node.
    ChildView c[3] = {{4, 4}, {0, kInf}, {kInf, 0}};
    Summary s = Summarize<true, false>(c, 3);
    CHECK(s.proved && !s.disproved, "proved");
    CHECK(s.abn == 0 && s.obn == kInf, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.best == 1, "best=%d", s.best);
  }
  {
    // All children disproved: OR node disproved.
    ChildView c[2] = {{kInf, 0}, {kInf, 0}};
    Summary s = Summarize<true, false>(c, 2);
    CHECK(s.disproved && !s.proved, "disproved");
    CHECK(s.abn == kInf && s.obn == 0, "abn=%u obn=%u", s.abn, s.obn);
  }
  {
    // Disproved siblings do not count into k.
    ChildView c[3] = {{kInf, 0}, {3, 5}, {kInf, 0}};
    Summary s = Summarize<true, false>(c, 3);
    CHECK(s.k == 1, "k=%u", s.k);
    CHECK(s.abn == 3 && s.obn == 5, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.second == kInf, "second=%u", s.second);
  }
  {
    // Saturation: obn = best.obn + (k-1) clamps at kInf.
    ChildView c[2] = {{1, kInf - 1}, {2, 1}};
    Summary s = Summarize<true, false>(c, 2);
    CHECK(s.best == 0, "sat best=%d", s.best);
    CHECK(s.obn == kInf, "sat obn=%u", s.obn);
  }
}

static void TestSummarizeAnd() {
  using namespace bns;
  {
    // AND node: dual of OR with obn selected.
    ChildView c[3] = {{9, 5}, {7, 2}, {1, 4}};
    Summary s = Summarize<false, false>(c, 3);
    CHECK(s.best == 1, "best=%d", s.best);
    CHECK(s.obn == 2 && s.abn == 7 + 2, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.second == 4, "second=%u", s.second);
  }
  {
    // The stable tie rule is symmetric at AND nodes.
    ChildView c[3] = {{9, 2}, {3, 2}, {1, 6}};
    Summary s = Summarize<false, false>(c, 3);
    CHECK(s.best == 0, "tie best=%d", s.best);
    CHECK(s.second == 2, "tie second=%u", s.second);
    CHECK(s.abn == 9 + 2, "tie abn=%u", s.abn);
  }
  {
    // A disproved child (obn=0) disproves the AND node.
    ChildView c[2] = {{2, 2}, {kInf, 0}};
    Summary s = Summarize<false, false>(c, 2);
    CHECK(s.disproved, "and disproved");
    CHECK(s.abn == kInf && s.obn == 0, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.best == 1, "best=%d", s.best);
  }
  {
    // All children proved: AND node proved (every defense mated).
    ChildView c[2] = {{0, kInf}, {0, kInf}};
    Summary s = Summarize<false, false>(c, 2);
    CHECK(s.proved, "and proved");
    CHECK(s.abn == 0 && s.obn == kInf, "abn=%u obn=%u", s.abn, s.obn);
  }
  {
    // Proved siblings do not count into k; abn = best.abn + (k-1).
    ChildView c[4] = {{0, kInf}, {3, 2}, {5, 6}, {0, kInf}};
    Summary s = Summarize<false, false>(c, 4);
    CHECK(s.k == 2, "k=%u", s.k);
    CHECK(s.obn == 2 && s.abn == 3 + 1, "abn=%u obn=%u", s.abn, s.obn);
    CHECK(s.second == 6, "second=%u", s.second);
  }
}

static void TestThresholds() {
  using namespace bns;
  {
    // OR node: ABN' = min(second+1, ABN), OBN' = OBN - (k-1).
    ChildView c[3] = {{2, 7}, {4, 1}, {5, 9}};
    Summary s = Summarize<true, false>(c, 3);
    // node: abn=2, obn=7+2=9, second=4, k=3.
    uint32_t a, o;
    ChildThresholds<true>(s, c[s.best], /*ABN=*/100, /*OBN=*/50, &a, &o);
    CHECK(a == 5, "ABN'=%u (min(4+1,100))", a);
    CHECK(o == 48, "OBN'=%u (50-(9-7))", o);
    // Tight parent ABN wins the min.
    ChildThresholds<true>(s, c[s.best], /*ABN=*/4, /*OBN=*/50, &a, &o);
    CHECK(a == 4, "ABN'=%u", a);
    // Infinite OBN propagates as infinite.
    ChildThresholds<true>(s, c[s.best], kInf, kInf, &a, &o);
    CHECK(a == 5 && o == kInf, "ABN'=%u OBN'=%u", a, o);
  }
  {
    // Single active child: second = INF, so ABN' = ABN.
    ChildView c[1] = {{3, 4}};
    Summary s = Summarize<true, false>(c, 1);
    uint32_t a, o;
    ChildThresholds<true>(s, c[0], 77, 20, &a, &o);
    CHECK(a == 77, "ABN'=%u", a);
    CHECK(o == 20, "OBN'=%u (k=1)", o);
  }
  {
    // AND node duals.
    ChildView c[3] = {{7, 2}, {1, 4}, {9, 5}};
    Summary s = Summarize<false, false>(c, 3);
    // node: obn=2, abn=7+2=9, second=4, k=3.
    uint32_t a, o;
    ChildThresholds<false>(s, c[s.best], /*ABN=*/50, /*OBN=*/100, &a, &o);
    CHECK(o == 5, "OBN'=%u (min(4+1,100))", o);
    CHECK(a == 48, "ABN'=%u (50-(9-7))", a);
  }
  {
    // pn/dn mode: sums instead of counts.
    ChildView c[3] = {{2, 7}, {4, 1}, {5, 9}};
    Summary s = Summarize<true, true>(c, 3);
    CHECK(s.abn == 2, "pn=%u", s.abn);
    CHECK(s.obn == 17, "dn=%u (7+1+9)", s.obn);
    uint32_t a, o;
    ChildThresholds<true>(s, c[s.best], 100, 50, &a, &o);
    CHECK(a == 5, "thpn=%u", a);
    CHECK(o == 50 - (17 - 7), "thdn=%u", o);
  }
  {
    // Simulated iteration on a two-level graph: OR -> {AND1, AND2},
    // AND1 -> {leaf a, leaf b}, AND2 -> {leaf c}. Verify the return /
    // continue conditions compose (regression of the paper's rules).
    ChildView and1_children[2] = {{1, 1}, {1, 1}};
    Summary and1 = Summarize<false, false>(and1_children, 2);
    CHECK(and1.abn == 2 && and1.obn == 1, "and1 %u/%u", and1.abn, and1.obn);
    ChildView and2_children[1] = {{1, 1}};
    Summary and2 = Summarize<false, false>(and2_children, 1);
    CHECK(and2.abn == 1 && and2.obn == 1, "and2 %u/%u", and2.abn, and2.obn);
    ChildView or_children[2] = {{and1.abn, and1.obn}, {and2.abn, and2.obn}};
    Summary root = Summarize<true, false>(or_children, 2);
    // Best = AND2 (abn 1 < 2), obn = 1 + 1 = 2.
    CHECK(root.best == 1 && root.abn == 1 && root.obn == 2, "root %u/%u",
          root.abn, root.obn);
    // Root thresholds INF/INF: continue condition holds.
    CHECK(kInf > root.abn && kInf > root.obn, "continue at root");
    uint32_t a, o;
    ChildThresholds<true>(root, or_children[1], kInf, kInf, &a, &o);
    CHECK(a == 3, "child ABN'=%u (second=2 +1)", a);
    CHECK(o == kInf, "child OBN'=%u", o);
    // AND2 under (3, INF): its abn=1 < 3 and obn=1 < INF -> continues.
    CHECK(a > and2.abn && o > and2.obn, "and2 continues");
    // If AND2's leaf becomes disproved, AND2 disproves; root falls back
    // to AND1 and its obn drops to 1.
    and2_children[0] = NoMateView();
    and2 = Summarize<false, false>(and2_children, 1);
    CHECK(and2.disproved, "and2 disproved");
    or_children[1] = {and2.abn, and2.obn};
    root = Summarize<true, false>(or_children, 2);
    CHECK(root.best == 0 && root.abn == 2 && root.obn == 1, "root2 %u/%u",
          root.abn, root.obn);
    // If AND1's leaves both become proved, AND1 proves and so does root.
    and1_children[0] = MateView();
    and1_children[1] = MateView();
    and1 = Summarize<false, false>(and1_children, 2);
    CHECK(and1.proved, "and1 proved");
    or_children[0] = {and1.abn, and1.obn};
    root = Summarize<true, false>(or_children, 2);
    CHECK(root.proved && root.abn == 0 && root.obn == kInf, "root proved");
  }
}

// =====================================================================
// 2. HashAfter property test
// =====================================================================

static void TestDominates() {
  Hand a, b;
  CHECK(a.Dominates(b) && b.Dominates(a), "empty hands");
  a.Add(kPawn);
  CHECK(a.Dominates(b) && !b.Dominates(a), "one pawn");
  b.Add(kRook);
  CHECK(!a.Dominates(b) && !b.Dominates(a), "incomparable");
  for (int i = 0; i < 17; i++) a.Add(kPawn);  // 18 pawns
  a.Add(kGold); a.Add(kGold); a.Add(kGold); a.Add(kGold);
  Hand c = a;
  CHECK(a.Dominates(c) && c.Dominates(a), "max fields equal");
  c.Sub(kGold);
  CHECK(a.Dominates(c) && !c.Dominates(a), "gold boundary (bit 31)");
  Hand d, e;
  d.Add(kBishop); d.Add(kBishop);
  e.Add(kBishop);
  CHECK(d.Dominates(e) && !e.Dominates(d), "bishop field");
}

static void TestBoardKeyAfter() {
  std::mt19937_64 rng(999);
  ShogiBoard board;
  int checked = 0;
  for (int game = 0; game < 20; game++) {
    board.SetStartPos();
    for (int step = 0; step < 100; step++) {
      MoveList moves = board.GenerateLegalMoves();
      if (moves.empty()) break;
      for (const Move& m : moves) {
        uint64_t predicted = board.BoardKeyAfter(m);
        UndoInfo u = board.DoMove(m);
        uint64_t actual = board.BoardKey();
        board.UndoMove(m, u);
        CHECK(predicted == actual, "BoardKeyAfter mismatch %s",
              m.ToString().c_str());
        checked++;
      }
      board.DoMove(moves[rng() % moves.size()]);
    }
  }
  printf("  BoardKeyAfter verified on %d moves\n", checked);
}

// Soundness of the approximate mate-in-1: every move it returns must be
// a legal immediate checkmate; and the exact routine must agree a mate
// exists. Also reports its coverage relative to the exact routine.
static void TestMate1Approx() {
  std::mt19937_64 rng(4242);
  ShogiBoard board;
  long approx_found = 0, exact_found = 0, positions = 0;
  for (int game = 0; game < 400; game++) {
    board.SetStartPos();
    // Random playout; check every position along the way.
    for (int step = 0; step < 160; step++) {
      MoveList moves = board.GenerateLegalMoves();
      if (moves.empty()) break;
      if (!board.InCheck()) {
        positions++;
        Move a = board.FindMateInOneApprox();
        Move e = board.FindMateInOneNonCheck();
        if (!e.is_null()) exact_found++;
        if (!a.is_null()) {
          approx_found++;
          // Approx move must be legal and deliver immediate mate.
          bool legal = false;
          for (const Move& m : moves)
            if (m == a) legal = true;
          CHECK(legal, "approx move %s illegal in %s", a.ToString().c_str(),
                board.ToSfen().c_str());
          CHECK(!e.is_null(), "approx found mate, exact did not: %s in %s",
                a.ToString().c_str(), board.ToSfen().c_str());
          if (legal) {
            UndoInfo u = board.DoMove(a);
            CHECK(board.InCheck() && board.GenerateLegalMoves().empty(),
                  "approx move %s is not mate in %s", a.ToString().c_str(),
                  board.ToSfen().c_str());
            board.UndoMove(a, u);
          }
        }
      }
      board.DoMove(moves[rng() % moves.size()]);
    }
  }
  printf("  mate1 approx: %ld positions, exact found %ld, approx found %ld"
         " (%.1f%% coverage)\n",
         positions, exact_found, approx_found,
         exact_found ? 100.0 * approx_found / exact_found : 100.0);
}

static void TestHashAfter() {
  std::mt19937_64 rng(12345);
  ShogiBoard board;
  board.SetStartPos();

  int checked = 0;
  for (int game = 0; game < 40; game++) {
    board.SetStartPos();
    for (int step = 0; step < 120; step++) {
      MoveList moves = board.GenerateLegalMoves();
      if (moves.empty()) break;
      for (const Move& m : moves) {
        uint64_t predicted = board.HashAfter(m);
        UndoInfo u = board.DoMove(m);
        uint64_t actual = board.Hash();
        board.UndoMove(m, u);
        CHECK(predicted == actual, "HashAfter mismatch move %s",
              m.ToString().c_str());
        checked++;
      }
      Move pick = moves[rng() % moves.size()];
      board.DoMove(pick);
    }
  }
  printf("  HashAfter verified on %d moves\n", checked);
}

// =====================================================================
// 3. Fixed positions
// =====================================================================

static bool ContainsMove(const MoveList& moves, Move target) {
  for (const Move& m : moves)
    if (m == target) return true;
  return false;
}

static bool IsValidMatePv(ShogiBoard board, Move result,
                          const std::vector<Move>& pv) {
  if (pv.empty() || pv.front() != result) return false;
  for (size_t ply = 0; ply < pv.size(); ++ply) {
    const MoveList legal = board.GenerateLegalMoves();
    if (!ContainsMove(legal, pv[ply])) return false;
    board.DoMove(pv[ply]);
    if (ply % 2 == 0 && !board.InCheck()) return false;
  }
  return board.InCheck() && board.GenerateLegalMoves().empty();
}

struct FixedCase {
  const char* sfen;
  int expect;  // >0: mate in <=expect plies; 0: no mate
  const char* what;
};

static void TestFixedPositions() {
  // Hand-constructed textbook shapes.
  static const FixedCase cases[] = {
      // Mate in 1: G*5b supported by the gold on 5c (atama-kin).
      {"4k4/9/4G4/9/9/9/9/9/8K b G 1", 1, "mate1 gold drop"},
      // Same shape without the hand gold: a lone gold cannot force mate.
      {"4k4/9/4G4/9/9/9/9/9/8K b - 1", 0, "no mate without hand gold"},
      // P*1b would mate (1b guarded by S2c, 2a by N3c, 2b by S2c) but a
      // pawn-drop mate is illegal (uchifuzume); every legal check fails.
      {"8k/9/6NS1/9/9/9/9/9/K8 b P 1", 0, "uchifuzume"},
      // The same mating pattern delivered by a pawn PUSH is legal.
      {"8k/9/6NSP/9/9/9/9/9/K8 b - 1", 1, "pawn push mate"},
      // Mate in 3: R*6b (guarded by P6c), K5a/7a, G*5b/7b (guarded by
      // the rook).
      {"3k5/9/3P5/9/9/9/9/9/8K b RG 1", 3, "mate3 rook+gold"},
  };

  for (const auto& fc : cases) {
    ShogiBoard board;
    if (!board.SetFromSfen(fc.sfen)) {
      CHECK(false, "bad sfen %s", fc.sfen);
      continue;
    }

    MateBnsSolver bns(/*tt_mb=*/16);
    bns.set_arith(MateBnsSolver::Arith::kBns);
    Move bm = bns.search(board, 2000000);

    MateDfpnSolver dfpn(2000000);
    Move dm = dfpn.search(board, 2000000);

    const bool bns_mate = !bm.is_null() && !MateBnsSolver::IsNoMate(bm);
    const bool bns_nomate = MateBnsSolver::IsNoMate(bm);
    const bool dfpn_mate = !dm.is_null() && !MateDfpnSolver::IsNoMate(dm);
    const bool dfpn_nomate = MateDfpnSolver::IsNoMate(dm);

    CHECK(bns_mate == dfpn_mate && bns_nomate == dfpn_nomate,
          "%s: bns %d/%d dfpn %d/%d", fc.what, bns_mate, bns_nomate,
          dfpn_mate, dfpn_nomate);

    if (fc.expect > 0) {
      CHECK(bns_mate, "%s: expected mate, bns says %s", fc.what,
            bns_nomate ? "nomate" : "unsolved");
      if (bns_mate) {
        CHECK(IsValidMatePv(board, bm, bns.get_pv()), "%s: invalid PV",
              fc.what);
        CHECK(bns.get_mate_ply() <= fc.expect && bns.get_mate_ply() % 2 == 1,
              "%s: mate ply %d", fc.what, bns.get_mate_ply());
      }
    } else {
      CHECK(bns_nomate, "%s: expected nomate, bns says %s", fc.what,
            bns_mate ? "mate" : "unsolved");
    }

    // pn/dn control mode must agree.
    MateBnsSolver pndn(/*tt_mb=*/16);
    pndn.set_arith(MateBnsSolver::Arith::kPnDn);
    Move pm = pndn.search(board, 2000000);
    const bool pndn_mate = !pm.is_null() && !MateBnsSolver::IsNoMate(pm);
    CHECK(pndn_mate == bns_mate, "%s: pndn disagrees", fc.what);
    if (pndn_mate) {
      CHECK(IsValidMatePv(board, pm, pndn.get_pv()), "%s: pndn invalid PV",
            fc.what);
    }
  }
}

int main() {
  ShogiTables::Init();

  printf("TestSummarizeOr...\n");
  TestSummarizeOr();
  printf("TestSummarizeAnd...\n");
  TestSummarizeAnd();
  printf("TestThresholds...\n");
  TestThresholds();
  printf("TestDominates...\n");
  TestDominates();
  printf("TestBoardKeyAfter...\n");
  TestBoardKeyAfter();
  printf("TestMate1Approx...\n");
  TestMate1Approx();
  printf("TestHashAfter...\n");
  TestHashAfter();
  printf("TestFixedPositions...\n");
  TestFixedPositions();

  if (g_failures) {
    printf("FAILED: %d checks\n", g_failures);
    return 1;
  }
  printf("All BNS tests passed.\n");
  return 0;
}
