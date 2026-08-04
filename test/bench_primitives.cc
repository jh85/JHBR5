/*
  JHBR3 — solver-relevant board primitive microbenchmarks.
  Mirrors cshogi_bench/bench_cshogi_core.cpp for direct comparison:
  attack lookups, DoMove/UndoMove, checking-move generation, evasion
  generation, mate-in-1 probe.
*/
#include <chrono>
#include <cstdio>
#include <string>

#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace lczero;
using Clock = std::chrono::steady_clock;

template <typename F>
static double NsPerIter(int iters, F&& f) {
  auto t0 = Clock::now();
  for (int i = 0; i < iters; i++) f(i);
  return std::chrono::duration<double>(Clock::now() - t0).count() / iters *
         1e9;
}

int main() {
  ShogiTables::Init();

  const std::string midgame2 =
      "l3k2nl/4g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL b "
      "RNPrnp 1";
  const std::string checkpos =
      "l3k1Rnl/4g2b1/p1sppsppp/2p3p2/1p7/2P1P4/PP1PSP1PP/1BG4R1/LN2KG1NL w "
      "NPrnp 2";

  ShogiBoard b;
  b.SetFromSfen(midgame2);

  {
    const Bitboard occ = b.occupied();
    volatile uint64_t sink = 0;
    double ns = NsPerIter(20000000, [&](int i) {
      Square sq = Square::FromIdx(i % 81);
      Bitboard r = ShogiTables::RookEffect(sq, occ);
      Bitboard bb = ShogiTables::BishopEffect(sq, occ);
      sink += r.Lo() ^ bb.Lo();
    });
    printf("rook+bishop attack pair:   %.1f ns\n", ns);
  }
  {
    MoveList moves = b.GenerateLegalMoves();
    Move m = moves[0];
    double ns = NsPerIter(10000000, [&](int) {
      UndoInfo u = b.DoMove(m);
      b.UndoMove(m, u);
    });
    printf("DoMove+UndoMove:           %.1f ns\n", ns);
  }
  {
    volatile int sink = 0;
    double ns = NsPerIter(1000000, [&](int) {
      MoveList ml = b.GenerateCheckingMoves();
      sink += ml.size();
    });
    printf("GenerateCheckingMoves:     %.1f ns (n=%d)\n", ns,
           b.GenerateCheckingMoves().size());
  }
  {
    volatile uint64_t sink = 0;
    double ns = NsPerIter(5000000, [&](int) {
      sink += b.ComputeBlockersForKing(WHITE).Lo();
      sink += b.ComputeBlockersForKing(BLACK).Lo();
    });
    printf("ComputeBlockersForKing x2: %.1f ns\n", ns);
  }
  {
    volatile int sink = 0;
    double ns = NsPerIter(1000000, [&](int) {
      MoveList ml = b.GenerateLegalMoves();
      sink += ml.size();
    });
    printf("GenerateLegalMoves:        %.1f ns (n=%d)\n", ns,
           b.GenerateLegalMoves().size());
  }
  {
    volatile int sink = 0;
    double ns = NsPerIter(2000000, [&](int) {
      sink += b.FindMateInOne().raw();
    });
    printf("FindMateInOne (no mate):   %.1f ns\n", ns);
  }
  {
    volatile int sink = 0;
    double ns = NsPerIter(2000000, [&](int) {
      sink += b.FindMateInOneApprox().raw();
    });
    printf("FindMateInOneApprox:       %.1f ns\n", ns);
  }

  ShogiBoard c;
  c.SetFromSfen(checkpos);
  {
    volatile int sink = 0;
    double ns = NsPerIter(2000000, [&](int) {
      MoveList ml = c.GenerateEvasionMoves();
      sink += ml.size();
    });
    printf("GenerateEvasionMoves:      %.1f ns (n=%d)\n", ns,
           c.GenerateEvasionMoves().size());
  }
  {
    volatile int sink = 0;
    double ns = NsPerIter(5000000, [&](int) { sink += c.InCheck(); });
    printf("InCheck:                   %.1f ns\n", ns);
  }
  return 0;
}
