// NNUE throughput benchmark.
//
//   bench_nnue <sfen file> [--value V.nn --policy P.nn | --l1 N --l1p N] [--walk K] [--secs S]
//
// Without net paths, random nets of the given widths are generated in /tmp.
// "walk" mode plays K random plies from each position, evaluating every one,
// which approximates the locality the finny caches see in MCTS.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nnue/evaluator.h"
#include "nnue/random_net.h"
#include "nnue/simd.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace jhbr5;
using lczero::MoveList;
using lczero::ShogiBoard;

namespace {
struct Rng {
  uint64_t s;
  uint64_t Next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};
using Clock = std::chrono::steady_clock;
double Secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <sfen file> [--value V --policy P] [--l1 N] [--l1p N] [--walk K] [--secs S]\n", argv[0]);
    return 2;
  }
  std::string vpath, ppath;
  int l1 = 1024, l1p = 4096, walk = 8;
  double secs = 3.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--value") && i + 1 < argc) vpath = argv[++i];
    else if (!std::strcmp(argv[i], "--policy") && i + 1 < argc) ppath = argv[++i];
    else if (!std::strcmp(argv[i], "--l1") && i + 1 < argc) l1 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--l1p") && i + 1 < argc) l1p = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--walk") && i + 1 < argc) walk = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--secs") && i + 1 < argc) secs = std::atof(argv[++i]);
  }
  lczero::ShogiTables::Init();
  nnue::Init();
  std::string err;
  if (vpath.empty()) {
    vpath = "/tmp/jhbr5_bench_value.nn";
    ppath = "/tmp/jhbr5_bench_policy.nn";
    std::printf("generating random nets l1=%d l1p=%d ...\n", l1, l1p);
    if (!nnue::WriteRandomValueNet(vpath, l1, 1, &err) ||
        !nnue::WriteRandomPolicyNet(ppath, l1p, true, 1, &err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
  }
  nnue::NetworkSet nets;
  if (!nets.Load(vpath, ppath, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::printf("backend=%s value l1=%d policy l1=%d rows=%d\n", simd::BackendName(),
              nets.value.l1(), nets.policy.l1(), nets.policy.num_rows());

  std::vector<ShogiBoard> boards;
  {
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line) && boards.size() < 256) {
      const size_t tab = line.find('\t');
      ShogiBoard b;
      if (b.SetFromSfen(tab == std::string::npos ? line : line.substr(0, tab))) boards.push_back(b);
    }
  }
  if (boards.empty()) return 1;

  nnue::Evaluator ev(nets);
  Rng rng{1};
  std::vector<float> logits(lczero::kMaxLegalMoves);

  // Value: random walks.
  {
    uint64_t evals = 0;
    double dummy = 0;
    const auto t0 = Clock::now();
    while (Secs(t0, Clock::now()) < secs) {
      ShogiBoard b = boards[rng.Next() % boards.size()];
      for (int ply = 0; ply <= walk; ++ply) {
        dummy += ev.Evaluate(b).win;
        ++evals;
        MoveList moves = b.GenerateLegalMoves();
        if (moves.empty()) break;
        b.DoMove(moves[static_cast<int>(rng.Next() % moves.size())]);
      }
    }
    const double dt = Secs(t0, Clock::now());
    std::printf("value: %.0f evals/s  (%.2f us/eval, %.1f A-rows + %.1f B-rows per eval, walk=%d) [%g]\n",
                evals / dt, 1e6 * dt / evals,
                static_cast<double>(ev.value().rows_a()) / evals,
                static_cast<double>(ev.value().rows_b()) / evals, walk, dummy);
  }
  // Policy: expansions (movegen + hidden + one logit per legal move).
  {
    uint64_t exps = 0, moves_total = 0;
    double dummy = 0;
    const auto t0 = Clock::now();
    while (Secs(t0, Clock::now()) < secs) {
      ShogiBoard b = boards[rng.Next() % boards.size()];
      for (int ply = 0; ply <= walk; ++ply) {
        MoveList moves = b.GenerateLegalMoves();
        if (moves.empty()) break;
        ev.Policy(b, moves, logits.data());
        dummy += logits[0];
        ++exps;
        moves_total += moves.size();
        b.DoMove(moves[static_cast<int>(rng.Next() % moves.size())]);
      }
    }
    const double dt = Secs(t0, Clock::now());
    std::printf("policy: %.0f expansions/s (%.2f us/expansion, %.1f legal moves avg) [%g]\n",
                exps / dt, 1e6 * dt / exps, static_cast<double>(moves_total) / exps, dummy);
  }
  return 0;
}
