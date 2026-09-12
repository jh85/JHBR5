// End-to-end network tests on deterministic random nets:
//   * scalar vs ISA forward pass bit-identical (value WDL and policy logits)
//   * finny-cached evaluation == from-scratch evaluation along a sequence
//   * WDL is a distribution; outputs are deterministic
//   * regression values for fixed positions and seed
//
//   test_nnue_net <sfen file> [tmpdir]

#include <cmath>
#include <cstdio>
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

int failures = 0;
void Fail(const char* what, const std::string& detail = "") {
  if (failures < 30) std::printf("FAIL %s %s\n", what, detail.c_str());
  ++failures;
}

struct Rng {
  uint64_t s;
  uint64_t Next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};

bool SameWdl(const nnue::Wdl& a, const nnue::Wdl& b) {
  return a.win == b.win && a.draw == b.draw && a.loss == b.loss;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <sfen file> [tmpdir]\n", argv[0]);
    return 2;
  }
  const std::string tmp = argc > 2 ? argv[2] : "/tmp";
  lczero::ShogiTables::Init();
  nnue::Init();

  const std::string vpath = tmp + "/jhbr5_test_value.nn";
  const std::string ppath = tmp + "/jhbr5_test_policy.nn";
  std::string err;
  if (!nnue::WriteRandomValueNet(vpath, 256, 1, &err) ||
      !nnue::WriteRandomPolicyNet(ppath, 512, true, 1, &err)) {
    std::printf("FAIL write nets: %s\n", err.c_str());
    return 1;
  }
  nnue::NetworkSet nets;
  if (!nets.Load(vpath, ppath, &err)) {
    std::printf("FAIL load nets: %s\n", err.c_str());
    return 1;
  }

  std::vector<ShogiBoard> boards;
  {
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line) && boards.size() < 64) {
      const size_t tab = line.find('\t');
      ShogiBoard b;
      if (b.SetFromSfen(tab == std::string::npos ? line : line.substr(0, tab))) boards.push_back(b);
    }
  }
  if (boards.empty()) {
    std::printf("FAIL no positions\n");
    return 1;
  }

  // 1. Scalar vs ISA, fresh scratch each time (so the finny path is a full refresh).
  std::vector<nnue::Wdl> ref_wdl;
  for (const ShogiBoard& b : boards) {
    simd::force_scalar = true;
    nnue::Evaluator es(nets);
    const nnue::Wdl ws = es.Evaluate(b);
    MoveList moves = const_cast<ShogiBoard&>(b).GenerateLegalMoves();
    std::vector<float> ls(moves.size()), lv(moves.size());
    es.Policy(b, moves, ls.data());
    simd::force_scalar = false;
    nnue::Evaluator ev(nets);
    const nnue::Wdl wv = ev.Evaluate(b);
    ev.Policy(b, moves, lv.data());
    if (!SameWdl(ws, wv)) Fail("scalar/isa wdl", b.ToSfen());
    if (std::memcmp(ls.data(), lv.data(), moves.size() * sizeof(float)) != 0) Fail("scalar/isa policy", b.ToSfen());
    const float sum = wv.win + wv.draw + wv.loss;
    if (!(std::fabs(sum - 1.0f) < 1e-5f) || wv.win < 0 || wv.draw < 0 || wv.loss < 0) Fail("wdl distribution", b.ToSfen());
    ref_wdl.push_back(wv);
  }

  // 2. Finny caching along random walks == fresh evaluation at every step.
  {
    nnue::Evaluator cached(nets);
    Rng rng{7};
    int steps = 0;
    for (ShogiBoard b : boards) {
      for (int ply = 0; ply < 12; ++ply) {
        const nnue::Wdl wc = cached.Evaluate(b);
        nnue::Evaluator fresh(nets);
        const nnue::Wdl wf = fresh.Evaluate(b);
        if (!SameWdl(wc, wf)) Fail("finny == scratch", b.ToSfen());
        ++steps;
        MoveList moves = b.GenerateLegalMoves();
        if (moves.empty()) break;
        b.DoMove(moves[static_cast<int>(rng.Next() % moves.size())]);
      }
    }
    std::printf("finny walk: %d evaluations, %.1f A-rows/eval, %.1f B-rows/eval\n", steps,
                static_cast<double>(cached.value().rows_a()) / steps,
                static_cast<double>(cached.value().rows_b()) / steps);
  }

  // 3. Determinism across repeated evaluation with the same scratch.
  {
    nnue::Evaluator ev(nets);
    for (size_t i = 0; i < boards.size(); ++i) {
      ev.Evaluate(boards[i]);
      if (!SameWdl(ev.Evaluate(boards[i]), ref_wdl[i])) Fail("determinism", boards[i].ToSfen());
    }
  }

  // 4. Regression values (seed 1, L1 256/512, first three positions of legal100).
  {
    struct Exp { const char* sfen; float w, d, l; };
    // Filled in from the first validated run; see docs/CHANGELOG.md.
    static const Exp kExpected[] = {
        {"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", 0.467709f, 0.157542f, 0.374750f},
    };
    nnue::Evaluator ev(nets);
    for (const Exp& e : kExpected) {
      ShogiBoard b;
      b.SetFromSfen(e.sfen);
      const nnue::Wdl w = ev.Evaluate(b);
      std::printf("regression %s -> W %.6f D %.6f L %.6f\n", e.sfen, w.win, w.draw, w.loss);
      if (e.w >= 0 && (std::fabs(w.win - e.w) > 1e-5f || std::fabs(w.draw - e.d) > 1e-5f ||
                       std::fabs(w.loss - e.l) > 1e-5f)) {
        Fail("regression value", e.sfen);
      }
    }
  }

  // 5. v2 architecture (docs/NNUE_V2_DESIGN.md): same properties on random v2 nets.
  const std::string v2path = tmp + "/jhbr5_test_value_v2.nn";
  const std::string p2path = tmp + "/jhbr5_test_policy_v2.nn";
  if (!nnue::WriteRandomValueNetV2(v2path, 256, 1, &err) ||
      !nnue::WriteRandomPolicyNetV2(p2path, 512, true, 1, &err)) {
    std::printf("FAIL write v2 nets: %s\n", err.c_str());
    return 1;
  }
  nnue::NetworkSet nets2;
  if (!nets2.Load(v2path, p2path, &err)) {
    std::printf("FAIL load v2 nets: %s\n", err.c_str());
    return 1;
  }

  // 5a. Scalar vs ISA, fresh scratch each time.
  std::vector<nnue::Wdl> ref2_wdl;
  for (const ShogiBoard& b : boards) {
    simd::force_scalar = true;
    nnue::Evaluator es(nets2);
    const nnue::Wdl ws = es.Evaluate(b);
    MoveList moves = const_cast<ShogiBoard&>(b).GenerateLegalMoves();
    std::vector<float> ls(moves.size()), lv(moves.size());
    es.Policy(b, moves, ls.data());
    simd::force_scalar = false;
    nnue::Evaluator ev(nets2);
    const nnue::Wdl wv = ev.Evaluate(b);
    ev.Policy(b, moves, lv.data());
    if (!SameWdl(ws, wv)) Fail("v2 scalar/isa wdl", b.ToSfen());
    if (std::memcmp(ls.data(), lv.data(), moves.size() * sizeof(float)) != 0) Fail("v2 scalar/isa policy", b.ToSfen());
    const float sum = wv.win + wv.draw + wv.loss;
    if (!(std::fabs(sum - 1.0f) < 1e-5f) || wv.win < 0 || wv.draw < 0 || wv.loss < 0) Fail("v2 wdl distribution", b.ToSfen());
    ref2_wdl.push_back(wv);
  }

  // 5b. Finny caching along random walks == fresh evaluation at every step.
  {
    nnue::Evaluator cached(nets2);
    Rng rng{7};
    int steps = 0;
    for (ShogiBoard b : boards) {
      for (int ply = 0; ply < 12; ++ply) {
        const nnue::Wdl wc = cached.Evaluate(b);
        nnue::Evaluator fresh(nets2);
        const nnue::Wdl wf = fresh.Evaluate(b);
        if (!SameWdl(wc, wf)) Fail("v2 finny == scratch", b.ToSfen());
        ++steps;
        MoveList moves = b.GenerateLegalMoves();
        if (moves.empty()) break;
        b.DoMove(moves[static_cast<int>(rng.Next() % moves.size())]);
      }
    }
    std::printf("v2 finny walk: %d evaluations, %.1f A-rows/eval, %.1f B-rows/eval\n", steps,
                static_cast<double>(cached.value().rows_a()) / steps,
                static_cast<double>(cached.value().rows_b()) / steps);
  }

  // 5c. King moves within and across buckets: finny == fresh at every step.
  {
    nnue::Evaluator cached(nets2);
    ShogiBoard b;
    b.SetFromSfen("4k4/9/9/9/4K4/9/9/9/9 b - 1");
    int bucket_changes = 0, steps = 0;
    for (int ply = 0; ply < 16; ++ply) {
      const nnue::Wdl wc = cached.Evaluate(b);
      nnue::Evaluator fresh(nets2);
      if (!SameWdl(wc, fresh.Evaluate(b))) Fail("v2 finny == fresh (king walk)", b.ToSfen());
      ++steps;
      const int before = nnue::KingBucket(nnue::KingFrameSq(b, b.side_to_move()));
      MoveList moves = b.GenerateLegalMoves();
      if (moves.empty()) break;
      // Directed stroll towards square 0: crosses 3x3 grid lines on the way.
      int best = 0;
      for (int i = 1; i < moves.size(); ++i) {
        if (moves[i].to().as_idx() < moves[best].to().as_idx()) best = i;
      }
      b.DoMove(moves[best]);
      if (nnue::KingBucket(nnue::KingFrameSq(b, b.side_to_move())) != before) ++bucket_changes;
    }
    if (steps == 0 || bucket_changes == 0) Fail("v2 king walk coverage", std::to_string(bucket_changes));
    std::printf("v2 king walk: %d evaluations, %d bucket changes\n", steps, bucket_changes);
  }

  // 5d. Determinism across repeated evaluation with the same scratch.
  {
    nnue::Evaluator ev(nets2);
    for (size_t i = 0; i < boards.size(); ++i) {
      ev.Evaluate(boards[i]);
      if (!SameWdl(ev.Evaluate(boards[i]), ref2_wdl[i])) Fail("v2 determinism", boards[i].ToSfen());
    }
  }

  // 5e. v1 and v2 networks coexist: interleaved evaluation stays deterministic.
  {
    nnue::Evaluator ev1(nets), ev2(nets2);
    for (size_t i = 0; i < boards.size(); ++i) {
      if (!SameWdl(ev2.Evaluate(boards[i]), ref2_wdl[i])) Fail("v2 interleaved", boards[i].ToSfen());
      if (!SameWdl(ev1.Evaluate(boards[i]), ref_wdl[i])) Fail("v1 interleaved", boards[i].ToSfen());
    }
  }

  // 5f. v2 regression values (seed 1, L1 256/512, start position).
  {
    struct Exp { const char* sfen; float w, d, l; };
    static const Exp kExpected2[] = {
        {"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", 0.328225f, 0.363114f, 0.308661f},
    };
    nnue::Evaluator ev(nets2);
    for (const Exp& e : kExpected2) {
      ShogiBoard b;
      b.SetFromSfen(e.sfen);
      const nnue::Wdl w = ev.Evaluate(b);
      std::printf("v2 regression %s -> W %.6f D %.6f L %.6f\n", e.sfen, w.win, w.draw, w.loss);
      if (e.w >= 0 && (std::fabs(w.win - e.w) > 1e-5f || std::fabs(w.draw - e.d) > 1e-5f ||
                       std::fabs(w.loss - e.l) > 1e-5f)) {
        Fail("v2 regression value", e.sfen);
      }
    }
  }

  std::remove(v2path.c_str());
  std::remove(p2path.c_str());
  std::remove(vpath.c_str());
  std::remove(ppath.c_str());
  std::printf("test_nnue_net backend=%s: %zu positions, %s\n", simd::BackendName(), boards.size(),
              failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
