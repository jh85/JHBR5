// Prints value and policy outputs for positions, for the trainer round-trip
// test (test/test_net_roundtrip.py).
//
//   eval_positions <value.nn> <policy.nn> <sfen file>
//
// Output per position:  S <sfen> / W <win> <draw> <loss> / M <usi> <logit>... / E

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nnue/evaluator.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s value.nn policy.nn sfens\n", argv[0]);
    return 2;
  }
  lczero::ShogiTables::Init();
  jhbr5::nnue::Init();
  jhbr5::nnue::NetworkSet nets;
  std::string err;
  if (!nets.Load(argv[1], argv[2], &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  jhbr5::nnue::Evaluator ev(nets);
  std::ifstream in(argv[3]);
  std::string line;
  std::vector<float> logits(lczero::kMaxLegalMoves);
  while (std::getline(in, line)) {
    const size_t tab = line.find('\t');
    const std::string sfen = tab == std::string::npos ? line : line.substr(0, tab);
    lczero::ShogiBoard board;
    if (sfen.empty() || !board.SetFromSfen(sfen)) continue;
    const jhbr5::nnue::Wdl w = ev.Evaluate(board);
    std::printf("S %s\nW %.8f %.8f %.8f\n", sfen.c_str(), w.win, w.draw, w.loss);
    lczero::MoveList moves = board.GenerateLegalMoves();
    ev.Policy(board, moves, logits.data());
    for (int i = 0; i < moves.size(); ++i) {
      std::printf("M %s %.6f\n", moves[i].ToString().c_str(), logits[i]);
    }
    std::printf("E\n");
  }
  return 0;
}
