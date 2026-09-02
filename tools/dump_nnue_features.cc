// Dumps feature indices and move buckets for positions, one block per sfen.
// Used by test/test_nnue_features_ref.py to compare against the pure-Python
// reference implementation.
//
//   dump_nnue_features <sfen file>
//
// Output per position:
//   S <sfen>
//   A_us <n> <idx...>
//   A_them <n> <idx...>
//   B <n> <idx...>
//   P <n> <idx...>
//   M <usi> <bucket> <see_good>      (one line per legal move)
//   E

#include <cstdio>
#include <fstream>
#include <string>

#include "nnue/features.h"
#include "nnue/move_buckets.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace jhbr5::nnue;

template <int N>
void Print(const char* tag, const FeatureList<N>& f) {
  std::printf("%s %d", tag, f.n);
  for (int i = 0; i < f.n; ++i) std::printf(" %d", f.idx[i]);
  std::printf("\n");
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  lczero::ShogiTables::Init();
  Init();
  std::printf("FS %u BT %u %u\n", FeatureSetId(), BucketTableId(false), BucketTableId(true));
  std::ifstream in(argv[1]);
  std::string line;
  while (std::getline(in, line)) {
    const size_t tab = line.find('\t');
    const std::string sfen = tab == std::string::npos ? line : line.substr(0, tab);
    lczero::ShogiBoard board;
    if (sfen.empty() || !board.SetFromSfen(sfen)) continue;
    std::printf("S %s\n", sfen.c_str());
    FeatureList<kMaxActiveA> a;
    GroupAFeatures(board, board.side_to_move(), &a);
    Print("A_us", a);
    GroupAFeatures(board, ~board.side_to_move(), &a);
    Print("A_them", a);
    FeatureList<kMaxActiveB> b;
    GroupBFeatures(board, &b);
    Print("B", b);
    FeatureList<kMaxActivePolicy> p;
    PolicyFeatures(board, &p);
    Print("P", p);
    lczero::MoveList moves = board.GenerateLegalMoves();
    for (int i = 0; i < moves.size(); ++i) {
      std::printf("M %s %d %d\n", moves[i].ToString().c_str(), MoveBucket(board, moves[i]),
                  MoveSeeGood(board, moves[i]) ? 1 : 0);
    }
    std::printf("E\n");
  }
  return 0;
}
