// Feature/bucket mapping self-consistency:
//   * table sizes match the constants in nnue/types.h
//   * indices are in range and unique per position
//   * rotating the board 180° and swapping colours leaves every feature and
//     bucket of the side to move unchanged (frame invariance)
//   * GroupADiff from an empty state reproduces GroupAFeatures
//
//   test_nnue_features <sfen file>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nnue/features.h"
#include "nnue/move_buckets.h"
#include "nnue/types.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace jhbr5::nnue;
using lczero::Move;
using lczero::MoveList;

namespace {

int failures = 0;
void Fail(const std::string& what, const std::string& sfen) {
  if (failures < 20) std::printf("FAIL %s: %s\n", what.c_str(), sfen.c_str());
  ++failures;
}

template <int N>
std::vector<int> Sorted(const FeatureList<N>& f) {
  std::vector<int> v(f.idx, f.idx + f.n);
  std::sort(v.begin(), v.end());
  return v;
}

bool InRangeUnique(const std::vector<int>& v, int limit) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i] < 0 || v[i] >= limit) return false;
    if (i && v[i] == v[i - 1]) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <sfen file>\n", argv[0]);
    return 2;
  }
  lczero::ShogiTables::Init();
  Init();

  if (Pairs().total != kPairsPerOwner) Fail("pair count", "");
  const BucketTables& bt = Buckets();
  if (bt.drop_off[kHandKinds - 1] + bt.drop_dest[kHandKinds - 1].PopCount() != kNumBuckets) {
    Fail("bucket count", "");
  }

  // v2: the 3x3 king-bucket grid covers 0..8 with 9 squares each and rotates
  // with the board (bucket(80-s) == 8 - bucket(s)).
  {
    int count[kKingBuckets] = {};
    for (int s = 0; s < 81; ++s) {
      const int b = KingBucket(s);
      if (b < 0 || b >= kKingBuckets) Fail("king bucket range", std::to_string(s));
      if (KingBucket(80 - s) != kKingBuckets - 1 - b) Fail("king bucket flip", std::to_string(s));
      ++count[b];
    }
    for (int b = 0; b < kKingBuckets; ++b) {
      if (count[b] != 9) Fail("king bucket coverage", std::to_string(b));
    }
  }

  std::ifstream in(argv[1]);
  std::string line;
  int positions = 0, max_b = 0, max_p = 0, max_p2 = 0;
  long phase_hist[kPhaseBuckets] = {};
  // SEE picks the lowest-square attacker among equal piece types, which is not
  // frame invariant when two same-type pieces attack the target and x-rays
  // differ (Stockfish and Monty share this property). Such cases are counted
  // and tolerated up to a tiny fraction of all moves.
  long see_checked = 0, see_mismatch = 0;
  while (std::getline(in, line)) {
    const size_t tab = line.find('\t');
    const std::string sfen = tab == std::string::npos ? line : line.substr(0, tab);
    if (sfen.empty()) continue;
    lczero::ShogiBoard board;
    if (!board.SetFromSfen(sfen)) {
      Fail("sfen parse", sfen);
      continue;
    }
    ++positions;
    const Color stm = board.side_to_move();
    lczero::ShogiBoard flipped = board.Flipped();

    FeatureList<kMaxActiveA> a_us, a_them, fa_us, fa_them;
    GroupAFeatures(board, stm, &a_us);
    GroupAFeatures(board, ~stm, &a_them);
    GroupAFeatures(flipped, flipped.side_to_move(), &fa_us);
    GroupAFeatures(flipped, ~flipped.side_to_move(), &fa_them);
    if (!InRangeUnique(Sorted(a_us), kGroupAInputs)) Fail("A_us range", sfen);
    if (!InRangeUnique(Sorted(a_them), kGroupAInputs)) Fail("A_them range", sfen);
    if (Sorted(a_us) != Sorted(fa_us)) Fail("A_us flip invariance", sfen);
    if (Sorted(a_them) != Sorted(fa_them)) Fail("A_them flip invariance", sfen);

    // Every non-king piece is exactly one slot.
    int pieces = 0;
    for (Color c : {lczero::BLACK, lczero::WHITE}) {
      pieces += board.pieces(c).PopCount() - (board.king_square(c).IsValid() ? 1 : 0);
      for (int h = 0; h < kHandKinds; ++h) {
        pieces += board.hand(c).Count(lczero::PieceType::FromIdx(static_cast<uint8_t>(h + 1)));
      }
    }
    if (a_us.n != pieces || a_them.n != pieces) Fail("A active count " + std::to_string(a_us.n) + " vs " + std::to_string(pieces), sfen);

    FeatureList<kMaxActiveB> b, fb;
    GroupBFeatures(board, &b);
    GroupBFeatures(flipped, &fb);
    max_b = std::max(max_b, b.n);
    if (!InRangeUnique(Sorted(b), kGroupBInputs)) Fail("B range/unique", sfen);
    if (Sorted(b) != Sorted(fb)) Fail("B flip invariance", sfen);

    FeatureList<kMaxActivePolicy> p, fp;
    PolicyFeatures(board, &p);
    PolicyFeatures(flipped, &fp);
    max_p = std::max(max_p, p.n);
    if (!InRangeUnique(Sorted(p), kPolicyInputs)) Fail("P range/unique", sfen);
    if (Sorted(p) != Sorted(fp)) Fail("P flip invariance", sfen);

    // Diff from an empty state == from scratch.
    FrameState st{};
    FeatureList<128> adds, subs;
    GroupADiff(board, stm, &st, &adds, &subs);
    if (subs.n != 0 || Sorted(adds) != Sorted(a_us)) Fail("A diff==scratch", sfen);
    // Diff against itself is empty.
    GroupADiff(board, stm, &st, &adds, &subs);
    if (adds.n || subs.n) Fail("A diff idempotent", sfen);

    // v2 group A: range/uniqueness/active-count/flip-invariance as v1.
    FeatureList<kMaxActiveA> a2_us, a2_them, fa2_us, fa2_them;
    GroupA2Features(board, stm, &a2_us);
    GroupA2Features(board, ~stm, &a2_them);
    GroupA2Features(flipped, flipped.side_to_move(), &fa2_us);
    GroupA2Features(flipped, ~flipped.side_to_move(), &fa2_them);
    if (!InRangeUnique(Sorted(a2_us), kGroupA2Inputs)) Fail("A2_us range", sfen);
    if (!InRangeUnique(Sorted(a2_them), kGroupA2Inputs)) Fail("A2_them range", sfen);
    if (Sorted(a2_us) != Sorted(fa2_us)) Fail("A2_us flip invariance", sfen);
    if (Sorted(a2_them) != Sorted(fa2_them)) Fail("A2_them flip invariance", sfen);
    if (a2_us.n != pieces || a2_them.n != pieces) {
      Fail("A2 active count " + std::to_string(a2_us.n) + " vs " + std::to_string(pieces), sfen);
    }
    FrameState st2{};
    FeatureList<128> adds2, subs2;
    GroupA2Diff(board, stm, &st2, &adds2, &subs2);
    if (subs2.n != 0 || Sorted(adds2) != Sorted(a2_us)) Fail("A2 diff==scratch", sfen);
    GroupA2Diff(board, stm, &st2, &adds2, &subs2);
    if (adds2.n || subs2.n) Fail("A2 diff idempotent", sfen);

    // v2 policy inputs.
    FeatureList<kMaxActivePolicy2> p2, fp2;
    Policy2Features(board, &p2);
    Policy2Features(flipped, &fp2);
    max_p2 = std::max(max_p2, p2.n);
    if (!InRangeUnique(Sorted(p2), kPolicy2Inputs)) Fail("P2 range/unique", sfen);
    if (Sorted(p2) != Sorted(fp2)) Fail("P2 flip invariance", sfen);

    // v2 material phase: bounded and color-symmetric.
    const int ph = PhaseBucket(board);
    if (ph < 0 || ph >= kPhaseBuckets) Fail("phase range", sfen);
    if (PhaseBucket(flipped) != ph) Fail("phase flip invariance", sfen);
    ++phase_hist[ph < 0 || ph >= kPhaseBuckets ? 0 : ph];

    // Buckets: unique over legal moves, in range, flip invariant (incl. SEE).
    MoveList moves = board.GenerateLegalMoves();
    std::vector<int> buckets;
    for (int i = 0; i < moves.size(); ++i) {
      const int bk = MoveBucket(board, moves[i]);
      if (bk < 0 || bk >= kNumBuckets) Fail("bucket range", sfen);
      buckets.push_back(bk);
      Move fm = moves[i];
      fm.Flip();
      if (MoveBucket(flipped, fm) != bk) Fail("bucket flip invariance", sfen);
      ++see_checked;
      if (MoveSeeGood(flipped, fm) != MoveSeeGood(board, moves[i])) ++see_mismatch;
      const int sb = MoveBucketSee(board, moves[i]);
      if (sb < 0 || sb >= kNumBucketsSee) Fail("see bucket range", sfen);
    }
    std::sort(buckets.begin(), buckets.end());
    if (std::adjacent_find(buckets.begin(), buckets.end()) != buckets.end()) {
      Fail("bucket uniqueness", sfen);
    }
  }
  if (see_mismatch * 10000 > see_checked) Fail("see flip invariance rate", std::to_string(see_mismatch));
  std::printf("test_nnue_features: %d positions, max active B=%d P=%d P2=%d, phases [",
              positions, max_b, max_p, max_p2);
  for (int i = 0; i < kPhaseBuckets; ++i) std::printf("%s%ld", i ? " " : "", phase_hist[i]);
  std::printf("], see frame mismatches %ld/%ld, %s\n",
              see_mismatch, see_checked, failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
