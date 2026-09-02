// Hand-made static exchange evaluation cases.

#include <cstdio>
#include <string>

#include "shogi/bitboard.h"
#include "shogi/board.h"

using lczero::Move;
using lczero::ShogiBoard;

namespace {

int failures = 0;

void Expect(const char* sfen, const char* usi, int threshold, bool expected) {
  ShogiBoard b;
  if (!b.SetFromSfen(sfen)) {
    std::printf("FAIL parse %s\n", sfen);
    ++failures;
    return;
  }
  const bool got = b.SeeGe(Move::Parse(usi), threshold);
  if (got != expected) {
    std::printf("FAIL %s %s thr=%d: got %d expected %d\n", sfen, usi, threshold, got, expected);
    ++failures;
  }
}

}  // namespace

int main() {
  lczero::ShogiTables::Init();

  // A: R5i x p5e, gold on 5d recaptures: 90 - 990 = -900.
  const char* a = "4k4/9/9/4g4/4p4/9/9/9/4R3K b - 1";
  Expect(a, "5i5e", 0, false);
  Expect(a, "5i5e", -900, true);
  Expect(a, "5i5e", -899, false);

  // B: undefended pawn: +90.
  const char* bpos = "4k4/9/9/9/4p4/9/9/9/4R3K b - 1";
  Expect(bpos, "5i5e", 0, true);
  Expect(bpos, "5i5e", 90, true);
  Expect(bpos, "5i5e", 91, false);

  // C: drops. P*5e is attacked by the gold on 5d (white gold attacks 5e):
  // the dropped pawn is lost (-90). P*5f is safe.
  const char* c = "4k4/9/9/4g4/9/9/9/9/4K4 b P 1";
  Expect(c, "P*5e", -90, true);
  Expect(c, "P*5e", -89, false);
  Expect(c, "P*5f", 0, true);

  // D: promotion gain 540 - 90 = 450 with nothing recapturing on 5c.
  const char* d = "4k4/9/9/4P4/9/9/9/9/4K4 b - 1";
  Expect(d, "5d5c+", 450, true);
  Expect(d, "5d5c+", 451, false);
  Expect(d, "5d5c", 0, true);
  Expect(d, "5d5c", 1, false);

  // E: x-ray. R5h x p5e, g5d x R, R5i x g (through the vacated 5h):
  // 90 - 990 + 540 = -360.
  const char* e = "4k4/9/9/4g4/4p4/9/9/4R4/4R3K b - 1";
  Expect(e, "5h5e", 0, false);
  Expect(e, "5h5e", -360, true);
  Expect(e, "5h5e", -359, false);

  // F: white to move, mirrored version of B (frame independence).
  const char* f = "k3r4/9/9/9/4P4/9/9/9/4K4 w - 1";
  Expect(f, "5a5e", 90, true);
  Expect(f, "5a5e", 91, false);

  // G: king cannot capture a defended piece (result must be false).
  const char* g = "4k4/9/9/9/9/9/4g4/4p4/4K4 b - 1";
  Expect(g, "5i5h", -10000, false);

  std::printf("test_see: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
