// Record file round trip and PSV-compatible layout.

#include <cstdio>
#include <cstring>
#include <string>

#include "nnue/record.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace jhbr5::data;

int main(int argc, char** argv) {
  const std::string path = std::string(argc > 1 ? argv[1] : "/tmp") + "/jhbr5_test.rec";
  lczero::ShogiTables::Init();
  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      std::printf("FAIL %s\n", what);
      ++failures;
    }
  };
  check(sizeof(RecordHead) == 44 && offsetof(RecordHead, score) == 32 &&
            offsetof(RecordHead, move) == 34 && offsetof(RecordHead, game_ply) == 36 &&
            offsetof(RecordHead, result) == 38 && offsetof(RecordHead, n_dist) == 40,
        "PSV-compatible layout");

  lczero::ShogiBoard board;
  board.SetStartPos();
  lczero::PackedSfen packed;
  board.ToPackedSfen(&packed);

  std::string err;
  {
    RecordWriter w;
    check(w.Open(path, &err), "open writer");
    for (int i = 0; i < 100; ++i) {
      Record r;
      std::memcpy(r.head.sfen, packed.data.data(), 32);
      r.head.score = static_cast<int16_t>(i * 7 - 300);
      r.head.move = static_cast<uint16_t>(0x1234 + i);
      r.head.game_ply = static_cast<uint16_t>(i + 1);
      r.head.result = static_cast<int8_t>((i % 3) - 1);
      r.head.flags = kSelfPlay;
      for (int k = 0; k < i % 5; ++k) r.dist.push_back({static_cast<uint16_t>(100 + k), static_cast<uint16_t>(65535 - k)});
      check(w.Write(r), "write record");
    }
  }
  {
    RecordReader rd;
    check(rd.Open(path, &err), "open reader");
    Record r;
    int n = 0;
    while (rd.Next(&r)) {
      check(r.head.score == n * 7 - 300, "score round trip");
      check(r.head.move == 0x1234 + n, "move round trip");
      check(static_cast<int>(r.dist.size()) == n % 5, "dist size");
      check((r.head.flags & kHasDist) != 0 || r.dist.empty(), "has-dist flag");
      if (!r.dist.empty()) check(r.dist[0].visits == 65535, "dist visits");
      lczero::PackedSfen p;
      std::memcpy(p.data.data(), r.head.sfen, 32);
      lczero::ShogiBoard b;
      // The packed sfen does not carry the ply, so compare without the move counter.
      auto strip = [](std::string t) { return t.substr(0, t.rfind(' ')); };
      check(b.SetFromPackedSfen(p, r.head.game_ply) && strip(b.ToSfen()) == strip(board.ToSfen()),
            "sfen round trip");
      ++n;
    }
    check(n == 100, "record count");
  }
  std::remove(path.c_str());
  std::printf("test_record_io: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
