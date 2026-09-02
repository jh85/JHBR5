// Writes deterministic random JHBR5 networks (for tests and benchmarks).
//
//   make_random_net value  <out> [--l1 N] [--seed S]
//   make_random_net policy <out> [--l1 N] [--seed S] [--no-see]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "nnue/random_net.h"
#include "nnue/types.h"
#include "shogi/bitboard.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s value|policy <out> [--l1 N] [--seed S] [--no-see]\n", argv[0]);
    return 2;
  }
  const std::string kind = argv[1];
  const std::string out = argv[2];
  int l1 = kind == "value" ? 1024 : 4096;
  uint64_t seed = 1;
  bool see = true;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--l1") && i + 1 < argc) l1 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--no-see")) see = false;
  }
  lczero::ShogiTables::Init();
  jhbr5::nnue::Init();
  std::string err;
  bool ok = kind == "value"
                ? jhbr5::nnue::WriteRandomValueNet(out, l1, seed, &err)
                : jhbr5::nnue::WriteRandomPolicyNet(out, l1, see, seed, &err);
  if (!ok) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  std::printf("wrote %s (%s, l1=%d, seed=%llu)\n", out.c_str(), kind.c_str(), l1,
              static_cast<unsigned long long>(seed));
  return 0;
}
