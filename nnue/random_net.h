// Deterministic random networks for tests and benchmarks (not for play).

#pragma once

#include <cstdint>
#include <string>

namespace jhbr5::nnue {

bool WriteRandomValueNet(const std::string& path, int l1, uint64_t seed,
                         std::string* err);
bool WriteRandomPolicyNet(const std::string& path, int l1, bool see_doubling,
                          uint64_t seed, std::string* err);

}  // namespace jhbr5::nnue
