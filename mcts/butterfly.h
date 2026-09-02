// Butterfly history table (Monty `ButterflyTable`, re-implemented).
//
// [side][from_id][to] with from_id = from square for board moves and
// 81 + hand kind for drops. Entries are i16 with a gravity update; read as a
// policy-logit bonus at expansion (docs/DESIGN.md §4.4).

#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

#include "shogi/types.h"

namespace dlshogi_mcts {

class ButterflyTable {
 public:
  static constexpr int kFromIds = 81 + 7;

  ButterflyTable() : data_(2 * kFromIds * 81) { Clear(); }

  void Clear() {
    for (auto& e : data_) e.store(0, std::memory_order_relaxed);
  }

  static int Index(lczero::Color side, lczero::Move m) {
    const int from = m.is_drop() ? 81 + (m.drop_piece().idx - 1) : m.from().as_idx();
    return (static_cast<int>(side) * kFromIds + from) * 81 + m.to().as_idx();
  }

  float Bonus(lczero::Color side, lczero::Move m, int divisor) const {
    return static_cast<float>(data_[Index(side, m)].load(std::memory_order_relaxed)) /
           static_cast<float>(divisor < 1 ? 1 : divisor);
  }

  // score: backed-up expected score in [0,1] for `side` after playing m.
  void Update(lczero::Color side, lczero::Move m, float score, int reduction_factor) {
    if (!std::isfinite(score)) return;
    score = score < 0.001f ? 0.001f : (score > 0.999f ? 0.999f : score);
    int cp = static_cast<int>(std::lround(-400.0 * std::log(1.0 / score - 1.0)));
    cp = cp < -32767 ? -32767 : (cp > 32767 ? 32767 : cp);
    const int rf = reduction_factor < 1 ? 1 : reduction_factor;
    auto& cell = data_[Index(side, m)];
    int16_t cur = cell.load(std::memory_order_relaxed);
    while (true) {
      const int adjusted = cp - static_cast<int>(cur) * std::abs(cp) / rf;
      int nv = static_cast<int>(cur) + (adjusted < -32767 ? -32767 : (adjusted > 32767 ? 32767 : adjusted));
      nv = nv < -32767 ? -32767 : (nv > 32767 ? 32767 : nv);
      if (cell.compare_exchange_weak(cur, static_cast<int16_t>(nv), std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }

 private:
  std::vector<std::atomic<int16_t>> data_;
};

}  // namespace dlshogi_mcts
