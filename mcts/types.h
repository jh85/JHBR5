#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace dlshogi_mcts {

constexpr int kVirtualLoss = 1;

template <typename T>
inline void AtomicFetchAdd(std::atomic<T>* obj, T arg) {
  obj->fetch_add(arg, std::memory_order_acq_rel);
}

class Timer {
 public:
  using Clock = std::chrono::steady_clock;

  Timer() { Restart(); }
  void Restart() { start_ = Clock::now(); }
  void Restart(Clock::time_point start) { start_ = start; }
  int ElapsedMs() const {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_).count());
  }
  Clock::time_point start() const { return start_; }

 private:
  Clock::time_point start_;
};

}  // namespace dlshogi_mcts
