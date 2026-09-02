// JHBR5 — lock-free value cache keyed by position hash (Monty-style).
//
// One 64-bit atomic per entry: [key32 | win16 | draw16]. A hit skips the
// value network. Policy is never cached (docs/DESIGN.md §7.5).

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace jhbr5::nnue {

class EvalCache {
 public:
  struct Stats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t occupied = 0;  // entries written at least once since Clear()
    size_t capacity = 0;
    int hashfull() const {
      return capacity ? static_cast<int>(occupied * 1000 / capacity) : 0;
    }
  };

  EvalCache() = default;
  explicit EvalCache(size_t mb) { Resize(mb); }

  void Resize(size_t mb) {
    size_t n = mb * 1024 * 1024 / sizeof(std::atomic<uint64_t>);
    size_t cap = 1;
    while (cap * 2 <= n) cap *= 2;
    if (n == 0) cap = 0;
    capacity_ = cap;
    table_.reset(cap ? new std::atomic<uint64_t>[cap] : nullptr);
    Clear();
  }

  void Clear() {
    for (size_t i = 0; i < capacity_; ++i) table_[i].store(0, std::memory_order_relaxed);
    lookups_.store(0, std::memory_order_relaxed);
    hits_.store(0, std::memory_order_relaxed);
    occupied_.store(0, std::memory_order_relaxed);
  }

  bool Enabled() const { return capacity_ != 0; }

  bool Probe(uint64_t key, float* win, float* draw) {
    if (!capacity_) return false;
    lookups_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t v = table_[key & (capacity_ - 1)].load(std::memory_order_relaxed);
    if (static_cast<uint32_t>(v >> 32) != Tag(key)) return false;
    *win = static_cast<float>((v >> 16) & 0xFFFF) / 65535.0f;
    *draw = static_cast<float>(v & 0xFFFF) / 65535.0f;
    hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void Store(uint64_t key, float win, float draw) {
    if (!capacity_) return;
    const auto q = [](float x) {
      x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
      return static_cast<uint64_t>(x * 65535.0f + 0.5f);
    };
    const uint64_t v = (static_cast<uint64_t>(Tag(key)) << 32) | (q(win) << 16) | q(draw);
    auto& slot = table_[key & (capacity_ - 1)];
    if (slot.load(std::memory_order_relaxed) == 0) occupied_.fetch_add(1, std::memory_order_relaxed);
    slot.store(v, std::memory_order_relaxed);
  }

  Stats GetStats() const {
    Stats s;
    s.lookups = lookups_.load(std::memory_order_relaxed);
    s.hits = hits_.load(std::memory_order_relaxed);
    s.occupied = occupied_.load(std::memory_order_relaxed);
    s.capacity = capacity_;
    return s;
  }

  void ResetStats() {
    lookups_.store(0, std::memory_order_relaxed);
    hits_.store(0, std::memory_order_relaxed);
  }

 private:
  static uint32_t Tag(uint64_t key) {
    const uint32_t t = static_cast<uint32_t>(key >> 32);
    return t == 0 ? 1u : t;  // 0 marks an empty slot
  }

  std::unique_ptr<std::atomic<uint64_t>[]> table_;
  size_t capacity_ = 0;
  std::atomic<uint64_t> lookups_{0}, hits_{0}, occupied_{0};
};

}  // namespace jhbr5::nnue
