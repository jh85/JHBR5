#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>

namespace jhbr5::nnue {

// 64-byte aligned, zero-initialised heap array.
template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(size_t count) { Allocate(count); }
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& o) noexcept : data_(o.data_), count_(o.count_) {
    o.data_ = nullptr;
    o.count_ = 0;
  }
  AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
    if (this != &o) {
      std::free(data_);
      data_ = o.data_;
      count_ = o.count_;
      o.data_ = nullptr;
      o.count_ = 0;
    }
    return *this;
  }

  void Allocate(size_t count) {
    std::free(data_);
    count_ = count;
    const size_t bytes = ((count * sizeof(T)) + 63) / 64 * 64;
    data_ = static_cast<T*>(std::aligned_alloc(64, bytes == 0 ? 64 : bytes));
    if (!data_) throw std::bad_alloc();
    std::memset(data_, 0, bytes == 0 ? 64 : bytes);
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return count_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

}  // namespace jhbr5::nnue
