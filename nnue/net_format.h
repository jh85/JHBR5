// JHBR5 NNUE — weight file format (docs/DESIGN.md §6.3).
//
//   NetHeader (256 B) | TensorDesc[n_tensors] (64 B each) | payload (64-B aligned tensors)
//
// Little endian. `payload_crc32c` covers the payload bytes; `header_crc32c`
// covers the header with that field zeroed. Tensors are located by name.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nnue/aligned.h"

namespace jhbr5::nnue {

constexpr char kNetMagic[8] = {'J', 'H', 'B', 'R', '5', 'N', 'N', '\0'};
constexpr uint32_t kNetVersion = 1;
constexpr uint32_t kNetVersion2 = 2;  // v2 architecture (docs/NNUE_V2_DESIGN.md)
constexpr uint32_t kNetKindValue = 1;
constexpr uint32_t kNetKindPolicy = 2;

enum class DType : uint32_t { kI8 = 1, kI16 = 2, kI32 = 3, kF32 = 4 };
size_t DTypeSize(DType t);

struct NetHeader {
  char magic[8];
  uint32_t version;
  uint32_t kind;
  uint32_t feature_set_id;
  uint32_t bucket_table_id;  // policy only; 0 for value
  uint32_t l1[4];            // value: [A, A(shared), B, 0]; policy: [L1, 0, 0, 0]
  uint32_t l2;
  uint32_t l3;
  uint32_t qa;
  uint32_t qb;
  uint32_t q_pst;
  uint32_t n_tensors;
  uint64_t payload_bytes;
  uint32_t payload_crc32c;
  uint32_t header_crc32c;
  uint16_t n_phase;      // v2 value nets: phase buckets (8); 0 for v1 and policy
  uint8_t reserved[174];
};
static_assert(sizeof(NetHeader) == 256, "NetHeader must be 256 bytes");

struct TensorDesc {
  char name[16];
  uint32_t dtype;
  uint32_t rank;
  uint64_t shape[3];
  uint64_t offset;  // from payload start, 64-byte aligned
  uint8_t reserved[8];
};
static_assert(sizeof(TensorDesc) == 64, "TensorDesc must be 64 bytes");

uint32_t Crc32c(const void* data, size_t bytes, uint32_t crc = 0);

class NetFile {
 public:
  bool Load(const std::string& path, std::string* err);

  const NetHeader& header() const { return header_; }

  // Returns the tensor payload or nullptr (with *err set) if the name, dtype
  // or shape do not match. `shape` entries of 0 are wildcards.
  const void* Tensor(const char* name, DType dtype,
                     std::vector<uint64_t> shape, std::string* err) const;

  template <typename T>
  const T* TensorAs(const char* name, DType dtype, std::vector<uint64_t> shape,
                    std::string* err) const {
    return static_cast<const T*>(Tensor(name, dtype, std::move(shape), err));
  }

 private:
  NetHeader header_{};
  std::vector<TensorDesc> descs_;
  AlignedBuffer<uint8_t> payload_;
};

class NetWriter {
 public:
  void AddTensor(const char* name, DType dtype, std::vector<uint64_t> shape,
                 const void* data);
  // Fills n_tensors, payload_bytes and both CRCs; other header fields must be
  // set by the caller. `version` may be left 0 for the v1 default (kNetVersion);
  // v2 writers set header.version = kNetVersion2 (and n_phase) themselves.
  bool Write(const std::string& path, NetHeader header, std::string* err) const;

 private:
  struct Entry {
    TensorDesc desc;
    const void* data;
    size_t bytes;
  };
  std::vector<Entry> entries_;
  uint64_t payload_bytes_ = 0;
};

}  // namespace jhbr5::nnue
