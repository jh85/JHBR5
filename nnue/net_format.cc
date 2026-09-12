#include "nnue/net_format.h"

#include <cstdio>
#include <cstring>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace jhbr5::nnue {

size_t DTypeSize(DType t) {
  switch (t) {
    case DType::kI8: return 1;
    case DType::kI16: return 2;
    case DType::kI32: return 4;
    case DType::kF32: return 4;
  }
  return 0;
}

namespace {

uint32_t g_crc_table[256];
bool g_crc_table_ready = false;

void InitCrcTable() {
  if (g_crc_table_ready) return;
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
    g_crc_table[i] = c;
  }
  g_crc_table_ready = true;
}

}  // namespace

uint32_t Crc32c(const void* data, size_t bytes, uint32_t crc) {
  const auto* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
#if defined(__SSE4_2__)
  while (bytes >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    crc = static_cast<uint32_t>(_mm_crc32_u64(crc, v));
    p += 8;
    bytes -= 8;
  }
  while (bytes--) crc = _mm_crc32_u8(crc, *p++);
#else
  InitCrcTable();
  while (bytes--) crc = g_crc_table[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
#endif
  return ~crc;
}

bool NetFile::Load(const std::string& path, std::string* err) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    *err = "cannot open " + path;
    return false;
  }
  auto fail = [&](const std::string& why) {
    std::fclose(f);
    *err = path + ": " + why;
    return false;
  };
  if (std::fread(&header_, sizeof(header_), 1, f) != 1) return fail("short header");
  if (std::memcmp(header_.magic, kNetMagic, 8) != 0) return fail("bad magic");
  if (header_.version != kNetVersion && header_.version != kNetVersion2) {
    return fail("unsupported version");
  }
  {
    NetHeader h = header_;
    h.header_crc32c = 0;
    if (Crc32c(&h, sizeof(h)) != header_.header_crc32c) return fail("header CRC mismatch");
  }
  if (header_.n_tensors == 0 || header_.n_tensors > 64) return fail("bad tensor count");
  descs_.resize(header_.n_tensors);
  if (std::fread(descs_.data(), sizeof(TensorDesc), descs_.size(), f) != descs_.size()) {
    return fail("short tensor table");
  }
  payload_.Allocate(header_.payload_bytes);
  if (header_.payload_bytes > 0 &&
      std::fread(payload_.data(), 1, header_.payload_bytes, f) != header_.payload_bytes) {
    return fail("short payload");
  }
  std::fclose(f);
  if (Crc32c(payload_.data(), header_.payload_bytes) != header_.payload_crc32c) {
    *err = path + ": payload CRC mismatch";
    return false;
  }
  for (const TensorDesc& d : descs_) {
    size_t n = DTypeSize(static_cast<DType>(d.dtype));
    if (n == 0 || d.rank == 0 || d.rank > 3) {
      *err = path + ": bad tensor descriptor";
      return false;
    }
    for (uint32_t r = 0; r < d.rank; ++r) n *= d.shape[r];
    if (d.offset % 64 != 0 || d.offset + n > header_.payload_bytes) {
      *err = path + ": tensor out of bounds";
      return false;
    }
  }
  return true;
}

const void* NetFile::Tensor(const char* name, DType dtype,
                            std::vector<uint64_t> shape, std::string* err) const {
  for (const TensorDesc& d : descs_) {
    if (std::strncmp(d.name, name, sizeof(d.name)) != 0) continue;
    if (static_cast<DType>(d.dtype) != dtype) {
      *err = std::string("tensor ") + name + ": dtype mismatch";
      return nullptr;
    }
    if (d.rank != shape.size()) {
      *err = std::string("tensor ") + name + ": rank mismatch";
      return nullptr;
    }
    for (size_t r = 0; r < shape.size(); ++r) {
      if (shape[r] != 0 && shape[r] != d.shape[r]) {
        *err = std::string("tensor ") + name + ": shape mismatch";
        return nullptr;
      }
    }
    return payload_.data() + d.offset;
  }
  *err = std::string("tensor ") + name + ": missing";
  return nullptr;
}

void NetWriter::AddTensor(const char* name, DType dtype,
                          std::vector<uint64_t> shape, const void* data) {
  Entry e{};
  std::strncpy(e.desc.name, name, sizeof(e.desc.name) - 1);
  e.desc.dtype = static_cast<uint32_t>(dtype);
  e.desc.rank = static_cast<uint32_t>(shape.size());
  size_t n = DTypeSize(dtype);
  for (size_t r = 0; r < shape.size() && r < 3; ++r) {
    e.desc.shape[r] = shape[r];
    n *= shape[r];
  }
  e.desc.offset = payload_bytes_;
  e.data = data;
  e.bytes = n;
  payload_bytes_ += (n + 63) / 64 * 64;
  entries_.push_back(e);
}

bool NetWriter::Write(const std::string& path, NetHeader header,
                      std::string* err) const {
  std::memcpy(header.magic, kNetMagic, 8);
  if (header.version == 0) header.version = kNetVersion;
  header.n_tensors = static_cast<uint32_t>(entries_.size());
  header.payload_bytes = payload_bytes_;

  uint32_t crc = 0;
  static const uint8_t zeros[64] = {0};
  for (const Entry& e : entries_) {
    crc = Crc32c(e.data, e.bytes, crc);
    const size_t pad = (e.bytes + 63) / 64 * 64 - e.bytes;
    if (pad) crc = Crc32c(zeros, pad, crc);
  }
  header.payload_crc32c = crc;
  header.header_crc32c = 0;
  header.header_crc32c = Crc32c(&header, sizeof(header));

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    *err = "cannot create " + path;
    return false;
  }
  bool ok = std::fwrite(&header, sizeof(header), 1, f) == 1;
  for (const Entry& e : entries_) ok = ok && std::fwrite(&e.desc, sizeof(e.desc), 1, f) == 1;
  for (const Entry& e : entries_) {
    ok = ok && (e.bytes == 0 || std::fwrite(e.data, 1, e.bytes, f) == e.bytes);
    const size_t pad = (e.bytes + 63) / 64 * 64 - e.bytes;
    if (pad) ok = ok && std::fwrite(zeros, 1, pad, f) == pad;
  }
  ok = std::fclose(f) == 0 && ok;
  if (!ok) *err = "write failed: " + path;
  return ok;
}

}  // namespace jhbr5::nnue
