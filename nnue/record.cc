#include "nnue/record.h"

#include <cstring>

namespace jhbr5::data {

bool RecordWriter::Open(const std::string& path, std::string* err) {
  Close();
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) {
    *err = "cannot create " + path;
    return false;
  }
  FileHeader h{};
  std::memcpy(h.magic, kRecordMagic, 8);
  h.version = kRecordVersion;
  if (std::fwrite(&h, sizeof(h), 1, f_) != 1) {
    *err = "cannot write header to " + path;
    Close();
    return false;
  }
  written_ = 0;
  return true;
}

bool RecordWriter::Write(const Record& r) {
  if (!f_) return false;
  RecordHead head = r.head;
  head.n_dist = static_cast<uint16_t>(r.dist.size());
  if (!r.dist.empty()) head.flags |= kHasDist;
  if (std::fwrite(&head, sizeof(head), 1, f_) != 1) return false;
  if (!r.dist.empty() &&
      std::fwrite(r.dist.data(), sizeof(DistEntry), r.dist.size(), f_) != r.dist.size()) {
    return false;
  }
  ++written_;
  return true;
}

void RecordWriter::Close() {
  if (f_) std::fclose(f_);
  f_ = nullptr;
}

bool RecordReader::Open(const std::string& path, std::string* err) {
  Close();
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) {
    *err = "cannot open " + path;
    return false;
  }
  FileHeader h{};
  if (std::fread(&h, sizeof(h), 1, f_) != 1 || std::memcmp(h.magic, kRecordMagic, 8) != 0) {
    *err = path + ": not a JHBR5 record file";
    Close();
    return false;
  }
  if (h.version != kRecordVersion) {
    *err = path + ": unsupported record version";
    Close();
    return false;
  }
  read_ = 0;
  return true;
}

bool RecordReader::Next(Record* r) {
  if (!f_) return false;
  if (std::fread(&r->head, sizeof(RecordHead), 1, f_) != 1) return false;
  r->dist.resize(r->head.n_dist);
  if (r->head.n_dist &&
      std::fread(r->dist.data(), sizeof(DistEntry), r->head.n_dist, f_) != r->head.n_dist) {
    return false;
  }
  ++read_;
  return true;
}

void RecordReader::Close() {
  if (f_) std::fclose(f_);
  f_ = nullptr;
}

}  // namespace jhbr5::data
