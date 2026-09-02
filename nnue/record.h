// JHBR5 training record format (docs/DATA_FORMAT.md is the normative spec).
//
// File: FileHeader (16 B) then records. Record: RecordHead (44 B, whose first
// 40 bytes are exactly a YaneuraOu PackedSfenValue) followed by n_dist
// DistEntry (4 B each): the root visit distribution as (move16, visits).

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace jhbr5::data {

constexpr char kRecordMagic[8] = {'J', 'H', 'B', 'R', '5', 'R', 'C', '\0'};
constexpr uint32_t kRecordVersion = 1;

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t reserved;
};
static_assert(sizeof(FileHeader) == 16);

#pragma pack(push, 1)
struct RecordHead {
  uint8_t sfen[32];   // YaneuraOu packed sfen
  int16_t score;      // search score, centipawns, side to move
  uint16_t move;      // played/best move, JHBR3 Move16 (0 = none)
  uint16_t game_ply;
  int8_t result;      // +1 win, 0 draw, -1 loss, side to move
  uint8_t flags;      // RecordFlags
  uint16_t n_dist;    // number of DistEntry following
  uint16_t reserved;
};
struct DistEntry {
  uint16_t move;    // JHBR3 Move16
  uint16_t visits;  // scaled so that the maximum is 65535
};
#pragma pack(pop)
static_assert(sizeof(RecordHead) == 44);
static_assert(sizeof(DistEntry) == 4);

enum RecordFlags : uint8_t {
  kHasDist = 1,
  kTeacherJhbr3 = 2,
  kSelfPlay = 4,
  kResignAdjudicated = 8,
  kImportedPsv = 16,
};

struct Record {
  RecordHead head{};
  std::vector<DistEntry> dist;
};

class RecordWriter {
 public:
  ~RecordWriter() { Close(); }
  bool Open(const std::string& path, std::string* err);
  bool Write(const Record& r);
  void Close();
  uint64_t written() const { return written_; }

 private:
  std::FILE* f_ = nullptr;
  uint64_t written_ = 0;
};

class RecordReader {
 public:
  ~RecordReader() { Close(); }
  bool Open(const std::string& path, std::string* err);
  // Returns false at end of file or on a truncated record.
  bool Next(Record* r);
  void Close();
  uint64_t read() const { return read_; }

 private:
  std::FILE* f_ = nullptr;
  uint64_t read_ = 0;
};

}  // namespace jhbr5::data
