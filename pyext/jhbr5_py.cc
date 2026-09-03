// JHBR5 Python bindings: the engine's feature/bucket mapping, record I/O and
// batch preparation for the trainer. The trainer must not re-implement any
// mapping; everything below calls the same C++ the engine runs.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nnue/features.h"
#include "nnue/move_buckets.h"
#include "nnue/net_format.h"
#include "nnue/record.h"
#include "nnue/types.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

namespace py = pybind11;
using namespace jhbr5;
using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;

namespace {

void EnsureInit() {
  static bool done = false;
  if (!done) {
    lczero::ShogiTables::Init();
    nnue::Init();
    done = true;
  }
}

ShogiBoard BoardFromSfen(const std::string& sfen) {
  ShogiBoard b;
  if (!b.SetFromSfen(sfen)) throw std::invalid_argument("bad sfen: " + sfen);
  return b;
}

template <int N>
py::array_t<int64_t> ToArray(const nnue::FeatureList<N>& f) {
  py::array_t<int64_t> a(f.n);
  auto r = a.mutable_unchecked<1>();
  for (int i = 0; i < f.n; ++i) r(i) = f.idx[i];
  return a;
}

template <typename T>
py::array_t<T> Vec(const std::vector<T>& v) {
  py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

// Streams records from a list of shard files into training batches.
class BatchReader {
 public:
  BatchReader(std::vector<std::string> paths, int batch_size, int shuffle_buffer,
              uint64_t seed, bool require_dist, bool see, bool loop, bool move_fallback)
      : paths_(std::move(paths)),
        batch_size_(batch_size),
        shuffle_buffer_(std::max(shuffle_buffer, batch_size)),
        rng_(seed),
        require_dist_(require_dist),
        see_(see),
        loop_(loop),
        move_fallback_(move_fallback) {
    EnsureInit();
    if (paths_.empty()) throw std::invalid_argument("no shard paths");
  }

  py::object Next() {
    std::vector<data::Record> batch;
    {
      py::gil_scoped_release release;
      batch = Take();
    }
    if (batch.empty()) return py::none();
    return Prepare(batch);
  }

  uint64_t records_read() const { return records_read_; }

 private:
  // Up to kOpenReaders shards are read round-robin so that a batch mixes
  // records from many files (shards are time-ordered chunks of games).
  static constexpr size_t kOpenReaders = 16;

  // Opens the next shard into slot `slot`; false when no shard is left.
  bool OpenNextInto(size_t slot) {
    while (true) {
      if (file_index_ >= paths_.size()) {
        if (!loop_ || paths_.empty()) return false;
        file_index_ = 0;
        std::shuffle(paths_.begin(), paths_.end(), rng_);
        ++epoch_;
      }
      std::string err;
      if (readers_[slot].Open(paths_[file_index_++], &err)) return true;
    }
  }

  bool ReadOne(data::Record* r) {
    if (readers_.empty()) {
      std::shuffle(paths_.begin(), paths_.end(), rng_);
      readers_.resize(std::min(kOpenReaders, paths_.size()));
      live_.assign(readers_.size(), false);
      for (size_t i = 0; i < readers_.size(); ++i) live_[i] = OpenNextInto(i);
    }
    for (size_t tries = 0; tries < 2 * readers_.size() + 2; ++tries) {
      const size_t slot = next_slot_++ % readers_.size();
      if (!live_[slot]) continue;
      if (readers_[slot].Next(r)) {
        ++records_read_;
        // A record without a distribution can still train the policy on its
        // played move (game records); require_dist without fallback skips it.
        if (require_dist_ && r->head.n_dist == 0 && !(move_fallback_ && r->head.move != 0)) {
          --tries;
          continue;
        }
        return true;
      }
      live_[slot] = OpenNextInto(slot);
      if (std::none_of(live_.begin(), live_.end(), [](bool b) { return b; })) return false;
    }
    return false;
  }

  std::vector<data::Record> Take() {
    // Fill the shuffle buffer, then pop random records.
    data::Record r;
    while (buffer_.size() < static_cast<size_t>(shuffle_buffer_) && !exhausted_) {
      if (ReadOne(&r)) {
        buffer_.push_back(std::move(r));
      } else {
        exhausted_ = true;
      }
    }
    std::vector<data::Record> out;
    while (!buffer_.empty() && static_cast<int>(out.size()) < batch_size_) {
      const size_t i = rng_() % buffer_.size();
      out.push_back(std::move(buffer_[i]));
      buffer_[i] = std::move(buffer_.back());
      buffer_.pop_back();
    }
    return out;
  }

  py::dict Prepare(const std::vector<data::Record>& batch) {
    const size_t n = batch.size();
    std::vector<int64_t> a_us, a_them, b, p, a_us_off{0}, a_them_off{0}, b_off{0}, p_off{0};
    std::vector<int64_t> a_us_kp, a_them_kp;
    std::vector<int64_t> mv_bucket, mv_off{0};
    std::vector<float> mv_visits, score(n), result(n);
    std::vector<int32_t> ply(n), n_moves(n);
    std::vector<uint8_t> has_dist(n);
    nnue::FeatureList<nnue::kMaxActiveA> fa;
    nnue::FeatureList<nnue::kMaxActiveB> fb;
    nnue::FeatureList<nnue::kMaxActivePolicy> fp;
    {
      py::gil_scoped_release release;
      ShogiBoard board;
      for (size_t i = 0; i < n; ++i) {
        const data::Record& r = batch[i];
        lczero::PackedSfen packed;
        std::memcpy(packed.data.data(), r.head.sfen, 32);
        if (!board.SetFromPackedSfen(packed, std::max<int>(1, r.head.game_ply))) {
          throw std::runtime_error("record has an undecodable packed sfen");
        }
        const lczero::Color stm = board.side_to_move();
        nnue::GroupAFeatures(board, stm, &fa);
        for (int f = 0; f < fa.n; ++f) {
          a_us.push_back(fa.idx[f]);
          a_us_kp.push_back(nnue::KPrelIndex(fa.idx[f]));
        }
        a_us_off.push_back(static_cast<int64_t>(a_us.size()));
        nnue::GroupAFeatures(board, ~stm, &fa);
        for (int f = 0; f < fa.n; ++f) {
          a_them.push_back(fa.idx[f]);
          a_them_kp.push_back(nnue::KPrelIndex(fa.idx[f]));
        }
        a_them_off.push_back(static_cast<int64_t>(a_them.size()));
        nnue::GroupBFeatures(board, &fb);
        b.insert(b.end(), fb.idx, fb.idx + fb.n);
        b_off.push_back(static_cast<int64_t>(b.size()));
        nnue::PolicyFeatures(board, &fp);
        p.insert(p.end(), fp.idx, fp.idx + fp.n);
        p_off.push_back(static_cast<int64_t>(p.size()));

        score[i] = static_cast<float>(r.head.score);
        result[i] = static_cast<float>(r.head.result);
        ply[i] = r.head.game_ply;
        has_dist[i] = r.head.n_dist > 0 || (move_fallback_ && r.head.move != 0);

        // Policy target: every legal move gets a bucket; visits from the
        // distribution (0 if absent).
        MoveList moves = board.GenerateLegalMoves();
        n_moves[i] = moves.size();
        for (int m = 0; m < moves.size(); ++m) {
          mv_bucket.push_back(see_ ? nnue::MoveBucketSee(board, moves[m])
                                   : nnue::MoveBucket(board, moves[m]));
          float visits = 0.0f;
          if (r.dist.empty()) {
            if (move_fallback_ && r.head.move == moves[m].raw()) visits = 1.0f;
          } else {
            for (const auto& d : r.dist) {
              if (d.move == moves[m].raw()) {
                visits = static_cast<float>(d.visits);
                break;
              }
            }
          }
          mv_visits.push_back(visits);
        }
        mv_off.push_back(static_cast<int64_t>(mv_bucket.size()));
      }
    }
    py::dict d;
    d["n"] = static_cast<int>(n);
    d["a_us_idx"] = Vec(a_us);
    d["a_us_off"] = Vec(a_us_off);
    d["a_them_idx"] = Vec(a_them);
    d["a_them_off"] = Vec(a_them_off);
    d["a_us_kp"] = Vec(a_us_kp);
    d["a_them_kp"] = Vec(a_them_kp);
    d["b_idx"] = Vec(b);
    d["b_off"] = Vec(b_off);
    d["p_idx"] = Vec(p);
    d["p_off"] = Vec(p_off);
    d["score"] = Vec(score);
    d["result"] = Vec(result);
    d["ply"] = Vec(ply);
    d["has_dist"] = Vec(has_dist);
    d["mv_bucket"] = Vec(mv_bucket);
    d["mv_off"] = Vec(mv_off);
    d["mv_visits"] = Vec(mv_visits);
    d["n_moves"] = Vec(n_moves);
    return d;
  }

  std::vector<std::string> paths_;
  int batch_size_;
  int shuffle_buffer_;
  std::mt19937_64 rng_;
  bool require_dist_;
  bool see_;
  bool loop_;
  bool move_fallback_;
  std::vector<data::RecordReader> readers_;
  std::vector<bool> live_;
  size_t next_slot_ = 0;
  size_t file_index_ = 0;
  int epoch_ = 0;
  bool exhausted_ = false;
  std::vector<data::Record> buffer_;
  uint64_t records_read_ = 0;
};

class PyRecordWriter {
 public:
  explicit PyRecordWriter(const std::string& path) {
    EnsureInit();
    std::string err;
    if (!writer_.Open(path, &err)) throw std::runtime_error(err);
  }
  void Write(const std::string& sfen, int score, const std::string& move_usi, int ply,
             int result, const std::vector<std::pair<std::string, int>>& dist, int flags) {
    ShogiBoard board = BoardFromSfen(sfen);
    data::Record r;
    lczero::PackedSfen packed;
    if (!board.ToPackedSfen(&packed)) throw std::runtime_error("cannot pack " + sfen);
    std::memcpy(r.head.sfen, packed.data.data(), 32);
    r.head.score = static_cast<int16_t>(std::clamp(score, -32000, 32000));
    r.head.move = move_usi.empty() ? 0 : Move::Parse(move_usi).raw();
    r.head.game_ply = static_cast<uint16_t>(std::max(ply, 0));
    r.head.result = static_cast<int8_t>(std::clamp(result, -1, 1));
    r.head.flags = static_cast<uint8_t>(flags);
    int max_v = 0;
    for (const auto& [usi, v] : dist) max_v = std::max(max_v, v);
    for (const auto& [usi, v] : dist) {
      if (v <= 0) continue;
      data::DistEntry e;
      e.move = Move::Parse(usi).raw();
      e.visits = static_cast<uint16_t>(std::max<long long>(1, 65535LL * v / max_v));
      r.dist.push_back(e);
    }
    if (!writer_.Write(r)) throw std::runtime_error("write failed");
  }
  void Close() { writer_.Close(); }
  uint64_t written() const { return writer_.written(); }

 private:
  data::RecordWriter writer_;
};

nnue::DType DTypeOf(const py::array& a) {
  const auto dt = a.dtype();
  if (dt.is(py::dtype::of<int8_t>())) return nnue::DType::kI8;
  if (dt.is(py::dtype::of<int16_t>())) return nnue::DType::kI16;
  if (dt.is(py::dtype::of<int32_t>())) return nnue::DType::kI32;
  if (dt.is(py::dtype::of<float>())) return nnue::DType::kF32;
  throw std::invalid_argument("tensor dtype must be int8, int16, int32 or float32");
}

void WriteNet(const std::string& path, int kind, const std::vector<uint32_t>& l1, int l2,
              int l3, int qa, int qb, int q_pst, bool see, const py::dict& tensors) {
  EnsureInit();
  nnue::NetWriter w;
  std::vector<py::array> keep;  // keep contiguous copies alive until written
  for (auto item : tensors) {
    const std::string name = py::str(item.first);
    py::array a = py::array::ensure(item.second, py::array::c_style | py::array::forcecast);
    keep.push_back(a);
  }
  size_t i = 0;
  for (auto item : tensors) {
    const std::string name = py::str(item.first);
    const py::array& a = keep[i++];
    std::vector<uint64_t> shape(a.shape(), a.shape() + a.ndim());
    w.AddTensor(name.c_str(), DTypeOf(a), shape, a.data());
  }
  nnue::NetHeader h{};
  h.kind = static_cast<uint32_t>(kind);
  h.feature_set_id = nnue::FeatureSetId();
  h.bucket_table_id = kind == nnue::kNetKindPolicy ? nnue::BucketTableId(see) : 0;
  for (size_t k = 0; k < l1.size() && k < 4; ++k) h.l1[k] = l1[k];
  h.l2 = static_cast<uint32_t>(l2);
  h.l3 = static_cast<uint32_t>(l3);
  h.qa = static_cast<uint32_t>(qa);
  h.qb = static_cast<uint32_t>(qb);
  h.q_pst = static_cast<uint32_t>(q_pst);
  std::string err;
  if (!w.Write(path, h, &err)) throw std::runtime_error(err);
}

}  // namespace

PYBIND11_MODULE(jhbr5, m) {
  m.doc() = "JHBR5 engine bindings: feature mapping, move buckets, record I/O";
  EnsureInit();

  m.attr("GROUP_A_INPUTS") = nnue::kGroupAInputs;
  m.attr("GROUP_B_INPUTS") = nnue::kGroupBInputs;
  m.attr("POLICY_INPUTS") = nnue::kPolicyInputs;
  m.attr("NUM_SLOTS") = nnue::kNumSlots;
  m.attr("NUM_BUCKETS") = nnue::kNumBuckets;
  m.attr("NUM_BUCKETS_SEE") = nnue::kNumBucketsSee;
  m.attr("KPREL_INPUTS") = nnue::kKPrelInputs;
  m.attr("MAX_ACTIVE_A") = nnue::kMaxActiveA;
  m.attr("MAX_ACTIVE_B") = nnue::kMaxActiveB;
  m.attr("MAX_ACTIVE_POLICY") = nnue::kMaxActivePolicy;
  m.attr("QA") = nnue::kQA;
  m.attr("POLICY_SHIFT") = nnue::kPolicyShift;
  m.attr("POLICY_FACTOR") = nnue::kPolicyFactor;
  m.attr("POLICY_QB") = nnue::kPolicyQB;
  m.attr("Q_PST") = nnue::kQPst;
  m.attr("VALUE_L2") = nnue::kValueL2;
  m.attr("VALUE_L3") = nnue::kValueL3;
  m.attr("NET_KIND_VALUE") = nnue::kNetKindValue;
  m.attr("NET_KIND_POLICY") = nnue::kNetKindPolicy;
  m.attr("MAX_LEGAL_MOVES") = lczero::kMaxLegalMoves;

  m.def("feature_set_id", [] { return nnue::FeatureSetId(); });
  m.def("kprel_index", [](py::array_t<int64_t> idx) {
    auto r = idx.unchecked<1>();
    py::array_t<int64_t> out(r.shape(0));
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < r.shape(0); ++i) o(i) = nnue::KPrelIndex(static_cast<int>(r(i)));
    return out;
  }, "King-relative factoriser index for group-A feature indices (training only)");
  m.def("bucket_table_id", [](bool see) { return nnue::BucketTableId(see); }, py::arg("see") = true);

  m.def("value_features", [](const std::string& sfen) {
    ShogiBoard b = BoardFromSfen(sfen);
    nnue::FeatureList<nnue::kMaxActiveA> a1, a2;
    nnue::FeatureList<nnue::kMaxActiveB> fb;
    nnue::GroupAFeatures(b, b.side_to_move(), &a1);
    nnue::GroupAFeatures(b, ~b.side_to_move(), &a2);
    nnue::GroupBFeatures(b, &fb);
    return py::make_tuple(ToArray(a1), ToArray(a2), ToArray(fb));
  }, "Group A (stm frame), group A (opponent frame) and group B indices for an sfen");
  m.def("policy_features", [](const std::string& sfen) {
    ShogiBoard b = BoardFromSfen(sfen);
    nnue::FeatureList<nnue::kMaxActivePolicy> f;
    nnue::PolicyFeatures(b, &f);
    return ToArray(f);
  });
  m.def("legal_moves", [](const std::string& sfen) {
    ShogiBoard b = BoardFromSfen(sfen);
    MoveList moves = b.GenerateLegalMoves();
    std::vector<std::string> out;
    for (int i = 0; i < moves.size(); ++i) out.push_back(moves[i].ToString());
    return out;
  });
  m.def("move_bucket", [](const std::string& sfen, const std::string& usi, bool see) {
    ShogiBoard b = BoardFromSfen(sfen);
    const Move m = Move::Parse(usi);
    return see ? nnue::MoveBucketSee(b, m) : nnue::MoveBucket(b, m);
  }, py::arg("sfen"), py::arg("usi"), py::arg("see") = true);
  m.def("see_good", [](const std::string& sfen, const std::string& usi) {
    return nnue::MoveSeeGood(BoardFromSfen(sfen), Move::Parse(usi));
  });
  m.def("move16", [](const std::string& usi) { return static_cast<int>(Move::Parse(usi).raw()); });
  m.def("packed_sfen", [](const std::string& sfen) {
    ShogiBoard b = BoardFromSfen(sfen);
    lczero::PackedSfen p;
    if (!b.ToPackedSfen(&p)) throw std::runtime_error("cannot pack");
    return py::bytes(reinterpret_cast<const char*>(p.data.data()), 32);
  });
  m.def("sfen_from_packed", [](const py::bytes& raw, int ply) {
    std::string s = raw;
    if (s.size() != 32) throw std::invalid_argument("packed sfen must be 32 bytes");
    lczero::PackedSfen p;
    std::memcpy(p.data.data(), s.data(), 32);
    ShogiBoard b;
    if (!b.SetFromPackedSfen(p, ply)) throw std::runtime_error("undecodable packed sfen");
    return b.ToSfen();
  }, py::arg("raw"), py::arg("ply") = 1);
  m.def("crc32c", [](const py::bytes& raw) {
    std::string s = raw;
    return nnue::Crc32c(s.data(), s.size());
  });
  m.def("write_net", &WriteNet, py::arg("path"), py::arg("kind"), py::arg("l1"), py::arg("l2"),
        py::arg("l3"), py::arg("qa"), py::arg("qb"), py::arg("q_pst"), py::arg("see"),
        py::arg("tensors"),
        "Write a JHBR5 .nn file (header fields + dict of name -> numpy array)");

  py::class_<BatchReader>(m, "BatchReader")
      .def(py::init<std::vector<std::string>, int, int, uint64_t, bool, bool, bool, bool>(),
           py::arg("paths"), py::arg("batch_size"), py::arg("shuffle_buffer") = 100000,
           py::arg("seed") = 1, py::arg("require_dist") = false, py::arg("see") = true,
           py::arg("loop") = false, py::arg("move_fallback") = false)
      .def("next", &BatchReader::Next, "Next batch as a dict of numpy arrays, or None")
      .def_property_readonly("records_read", &BatchReader::records_read);

  py::class_<PyRecordWriter>(m, "RecordWriter")
      .def(py::init<const std::string&>())
      .def("write", &PyRecordWriter::Write, py::arg("sfen"), py::arg("score"), py::arg("move"),
           py::arg("ply"), py::arg("result"), py::arg("dist") = std::vector<std::pair<std::string, int>>{},
           py::arg("flags") = 0)
      .def("close", &PyRecordWriter::Close)
      .def_property_readonly("written", &PyRecordWriter::written);
}
