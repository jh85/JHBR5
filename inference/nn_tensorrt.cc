/*
  JHBR2 Shogi Engine — Native TensorRT Backend

  Uses TensorRT C++ API directly. Supports multiple execution slots
  per GPU: each slot owns its own IExecutionContext + CUDA stream +
  pinned host buffers + device buffers, allowing N workers to keep
  the GPU busy concurrently (dlshogi-style design).
*/

#ifdef USE_TENSORRT

#include "inference/nn_tensorrt.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "inference/nn_diagnostics.h"
#include "shogi/encoder.h"
#include "shogi/encoder_unpack.h"

#include <cstring>

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// TensorRT logger
// =====================================================================

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

// =====================================================================
// CUDA helpers
// =====================================================================

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "[CUDA] Error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
  } \
} while(0)

namespace {

// Lightweight RAII wrappers for CUDA allocations. They are movable but not
// copyable, and they free the underlying resource in the destructor.

class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  explicit CudaDeviceBuffer(size_t size) { Allocate(size); }
  ~CudaDeviceBuffer() { Free(); }

  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  bool Allocate(size_t size) {
    Free();
    if (size == 0) return true;
    cudaError_t err = cudaMalloc(&data_, size);
    if (err != cudaSuccess) {
      fprintf(stderr, "[CUDA] cudaMalloc failed: %s\n",
              cudaGetErrorString(err));
      data_ = nullptr;
      return false;
    }
    size_ = size;
    return true;
  }

  void Free() {
    if (data_) {
      cudaFree(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void* data() const { return data_; }
  size_t size() const { return size_; }
  bool valid() const { return data_ != nullptr; }
  explicit operator bool() const { return valid(); }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class CudaHostBuffer {
 public:
  CudaHostBuffer() = default;
  explicit CudaHostBuffer(size_t size) { Allocate(size); }
  ~CudaHostBuffer() { Free(); }

  CudaHostBuffer(CudaHostBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  CudaHostBuffer& operator=(CudaHostBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  CudaHostBuffer(const CudaHostBuffer&) = delete;
  CudaHostBuffer& operator=(const CudaHostBuffer&) = delete;

  bool Allocate(size_t size) {
    Free();
    if (size == 0) return true;
    cudaError_t err = cudaMallocHost(&data_, size);
    if (err != cudaSuccess) {
      fprintf(stderr, "[CUDA] cudaMallocHost failed: %s\n",
              cudaGetErrorString(err));
      data_ = nullptr;
      return false;
    }
    size_ = size;
    return true;
  }

  void Free() {
    if (data_) {
      cudaFreeHost(data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void* data() const { return data_; }
  size_t size() const { return size_; }
  bool valid() const { return data_ != nullptr; }
  explicit operator bool() const { return valid(); }

  uint8_t* as_u8() const { return static_cast<uint8_t*>(data_); }
  float* as_float() const { return static_cast<float*>(data_); }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { Destroy(); }

  CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      Destroy();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  bool Create() {
    Destroy();
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      fprintf(stderr, "[CUDA] cudaStreamCreate failed: %s\n",
              cudaGetErrorString(err));
      stream_ = nullptr;
      return false;
    }
    return true;
  }

  void Destroy() {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  cudaStream_t get() const { return stream_; }
  explicit operator bool() const { return stream_ != nullptr; }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace

// =====================================================================
// Per-slot resources: one IExecutionContext + stream + buffers.
// =====================================================================

struct Slot {
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  CudaStream stream;
  CudaDeviceBuffer d_input;       // dlshogi input1 / native unpack target (float)
  CudaDeviceBuffer d_input2;      // dlshogi input2 (float)
  CudaDeviceBuffer d_packed_f1;   // native: packed features1 bits
  CudaDeviceBuffer d_packed_f2;   // native: packed features2 bits
  CudaDeviceBuffer d_policy;
  CudaDeviceBuffer d_wdl;
  CudaDeviceBuffer d_mlh;
  CudaHostBuffer h_input;         // dlshogi input1 (pinned float)
  CudaHostBuffer h_input2;        // dlshogi input2 (pinned float)
  CudaHostBuffer h_packed_f1;     // native: packed features1 bits (pinned)
  CudaHostBuffer h_packed_f2;     // native: packed features2 bits (pinned)
  CudaHostBuffer h_policy;
  CudaHostBuffer h_wdl;
  CudaHostBuffer h_mlh;
  std::atomic_flag in_use = ATOMIC_FLAG_INIT;
  std::atomic<uint64_t> sequence{0};
};

namespace {

class SlotUseGuard {
 public:
  explicit SlotUseGuard(Slot& slot)
      : slot_(slot), acquired_(!slot_.in_use.test_and_set(
                         std::memory_order_acquire)) {}
  ~SlotUseGuard() {
    if (acquired_) slot_.in_use.clear(std::memory_order_release);
  }
  bool acquired() const { return acquired_; }

 private:
  Slot& slot_;
  bool acquired_;
};

std::vector<NNOutput> InvalidResults(int batch_size) {
  std::vector<NNOutput> results(batch_size);
  for (auto& result : results) result.valid = false;
  return results;
}

uint64_t Fnv1a(const void* data, size_t size,
               uint64_t hash = 1469598103934665603ULL) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

// =====================================================================
// Implementation
// =====================================================================

struct NNEvaluator::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::vector<std::unique_ptr<Slot>> slots;

  ModelFormat requested_format = ModelFormat::kAuto;
  ModelFormat model_format = ModelFormat::kJHBR2;

  int input_idx = -1;
  int input2_idx = -1;
  int policy_idx = -1;
  int wdl_idx = -1;
  int mlh_idx = -1;

  int input_channels = 48;      // overwritten from the engine at load
  int input2_channels = 0;
  int max_batch_size = 32;
  int policy_size = 0;
  bool dynamic_batch = false;

  int device_id = 0;
};

namespace {

const char* ModelFormatName(ModelFormat format) {
  switch (format) {
    case ModelFormat::kAuto:
      return "auto";
    case ModelFormat::kJHBR2:
      return "jhbr2";
    case ModelFormat::kDlshogi:
      return "dlshogi";
  }
  return "unknown";
}

int FindTensorIndex(nvinfer1::ICudaEngine* engine, const char* target) {
  const int nb = engine->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    if (name && std::string(name) == target) return i;
  }
  return -1;
}

int GetMaxBatch(nvinfer1::ICudaEngine* engine, const char* input_name,
                bool* dynamic_batch) {
  auto dims = engine->getTensorShape(input_name);
  if (dims.nbDims >= 1 && dims.d[0] > 0) {
    *dynamic_batch = false;
    return dims.d[0];
  }

  *dynamic_batch = true;
  int max_batch = 32;
  if (engine->getNbOptimizationProfiles() > 0) {
    auto max_dims = engine->getProfileShape(
        input_name, 0, nvinfer1::OptProfileSelector::kMAX);
    if (max_dims.nbDims >= 1 && max_dims.d[0] > 0) {
      max_batch = max_dims.d[0];
    }
  }
  return max_batch;
}

}  // namespace


NNEvaluator::NNEvaluator(const std::string& engine_path, bool /*use_gpu*/,
                         int device_id, int num_slots,
                         ModelFormat model_format)
    : impl_(std::make_unique<Impl>()) {

  CUDA_CHECK(cudaSetDevice(device_id));
  impl_->device_id = device_id;
  impl_->requested_format = model_format;
  ShogiEncoderTables::Init();

  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    fprintf(stderr, "[TRT] Cannot open engine file: %s\n", engine_path.c_str());
    return;
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(file_size);
  file.read(engine_data.data(), file_size);

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size()));

  if (!impl_->engine) {
    fprintf(stderr, "[TRT] Failed to deserialize engine\n");
    return;
  }

  impl_->input_idx = FindTensorIndex(impl_->engine.get(), "input_planes");
  impl_->policy_idx = FindTensorIndex(impl_->engine.get(), "policy");
  impl_->wdl_idx = FindTensorIndex(impl_->engine.get(), "wdl");
  impl_->mlh_idx = FindTensorIndex(impl_->engine.get(), "mlh");
  impl_->input2_idx = FindTensorIndex(impl_->engine.get(), "input2");
  const int dlshogi_input1_idx = FindTensorIndex(impl_->engine.get(), "input1");
  const int dlshogi_policy_idx =
      FindTensorIndex(impl_->engine.get(), "output_policy");
  const int dlshogi_value_idx =
      FindTensorIndex(impl_->engine.get(), "output_value");

  const bool has_jhbr2 = impl_->input_idx >= 0 && impl_->policy_idx >= 0 &&
                         impl_->wdl_idx >= 0;
  const bool has_dlshogi = dlshogi_input1_idx >= 0 && impl_->input2_idx >= 0 &&
                           dlshogi_policy_idx >= 0 && dlshogi_value_idx >= 0;

  if (model_format == ModelFormat::kAuto) {
    impl_->model_format = has_dlshogi ? ModelFormat::kDlshogi
                                      : ModelFormat::kJHBR2;
  } else {
    impl_->model_format = model_format;
  }

  if (impl_->model_format == ModelFormat::kDlshogi) {
    impl_->input_idx = dlshogi_input1_idx;
    impl_->policy_idx = dlshogi_policy_idx;
    impl_->wdl_idx = dlshogi_value_idx;
    impl_->mlh_idx = -1;
    if (!has_dlshogi) {
      fprintf(stderr, "[TRT] Missing dlshogi tensor names "
              "(input1,input2,output_policy,output_value)\n");
      return;
    }
  } else if (!has_jhbr2) {
    fprintf(stderr, "[TRT] Missing JHBR2 tensor names "
            "(input_planes,policy,wdl)\n");
    return;
  }

  if (impl_->model_format == ModelFormat::kDlshogi) {
    auto input1_dims = impl_->engine->getTensorShape("input1");
    auto input2_dims = impl_->engine->getTensorShape("input2");
    if (input1_dims.nbDims >= 2) impl_->input_channels = input1_dims.d[1];
    if (input2_dims.nbDims >= 2) impl_->input2_channels = input2_dims.d[1];
    impl_->max_batch_size =
        GetMaxBatch(impl_->engine.get(), "input1", &impl_->dynamic_batch);
    auto policy_dims = impl_->engine->getTensorShape("output_policy");
    impl_->policy_size = (policy_dims.nbDims >= 2) ? policy_dims.d[1] : 2187;
  } else {
    auto input_dims = impl_->engine->getTensorShape("input_planes");
    if (input_dims.nbDims >= 2) {
      impl_->input_channels = input_dims.d[1];
    }
    if (impl_->input_channels != kShogiInputPlanes) {
      fprintf(stderr,
              "[TRT] WARNING: engine expects %d input channels, but the encoder "
              "produces %d (kShogiInputPlanes). Rebuild the engine from a model "
              "trained with the current encoder, or the unpack will mismatch.\n",
              impl_->input_channels, kShogiInputPlanes);
    }
    impl_->max_batch_size =
        GetMaxBatch(impl_->engine.get(), "input_planes", &impl_->dynamic_batch);
    auto policy_dims = impl_->engine->getTensorShape("policy");
    impl_->policy_size = (policy_dims.nbDims >= 2) ? policy_dims.d[1] : 2187;
  }

  if (num_slots < 1) num_slots = 1;
  fprintf(stderr,
          "[TRT] Engine loaded gpu=%d: format=%s channels=%d/%d "
          "max_batch=%d policy=%d dynamic=%s slots=%d\n",
          device_id, ModelFormatName(impl_->model_format),
          impl_->input_channels, impl_->input2_channels, impl_->max_batch_size,
          impl_->policy_size, impl_->dynamic_batch ? "yes" : "no", num_slots);

  // Allocate per-slot resources: one execution context + stream + buffers.
  impl_->slots.reserve(num_slots);
  const int B = impl_->max_batch_size;
  const int C = impl_->input_channels;
  const int C2 = impl_->input2_channels;
  const int P = impl_->policy_size;
  const int value_planes =
      (impl_->model_format == ModelFormat::kDlshogi) ? 1 : 3;
  for (int s = 0; s < num_slots; s++) {
    auto slot = std::make_unique<Slot>();
    slot->context.reset(impl_->engine->createExecutionContext());
    if (!slot->context) {
      fprintf(stderr, "[TRT] Failed to create execution context (slot=%d)\n", s);
      continue;
    }
    slot->stream.Create();
    slot->d_input.Allocate(static_cast<size_t>(B) * C * 81 * sizeof(float));
    if (impl_->model_format == ModelFormat::kDlshogi) {
      slot->d_input2.Allocate(static_cast<size_t>(B) * C2 * 81 * sizeof(float));
    } else {
      slot->d_packed_f1.Allocate(static_cast<size_t>(B) * kPackedF1Bytes);
      slot->d_packed_f2.Allocate(static_cast<size_t>(B) * kPackedF2Bytes);
    }
    slot->d_policy.Allocate(static_cast<size_t>(B) * P * sizeof(float));
    slot->d_wdl.Allocate(static_cast<size_t>(B) * value_planes * sizeof(float));
    if (impl_->mlh_idx >= 0) {
      slot->d_mlh.Allocate(static_cast<size_t>(B) * sizeof(float));
    }
    if (impl_->model_format == ModelFormat::kDlshogi) {
      slot->h_input.Allocate(static_cast<size_t>(B) * C * 81 * sizeof(float));
      slot->h_input2.Allocate(static_cast<size_t>(B) * C2 * 81 * sizeof(float));
    } else {
      slot->h_packed_f1.Allocate(static_cast<size_t>(B) * kPackedF1Bytes);
      slot->h_packed_f2.Allocate(static_cast<size_t>(B) * kPackedF2Bytes);
    }
    slot->h_policy.Allocate(static_cast<size_t>(B) * P * sizeof(float));
    slot->h_wdl.Allocate(static_cast<size_t>(B) * value_planes * sizeof(float));
    if (impl_->mlh_idx >= 0) {
      slot->h_mlh.Allocate(static_cast<size_t>(B) * sizeof(float));
    }
    impl_->slots.push_back(std::move(slot));
  }
}

NNEvaluator::~NNEvaluator() {
  if (impl_) cudaSetDevice(impl_->device_id);
}

int NNEvaluator::num_slots() const {
  return impl_ ? static_cast<int>(impl_->slots.size()) : 0;
}

bool NNEvaluator::has_moves_left() const {
  return impl_ && impl_->mlh_idx >= 0;
}

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  std::vector<std::pair<ShogiBoard, MoveList>> batch;
  batch.emplace_back(board, legal_moves);
  return EvaluateBatchSlot(0, batch)[0];
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  return EvaluateBatchSlot(0, batch);
}

std::vector<NNOutput> NNEvaluator::EvaluateBatchSlot(
    int slot_id,
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  const int batch_size = static_cast<int>(batch.size());
  if (batch_size == 0) return {};

  const int C = impl_->input_channels;
  const int C2 = impl_->input2_channels;
  const int P = impl_->policy_size;
  const bool is_dlshogi = impl_->model_format == ModelFormat::kDlshogi;
  const int value_planes = is_dlshogi ? 1 : 3;
  constexpr int sq = 81;

  if (!impl_->engine || impl_->slots.empty()) {
    nn_diagnostics::LogOnce(
        "engine_unavailable",
        "backend=tensorrt gpu=" + std::to_string(impl_->device_id) +
            " batch=" + std::to_string(batch_size));
    return InvalidResults(batch_size);
  }

  if (slot_id < 0 || slot_id >= static_cast<int>(impl_->slots.size())) {
    nn_diagnostics::LogOnce(
        "slot_id",
        "backend=tensorrt gpu=" + std::to_string(impl_->device_id) +
            " slot=" + std::to_string(slot_id) +
            " slots=" + std::to_string(impl_->slots.size()) +
            " batch=" + std::to_string(batch_size));
    return InvalidResults(batch_size);
  }

  // Buffers are allocated at the engine profile's maximum batch size, so
  // split oversized requests before preparing inputs or setting shapes.
  if (batch_size > impl_->max_batch_size) {
    std::vector<NNOutput> results;
    results.reserve(batch_size);
    for (int start = 0; start < batch_size;
         start += impl_->max_batch_size) {
      int end = std::min(start + impl_->max_batch_size, batch_size);
      std::vector<std::pair<ShogiBoard, MoveList>> chunk(
          batch.begin() + start, batch.begin() + end);
      auto chunk_results = EvaluateBatchSlot(slot_id, chunk);
      results.insert(results.end(), chunk_results.begin(), chunk_results.end());
    }
    return results;
  }

  Slot& slot = *impl_->slots[slot_id];
  SlotUseGuard slot_guard(slot);
  const uint64_t sequence =
      slot.sequence.fetch_add(1, std::memory_order_relaxed) + 1;
  if (!slot_guard.acquired()) {
    std::ostringstream details;
    details << "backend=tensorrt reason=concurrent_slot_use"
            << " gpu=" << impl_->device_id << " slot=" << slot_id
            << " sequence=" << sequence << " batch=" << batch_size
            << " thread=" << std::hash<std::thread::id>{}(
                                   std::this_thread::get_id());
    nn_diagnostics::LogOnce("slot_ownership", details.str());
    return InvalidResults(batch_size);
  }

  // For dynamic engines use the actual batch size. Static engines require
  // padding to their fixed batch size.
  const int run_batch =
      impl_->dynamic_batch ? batch_size : impl_->max_batch_size;

  bool input_prepared = false;
  bool outputs_ready = false;
  auto log_failure = [&](const char* stage, const std::string& reason,
                         int element) {
    std::ostringstream details;
    details << std::setprecision(9)
            << "backend=tensorrt reason=\"" << reason << "\""
            << " gpu=" << impl_->device_id << " slot=" << slot_id
            << " sequence=" << sequence << " batch=" << batch_size
            << " run_batch=" << run_batch
            << " thread=" << std::hash<std::thread::id>{}(
                                   std::this_thread::get_id());
    if (element >= 0 && element < batch_size) {
      const auto& board = batch[element].first;
      details << " element=" << element << " position_hash=0x" << std::hex
              << board.Hash() << std::dec
              << " legal_moves=" << batch[element].second.size()
              << " sfen=\"" << board.ToSfen() << "\"";
      if (input_prepared) {
        uint64_t input_hash = 1469598103934665603ULL;
        if (is_dlshogi) {
          input_hash = Fnv1a(
              slot.h_input.as_float() + static_cast<size_t>(element) * C * sq,
              static_cast<size_t>(C) * sq * sizeof(float), input_hash);
          input_hash = Fnv1a(
              slot.h_input2.as_float() + static_cast<size_t>(element) * C2 * sq,
              static_cast<size_t>(C2) * sq * sizeof(float), input_hash);
        } else {
          input_hash = Fnv1a(
              slot.h_packed_f1.as_u8() +
                  static_cast<size_t>(element) * kPackedF1Bytes,
              kPackedF1Bytes, input_hash);
          input_hash = Fnv1a(
              slot.h_packed_f2.as_u8() +
                  static_cast<size_t>(element) * kPackedF2Bytes,
              kPackedF2Bytes, input_hash);
        }
        details << " input_fnv1a=0x" << std::hex << input_hash << std::dec;
      }
      if (outputs_ready) {
        details << " raw_wdl=";
        for (int i = 0; i < value_planes; ++i) {
          if (i != 0) details << ',';
          details << slot.h_wdl.as_float()[element * value_planes + i];
        }
        if (impl_->mlh_idx >= 0) {
          details << " raw_mlh=" << slot.h_mlh.as_float()[element];
        }
        const float* policy = slot.h_policy.as_float() + static_cast<size_t>(element) * P;
        int policy_nonfinite = 0;
        float policy_min = std::numeric_limits<float>::infinity();
        float policy_max = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < P; ++i) {
          if (!std::isfinite(policy[i])) {
            ++policy_nonfinite;
          } else {
            policy_min = std::min(policy_min, policy[i]);
            policy_max = std::max(policy_max, policy[i]);
          }
        }
        details << " policy_nonfinite=" << policy_nonfinite
                << " policy_min=" << policy_min
                << " policy_max=" << policy_max;
      }
    }
    nn_diagnostics::LogOnce(stage, details.str());
  };

  // Multi-GPU contexts: ensure CUDA calls in this thread go to our device.
  cudaError_t cuda_error = cudaSetDevice(impl_->device_id);
  if (cuda_error != cudaSuccess) {
    log_failure("cuda_set_device", cudaGetErrorString(cuda_error), 0);
    return InvalidResults(batch_size);
  }

  if (is_dlshogi) {
    std::fill(slot.h_input.as_float(),
              slot.h_input.as_float() + static_cast<size_t>(run_batch) * C * sq, 0.0f);
    std::fill(slot.h_input2.as_float(),
              slot.h_input2.as_float() + static_cast<size_t>(run_batch) * C2 * sq, 0.0f);

    for (int b = 0; b < batch_size; b++) {
      EncodeDlshogiPosition(batch[b].first,
                            slot.h_input.as_float() + static_cast<size_t>(b) * C * sq,
                            slot.h_input2.as_float() + static_cast<size_t>(b) * C2 * sq);
    }
    input_prepared = true;

    nvinfer1::Dims4 input1_dims{run_batch, C, 9, 9};
    nvinfer1::Dims4 input2_dims{run_batch, C2, 9, 9};
    const bool shapes_ok = slot.context->setInputShape("input1", input1_dims) &&
                           slot.context->setInputShape("input2", input2_dims);
    const bool addresses_ok =
        slot.context->setTensorAddress("input1", slot.d_input.data()) &&
        slot.context->setTensorAddress("input2", slot.d_input2.data()) &&
        slot.context->setTensorAddress("output_policy", slot.d_policy.data()) &&
        slot.context->setTensorAddress("output_value", slot.d_wdl.data());
    if (!shapes_ok || !addresses_ok) {
      log_failure("tensor_binding", "setInputShape/setTensorAddress failed", 0);
      return InvalidResults(batch_size);
    }

    cuda_error = cudaMemcpyAsync(slot.d_input.data(), slot.h_input.as_float(),
        static_cast<size_t>(run_batch) * C * sq * sizeof(float),
        cudaMemcpyHostToDevice, slot.stream.get());
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_h2d_input1", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
    cuda_error = cudaMemcpyAsync(slot.d_input2.data(), slot.h_input2.as_float(),
        static_cast<size_t>(run_batch) * C2 * sq * sizeof(float),
        cudaMemcpyHostToDevice, slot.stream.get());
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_h2d_input2", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
  } else {
    // Native JHBR2: pack each position into the small bit buffers (~300 B/pos
    // vs ~48 KB of float planes), transfer, and expand on the GPU into d_input.
    // Zero first: padding rows (run_batch > batch_size) must read as empty, and
    // packing ORs bits in.
    std::memset(slot.h_packed_f1.as_u8(), 0,
                static_cast<size_t>(run_batch) * kPackedF1Bytes);
    std::memset(slot.h_packed_f2.as_u8(), 0,
                static_cast<size_t>(run_batch) * kPackedF2Bytes);
    for (int b = 0; b < batch_size; b++) {
      PackShogiPosition(batch[b].first,
                        slot.h_packed_f1.as_u8() + static_cast<size_t>(b) * kPackedF1Bytes,
                        slot.h_packed_f2.as_u8() + static_cast<size_t>(b) * kPackedF2Bytes);
    }
    input_prepared = true;

    nvinfer1::Dims4 input_dims{run_batch, C, 9, 9};
    bool bindings_ok = slot.context->setInputShape("input_planes", input_dims);
    bindings_ok = slot.context->setTensorAddress("input_planes", slot.d_input.data()) &&
                  bindings_ok;
    bindings_ok = slot.context->setTensorAddress("policy", slot.d_policy.data()) &&
                  bindings_ok;
    bindings_ok = slot.context->setTensorAddress("wdl", slot.d_wdl.data()) &&
                  bindings_ok;
    if (impl_->mlh_idx >= 0) {
      bindings_ok = slot.context->setTensorAddress("mlh", slot.d_mlh.data()) &&
                    bindings_ok;
    }
    if (!bindings_ok) {
      log_failure("tensor_binding", "setInputShape/setTensorAddress failed", 0);
      return InvalidResults(batch_size);
    }

    cuda_error = cudaMemcpyAsync(slot.d_packed_f1.data(), slot.h_packed_f1.as_u8(),
        static_cast<size_t>(run_batch) * kPackedF1Bytes,
        cudaMemcpyHostToDevice, slot.stream.get());
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_h2d_packed_f1", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
    cuda_error = cudaMemcpyAsync(slot.d_packed_f2.data(), slot.h_packed_f2.as_u8(),
        static_cast<size_t>(run_batch) * kPackedF2Bytes,
        cudaMemcpyHostToDevice, slot.stream.get());
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_h2d_packed_f2", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
    LaunchUnpackFeatures(run_batch, C,
        kShogiNumF1Planes, kPackedF1Bytes,
        kShogiNumF2Planes, kPackedF2Bytes,
        static_cast<const uint8_t*>(slot.d_packed_f1.data()),
        static_cast<const uint8_t*>(slot.d_packed_f2.data()),
        static_cast<float*>(slot.d_input.data()), slot.stream.get());
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_unpack_launch", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
  }

  if (!slot.context->enqueueV3(slot.stream.get())) {
    log_failure("tensorrt_enqueue", "enqueueV3 returned false", 0);
    return InvalidResults(batch_size);
  }

  cuda_error = cudaMemcpyAsync(slot.h_policy.as_float(), slot.d_policy.data(),
      static_cast<size_t>(run_batch) * P * sizeof(float),
      cudaMemcpyDeviceToHost, slot.stream.get());
  if (cuda_error != cudaSuccess) {
    log_failure("cuda_d2h_policy", cudaGetErrorString(cuda_error), 0);
    return InvalidResults(batch_size);
  }
  cuda_error = cudaMemcpyAsync(slot.h_wdl.as_float(), slot.d_wdl.data(),
      static_cast<size_t>(run_batch) * value_planes * sizeof(float),
      cudaMemcpyDeviceToHost, slot.stream.get());
  if (cuda_error != cudaSuccess) {
    log_failure("cuda_d2h_wdl", cudaGetErrorString(cuda_error), 0);
    return InvalidResults(batch_size);
  }
  if (impl_->mlh_idx >= 0) {
    cuda_error = cudaMemcpyAsync(slot.h_mlh.as_float(), slot.d_mlh.data(),
        static_cast<size_t>(run_batch) * 1 * sizeof(float),
        cudaMemcpyDeviceToHost, slot.stream.get());
    if (cuda_error != cudaSuccess) {
      log_failure("cuda_d2h_mlh", cudaGetErrorString(cuda_error), 0);
      return InvalidResults(batch_size);
    }
  }

  cuda_error = cudaStreamSynchronize(slot.stream.get());
  if (cuda_error != cudaSuccess) {
    log_failure("cuda_stream_sync", cudaGetErrorString(cuda_error), 0);
    return InvalidResults(batch_size);
  }
  outputs_ready = true;

  std::vector<NNOutput> results(batch_size);
  for (int b = 0; b < batch_size; b++) {
    auto& result = results[b];
    const auto& board = batch[b].first;
    const auto& legal_moves = batch[b].second;

    result.moves_left = (impl_->mlh_idx >= 0) ? slot.h_mlh.as_float()[b] : 0.0f;
    if (!std::isfinite(result.moves_left)) {
      log_failure("raw_output", "non-finite moves-left output", b);
      return InvalidResults(batch_size);
    }

    if (is_dlshogi) {
      float value_win = slot.h_wdl.as_float()[b];
      if (!std::isfinite(value_win)) {
        log_failure("raw_output", "non-finite value output", b);
        return InvalidResults(batch_size);
      }
      value_win = std::clamp(value_win, 0.0f, 1.0f);
      result.wdl[0] = value_win;
      result.wdl[1] = 0.0f;
      result.wdl[2] = 1.0f - value_win;
      result.value = value_win * 2.0f - 1.0f;
      result.draw = 0.0f;
    } else {
      float wdl[3];
      std::copy(slot.h_wdl.as_float() + b * 3, slot.h_wdl.as_float() + b * 3 + 3, wdl);
      if (!SoftmaxInPlace(wdl, 3)) {
        log_failure("raw_output", "non-finite or invalid WDL logits", b);
        return InvalidResults(batch_size);
      }
      result.wdl[0] = wdl[0];
      result.wdl[1] = wdl[1];
      result.wdl[2] = wdl[2];
      result.value = wdl[0] - wdl[2];
      result.draw = wdl[1];
    }

    float* logits = slot.h_policy.as_float() + b * P;
    bool policy_ok;
    if (is_dlshogi) {
      policy_ok = LegalPolicySoftmax(
          logits, P, legal_moves,
          [&board](Move move) {
            return DlshogiMoveToNNIndex(move, board.side_to_move());
          },
          &result.policy);
    } else {
      policy_ok = LegalPolicySoftmax(logits, P, legal_moves,
                                     board.side_to_move(), &result.policy);
    }
    if (!policy_ok) {
      log_failure("policy_softmax", "invalid policy normalization", b);
      return InvalidResults(batch_size);
    }

    const float wdl_sum = result.wdl[0] + result.wdl[1] + result.wdl[2];
    if (!std::isfinite(result.value) || !std::isfinite(result.draw) ||
        !std::isfinite(wdl_sum) || std::abs(wdl_sum - 1.0f) > 1.0e-3f) {
      log_failure("postprocess", "invalid processed value/WDL", b);
      return InvalidResults(batch_size);
    }
  }

  return results;
}

}  // namespace jhbr2

#endif  // USE_TENSORRT
