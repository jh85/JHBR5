/*
  JHBR3 Shogi Engine — USI Protocol Implementation
  Uses the dlshogi-style MCTS search.
*/

#include "usi/usi_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "book/book_selection.h"
#include "mate/bns.h"
#include "mate/dfpn.h"
#include "shogi/encoder.h"
#include "usi/root_mate_state.h"
#include "usi/search_info.h"
#include "usi/time_manager.h"

namespace jhbr2 {

using namespace lczero;

namespace {

// TensorRT creates one execution context and one set of buffers per worker.
// Production has successfully used 16 workers/GPU on RTX 5090. Keep a
// generous USI safety ceiling so higher-end hardware can be benchmarked,
// while still preventing an accidental unbounded allocation.
constexpr int kMaxWorkersPerGpu = 64;

// Search-wide counters are signed 32-bit integers. One billion leaves ample
// overflow headroom while making MaxNodes an effectively non-binding safety
// ceiling for normal clock-based play.
constexpr int kMaxMctsNodes = 1'000'000'000;

// BNS uses fixed-size tables, so its playing-strength budget follows the MCTS
// lifetime instead of an independent node tier. The legacy tree df-pn cap is
// defined in usi/root_mate_state.h because it is part of the root mate solver
// configuration.
constexpr int kRootMateDeadlineMarginMs = 50;

const char* RepetitionResultName(ShogiBoard::RepetitionResult result) {
  switch (result) {
    case ShogiBoard::RepetitionResult::kNone:
      return "none";
    case ShogiBoard::RepetitionResult::kDraw:
      return "draw";
    case ShogiBoard::RepetitionResult::kWin:
      return "opponent-win";
    case ShogiBoard::RepetitionResult::kLoss:
      return "opponent-loss";
  }
  return "unknown";
}

ShogiBoard::RepetitionResult RootMoveRepetitionResult(
    const ShogiBoard& board, Move move) {
  ShogiBoard child = board;
  child.DoMove(move);
  return child.CheckRepetition(MateDfpnSolver::kRepetitionLookbackPly);
}

}  // namespace

// =====================================================================
// Helpers
// =====================================================================

static std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream iss(s);
  std::string token;
  while (iss >> token) parts.push_back(token);
  return parts;
}

static std::string ToLower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return r;
}

static int ParseInt(const std::string& value) {
  std::size_t consumed = 0;
  const long long parsed = std::stoll(value, &consumed);
  if (consumed != value.size() ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("expected an integer");
  }
  return static_cast<int>(parsed);
}

static std::size_t ParseSize(const std::string& value) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("expected a non-negative integer");
  }
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("expected a non-negative integer");
  }
  return static_cast<std::size_t>(parsed);
}

static float ParseFiniteFloat(const std::string& value) {
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument("expected a finite number");
  }
  return parsed;
}

static ModelFormat ParseModelFormat(const std::string& s) {
  std::string v = ToLower(s);
  if (v == "dlshogi" || v == "dlshogimodel") return ModelFormat::kDlshogi;
  if (v == "jhbr2" || v == "default") return ModelFormat::kJHBR2;
  return ModelFormat::kAuto;
}

static TimeManagementMode ParseTimeManagementMode(const std::string& s) {
  const std::string value = ToLower(s);
  if (value == "off") return TimeManagementMode::kOff;
  if (value == "shadow") return TimeManagementMode::kShadow;
  if (value == "on") return TimeManagementMode::kOn;
  throw std::invalid_argument("expected off, shadow, or on");
}

static std::string ModelFormatToString(ModelFormat format) {
  switch (format) {
    case ModelFormat::kAuto:
      return "auto";
    case ModelFormat::kJHBR2:
      return "jhbr2";
    case ModelFormat::kDlshogi:
      return "dlshogi";
  }
  return "auto";
}

static std::string FormatNNCacheStats(const NNCacheStats& stats) {
  const double hit_rate = stats.lookups == 0
                              ? 0.0
                              : 100.0 * static_cast<double>(stats.hits) /
                                    static_cast<double>(stats.lookups);
  const double reuse_rate =
      stats.lookups == 0
          ? 0.0
          : 100.0 * static_cast<double>(stats.hits + stats.in_flight_waits) /
                static_cast<double>(stats.lookups);
  std::ostringstream out;
  out << "nncache size " << stats.size << "/" << stats.capacity
      << " probes " << stats.lookups << " hits " << stats.hits
      << " hitrate " << std::fixed << std::setprecision(1) << hit_rate << "%"
      << " reuse_rate " << reuse_rate << "%"
      << " inserts " << stats.inserts
      << " duplicate_inserts " << stats.duplicate_inserts
      << " evictions " << stats.evictions
      << " in_flight_owners " << stats.in_flight_owners
      << " in_flight_waits " << stats.in_flight_waits
      << " lock_contentions " << stats.lock_contentions
      << " lock_wait_us " << stats.lock_wait_ns / 1000;
  return out.str();
}

// USI hashfull is expressed in permill. JHBR3 does not have a fixed-size
// MCTS node arena like dlshogi, so its bounded NN position cache is the only
// meaningful hash-table occupancy to report.
static int NNCacheHashfull(const NNCacheStats& stats) {
  if (stats.capacity == 0) return 0;
  const size_t used = std::min(stats.size, stats.capacity);
  return static_cast<int>(used * 1000 / stats.capacity);
}

// =====================================================================
// Constructor
// =====================================================================

USIEngine::USIEngine() {
  board_.SetStartPos();
  position_start_key_ = board_.Hash();
  RegisterOptionParsers();
}

// =====================================================================
// Main loop
// =====================================================================

void USIEngine::Run() {
  std::string line;
  while (std::getline(std::cin, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (line.empty()) continue;

    auto parts = Split(line);
    if (parts.empty()) continue;

    const auto& cmd = parts[0];

    if (cmd == "usi")         CmdUsi();
    else if (cmd == "isready")    CmdIsReady();
    else if (cmd == "setoption")  CmdSetOption(parts);
    else if (cmd == "usinewgame") CmdUsiNewGame();
    else if (cmd == "position")   CmdPosition(parts);
    else if (cmd == "go")         CmdGo(parts);
    else if (cmd == "stop")       CmdStop();
    else if (cmd == "quit")       break;
    else if (cmd == "gameover")   CmdGameOver(parts);
    else if (cmd == "d")          CmdDebug();
  }
}

// =====================================================================
// USI command handlers
// =====================================================================

void USIEngine::Send(const std::string& msg) {
  std::cout << msg << std::endl;
}

void USIEngine::Log(const std::string& msg) {
  std::cout << "info string " << msg << std::endl;
}

void USIEngine::EnsureSearch() {
  if (search_ || evaluators_.empty()) return;

  std::vector<jhbr2::NNEvaluator*> eval_ptrs;
  eval_ptrs.reserve(evaluators_.size());
  for (auto& evaluator : evaluators_) eval_ptrs.push_back(evaluator.get());
  search_ =
      std::make_unique<dlshogi_mcts::Search>(eval_ptrs, search_config_);
}

void USIEngine::CmdUsi() {
  Send(std::string("id name ") + ENGINE_NAME);
  Send(std::string("id author ") + ENGINE_AUTHOR);

  Send("option name MaxNodes type spin default 800 min 1 max " +
       std::to_string(kMaxMctsNodes));
  Send("option name RootMateSolver type combo default bns var bns var dfpn");
  Send("option name OnnxModel type string default shogi_bt4.onnx");
  Send("option name ModelFormat type combo default auto var auto var jhbr2 var dlshogi");
  Send("option name DlshogiModel type check default false");
  Send("option name UseGPU type check default true");
  // Threads is kept as an alias for WorkersPerGpu (backward compat).
  Send("option name Threads type spin default 2 min 1 max 64");
  Send("option name WorkersPerGpu type spin default 2 min 1 max 64");
  Send("option name MinibatchSize type spin default 128 min 1 max 4096");
  Send("option name CInit type string default 1.25");
  Send("option name CBase type string default 19652.0");
  Send("option name FpuReduction type string default 0.27");
  Send("option name CInitRoot type string default 1.25");
  Send("option name CBaseRoot type string default 19652.0");
  Send("option name FpuReductionRoot type string default 0.0");
  Send("option name DrawValueBlack type string default 0.5");
  Send("option name DrawValueWhite type string default 0.5");
  Send("option name ResignThreshold type string default 0.01");
  Send("option name InfoIntervalMs type spin default 1000 min 100 max 10000");
  Send("option name LeafMateMode type combo default shallow var off var shallow");
  Send("option name LeafMateDepth type spin default 5 min 1 max 7");
  Send("option name RootMateDepth type spin default 7 min 0 max 7");
  Send("option name NNCacheSize type spin default 0 min 0 max 100000000");
  Send("option name NumGPUs type spin default 1 min 1 max 8");
  Send("option name MaxMovesToDraw type spin default 100000 min 1 max 100000");
  Send("option name UseMovesLeft type check default false");
  Send("option name MovesLeftMaxEffect type string default 0.0345");
  Send("option name MovesLeftThreshold type string default 0.8");
  Send("option name MovesLeftSlope type string default 0.0027");
  Send("option name MovesLeftConstantFactor type string default 0.0");
  Send("option name MovesLeftScaledFactor type string default 1.6521");
  Send("option name MovesLeftQuadraticFactor type string default -0.6521");
  Send("option name MaxMoveTime type spin default 0 min 0 max 300000");
  Send("option name MaxMoveTime1m type spin default 0 min 0 max 60000");
  Send("option name TimeManagement type combo default shadow var off var shadow var on");
  Send("option name MoveOverheadMs type spin default 100 min 0 max 5000");
  Send("option name TimeMaxExtensionPercent type spin default 175 min 100 max 300");
  Send("option name TimeDebug type check default false");
  Send("option name BookFile type string default ");
  Send("option name UseGoteExitBook type check default false");
  Send("option name GoteExitBookFile type string default "
       "user_book1_gote_exit.ybb");

  Send("usiok");
}

void USIEngine::CmdIsReady() {
  if (evaluators_.empty()) {
    ShogiEncoderTables::Init();

    try {
      for (int g = 0; g < num_gpus_; g++) {
        Log("Loading model on GPU " + std::to_string(g) + ": " + onnx_path_);
        auto evaluator =
            std::make_unique<NNEvaluator>(onnx_path_, use_gpu_, g,
                                          search_config_.workers_per_gpu,
                                          model_format_);
        if (evaluator->num_slots() == 0) {
          throw std::runtime_error("inference backend created no worker slots");
        }
        evaluators_.push_back(std::move(evaluator));
      }
    } catch (const std::exception& error) {
      evaluators_.clear();
      Log("Model load failed: " + std::string(error.what()));
      Send("readyok");
      return;
    }

    const bool has_moves_left =
        !evaluators_.empty() &&
        std::all_of(evaluators_.begin(), evaluators_.end(),
                    [](const auto& evaluator) {
                      return evaluator && evaluator->has_moves_left();
                    });
    Log("Model loaded, GPUs=" + std::to_string(num_gpus_) +
        ", format=" + ModelFormatToString(model_format_) +
        ", mlh=" + (has_moves_left ? "yes" : "no") +
        ", max_nodes=" + std::to_string(max_nodes_));
    if (search_config_.moves_left.enabled && !has_moves_left) {
      Log("UseMovesLeft requested, but the loaded model has no MLH output; "
          "the search effect is disabled");
    }

  }

  // isready is the acknowledged per-game preparation barrier. Reuse the
  // Search allocation and GPU workers, but clear the previous game's tree and
  // NN entries before readyok so cleanup is never charged to a timed move.
  if (search_) {
    search_->PrepareForNewGame();
  } else {
    EnsureSearch();
  }
  new_game_prepared_ = true;

  // Book loading is independent of model loading so BookFile changes followed
  // by isready take effect without rebuilding the TensorRT evaluators.
  if (books_dirty_) {
    book_.Close();
    gote_exit_book_.Close();

    if (!book_path_.empty()) {
      const uint64_t book_count = book_.Load(book_path_);
      if (book_.is_loaded()) {
        Log("YBB book ready: " + std::to_string(book_count) +
            " positions from " + book_path_);
      } else {
        Log("YBB book error: " + book_.last_error());
      }
    }

    if (use_gote_exit_book_ && !gote_exit_book_path_.empty()) {
      const uint64_t book_count =
          gote_exit_book_.Load(gote_exit_book_path_);
      if (gote_exit_book_.is_loaded()) {
        Log("Gote exit YBB ready: " + std::to_string(book_count) +
            " positions from " + gote_exit_book_path_);
      } else if (use_gote_exit_book_) {
        Log("Gote exit YBB error: " + gote_exit_book_.last_error());
      }
    }
    books_dirty_ = false;
  }
  Send("readyok");
}

void USIEngine::RegisterOptionParsers() {
  const auto reset_search = [this]() { search_.reset(); };
  const auto reset_inference = [this]() {
    search_.reset();
    evaluators_.clear();
  };

  option_parsers_["maxnodes"] =
      [this](const std::string& name, const std::string& value) {
        max_nodes_ = std::clamp(ParseInt(value), 1, kMaxMctsNodes);
        // Unlike most options, make the diagnostic report the effective value
        // after clamping instead of echoing a potentially misleading request.
        Log("Set " + name + " = " + std::to_string(max_nodes_));
        return OptionSetResult::kAlreadyLogged;
      };

  option_parsers_["onnxmodel"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        onnx_path_ = value;
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["modelformat"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        model_format_ = ParseModelFormat(value);
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["dlshogimodel"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        model_format_ = ToLower(value) == "true" ? ModelFormat::kDlshogi
                                                  : ModelFormat::kAuto;
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["usegpu"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        use_gpu_ = ToLower(value) == "true" || value == "1";
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["threads"] = option_parsers_["workerspergpu"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        search_config_.workers_per_gpu =
            std::clamp(ParseInt(value), 1, kMaxWorkersPerGpu);
        // TensorRT allocates one execution slot per worker.
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["minibatchsize"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.minibatch_size =
            std::clamp(ParseInt(value), 1, 4096);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["cinit"] = option_parsers_["c_init"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.c_init =
            std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["cbase"] = option_parsers_["c_base"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.c_base =
            std::clamp(ParseFiniteFloat(value), 1.0f, 1.0e9f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["fpureduction"] = option_parsers_["c_fpu_reduction"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.c_fpu_reduction =
            std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["cinitroot"] = option_parsers_["c_init_root"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.c_init_root =
            std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["cbaseroot"] = option_parsers_["c_base_root"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.c_base_root =
            std::clamp(ParseFiniteFloat(value), 1.0f, 1.0e9f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["fpureductionroot"] =
      option_parsers_["c_fpu_reduction_root"] =
          [this, reset_search](const std::string& /*name*/,
                               const std::string& value) {
            search_config_.c_fpu_reduction_root =
                std::clamp(ParseFiniteFloat(value), 0.0f, 100.0f);
            reset_search();
            return OptionSetResult::kSetAndLog;
          };

  option_parsers_["drawvalueblack"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.draw_value_black =
            std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["drawvaluewhite"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.draw_value_white =
            std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["resignthreshold"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.resign_threshold =
            std::clamp(ParseFiniteFloat(value), 0.0f, 0.5f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["infointervalms"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.info_interval_ms =
            std::clamp(ParseInt(value), 100, 10000);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["leafmatemode"] =
      [this, reset_search](const std::string& name,
                           const std::string& value) {
        const std::string mode = ToLower(value);
        if (mode == "shallow") {
          if (search_config_.leaf_mate_depth <= 0 ||
              search_config_.leaf_mate_depth % 2 == 0) {
            search_config_.leaf_mate_depth = 5;
          }
        } else if (mode == "off") {
          search_config_.leaf_mate_depth = 0;
        } else if (mode == "dfpn") {
          // Historical behavior treated the unimplemented df-pn mode as off.
          search_config_.leaf_mate_depth = 0;
          Log("LeafMateMode=dfpn is retired; treating it as off");
          return OptionSetResult::kAlreadyLogged;
        } else {
          Log("Ignored unsupported LeafMateMode value: " + value);
          return OptionSetResult::kAlreadyLogged;
        }
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["leafmatedepth"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        int depth = std::clamp(ParseInt(value), 1, 7);
        if (depth % 2 == 0) --depth;
        search_config_.leaf_mate_depth = depth;
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["rootmatedepth"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        int depth = std::clamp(ParseInt(value), 0, 7);
        if (depth > 0 && depth % 2 == 0) --depth;
        search_config_.root_mate_depth = depth;
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["nncachesize"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.nn_cache_size =
            std::min<std::size_t>(ParseSize(value), 100000000);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["numgpus"] =
      [this, reset_inference](const std::string& /*name*/,
                              const std::string& value) {
        num_gpus_ = std::clamp(ParseInt(value), 1, 8);
        reset_inference();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["maxmovestodraw"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.max_moves_to_draw =
            std::clamp(ParseInt(value), 1, 100000);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["usemovesleft"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.enabled =
            ToLower(value) == "true" || value == "1";
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftmaxeffect"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.max_effect =
            std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftthreshold"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.threshold =
            std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftslope"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.slope =
            std::clamp(ParseFiniteFloat(value), 0.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftconstantfactor"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.constant_factor =
            std::clamp(ParseFiniteFloat(value), -1.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftscaledfactor"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.scaled_factor =
            std::clamp(ParseFiniteFloat(value), -2.0f, 2.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["movesleftquadraticfactor"] =
      [this, reset_search](const std::string& /*name*/,
                           const std::string& value) {
        search_config_.moves_left.quadratic_factor =
            std::clamp(ParseFiniteFloat(value), -1.0f, 1.0f);
        reset_search();
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["rootmatesolver"] =
      [this](const std::string& /*name*/, const std::string& value) {
        std::string mode = value;
        std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        root_mate_solver_bns_ = mode != "dfpn";
        Log("RootMateSolver=" + std::string(root_mate_solver_bns_ ? "bns"
                                                                  : "dfpn"));
        return OptionSetResult::kAlreadyLogged;
      };

  option_parsers_["dfpnmaxtime"] =
      [this](const std::string& /*name*/, const std::string& /*value*/) {
        // Accepted for old engine configuration files, but no longer advertised
        // or used. Root mate search now follows the MCTS lifetime.
        Log("DfPnMaxTime is retired; root mate search follows MCTS");
        return OptionSetResult::kAlreadyLogged;
      };

  option_parsers_["maxmovetime"] =
      [this](const std::string& /*name*/, const std::string& value) {
        max_move_time_ms_ = std::clamp(ParseInt(value), 0, 300000);
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["maxmovetime1m"] =
      [this](const std::string& /*name*/, const std::string& value) {
        max_move_time_1m_ms_ = std::clamp(ParseInt(value), 0, 60000);
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["timemanagement"] =
      [this](const std::string& /*name*/, const std::string& value) {
        time_management_mode_ = ParseTimeManagementMode(value);
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["moveoverheadms"] =
      [this](const std::string& /*name*/, const std::string& value) {
        move_overhead_ms_ = std::clamp(ParseInt(value), 0, 5000);
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["timemaxextensionpercent"] =
      [this](const std::string& /*name*/, const std::string& value) {
        time_max_extension_percent_ =
            std::clamp(ParseInt(value), 100, 300);
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["timedebug"] =
      [this](const std::string& /*name*/, const std::string& value) {
        time_debug_ = ToLower(value) == "true" || value == "1";
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["bookfile"] =
      [this](const std::string& /*name*/, const std::string& value) {
        book_path_ = value;
        books_dirty_ = true;
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["goteexitbookfile"] =
      [this](const std::string& /*name*/, const std::string& value) {
        gote_exit_book_path_ = value;
        books_dirty_ = true;
        return OptionSetResult::kSetAndLog;
      };

  option_parsers_["usegoteexitbook"] =
      [this](const std::string& /*name*/, const std::string& value) {
        const bool enabled = ToLower(value) == "true" || value == "1";
        if (enabled != use_gote_exit_book_) books_dirty_ = true;
        use_gote_exit_book_ = enabled;
        return OptionSetResult::kSetAndLog;
      };

  const auto retired_option = [this](const std::string& name,
                                     const std::string& /*value*/) {
    Log("Option " + name + " is retired and ignored");
    return OptionSetResult::kAlreadyLogged;
  };
  option_parsers_["noiseepsilon"] = retired_option;
  option_parsers_["perleafgathering"] = retired_option;
  option_parsers_["leafdfpnnodes"] = retired_option;
  option_parsers_["virtuallossweight"] = retired_option;
  option_parsers_["maxgpubatch"] = retired_option;
  option_parsers_["movesleftweight"] = retired_option;
  option_parsers_["movesleftcap"] = retired_option;
  option_parsers_["bookonthefly"] = retired_option;
}

void USIEngine::CmdSetOption(const std::vector<std::string>& parts) {
  std::string name, value;
  std::size_t value_marker = parts.size();
  for (std::size_t i = 1; i < parts.size(); ++i) {
    if (parts[i] == "value") {
      value_marker = i;
      break;
    }
  }
  const std::size_t name_begin =
      parts.size() > 1 && parts[1] == "name" ? 2 : parts.size();
  for (std::size_t i = name_begin; i < value_marker; ++i) {
    if (!name.empty()) name += ' ';
    name += parts[i];
  }
  if (value_marker < parts.size()) {
    for (std::size_t i = value_marker + 1; i < parts.size(); ++i) {
      if (!value.empty()) value += ' ';
      value += parts[i];
    }
  }

  if (name.empty()) {
    Log("Ignored malformed setoption without a name");
    return;
  }

  const std::string name_lower = ToLower(name);
  const auto it = option_parsers_.find(name_lower);
  if (it == option_parsers_.end()) {
    Log("Unknown option ignored: " + name);
    return;
  }

  try {
    const auto result = it->second(name, value);
    if (result == OptionSetResult::kSetAndLog) {
      Log("Set " + name + " = " + value);
    }
  } catch (const std::exception& error) {
    Log("Invalid value for " + name + ": " + error.what());
  }
}

void USIEngine::CmdUsiNewGame() {
  board_.SetStartPos();
  board_.ClearHistory();
  position_start_key_ = board_.Hash();
  position_moves_.clear();
  if (search_ && !new_game_prepared_) {
    // USI clients should issue isready/readyok before each game. Preserve
    // strict per-game cache isolation for older clients, but make the
    // unsynchronised slow path visible because it can delay the next go.
    Log("WARNING usinewgame received without per-game isready; "
        "clearing search state now");
    search_->PrepareForNewGame();
  }
  new_game_prepared_ = false;
}

void USIEngine::CmdPosition(const std::vector<std::string>& parts) {
  board_ = ShogiBoard();
  size_t idx = 1;

  if (idx >= parts.size()) return;

  if (parts[idx] == "startpos") {
    board_.SetStartPos();
    idx++;
  } else if (parts[idx] == "sfen") {
    idx++;
    std::string sfen;
    while (idx < parts.size() && parts[idx] != "moves") {
      if (!sfen.empty()) sfen += " ";
      sfen += parts[idx];
      idx++;
    }
    board_.SetFromSfen(sfen);
  }

  position_start_key_ = board_.Hash();
  position_moves_.clear();

  if (idx < parts.size() && parts[idx] == "moves") {
    idx++;
    while (idx < parts.size()) {
      Move m = Move::Parse(parts[idx]);
      board_.DoMove(m);
      position_moves_.push_back(m);
      idx++;
    }
  }

}

USIEngine::GoParameters USIEngine::ParseGoParameters(
    const std::vector<std::string>& parts) const {
  GoParameters params;
  params.nodes_limit = max_nodes_;
  int btime = 0, wtime = 0, byoyomi = 0, binc = 0, winc = 0;
  int move_time = 0;

  size_t i = 1;
  while (i < parts.size()) {
    if (parts[i] == "btime" && i + 1 < parts.size()) {
      btime = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "wtime" && i + 1 < parts.size()) {
      wtime = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "byoyomi" && i + 1 < parts.size()) {
      byoyomi = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "binc" && i + 1 < parts.size()) {
      binc = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "winc" && i + 1 < parts.size()) {
      winc = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "movetime" && i + 1 < parts.size()) {
      move_time = std::stoi(parts[i + 1]); i += 2;
    } else if (parts[i] == "nodes" && i + 1 < parts.size()) {
      params.nodes_limit = std::stoi(parts[i + 1]);
      params.time_control.has_explicit_nodes = true;
      i += 2;
    } else if (parts[i] == "infinite") {
      params.nodes_limit = kMaxMctsNodes;
      params.infinite = true;
      i++;
    } else if (parts[i] == "ponder") {
      params.ponder = true;
      i++;
    } else {
      i++;
    }
  }

  params.time_control.main_time_ms =
      board_.side_to_move() == BLACK ? btime : wtime;
  params.time_control.increment_ms =
      board_.side_to_move() == BLACK ? binc : winc;
  params.time_control.byoyomi_ms = byoyomi;
  params.time_control.move_time_ms = move_time;
  params.time_control.game_ply = board_.ply();
  params.time_control.has_main_time = btime > 0 || wtime > 0;
  return params;
}

std::optional<std::string> USIEngine::ProbeOpeningBook() {
  // Check entering-king declaration.
  if (board_.CanDeclareWin()) {
    return std::string("win");
  }

  // Probe the specialized policy only on Gote turns. A miss deliberately
  // falls through to MCTS instead of the normal book: it means this line has
  // left the generated Gote policy.
  OpeningBook* active_book = nullptr;
  const OpeningBookChoice book_choice = ChooseOpeningBook(
      board_.side_to_move(), use_gote_exit_book_, book_.is_loaded(),
      gote_exit_book_.is_loaded());
  if (book_choice == OpeningBookChoice::kGoteExit) {
    active_book = &gote_exit_book_;
  } else if (book_choice == OpeningBookChoice::kNormal) {
    active_book = &book_;
  }

  if (active_book != nullptr) {
    auto* entry = active_book->Probe(board_);
    if (entry) {
      const std::string move_usi = entry->move.ToString();
      Log(std::string(book_choice == OpeningBookChoice::kGoteExit
                          ? "Gote exit book hit: "
                          : "Book hit: ") +
          move_usi + " (eval=" +
          std::to_string(entry->eval) + ", depth=" +
          std::to_string(entry->depth) + ")");
      return move_usi;
    }
  }
  return std::nullopt;
}

std::string USIEngine::FormatTimeBudgetForLog(const TimeBudget& budget,
                                              bool compact) const {
  std::ostringstream out;
  out << "mode=" << TimeManagementModeName(budget.mode);
  if (compact) {
    out << " budget_ms=" << budget.earliest_stop_ms << "/"
        << budget.target_stop_ms << "/" << budget.latest_search_ms;
  } else {
    out << " earliest_ms=" << budget.earliest_stop_ms
        << " target_ms=" << budget.target_stop_ms
        << " latest_ms=" << budget.latest_search_ms;
  }
  out << " response_ms=" << budget.response_deadline_ms;
  return out.str();
}

USIEngine::RootMateLaunch USIEngine::LaunchRootMateSearch(
    const TimeBudget& time_budget,
    std::chrono::steady_clock::time_point move_start_time,
    dlshogi_mcts::Search* mcts_search,
    std::mutex& watchdog_mutex,
    std::condition_variable& watchdog_cv,
    bool& search_done,
    std::atomic<bool>& watchdog_fired) {
  RootMateLaunch launch;
  launch.state =
      std::make_shared<RootMateState>(root_mate_solver_bns_, board_);

  auto root_mate_deadline = MateDfpnSolver::Deadline::max();
  if (time_budget.hard_deadline_ms > 0) {
    const int safe_deadline_ms = std::max(
        time_budget.hard_deadline_ms - kRootMateDeadlineMarginMs, 1);
    root_mate_deadline =
        move_start_time + std::chrono::milliseconds(safe_deadline_ms);
  }

  // Search::Run releases this worker only after it has reset its stop flag.
  // Otherwise a fast mate proof could call Stop() just before Run() and have
  // that cancellation erased by search startup.
  launch.search_thread = std::thread(
      [root_mate = launch.state, root_mate_deadline, mcts_search]() {
        if (root_mate->WaitForStart()) {
          const auto started_at = std::chrono::steady_clock::now();
          root_mate->mate_move = root_mate->Search(root_mate_deadline);
          root_mate->elapsed_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - started_at)
                  .count();
          const bool proved_mate =
              !root_mate->mate_move.is_null() &&
              !MateDfpnSolver::IsNoMate(root_mate->mate_move);
          root_mate->done.store(true, std::memory_order_release);
          // Apply the same root repetition boundary before stopping MCTS. If
          // defense-in-depth rejects a solver regression, normal search must
          // still be allowed to finish and provide a usable fallback move.
          if (proved_mate &&
              RootMoveRepetitionResult(root_mate->board,
                                       root_mate->mate_move) ==
                  ShogiBoard::RepetitionResult::kNone) {
            root_mate->stopped_mcts.store(true, std::memory_order_release);
            mcts_search->Stop();
          }
        } else {
          root_mate->done.store(true, std::memory_order_release);
        }
      });

  // Exact watchdog: a condition-variable deadline avoids the old 0-50 ms
  // polling/join delay. Pure node-limited searches remain uncapped.
  if (time_budget.hard_deadline_ms > 0) {
    const auto hard_deadline =
        move_start_time +
        std::chrono::milliseconds(time_budget.hard_deadline_ms);
    launch.watchdog_thread = std::thread(
        [this, root_mate = launch.state, &watchdog_mutex, &watchdog_cv,
         &search_done, &watchdog_fired, hard_deadline]() {
          std::unique_lock<std::mutex> lock(watchdog_mutex);
          if (!watchdog_cv.wait_until(
                  lock, hard_deadline, [&search_done] { return search_done; })) {
            watchdog_fired.store(true, std::memory_order_release);
            if (search_) search_->Stop();
            root_mate->Stop();
          }
        });
  }

  return launch;
}

void USIEngine::LogRootMateResult(const RootMateState& state,
                                  bool finished_before_stop,
                                  bool watchdog_fired,
                                  std::int64_t join_ms) {
  const bool is_mate =
      !state.mate_move.is_null() &&
      !MateDfpnSolver::IsNoMate(state.mate_move);
  const bool is_nomate = MateDfpnSolver::IsNoMate(state.mate_move);
  const char* outcome =
      is_mate ? "mate"
              : is_nomate ? "nomate"
                          : finished_before_stop ? "limit" : "stopped";
  const char* stop_source = "mcts";
  if (state.stopped_mcts.load(std::memory_order_acquire)) {
    stop_source = "mate";
  } else if (is_nomate || finished_before_stop) {
    stop_source = "self";
  } else if (watchdog_fired) {
    stop_source = "watchdog";
  }
  Log("root_mate solver=" + std::string(state.SolverName()) +
      " outcome=" + outcome +
      " elapsed_ms=" + std::to_string(state.elapsed_ms) +
      " nodes=" + std::to_string(state.NodesSearched()) +
      " stop_source=" + stop_source +
      " join_ms=" + std::to_string(join_ms));
}

void USIEngine::LogTimeResult(const dlshogi_mcts::SearchResult& result) {
  if (time_management_mode_ == TimeManagementMode::kOff && !time_debug_) {
    return;
  }
  const auto& decision = result.time_decision;
  const auto& snapshot = decision.snapshot;
  std::ostringstream timing;
  timing << "time_result " << FormatTimeBudgetForLog(result.time_budget, true)
         << " reason=" << TimeStopReasonName(decision.reason)
         << " search_ms=" << static_cast<int>(result.time_sec * 1000.0f)
         << " effective_ms=" << decision.effective_deadline_ms
         << " playouts=" << snapshot.new_playouts
         << " best_visits=" << snapshot.best_visits
         << " second_visits=" << snapshot.second_visits
         << " best_q=" << std::fixed << std::setprecision(4) << snapshot.best_q
         << " second_q=" << snapshot.second_q
         << " stable_ms=" << decision.stable_ms
         << " projected=" << decision.projected_remaining
         << " best_changes=" << decision.best_changes
         << " extension=" << (decision.extension_active ? 1 : 0)
         << " root_guard_cancelled="
         << (result.root_guard_cancelled ? 1 : 0);
  Log(timing.str());
}

void USIEngine::LogTimeResponse(
    std::chrono::steady_clock::time_point move_start_time,
    const TimeBudget& time_budget,
    const RootMateState& state) {
  if (time_management_mode_ == TimeManagementMode::kOff && !time_debug_) {
    return;
  }
  const auto response_elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - move_start_time)
          .count();
  Log("time_response elapsed_ms=" + std::to_string(response_elapsed_ms) +
      " deadline_ms=" + std::to_string(time_budget.response_deadline_ms) +
      " root_mate_nodes=" + std::to_string(state.NodesSearched()));
}

void USIEngine::SelectAndReportBestMove(
    const dlshogi_mcts::SearchResult& result,
    const RootMateState& root_mate) {
  const bool root_mate_is_mate =
      !root_mate.mate_move.is_null() &&
      !MateDfpnSolver::IsNoMate(root_mate.mate_move);

  // --- Choose result ---
  bool use_mate = root_mate_is_mate;

  // Defense in depth: a root df-pn result must not replace MCTS when its very
  // first move enters a repetition. The solver now adjudicates these nodes
  // itself, but keeping the final boundary check prevents a future df-pn
  // regression from turning an OUTE_SENNICHITE loss into bestmove.
  if (use_mate) {
    const auto repetition =
        RootMoveRepetitionResult(board_, root_mate.mate_move);
    if (repetition != ShogiBoard::RepetitionResult::kNone) {
      Log("Rejected root mate move " + root_mate.mate_move.ToString() +
          ": repetition=" + RepetitionResultName(repetition));
      use_mate = false;
    }
  }

  if (use_mate) {
    auto pv = root_mate.Pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    if (pv_str.empty()) pv_str = root_mate.mate_move.ToString();

    int mate_ply = (int)pv.size();
    Log("Root mate solver found mate in " + std::to_string(mate_ply) +
        " ply");

    Send("info depth 1 score mate " + std::to_string((mate_ply + 1) / 2) +
         " nodes " + std::to_string(root_mate.NodesSearched()) +
         " pv " + pv_str);
    Send("bestmove " + root_mate.mate_move.ToString());
    return;
  }

  // --- Use MCTS result ---
  if (result.best_move.is_null()) {
    Send("bestmove resign");
    return;
  }

  if (result.nn_cache.capacity > 0) {
    Log(FormatNNCacheStats(result.nn_cache));
  }

  USISearchInfo usi_info;
  usi_info.pv = result.pv;
  if (usi_info.pv.empty()) usi_info.pv.push_back(result.best_move);
  usi_info.depth = static_cast<int>(usi_info.pv.size());
  usi_info.seldepth = usi_info.depth;
  usi_info.score_cp = result.score_cp;
  usi_info.nodes = std::max(result.nodes, 0);
  usi_info.nps = static_cast<std::uint64_t>(std::max(result.nps, 0.0f));
  usi_info.hashfull = NNCacheHashfull(result.nn_cache);
  usi_info.time_ms = static_cast<std::uint64_t>(
      std::max(result.time_sec, 0.0f) * 1000.0f);
  Send(FormatUSISearchInfo(usi_info));

  Send("bestmove " + result.best_move.ToString());
}

void USIEngine::CmdGo(const std::vector<std::string>& parts) {
  if (std::find(parts.begin(), parts.end(), "mate") != parts.end()) {
    CmdGoMate(parts);
    return;
  }

  const auto move_start_time = std::chrono::steady_clock::now();
  if (evaluators_.empty()) {
    Send("bestmove resign");
    return;
  }

  const auto params = ParseGoParameters(parts);

  TimeOptions time_options;
  time_options.max_move_time_ms = max_move_time_ms_;
  time_options.max_move_time_1m_ms = max_move_time_1m_ms_;
  time_options.move_overhead_ms = move_overhead_ms_;
  time_options.max_extension_percent = time_max_extension_percent_;
  time_options.mode = time_management_mode_;
  const TimeBudget time_budget =
      TimeManager::Compute(params.time_control, time_options);

  if (time_debug_) {
    Log("time_budget " + FormatTimeBudgetForLog(time_budget, false) +
        " ply=" + std::to_string(params.time_control.game_ply) +
        " actual_mcts_ms=" +
        std::to_string(static_cast<int>(time_budget.mcts_time_seconds * 1000.0f)) +
        " hard_ms=" + std::to_string(time_budget.hard_deadline_ms));
  }

  if (auto book_move = ProbeOpeningBook()) {
    Send("bestmove " + *book_move);
    return;
  }

  // Configure the dlshogi-style MCTS search.
  search_config_.max_nodes = params.nodes_limit;
  search_config_.max_time = time_budget.mcts_time_seconds;
  search_config_.time_budget = time_budget;

  // Set info callback for periodic GUI output during search.
  search_config_.info_callback = [this](const dlshogi_mcts::SearchInfo& info) {
    // Keep free-form diagnostics before the structured record.  Some GUIs
    // incorrectly treat `info string` as a new empty analysis record, so the
    // last line in each update must be the complete depth/score/PV record.
    if (info.nn_cache.capacity > 0) {
      Log(FormatNNCacheStats(info.nn_cache));
    }

    USISearchInfo usi_info;
    usi_info.depth = info.depth;
    usi_info.seldepth = info.depth;
    usi_info.score_cp = info.score_cp;
    usi_info.nodes = std::max(info.nodes, 0);
    usi_info.nps = std::max(info.nps, 0);
    usi_info.hashfull = NNCacheHashfull(info.nn_cache);
    usi_info.time_ms = std::max(info.time_ms, 0);
    usi_info.pv = info.pv;
    Send(FormatUSISearchInfo(usi_info));
  };

  // Persistent Search object across `go` commands and games.
  EnsureSearch();
  // Search holds its own config snapshot — push per-move
  // updates so max_time / max_nodes reflect THIS go command, not the
  // first one ever issued.
  search_->SetMaxTime(search_config_.max_time);
  search_->SetMaxNodes(search_config_.max_nodes);
  search_->SetTimeBudget(time_budget);

  std::mutex watchdog_mutex;
  std::condition_variable watchdog_cv;
  bool search_done = false;
  std::atomic<bool> watchdog_fired{false};
  auto launch = LaunchRootMateSearch(
      time_budget, move_start_time, search_.get(),
      watchdog_mutex, watchdog_cv, search_done, watchdog_fired);

  auto result =
      search_->Run(board_, position_start_key_, position_moves_,
                   move_start_time,
                   [state = launch.state] { state->Start(); });

  // MCTS is the single owner of move time. Do not grant a separate post-MCTS
  // grace period: stop and join the mate worker as soon as MCTS returns.
  const bool root_mate_finished_before_stop =
      launch.state->done.load(std::memory_order_acquire);
  const auto join_started_at = std::chrono::steady_clock::now();
  launch.state->Stop();
  if (launch.search_thread.joinable()) launch.search_thread.join();
  const auto root_mate_join_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - join_started_at)
          .count();

  {
    std::lock_guard<std::mutex> lock(watchdog_mutex);
    search_done = true;
  }
  watchdog_cv.notify_all();
  if (launch.watchdog_thread.joinable()) launch.watchdog_thread.join();

  Log(std::string("tree_reused ") + (result.tree_reused ? "true" : "false") +
      " root_visits_before " +
      std::to_string(result.root_visits_before));

  LogTimeResult(result);
  LogRootMateResult(*launch.state, root_mate_finished_before_stop,
                    watchdog_fired.load(std::memory_order_acquire),
                    root_mate_join_ms);
  LogTimeResponse(move_start_time, time_budget, *launch.state);

  SelectAndReportBestMove(result, *launch.state);
}

void USIEngine::CmdGoMate(const std::vector<std::string>& parts) {
  int time_limit_ms = 0;
  for (size_t i = 1; i < parts.size(); i++) {
    if (parts[i] == "mate") {
      if (i + 1 < parts.size() && parts[i + 1] != "infinite") {
        time_limit_ms = std::stoi(parts[i + 1]);
      }
      break;
    }
  }

  size_t max_nodes;
  if (time_limit_ms <= 0) {
    max_nodes = 10000000;
  } else {
    max_nodes = std::max((size_t)(time_limit_ms * 200), (size_t)100000);
  }
  MateDfpnSolver solver(max_nodes);

  std::atomic<bool> search_done{false};
  Move mate_move;
  const auto t0 = std::chrono::steady_clock::now();
  const auto deadline =
      time_limit_ms > 0
          ? t0 + std::chrono::milliseconds(time_limit_ms)
          : MateDfpnSolver::Deadline::max();

  auto search_thread = std::thread([&, deadline]() {
    mate_move = solver.search(board_, max_nodes, deadline);
    search_done.store(true, std::memory_order_release);
  });

  while (!search_done.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (time_limit_ms > 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed >= time_limit_ms) {
        solver.stop();
        break;
      }
    }
  }
  search_thread.join();

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0).count();

  if (!search_done.load(std::memory_order_acquire) && time_limit_ms > 0) {
    Log("Mate search timeout after " + std::to_string(elapsed) + " ms");
    Send("checkmate timeout");
  } else if (!mate_move.is_null() && !MateDfpnSolver::IsNoMate(mate_move)) {
    auto pv = solver.get_pv();
    std::string pv_str;
    for (const auto& m : pv) {
      if (!pv_str.empty()) pv_str += " ";
      pv_str += m.ToString();
    }
    Log("Mate found in " + std::to_string(pv.size()) + " ply");
    Send("checkmate " + pv_str);
  } else if (MateDfpnSolver::IsNoMate(mate_move)) {
    Send("checkmate nomate");
  } else {
    Send("checkmate timeout");
  }
}

void USIEngine::CmdStop() {
  if (search_) search_->Stop();
}

void USIEngine::CmdGameOver(const std::vector<std::string>& parts) {
  if (parts.size() > 1) Log("Game over: " + parts[1]);
}

void USIEngine::CmdDebug() {
  Log("Position: " + board_.ToSfen());
  auto moves = board_.GenerateLegalMoves();
  Log("Legal moves: " + std::to_string(moves.size()));
}

}  // namespace jhbr2
