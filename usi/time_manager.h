#pragma once

#include <cstddef>
#include <cstdint>

namespace jhbr2 {

enum class TimeManagementMode {
  kOff,
  kShadow,
  kOn,
};

enum class TimeStopReason;

const char* TimeManagementModeName(TimeManagementMode mode);
const char* TimeStopReasonName(TimeStopReason reason);

struct TimeControl {
  int main_time_ms = 0;
  int increment_ms = 0;
  int byoyomi_ms = 0;
  int move_time_ms = 0;
  int game_ply = 0;

  // CmdGo historically enables main-time allocation when either player's
  // main-time field is positive, even if main_time_ms is zero for this side.
  bool has_main_time = false;
  bool has_explicit_nodes = false;
  bool infinite = false;
  bool ponder = false;
};

struct TimeOptions {
  int max_move_time_ms = 0;
  int max_move_time_1m_ms = 0;
  int dfpn_max_time_ms = 4000;
  int move_overhead_ms = 100;
  int max_extension_percent = 175;
  TimeManagementMode mode = TimeManagementMode::kShadow;
};

struct TimeBudget {
  TimeManagementMode mode = TimeManagementMode::kOff;

  // Nominal MCTS duration. Zero leaves MCTS node-limited.
  float mcts_time_seconds = 0.0f;

  // Selected USI option cap and the absolute watchdog deadline. When no
  // option cap applies, the watchdog allows two seconds beyond nominal MCTS.
  int active_move_cap_ms = 0;
  int hard_deadline_ms = 0;

  // Concurrent root DFPN limits and allowed wait after MCTS completes.
  int root_dfpn_time_ms = 0;
  int root_dfpn_grace_ms = 0;
  std::size_t root_dfpn_nodes = 0;

  // Adaptive time points, all relative to receipt of `go`. They are populated
  // in shadow/on modes. In shadow mode the legacy fields above still control
  // the search, so the candidate policy can be measured without changing
  // moves. In on mode mcts_time_seconds is latest_search_ms / 1000.
  int earliest_stop_ms = 0;
  int target_stop_ms = 0;
  int latest_search_ms = 0;
  int root_guard_deadline_ms = 0;
  int response_deadline_ms = 0;
  bool allow_early_stop = false;
  bool allow_extension = false;

  bool HasAdaptiveDeadline() const {
    return target_stop_ms > 0 && latest_search_ms > 0;
  }
};

class TimeManager {
 public:
  static TimeBudget Compute(const TimeControl& control,
                            const TimeOptions& options);
};

// A lock-free snapshot of the root information used by the adaptive policy.
// Visits may include a reused tree; new_playouts is always from this search.
struct RootSearchSnapshot {
  int elapsed_ms = 0;
  std::int64_t new_playouts = 0;
  int best_index = -1;
  int second_index = -1;
  std::int64_t best_visits = 0;
  std::int64_t second_visits = 0;
  float best_q = 0.5f;
  float second_q = 0.5f;
  bool best_proven = false;
};

enum class TimeStopReason {
  kNone,
  kProven,
  kEarlyDominant,
  kTargetStable,
  kExtensionStable,
  kLatest,
  kLegacyLimit,
  kNodeLimit,
  kExternal,
};

struct AdaptiveTimeDecision {
  bool should_stop = false;
  bool extension_active = false;
  TimeStopReason reason = TimeStopReason::kNone;
  int effective_deadline_ms = 0;
  int best_changes = 0;
  int stable_ms = 0;
  std::int64_t projected_remaining = 0;
  RootSearchSnapshot snapshot;
};

// Stateful, deterministic policy evaluator. Search serializes calls to Update;
// the class itself intentionally contains no locks.
class AdaptiveTimeController {
 public:
  void Reset(const TimeBudget& budget, int in_flight_playouts);
  AdaptiveTimeDecision Update(const RootSearchSnapshot& snapshot);
  const AdaptiveTimeDecision& decision() const { return decision_; }

 private:
  std::int64_t ProjectRemaining(const RootSearchSnapshot& snapshot,
                                int deadline_ms);
  bool StableAndDominant(const RootSearchSnapshot& snapshot,
                         int deadline_ms, int required_stable_ms,
                         float q_tolerance);

  TimeBudget budget_;
  int in_flight_playouts_ = 0;
  int last_best_index_ = -1;
  int last_best_change_ms_ = 0;
  int last_sample_ms_ = 0;
  std::int64_t last_sample_playouts_ = 0;
  double ema_nps_ = 0.0;
  int extension_deadline_ms_ = 0;
  AdaptiveTimeDecision decision_;
};

}  // namespace jhbr2
