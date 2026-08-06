#include "usi/time_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace jhbr2 {

namespace {

int ActiveMoveCap(const TimeControl& control, const TimeOptions& options) {
  int cap = options.max_move_time_ms;
  if (options.max_move_time_1m_ms > 0 && control.main_time_ms > 0 &&
      control.main_time_ms < 60000) {
    cap = options.max_move_time_1m_ms;
  }
  return cap;
}

TimeBudget ComputeLegacy(const TimeControl& control,
                         const TimeOptions& options) {
  TimeBudget budget;

  if (control.move_time_ms > 0) {
    budget.mcts_time_seconds =
        std::max(control.move_time_ms - options.move_overhead_ms, 1) /
        1000.0f;
  } else if (control.byoyomi_ms > 0) {
    budget.mcts_time_seconds = control.byoyomi_ms / 1000.0f * 0.9f;
  } else if (control.has_main_time) {
    budget.mcts_time_seconds =
        (control.main_time_ms * 0.05f + control.increment_ms * 0.8f) /
        1000.0f;
    budget.mcts_time_seconds = std::max(budget.mcts_time_seconds, 0.1f);
  }

  budget.active_move_cap_ms = ActiveMoveCap(control, options);
  if (control.move_time_ms > 0) {
    budget.hard_deadline_ms = control.move_time_ms;
  } else if (budget.active_move_cap_ms > 0) {
    const float cap_seconds =
        std::max(budget.active_move_cap_ms / 1000.0f - 0.5f, 0.5f);
    if (budget.mcts_time_seconds <= 0.0f ||
        cap_seconds < budget.mcts_time_seconds) {
      budget.mcts_time_seconds = cap_seconds;
    }
  }

  if (control.move_time_ms > 0) {
    budget.hard_deadline_ms = control.move_time_ms;
  } else if (budget.active_move_cap_ms > 0) {
    budget.hard_deadline_ms = budget.active_move_cap_ms;
  } else if (budget.mcts_time_seconds > 0.0f) {
    budget.hard_deadline_ms =
        static_cast<int>(budget.mcts_time_seconds * 1000.0f) + 2000;
  }

  return budget;
}

double PhaseDivisor(int game_ply) {
  const double opening =
      std::clamp((40.0 - std::max(game_ply, 0)) / 40.0, 0.0, 1.0);
  return 14.0 + 26.0 * opening;
}

}  // namespace

const char* TimeManagementModeName(TimeManagementMode mode) {
  switch (mode) {
    case TimeManagementMode::kOff:
      return "off";
    case TimeManagementMode::kShadow:
      return "shadow";
    case TimeManagementMode::kOn:
      return "on";
  }
  return "off";
}

const char* TimeStopReasonName(TimeStopReason reason) {
  switch (reason) {
    case TimeStopReason::kNone:
      return "none";
    case TimeStopReason::kProven:
      return "proven";
    case TimeStopReason::kEarlyDominant:
      return "early_dominant";
    case TimeStopReason::kTargetStable:
      return "target_stable";
    case TimeStopReason::kExtensionStable:
      return "extension_stable";
    case TimeStopReason::kLatest:
      return "latest";
    case TimeStopReason::kLegacyLimit:
      return "legacy_limit";
    case TimeStopReason::kNodeLimit:
      return "node_limit";
    case TimeStopReason::kExternal:
      return "external";
  }
  return "none";
}

TimeBudget TimeManager::Compute(const TimeControl& control,
                                const TimeOptions& options) {
  TimeBudget budget = ComputeLegacy(control, options);
  budget.mode = options.mode;
  if (control.infinite) {
    budget.mcts_time_seconds = 0.0f;
    budget.hard_deadline_ms = 0;
    return budget;
  }
  if (options.mode == TimeManagementMode::kOff || control.ponder) {
    return budget;
  }

  const int overhead_ms = std::clamp(options.move_overhead_ms, 0, 5000);
  const int active_cap_ms = ActiveMoveCap(control, options);
  const bool pure_byoyomi =
      control.byoyomi_ms > 0 && !control.has_main_time &&
      control.increment_ms <= 0;

  int response_ms = 0;
  if (control.move_time_ms > 0) {
    response_ms = std::max(control.move_time_ms - overhead_ms, 1);
  } else if (control.main_time_ms > 0 || control.byoyomi_ms > 0) {
    // Increment is credited after the move and therefore is allocation input,
    // not part of the absolute flag-fall deadline.
    response_ms =
        std::max(control.main_time_ms + control.byoyomi_ms - overhead_ms, 1);
  } else if (control.has_main_time) {
    // At a reported zero main clock the only defensible deadline is immediate
    // response. Increment is not available before this move completes.
    response_ms = 1;
  }
  if (active_cap_ms > 0) {
    const int capped = std::max(active_cap_ms - overhead_ms, 1);
    response_ms = response_ms > 0 ? std::min(response_ms, capped) : capped;
  }
  if (response_ms <= 0) return budget;

  int target_ms = 0;
  const bool cap_only =
      active_cap_ms > 0 && control.move_time_ms <= 0 &&
      !control.has_main_time && control.main_time_ms <= 0 &&
      control.byoyomi_ms <= 0;
  if (control.move_time_ms > 0) {
    target_ms = response_ms;
  } else if (pure_byoyomi || cap_only) {
    target_ms = response_ms;
  } else {
    const double main_budget =
        control.main_time_ms / PhaseDivisor(control.game_ply) +
        0.8 * control.increment_ms;
    const int byoyomi_floor =
        std::max(control.byoyomi_ms - overhead_ms, 0);
    target_ms = static_cast<int>(std::lround(
        std::max(main_budget, static_cast<double>(byoyomi_floor))));
    target_ms = std::max(target_ms, 100);
    target_ms = std::min(target_ms, response_ms);
  }

  const int byoyomi_floor = std::max(control.byoyomi_ms - overhead_ms, 0);
  int earliest_ms =
      std::max({100, target_ms / 10, byoyomi_floor});
  earliest_ms = std::min(earliest_ms, target_ms);

  bool allow_extension =
      control.move_time_ms <= 0 && !pure_byoyomi && control.has_main_time &&
      control.main_time_ms >= 10000 && !control.has_explicit_nodes;
  int extension_percent =
      std::clamp(options.max_extension_percent, 100, 300);
  if (control.main_time_ms < 60000) {
    extension_percent = std::min(extension_percent, 125);
  }
  if (!allow_extension) extension_percent = 100;
  int latest_ms = static_cast<int>(
      static_cast<int64_t>(target_ms) * extension_percent / 100);
  latest_ms = std::clamp(latest_ms, target_ms, response_ms);

  budget.earliest_stop_ms = pure_byoyomi ? target_ms : earliest_ms;
  budget.target_stop_ms = target_ms;
  budget.latest_search_ms = latest_ms;
  budget.response_deadline_ms = response_ms;
  budget.root_guard_deadline_ms = response_ms;
  budget.allow_early_stop =
      !pure_byoyomi && !cap_only && control.move_time_ms <= 0 &&
      control.main_time_ms > 0 && !control.has_explicit_nodes;
  budget.allow_extension = allow_extension && latest_ms > target_ms;

  if (options.mode == TimeManagementMode::kOn) {
    budget.mcts_time_seconds = latest_ms / 1000.0f;
    budget.hard_deadline_ms = response_ms;
  }
  return budget;
}

void AdaptiveTimeController::Reset(const TimeBudget& budget,
                                   int in_flight_playouts) {
  budget_ = budget;
  in_flight_playouts_ = std::max(in_flight_playouts, 0);
  last_best_index_ = -1;
  last_best_change_ms_ = 0;
  last_sample_ms_ = 0;
  last_sample_playouts_ = 0;
  ema_nps_ = 0.0;
  extension_deadline_ms_ = 0;
  decision_ = {};
  decision_.effective_deadline_ms = budget.target_stop_ms;
}

std::int64_t AdaptiveTimeController::ProjectRemaining(
    const RootSearchSnapshot& snapshot, int deadline_ms) {
  const int elapsed_ms = std::max(snapshot.elapsed_ms, 1);
  const double whole_nps =
      snapshot.new_playouts * 1000.0 / elapsed_ms;
  double recent_nps = whole_nps;
  if (last_sample_ms_ > 0 && snapshot.elapsed_ms > last_sample_ms_ &&
      snapshot.new_playouts >= last_sample_playouts_) {
    recent_nps =
        (snapshot.new_playouts - last_sample_playouts_) * 1000.0 /
        (snapshot.elapsed_ms - last_sample_ms_);
  }
  if (ema_nps_ <= 0.0) {
    ema_nps_ = recent_nps;
  } else {
    ema_nps_ = 0.75 * ema_nps_ + 0.25 * recent_nps;
  }
  last_sample_ms_ = snapshot.elapsed_ms;
  last_sample_playouts_ = snapshot.new_playouts;

  const double upper_nps = std::max({whole_nps, recent_nps, ema_nps_, 0.0});
  const int remaining_ms = std::max(deadline_ms - snapshot.elapsed_ms, 0);
  return static_cast<std::int64_t>(
             std::ceil(1.20 * upper_nps * remaining_ms / 1000.0)) +
         in_flight_playouts_;
}

bool AdaptiveTimeController::StableAndDominant(
    const RootSearchSnapshot& snapshot, int deadline_ms,
    int required_stable_ms, float q_tolerance) {
  decision_.projected_remaining = ProjectRemaining(snapshot, deadline_ms);
  decision_.stable_ms =
      std::max(snapshot.elapsed_ms - last_best_change_ms_, 0);
  const std::int64_t gap =
      snapshot.best_visits - snapshot.second_visits;
  return snapshot.best_index >= 0 &&
         decision_.stable_ms >= required_stable_ms &&
         gap > decision_.projected_remaining &&
         snapshot.best_q >= snapshot.second_q - q_tolerance;
}

AdaptiveTimeDecision AdaptiveTimeController::Update(
    const RootSearchSnapshot& snapshot) {
  decision_.snapshot = snapshot;
  if (!budget_.HasAdaptiveDeadline() || decision_.should_stop) {
    return decision_;
  }

  if (snapshot.best_index != last_best_index_) {
    if (last_best_index_ >= 0) ++decision_.best_changes;
    last_best_index_ = snapshot.best_index;
    last_best_change_ms_ = snapshot.elapsed_ms;
  }
  decision_.stable_ms =
      std::max(snapshot.elapsed_ms - last_best_change_ms_, 0);

  if (snapshot.best_proven) {
    decision_.should_stop = true;
    decision_.reason = TimeStopReason::kProven;
    decision_.effective_deadline_ms = snapshot.elapsed_ms;
    return decision_;
  }

  const int stable_required =
      std::max(200, budget_.target_stop_ms / 10);
  if (budget_.allow_early_stop &&
      snapshot.elapsed_ms >= budget_.earliest_stop_ms &&
      snapshot.elapsed_ms < budget_.target_stop_ms &&
      StableAndDominant(snapshot, budget_.target_stop_ms, stable_required,
                        0.02f)) {
    decision_.should_stop = true;
    decision_.reason = TimeStopReason::kEarlyDominant;
    decision_.effective_deadline_ms = snapshot.elapsed_ms;
    return decision_;
  }

  if (snapshot.elapsed_ms < budget_.target_stop_ms) return decision_;

  if (!budget_.allow_extension ||
      budget_.latest_search_ms <= budget_.target_stop_ms) {
    decision_.should_stop = true;
    decision_.reason = TimeStopReason::kTargetStable;
    decision_.effective_deadline_ms = budget_.target_stop_ms;
    return decision_;
  }

  if (extension_deadline_ms_ == 0) {
    const bool q_inversion =
        snapshot.best_q + 0.02f < snapshot.second_q;
    const bool recent_change = decision_.stable_ms < stable_required;
    const double ratio =
        snapshot.best_visits /
        static_cast<double>(std::max<std::int64_t>(snapshot.second_visits, 1));
    const std::int64_t projected_latest =
        ProjectRemaining(snapshot, budget_.latest_search_ms);
    const bool catchable =
        snapshot.best_visits - snapshot.second_visits <= projected_latest;
    const bool visit_pressure = ratio < 1.5 || catchable;

    if (!q_inversion && !recent_change && !visit_pressure) {
      decision_.should_stop = true;
      decision_.reason = TimeStopReason::kTargetStable;
      decision_.effective_deadline_ms = budget_.target_stop_ms;
      return decision_;
    }

    if (q_inversion || recent_change) {
      extension_deadline_ms_ = budget_.latest_search_ms;
    } else {
      extension_deadline_ms_ =
          budget_.target_stop_ms +
          (budget_.latest_search_ms - budget_.target_stop_ms) / 2;
    }
    decision_.extension_active = true;
    decision_.effective_deadline_ms = extension_deadline_ms_;
  }

  if (snapshot.elapsed_ms >= extension_deadline_ms_) {
    decision_.should_stop = true;
    decision_.reason = TimeStopReason::kLatest;
    return decision_;
  }

  if (StableAndDominant(snapshot, extension_deadline_ms_, stable_required,
                        0.02f)) {
    decision_.should_stop = true;
    decision_.reason = TimeStopReason::kExtensionStable;
    decision_.effective_deadline_ms = snapshot.elapsed_ms;
  }
  return decision_;
}

}  // namespace jhbr2
