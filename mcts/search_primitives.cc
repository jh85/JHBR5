#include "mcts/search_primitives.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "inference/nn_diagnostics.h"

namespace dlshogi_mcts {
namespace {

void UpdateResult(child_node_t* child, float result, float moves_left,
                  uct_node_t* current) {
  AtomicFetchAdd(&current->win, result);
  // moves_left describes the position after traversing child. The current
  // position is one ply farther from the end.
  AtomicFetchAdd(&current->sum_m, moves_left + 1.0f);
  current->m_visits.fetch_add(1, std::memory_order_release);
  if constexpr (kVirtualLoss != 1) {
    current->move_count.fetch_add(1 - kVirtualLoss,
                                  std::memory_order_acq_rel);
  }
  AtomicFetchAdd(&child->win, result);
  AtomicFetchAdd(&child->sum_m, moves_left);
  child->m_visits.fetch_add(1, std::memory_order_release);
  if constexpr (kVirtualLoss != 1) {
    child->move_count.fetch_add(1 - kVirtualLoss,
                                std::memory_order_acq_rel);
  }
}

}  // namespace

float ComputeMovesLeftUtility(const MovesLeftParameters& params,
                              float parent_q_win, float parent_m,
                              float child_q_win, float child_m) {
  if (!params.enabled || params.max_effect <= 0.0f ||
      params.slope <= 0.0f) {
    return 0.0f;
  }

  const float parent_q = parent_q_win * 2.0f - 1.0f;
  if (std::fabs(parent_q) <= params.threshold) return 0.0f;

  float q = child_q_win * 2.0f - 1.0f;
  float utility = std::clamp(params.slope * (child_m - parent_m),
                             -params.max_effect, params.max_effect);
  utility *= static_cast<float>((-q > 0.0f) - (-q < 0.0f));

  if (params.threshold > 0.0f && params.threshold < 1.0f) {
    q = std::max(0.0f, std::fabs(q) - params.threshold) /
        (1.0f - params.threshold);
  } else {
    q = std::fabs(q);
  }
  utility *= params.constant_factor + params.scaled_factor * q +
             params.quadratic_factor * q * q;

  // Lc0 adds M to a [-1,1] Q. JHBR3 scores win probability in [0,1].
  return 0.5f * utility;
}

float ResolveTerminalEdge(child_node_t* edge, EdgeOutcome outcome,
                          float draw_value) {
  switch (outcome) {
    case EdgeOutcome::kWin:
      if (edge) edge->SetLose();
      return 1.0f;
    case EdgeOutcome::kLoss:
      if (edge) edge->SetWin();
      return 0.0f;
    case EdgeOutcome::kDraw:
      if (edge) edge->SetDraw();
      return draw_value;
  }
  return draw_value;
}

bool TryGetProvenEdgeValue(const child_node_t* edge, float draw_value,
                           float* value) {
  if (!edge || !value) return false;
  if (edge->IsWin()) {
    *value = 0.0f;
    return true;
  }
  if (edge->IsLose()) {
    *value = 1.0f;
    return true;
  }
  if (edge->IsDraw()) {
    *value = draw_value;
    return true;
  }
  return false;
}

unsigned SelectPuctChild(child_node_t* parent, uct_node_t* current,
                         const PuctParameters& params) {
  const int sum =
      std::max(0, current->move_count.load(std::memory_order_acquire));
  const float sum_win = current->win.load(std::memory_order_acquire);
  const float sqrt_sum = std::sqrt(static_cast<float>(sum));
  const float c =
      params.c_init +
      std::log((static_cast<float>(sum) + params.c_base + 1.0f) /
               params.c_base);
  const float visited =
      current->visited_nnrate.load(std::memory_order_acquire);
  const float fpu =
      params.fpu_reduction * std::sqrt(std::max(visited, 0.0f));
  const float parent_q =
      sum > 0 && sum_win > 0.0f
          ? std::max(0.0f, sum_win / static_cast<float>(sum) - fpu)
          : 0.0f;
  const float init_u = sum == 0 ? 1.0f : sqrt_sum;

  const int parent_m_visits =
      current->m_visits.load(std::memory_order_acquire);
  // One node M sample is its own NN evaluation. Every later sample corresponds
  // to one completed child backup and therefore one value in current->win.
  // This count deliberately excludes virtual visits in move_count.
  const int parent_completed_visits = std::max(0, parent_m_visits - 1);
  const bool use_m = params.moves_left.enabled &&
                     parent_completed_visits > 0;
  const float parent_m = use_m ? current->MeanMovesLeft() : 0.0f;
  const float parent_q_win = use_m
      ? std::clamp(sum_win / static_cast<float>(parent_completed_visits),
                   0.0f, 1.0f)
      : 0.5f;

  float best_score = -std::numeric_limits<float>::infinity();
  unsigned best = 0;
  bool all_children_win = true;
  bool all_children_win_or_draw = true;
  bool has_draw = false;

  for (int i = 0; i < current->child_num; ++i) {
    auto& child = current->child[i];

    // IsLose means that the opponent after this move is proven to lose, so
    // current has a winning move and the edge leading to current is a loss.
    if (child.IsLose()) {
      if (parent) parent->SetWin();
      return static_cast<unsigned>(i);
    }

    // IsWin means this move lets the opponent win and must not be selected
    // unless every move is proven losing.
    if (child.IsWin()) continue;

    all_children_win = false;
    if (child.IsDraw()) {
      has_draw = true;
    } else {
      all_children_win_or_draw = false;
    }

    const int move_count =
        child.move_count.load(std::memory_order_acquire);
    float q;
    float u;
    if (move_count == 0) {
      q = parent_q;
      u = init_u;
    } else {
      q = child.win.load(std::memory_order_acquire) /
          static_cast<float>(move_count);
      u = sqrt_sum / static_cast<float>(1 + move_count);
    }

    float moves_left_effect = 0.0f;
    const int child_m_visits =
        child.m_visits.load(std::memory_order_acquire);
    if (use_m && child_m_visits > 0) {
      const float child_q_win = std::clamp(
          child.win.load(std::memory_order_acquire) /
              static_cast<float>(child_m_visits),
          0.0f, 1.0f);
      moves_left_effect = ComputeMovesLeftUtility(
          params.moves_left, parent_q_win, parent_m, child_q_win,
          child.MeanMovesLeft());
    }

    const float score =
        q + c * u * child.nnrate + moves_left_effect;
    if (score > best_score) {
      best_score = score;
      best = static_cast<unsigned>(i);
    }
  }

  if (all_children_win) {
    if (parent) parent->SetLose();
  } else if (all_children_win_or_draw && has_draw) {
    if (parent) parent->SetDraw();
  } else {
    AtomicFetchAdd(&current->visited_nnrate, current->child[best].nnrate);
  }

  return best;
}

bool BackupTrajectory(const std::vector<trajectory_t>& trajectory,
                      float leaf_parent_value, float leaf_moves_left) {
  if (!std::isfinite(leaf_parent_value) ||
      !std::isfinite(leaf_moves_left)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite backup input\""
            << " value=" << leaf_parent_value
            << " moves_left=" << leaf_moves_left
            << " trajectory=" << trajectory.size();
    jhbr2::nn_diagnostics::LogOnce("mcts_backup_input", details.str());
    return false;
  }

  // Validate the entire path before updating any accumulator, so containment
  // cannot leave a partially backed-up trajectory.
  size_t step = 0;
  for (const auto& item : trajectory) {
    const auto& child = item.parent->child[item.child_idx];
    const float parent_win =
        item.parent->win.load(std::memory_order_acquire);
    const float parent_m =
        item.parent->sum_m.load(std::memory_order_acquire);
    const float child_win = child.win.load(std::memory_order_acquire);
    const float child_m = child.sum_m.load(std::memory_order_acquire);
    if (!std::isfinite(parent_win) || !std::isfinite(parent_m) ||
        !std::isfinite(child_win) ||
        !std::isfinite(child_m) || !std::isfinite(child.nnrate)) {
      std::ostringstream details;
      details << "backend=mcts reason=\"non-finite tree accumulator\""
              << " step=" << step << " trajectory=" << trajectory.size()
              << " parent_win=" << parent_win
              << " parent_sum_m=" << parent_m
              << " child_win=" << child_win << " child_sum_m=" << child_m
              << " child_prior=" << child.nnrate;
      jhbr2::nn_diagnostics::LogOnce("mcts_backup_tree", details.str());
      return false;
    }
    ++step;
  }

  float value = leaf_parent_value;
  float moves_left = leaf_moves_left;
  for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
    UpdateResult(&it->parent->child[it->child_idx], value, moves_left,
                 it->parent);
    value = 1.0f - value;
    moves_left += 1.0f;
  }
  return true;
}

}  // namespace dlshogi_mcts
