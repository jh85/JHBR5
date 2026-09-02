#include "mcts/search_primitives.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "mcts/nn_diagnostics.h"

namespace dlshogi_mcts {
namespace {

void UpdateResult(child_node_t* child, float result, uct_node_t* current) {
  AtomicFetchAdd(&current->win, result);
  if constexpr (kVirtualLoss != 1) {
    current->move_count.fetch_add(1 - kVirtualLoss,
                                  std::memory_order_acq_rel);
  }
  AtomicFetchAdd(&child->win, result);
  if constexpr (kVirtualLoss != 1) {
    child->move_count.fetch_add(1 - kVirtualLoss,
                                std::memory_order_acq_rel);
  }
}

}  // namespace

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

    const float score = q + c * u * child.nnrate;
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
                      float leaf_parent_value) {
  if (!std::isfinite(leaf_parent_value)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite backup input\""
            << " value=" << leaf_parent_value
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
    const float child_win = child.win.load(std::memory_order_acquire);
    if (!std::isfinite(parent_win) || !std::isfinite(child_win) ||
        !std::isfinite(child.nnrate)) {
      std::ostringstream details;
      details << "backend=mcts reason=\"non-finite tree accumulator\""
              << " step=" << step << " trajectory=" << trajectory.size()
              << " parent_win=" << parent_win << " child_win=" << child_win
              << " child_prior=" << child.nnrate;
      jhbr2::nn_diagnostics::LogOnce("mcts_backup_tree", details.str());
      return false;
    }
    ++step;
  }

  float value = leaf_parent_value;
  for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
    UpdateResult(&it->parent->child[it->child_idx], value, it->parent);
    value = 1.0f - value;
  }
  return true;
}

}  // namespace dlshogi_mcts
