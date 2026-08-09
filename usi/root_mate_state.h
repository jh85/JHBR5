/*
  JHBR3 Shogi Engine — Root mate solver state for parallel MCTS.

  Holds the BNS or tree df-pn solver that runs alongside the main MCTS search
  to prove a mate at the root, plus the synchronization primitives used to
  start/stop it safely.
*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "mate/bns.h"
#include "mate/dfpn.h"
#include "shogi/board.h"

namespace jhbr2 {

// The legacy tree df-pn keeps a fixed cap because its node pool grows linearly
// with this value.
inline constexpr size_t kTreeDfpnMaxNodes = 2'000'000;

struct RootMateState {
  std::unique_ptr<MateDfpnSolver> tree;
  std::unique_ptr<MateBnsSolver> bns;
  const bool use_bns;
  const size_t nodes_limit;
  std::atomic<bool> done{false};
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> stopped_mcts{false};
  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool started = false;
  lczero::Move mate_move;
  lczero::ShogiBoard board;
  std::int64_t elapsed_ms = 0;

  RootMateState(bool use_bns_solver, const lczero::ShogiBoard& b)
      : use_bns(use_bns_solver),
        nodes_limit(use_bns_solver
                        ? std::numeric_limits<size_t>::max()
                        : kTreeDfpnMaxNodes),
        board(b) {
    if (use_bns) {
      // Keep the two hot search tables inside the local cache slice. Their
      // fixed allocation does not grow with nodes_limit.
      bns = std::make_unique<MateBnsSolver>(/*tt_mb=*/4, nodes_limit);
      bns->set_move_cache_mb(2);
    } else {
      tree = std::make_unique<MateDfpnSolver>(nodes_limit);
    }
  }

  void Start() {
    {
      std::lock_guard<std::mutex> lock(start_mutex);
      started = true;
    }
    start_cv.notify_all();
  }

  bool WaitForStart() {
    std::unique_lock<std::mutex> lock(start_mutex);
    start_cv.wait(lock, [this] {
      return started || stop_requested.load(std::memory_order_acquire);
    });
    return started && !stop_requested.load(std::memory_order_acquire);
  }

  lczero::Move Search(MateDfpnSolver::Deadline deadline) {
    return bns ? bns->search(board, nodes_limit, deadline)
               : tree->search(board, nodes_limit, deadline);
  }

  void Stop() {
    stop_requested.store(true, std::memory_order_release);
    if (bns)
      bns->stop();
    else
      tree->stop();
    start_cv.notify_all();
  }

  size_t NodesSearched() const {
    return bns ? bns->get_nodes_searched() : tree->get_nodes_searched();
  }

  std::vector<lczero::Move> Pv() const {
    return bns ? bns->get_pv() : tree->get_pv();
  }

  const char* SolverName() const { return use_bns ? "bns" : "dfpn"; }
};

}  // namespace jhbr2
