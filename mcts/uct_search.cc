#include "mcts/uct_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include "mate/shallow_mate.h"
#include "mcts/nn_diagnostics.h"
#include "mcts/search_repetition.h"

namespace dlshogi_mcts {

using lczero::BLACK;
using lczero::Color;
using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;
using lczero::WHITE;

namespace {

constexpr uint64_t kMutexNum = 65536;
std::array<std::mutex, kMutexNum> g_position_mutexes;
std::mutex g_root_expand_mutex;

std::mutex& GetPositionMutex(const ShogiBoard* board) {
  return g_position_mutexes[board->Hash() & (kMutexNum - 1)];
}

void AddVirtualLoss(child_node_t* child, uct_node_t* current) {
  current->move_count.fetch_add(kVirtualLoss, std::memory_order_acq_rel);
  child->move_count.fetch_add(kVirtualLoss, std::memory_order_acq_rel);
}

void SubVirtualLoss(child_node_t* child, uct_node_t* current) {
  current->move_count.fetch_sub(kVirtualLoss, std::memory_order_acq_rel);
  child->move_count.fetch_sub(kVirtualLoss, std::memory_order_acq_rel);
}

float DrawValue(const SearchConfig& cfg, Color color) {
  return color == BLACK ? cfg.draw_value_black : cfg.draw_value_white;
}

// Monty's depth/Q dependent policy softmax temperature (MONTY_NOTES.md 3.6).
float PolicyTemperature(const SearchConfig& cfg, int depth, float q) {
  const float t = std::max(q - cfg.pst_win_threshold, 0.0f) /
                  std::max(1.0f - cfg.pst_win_threshold, 1e-6f);
  const float base = 1.0f - cfg.pst_base +
                     std::pow(std::max(static_cast<float>(depth) - cfg.pst_root, 1e-3f),
                              -cfg.pst_depth);
  return std::max(base + (cfg.pst_win_max - base) * t, 0.05f);
}

}  // namespace

struct visitor_t {
  std::vector<trajectory_t> trajectories;
  float value_win = 0.5f;       // leaf value, perspective of the leaf's side to move
  float terminal_value = 0.5f;  // perspective of the final edge's mover
  visitor_t() { trajectories.reserve(128); }
};

enum class PlayoutStatus { kTerminal, kEvaluated, kDiscarded };

class UCTSearcher {
 public:
  UCTSearcher(Search* owner, int thread_id)
      : owner_(owner), thread_id_(thread_id), evaluator_(*owner->nets_) {
    logits_.resize(lczero::kMaxLegalMoves);
  }

  void Run() { handle_ = std::thread([this] { ParallelUctSearch(); }); }
  void Join() {
    if (handle_.joinable()) handle_.join();
  }
  void ResetCaches() { evaluator_.Reset(); }

  // Writes policy priors for node->child[i] == moves[i].
  void ComputePriors(const ShogiBoard& board, uct_node_t* node,
                     const MoveList& moves, const child_node_t* parent_edge,
                     int depth);

 private:
  void ParallelUctSearch();
  PlayoutStatus UctSearch(ShogiBoard* board, child_node_t* parent,
                          uct_node_t* current, visitor_t& visitor, int depth);
  unsigned SelectMaxUcbChild(child_node_t* parent, uct_node_t* current);
  float EvaluateLeaf(const ShogiBoard& board);
  void UpdateButterfly(const visitor_t& visitor, float leaf_edge_value);

  Search* owner_;
  int thread_id_;
  jhbr5::nnue::Evaluator evaluator_;
  std::vector<float> logits_;
  std::thread handle_;
};

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

Search::Search(const jhbr5::nnue::NetworkSet* nets, const SearchConfig& config)
    : config_(config), nets_(nets), eval_cache_(config.eval_cache_mb) {
  const int threads = std::max(1, config_.threads);
  searchers_.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    searchers_.push_back(std::make_unique<UCTSearcher>(this, t));
  }
}

Search::~Search() {
  Stop();
  for (auto& s : searchers_) s->Join();
}

void Search::PrepareForNewGame() {
  Stop();
  for (auto& s : searchers_) s->Join();
  root_ = nullptr;
  tree_.DeallocateTree();
  tree_reused_ = false;
  root_visits_before_ = 0;
  playout_count_.store(0, std::memory_order_release);
  eval_cache_.Clear();
  butterfly_.Clear();
  for (auto& s : searchers_) s->ResetCaches();
}

bool Search::IsSearchActive() const {
  if (stop_.load(std::memory_order_acquire)) return false;
  if (adaptive_stop_.load(std::memory_order_acquire)) return false;
  if (tree_full_.load(std::memory_order_acquire)) return false;
  if (config_.max_nodes > 0 &&
      playout_count_.load(std::memory_order_acquire) >= config_.max_nodes) {
    return false;
  }
  if (config_.max_time > 0.0f &&
      timer_.ElapsedMs() >= static_cast<int>(config_.max_time * 1000.0f)) {
    return false;
  }
  return true;
}

void Search::ExpandRoot() {
  std::lock_guard<std::mutex> lk(g_root_expand_mutex);
  if (!root_->IsExpanded()) {
    MoveList moves = root_board_.GenerateLegalMoves();
    root_->ExpandNode(moves);
    searchers_[0]->ComputePriors(root_board_, root_, moves, nullptr, 1);
    root_->SetExpanded();
    expansions_.fetch_add(1, std::memory_order_relaxed);
  } else if (config_.use_policy_temperature || config_.use_butterfly) {
    // A reused root was expanded at a deeper level: relabel with root
    // temperature / current history (Monty relabel_policy).
    MoveList moves;
    for (int i = 0; i < root_->child_num; ++i) moves.push_back(root_->child[i].move);
    searchers_[0]->ComputePriors(root_board_, root_, moves, nullptr, 1);
  }
}

void Search::RejectRootMates() {
  if (config_.root_mate_depth <= 0 || !root_ || root_->child_num == 0) {
    return;
  }

  jhbr2::shallow_mate::SearchLimits limits;
  limits.stop = &stop_;
  if (config_.time_budget.mode == jhbr2::TimeManagementMode::kOn &&
      config_.time_budget.root_guard_deadline_ms > 0) {
    limits.deadline =
        timer_.start() + std::chrono::milliseconds(
                             config_.time_budget.root_guard_deadline_ms);
  }

  for (int attempt = 0; attempt < root_->child_num; ++attempt) {
    const unsigned idx = SelectBestChild(root_);
    auto& child = root_->child[idx];
    if (child.IsLose() || child.IsWin()) return;

    const auto undo = root_board_.DoMove(child.move);
    const auto probe = jhbr2::shallow_mate::ProbeMateWithin(
        root_board_, config_.root_mate_depth, &limits);
    root_board_.UndoMove(child.move, undo);
    if (probe == jhbr2::shallow_mate::ProbeResult::kCancelled) {
      root_guard_cancelled_ = true;
      return;
    }
    if (probe == jhbr2::shallow_mate::ProbeResult::kNoMate) {
      return;
    }
    child.SetWin();
  }
}

jhbr2::RootSearchSnapshot Search::CaptureRootSnapshot() const {
  jhbr2::RootSearchSnapshot snapshot;
  snapshot.elapsed_ms = timer_.ElapsedMs();
  snapshot.new_playouts = playout_count_.load(std::memory_order_acquire);
  if (!root_ || !root_->IsExpanded() || root_->child_num <= 0) {
    return snapshot;
  }

  snapshot.best_index = static_cast<int>(SelectBestChild(root_));
  int second = -1;
  int second_visits = std::numeric_limits<int>::min();
  float second_prior = -1.0f;
  for (int i = 0; i < root_->child_num; ++i) {
    if (i == snapshot.best_index || root_->child[i].IsWin()) continue;
    const int visits =
        root_->child[i].move_count.load(std::memory_order_acquire);
    if (visits > second_visits ||
        (visits == second_visits && root_->child[i].nnrate > second_prior)) {
      second = i;
      second_visits = visits;
      second_prior = root_->child[i].nnrate;
    }
  }
  snapshot.second_index = second;

  const auto read_child = [](const child_node_t& child,
                             std::int64_t* visits, float* q) {
    *visits =
        std::max(child.move_count.load(std::memory_order_acquire), 0);
    if (child.IsLose()) {
      *q = 1.0f;
    } else if (child.IsWin()) {
      *q = 0.0f;
    } else {
      const float wins = child.win.load(std::memory_order_acquire);
      *q = *visits > 0 ? wins / static_cast<float>(*visits) : 0.5f;
      if (!std::isfinite(*q)) *q = 0.5f;
    }
  };
  read_child(root_->child[snapshot.best_index], &snapshot.best_visits,
             &snapshot.best_q);
  snapshot.best_proven = root_->child[snapshot.best_index].IsLose() ||
                         root_->child[snapshot.best_index].IsWin();
  if (second >= 0) {
    read_child(root_->child[second], &snapshot.second_visits,
               &snapshot.second_q);
  }
  return snapshot;
}

void Search::MaybeManageTime(bool force) {
  if (config_.time_budget.mode == jhbr2::TimeManagementMode::kOff ||
      !config_.time_budget.HasAdaptiveDeadline()) {
    return;
  }

  const int elapsed_ms = timer_.ElapsedMs();
  int previous = last_time_check_ms_.load(std::memory_order_relaxed);
  if (!force && elapsed_ms - previous < 20) return;
  if (!last_time_check_ms_.compare_exchange_strong(
          previous, elapsed_ms, std::memory_order_acq_rel,
          std::memory_order_relaxed) &&
      !force) {
    return;
  }
  bool expected = false;
  if (!time_check_busy_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_relaxed)) {
    return;
  }

  const auto decision = time_controller_.Update(CaptureRootSnapshot());
  if (config_.time_budget.mode == jhbr2::TimeManagementMode::kOn &&
      decision.should_stop) {
    adaptive_stop_.store(true, std::memory_order_release);
  }
  time_check_busy_.store(false, std::memory_order_release);
}

unsigned Search::SelectBestChild(const uct_node_t* node) const {
  unsigned best = 0;
  int best_visits = std::numeric_limits<int>::min();
  float best_prior = -1.0f;
  for (int i = 0; i < node->child_num; ++i) {
    const auto& child = node->child[i];
    const int visits = child.move_count.load(std::memory_order_acquire);
    if (child.IsLose()) return i;
    if (child.IsWin()) continue;
    if (visits > best_visits ||
        (visits == best_visits && child.nnrate > best_prior)) {
      best_visits = visits;
      best_prior = child.nnrate;
      best = static_cast<unsigned>(i);
    }
  }
  return best;
}

SearchResult Search::Run(ShogiBoard board, uint64_t starting_pos_key,
                         const std::vector<Move>& moves,
                         Clock::time_point move_start,
                         SearchStartedCallback on_search_started) {
  stop_.store(false, std::memory_order_release);
  adaptive_stop_.store(false, std::memory_order_release);
  tree_full_.store(false, std::memory_order_release);
  playout_count_.store(0, std::memory_order_release);
  evals_.store(0, std::memory_order_release);
  expansions_.store(0, std::memory_order_release);
  eval_cache_.ResetStats();
  timer_.Restart(move_start);
  time_controller_.Reset(config_.time_budget,
                         static_cast<int>(searchers_.size()));
  last_time_check_ms_.store(-1000000, std::memory_order_release);
  time_check_busy_.store(false, std::memory_order_release);
  root_guard_cancelled_ = false;
  last_info_ms_ = 0;
  root_board_ = std::move(board);
  tree_reused_ = tree_.ResetToPosition(starting_pos_key, moves);
  root_ = tree_.GetCurrentHead();
  root_visits_before_ =
      std::max(0, root_->move_count.load(std::memory_order_acquire));

  if (on_search_started) on_search_started();

  auto root_legal = root_board_.GenerateLegalMoves();
  if (root_legal.empty()) return BuildResult();
  if (root_legal.size() == 1) {
    SearchResult result;
    result.best_move = root_legal[0];
    result.tree_reused = tree_reused_;
    result.root_visits_before = root_visits_before_;
    result.time_sec = timer_.ElapsedMs() / 1000.0f;
    result.cache = eval_cache_.GetStats();
    result.time_budget = config_.time_budget;
    result.time_decision = time_controller_.decision();
    return result;
  }

  ExpandRoot();
  for (auto& s : searchers_) s->Run();
  for (auto& s : searchers_) s->Join();
  MaybeManageTime(true);
  RejectRootMates();
  MaybeOutputInfo();
  return BuildResult();
}

// ---------------------------------------------------------------------------
// UCTSearcher
// ---------------------------------------------------------------------------

unsigned UCTSearcher::SelectMaxUcbChild(child_node_t* parent,
                                        uct_node_t* current) {
  const auto& cfg = owner_->config_;
  const bool is_root = parent == nullptr;
  PuctParameters params;
  params.c_init = is_root ? cfg.c_init_root : cfg.c_init;
  params.c_base = is_root ? cfg.c_base_root : cfg.c_base;
  params.fpu_reduction =
      is_root ? cfg.c_fpu_reduction_root : cfg.c_fpu_reduction;
  return SelectPuctChild(parent, current, params);
}

float UCTSearcher::EvaluateLeaf(const ShogiBoard& board) {
  const auto& cfg = owner_->config_;
  float win = 0.0f, draw = 0.0f;
  const uint64_t key = board.Hash();
  if (!owner_->eval_cache_.Probe(key, &win, &draw)) {
    jhbr5::nnue::Wdl wdl = evaluator_.Evaluate(board);
    owner_->evals_.fetch_add(1, std::memory_order_relaxed);
    if (cfg.draw_scale != 0.0f || cfg.draw_quadratic != 0.0f) {
      // Monty "sharpness": inflate the draw share (MONTY_NOTES.md 2.4).
      const float adj = std::max(
          wdl.draw * cfg.draw_scale + wdl.draw * wdl.draw * cfg.draw_quadratic, 0.0f);
      const float sum = wdl.win + wdl.draw + adj + wdl.loss;
      wdl.win /= sum;
      wdl.draw = (wdl.draw + adj) / sum;
      wdl.loss /= sum;
    }
    win = wdl.win;
    draw = wdl.draw;
    owner_->eval_cache_.Store(key, win, draw);
  }
  return win + 0.5f * draw;
}

void UCTSearcher::ComputePriors(const ShogiBoard& board, uct_node_t* node,
                                const MoveList& moves,
                                const child_node_t* parent_edge, int depth) {
  const auto& cfg = owner_->config_;
  const int n = moves.size();
  if (n == 0) return;
  float* logits = logits_.data();
  evaluator_.Policy(board, moves, logits);

  if (cfg.use_butterfly) {
    const Color stm = board.side_to_move();
    for (int i = 0; i < n; ++i) {
      logits[i] += owner_->butterfly_.Bonus(stm, moves[i], cfg.butterfly_divisor);
    }
  }

  float temperature = 1.0f;
  if (cfg.use_policy_temperature) {
    float q;
    if (parent_edge) {
      const int visits = std::max(1, parent_edge->move_count.load(std::memory_order_acquire));
      q = parent_edge->win.load(std::memory_order_acquire) / static_cast<float>(visits);
    } else {
      const int visits = node->move_count.load(std::memory_order_acquire);
      q = visits > 0 ? 1.0f - node->win.load(std::memory_order_acquire) / static_cast<float>(visits)
                     : 0.5f;
    }
    temperature = PolicyTemperature(cfg, depth, std::clamp(q, 0.0f, 1.0f));
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(logits[i])) logits[i] = 0.0f;
    max_logit = std::max(max_logit, logits[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    logits[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += logits[i];
  }
  const float inv = sum > 0.0f && std::isfinite(sum) ? 1.0f / sum : 0.0f;
  for (int i = 0; i < n; ++i) {
    node->child[i].nnrate = inv > 0.0f ? logits[i] * inv : 1.0f / static_cast<float>(n);
  }
  node->visited_nnrate.store(0.0f, std::memory_order_release);
}

PlayoutStatus UCTSearcher::UctSearch(ShogiBoard* board, child_node_t* parent,
                                     uct_node_t* current, visitor_t& visitor,
                                     int depth) {
  const auto& cfg = owner_->config_;
  const float parent_draw_value = DrawValue(cfg, ~board->side_to_move());

  if (TryGetProvenEdgeValue(parent, parent_draw_value,
                            &visitor.terminal_value)) {
    return PlayoutStatus::kTerminal;
  }
  if (board->CanDeclareWin()) {
    visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
    return PlayoutStatus::kTerminal;
  }

  switch (GetSearchRepetitionResult(*board, parent == nullptr)) {
    case ShogiBoard::RepetitionResult::kLoss:
      visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kWin);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kWin:
      visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kDraw:
      visitor.terminal_value =
          ResolveTerminalEdge(parent, EdgeOutcome::kDraw, parent_draw_value);
      return PlayoutStatus::kTerminal;
    case ShogiBoard::RepetitionResult::kNone:
      break;
  }

  if (board->ply() > cfg.max_moves_to_draw) {
    const EdgeOutcome outcome = board->GenerateLegalMoves().empty()
                                    ? EdgeOutcome::kWin
                                    : EdgeOutcome::kDraw;
    visitor.terminal_value =
        ResolveTerminalEdge(parent, outcome, parent_draw_value);
    return PlayoutStatus::kTerminal;
  }

  // First visit: terminal detection and value evaluation.
  if (!current->IsEvaled()) {
    std::lock_guard<std::mutex> lk(GetPositionMutex(board));
    if (!current->IsEvaled()) {
      if (cfg.leaf_mate_depth > 0 &&
          jhbr2::shallow_mate::HasMateWithin(*board, cfg.leaf_mate_depth)) {
        visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kLoss);
        return PlayoutStatus::kTerminal;
      }
      const bool has_move = board->InCheck()
                                ? !board->GenerateEvasionMoves().empty()
                                : !board->GenerateLegalMoves().empty();
      if (!has_move) {
        current->child_num = 0;
        current->SetExpanded();
        visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kWin);
        return PlayoutStatus::kTerminal;
      }
      visitor.value_win = EvaluateLeaf(*board);
      current->SetEvaled();
      return PlayoutStatus::kEvaluated;
    }
  }

  // Second visit: move generation and policy priors, then keep descending.
  if (!current->IsExpanded()) {
    std::lock_guard<std::mutex> lk(GetPositionMutex(board));
    if (!current->IsExpanded()) {
      MoveList moves = board->GenerateLegalMoves();
      if (moves.empty()) {
        current->child_num = 0;
        current->SetExpanded();
      } else {
        current->ExpandNode(moves);
        ComputePriors(*board, current, moves, parent, depth);
        current->SetExpanded();
        owner_->expansions_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  if (current->child_num == 0) {
    visitor.terminal_value = ResolveTerminalEdge(parent, EdgeOutcome::kWin);
    return PlayoutStatus::kTerminal;
  }

  const unsigned next = SelectMaxUcbChild(parent, current);
  child_node_t* edge = &current->child[next];
  AddVirtualLoss(edge, current);
  visitor.trajectories.push_back({current, next});
  uct_node_t* next_node = current->child_nodes[next].GetOrCreate();

  const Move next_move = edge->move;
  const auto undo = board->DoMove(next_move);
  const PlayoutStatus status =
      UctSearch(board, edge, next_node, visitor, depth + 1);
  board->UndoMove(next_move, undo);
  return status;
}

void UCTSearcher::UpdateButterfly(const visitor_t& visitor,
                                  float leaf_edge_value) {
  const auto& cfg = owner_->config_;
  const Color root_stm = owner_->root_board_.side_to_move();
  float value = leaf_edge_value;
  const auto& path = visitor.trajectories;
  for (size_t i = path.size(); i-- > 0;) {
    const child_node_t& edge = path[i].parent->child[path[i].child_idx];
    const Color side = (i % 2 == 0) ? root_stm : ~root_stm;
    if (!edge.IsWin() && !edge.IsLose() && !edge.IsDraw()) {
      owner_->butterfly_.Update(side, edge.move, value, cfg.butterfly_reduction);
    }
    value = 1.0f - value;
  }
}

void UCTSearcher::ParallelUctSearch() {
  auto* root = owner_->root_;
  if (!owner_->IsSearchActive() || !root) return;

  ShogiBoard board = owner_->root_board_;
  visitor_t visitor;
  const size_t tree_limit_bytes = owner_->config_.tree_memory_mb * 1024 * 1024;
  int iterations = 0;

  auto unwind = [](const std::vector<trajectory_t>& path) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      SubVirtualLoss(&it->parent->child[it->child_idx], it->parent);
    }
  };

  while (owner_->IsSearchActive()) {
    visitor.trajectories.clear();
    visitor.value_win = 0.5f;
    visitor.terminal_value = 0.5f;

    const PlayoutStatus status = UctSearch(&board, nullptr, root, visitor, 1);
    if (status == PlayoutStatus::kDiscarded) {
      unwind(visitor.trajectories);
      continue;
    }
    owner_->playout_count_.fetch_add(1, std::memory_order_acq_rel);

    // Value for the player who traversed the last edge.
    const float edge_value = status == PlayoutStatus::kTerminal
                                 ? visitor.terminal_value
                                 : 1.0f - visitor.value_win;
    if (!BackupTrajectory(visitor.trajectories, edge_value)) {
      unwind(visitor.trajectories);
      owner_->Stop();
      break;
    }
    if (owner_->config_.use_butterfly) UpdateButterfly(visitor, edge_value);

    if ((++iterations & 63) == 0) {
      if (tree_limit_bytes > 0 && TreeMemory::Bytes().load(std::memory_order_relaxed) > tree_limit_bytes) {
        owner_->tree_full_.store(true, std::memory_order_release);
      }
      owner_->MaybeOutputInfo();
      owner_->MaybeManageTime();
    }
  }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

int Search::QToCentipawns(float win_rate) const {
  if (!std::isfinite(win_rate)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite root win rate\""
            << " win_rate=" << win_rate;
    jhbr2::nn_diagnostics::LogOnce("cp_conversion", details.str());
    return 0;
  }
  win_rate = std::clamp(win_rate, 0.001f, 0.999f);
  return static_cast<int>(-std::log(1.0f / win_rate - 1.0f) * 756.0f);
}

std::vector<Move> Search::GetPV() const {
  std::vector<Move> pv;
  const uct_node_t* node = root_;
  while (node && node->IsExpanded() && node->child_num > 0 && node->child) {
    const unsigned idx = SelectBestChild(node);
    if (node->child[idx].move_count.load(std::memory_order_acquire) <= 0 &&
        !node->child[idx].IsLose()) {
      break;
    }
    pv.push_back(node->child[idx].move);
    if (!node->child_nodes) break;
    node = node->child_nodes[idx].get();
    if (pv.size() > 256) break;
  }
  return pv;
}

void Search::MaybeOutputInfo() {
  if (!config_.info_callback) return;
  const int elapsed = timer_.ElapsedMs();
  std::lock_guard<std::mutex> lk(info_mutex_);
  if (elapsed - last_info_ms_ < config_.info_interval_ms) {
    return;
  }
  last_info_ms_ = elapsed;
  const int nodes = playout_count_.load(std::memory_order_acquire);
  SearchInfo info;
  info.nodes = nodes;
  info.time_ms = elapsed;
  info.nps = elapsed > 0 ? static_cast<int>(nodes * 1000LL / elapsed) : 0;
  info.pv = GetPV();
  info.depth = static_cast<int>(info.pv.size());
  info.cache = eval_cache_.GetStats();
  if (root_ && root_->child_num > 0) {
    const unsigned best = SelectBestChild(root_);
    const auto& ch = root_->child[best];
    const int n = ch.move_count.load(std::memory_order_acquire);
    const float wp = ch.IsLose() ? 1.0f
                     : ch.IsWin() ? 0.0f
                     : n > 0 ? ch.win.load(std::memory_order_acquire) / n
                             : 0.5f;
    info.score_cp = QToCentipawns(wp);
  }
  config_.info_callback(info);
}

SearchResult Search::BuildResult() const {
  SearchResult result;
  result.tree_reused = tree_reused_;
  result.root_visits_before = root_visits_before_;
  result.nodes = playout_count_.load(std::memory_order_acquire);
  result.time_sec = timer_.ElapsedMs() / 1000.0f;
  result.nps = result.time_sec > 0.001f ? result.nodes / result.time_sec : 0.0f;
  result.evals = evals_.load(std::memory_order_acquire);
  result.expansions = expansions_.load(std::memory_order_acquire);
  result.cache = eval_cache_.GetStats();
  result.time_budget = config_.time_budget;
  result.time_decision = time_controller_.decision();
  result.root_guard_cancelled = root_guard_cancelled_;
  if (result.time_decision.reason == jhbr2::TimeStopReason::kNone) {
    if (tree_full_.load(std::memory_order_acquire)) {
      result.time_decision.reason = jhbr2::TimeStopReason::kTreeFull;
    } else if (config_.max_nodes > 0 && result.nodes >= config_.max_nodes) {
      result.time_decision.reason = jhbr2::TimeStopReason::kNodeLimit;
    } else if (stop_.load(std::memory_order_acquire)) {
      result.time_decision.reason = jhbr2::TimeStopReason::kExternal;
    } else if (config_.max_time > 0.0f &&
               timer_.ElapsedMs() >=
                   static_cast<int>(config_.max_time * 1000.0f)) {
      result.time_decision.reason =
          config_.time_budget.mode == jhbr2::TimeManagementMode::kOn
              ? jhbr2::TimeStopReason::kLatest
              : jhbr2::TimeStopReason::kLegacyLimit;
    }
  }
  if (!root_ || root_->child_num == 0) return result;

  const unsigned best = SelectBestChild(root_);
  const auto& child = root_->child[best];
  result.best_move = child.move;
  const int n = child.move_count.load(std::memory_order_acquire);
  const float wp = child.IsLose() ? 1.0f
                   : child.IsWin() ? 0.0f
                   : n > 0 ? child.win.load(std::memory_order_acquire) / n
                           : 0.5f;
  const float safe_wp = std::isfinite(wp) ? wp : 0.5f;
  if (!std::isfinite(wp)) {
    std::ostringstream details;
    details << "backend=mcts reason=\"non-finite final root win rate\""
            << " visits=" << n << " win="
            << child.win.load(std::memory_order_acquire);
    jhbr2::nn_diagnostics::LogOnce("build_result", details.str());
  }
  if (safe_wp < config_.resign_threshold) result.best_move = Move();
  result.score_cp = QToCentipawns(safe_wp);
  result.pv = GetPV();
  return result;
}

}  // namespace dlshogi_mcts
