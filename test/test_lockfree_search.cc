// Multi-threaded MCTS search on deterministic random networks:
//   * concurrent search returns legal moves/PVs and reuses the tree
//   * adaptive deadline and external stop are honoured
//   * root mate worker stops MCTS
//   * deferred expansion: fewer expansions than evaluations
//   * single-thread search is deterministic
//   * leaf terminal detection: a mate-in-one is found without the mate solver

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mate/bns.h"
#include "mcts/uct_search.h"
#include "nnue/random_net.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"
#include "shogi/types.h"

namespace {

using dlshogi_mcts::Search;
using dlshogi_mcts::SearchConfig;
using dlshogi_mcts::SearchResult;
using lczero::Move;
using lczero::ShogiBoard;

int failures = 0;
jhbr5::nnue::NetworkSet g_nets;

void Check(const char* name, bool condition) {
  if (!condition) {
    std::printf("  FAIL  %s\n", name);
    ++failures;
  }
}

bool IsLegalMove(ShogiBoard board, Move move) {
  const auto legal = board.GenerateLegalMoves();
  return std::find(legal.begin(), legal.end(), move) != legal.end();
}

bool IsLegalPv(ShogiBoard board, const std::vector<Move>& pv) {
  for (Move move : pv) {
    if (!IsLegalMove(board, move)) return false;
    board.DoMove(move);
  }
  return true;
}

SearchConfig BaseConfig(int threads) {
  SearchConfig config;
  config.threads = threads;
  config.max_time = 0.0f;
  config.leaf_mate_depth = 0;
  config.root_mate_depth = 0;
  config.eval_cache_mb = 4;
  config.info_interval_ms = 0;
  config.info_callback = [](const dlshogi_mcts::SearchInfo&) {};
  return config;
}

void CheckResult(const char* prefix, const SearchResult& result,
                 const ShogiBoard& board, int minimum_nodes) {
  const std::string best_name = std::string(prefix) + " returns a legal move";
  const std::string nodes_name =
      std::string(prefix) + " completes the requested search";
  const std::string cp_name = std::string(prefix) + " returns a bounded CP";
  const std::string pv_name = std::string(prefix) + " returns a legal PV";
  Check(best_name.c_str(),
        !result.best_move.is_null() && IsLegalMove(board, result.best_move));
  Check(nodes_name.c_str(), result.nodes >= minimum_nodes);
  Check(cp_name.c_str(),
        result.score_cp >= -100000 && result.score_cp <= 100000);
  Check(pv_name.c_str(), IsLegalPv(board, result.pv));
}

void TestConcurrentSearchAndReuse() {
  ShogiBoard start;
  start.SetStartPos();
  const uint64_t starting_key = start.Hash();

  SearchConfig config = BaseConfig(8);
  config.max_nodes = 12000;

  Search search(&g_nets, config);
  const SearchResult first = search.Run(start, starting_key, {});
  CheckResult("threaded search", first, start, config.max_nodes);
  Check("first search starts a fresh tree", !first.tree_reused);
  Check("deferred expansion: expansions < evaluations",
        first.expansions > 0 && first.expansions < first.evals + first.cache.hits);
  if (first.best_move.is_null() || !IsLegalMove(start, first.best_move)) return;

  ShogiBoard continued = start;
  continued.DoMove(first.best_move);
  const SearchResult reused =
      search.Run(continued, starting_key, {first.best_move});
  CheckResult("reused threaded search", reused, continued, config.max_nodes);
  Check("continued search reuses the selected subtree", reused.tree_reused);

  search.PrepareForNewGame();
  const SearchResult fresh = search.Run(start, starting_key, {});
  CheckResult("new-game search", fresh, start, config.max_nodes);
  Check("new-game search does not reuse the old tree", !fresh.tree_reused);
  Check("new-game search begins without retained visits",
        fresh.root_visits_before == 0);
}

void TestSingleThreadDeterminism() {
  ShogiBoard start;
  start.SetStartPos();
  SearchConfig config = BaseConfig(1);
  config.max_nodes = 3000;
  std::vector<Move> pv1, pv2;
  int cp1 = 0, cp2 = 0;
  {
    Search search(&g_nets, config);
    const SearchResult r = search.Run(start, start.Hash(), {});
    pv1 = r.pv;
    cp1 = r.score_cp;
  }
  {
    Search search(&g_nets, config);
    const SearchResult r = search.Run(start, start.Hash(), {});
    pv2 = r.pv;
    cp2 = r.score_cp;
  }
  Check("single-thread search is deterministic", pv1 == pv2 && cp1 == cp2 && !pv1.empty());
}

void TestOptionalMontyFeatures() {
  ShogiBoard start;
  start.SetStartPos();
  SearchConfig config = BaseConfig(2);
  config.max_nodes = 2000;
  config.use_butterfly = true;
  config.use_policy_temperature = true;
  config.draw_scale = 1.0f;
  Search search(&g_nets, config);
  const SearchResult r = search.Run(start, start.Hash(), {});
  CheckResult("butterfly+temperature search", r, start, config.max_nodes);
}

void TestTreeMemoryLimitStops() {
  ShogiBoard start;
  start.SetStartPos();
  SearchConfig config = BaseConfig(2);
  config.max_nodes = 1000000;
  config.tree_memory_mb = 1;
  Search search(&g_nets, config);
  const SearchResult r = search.Run(start, start.Hash(), {});
  Check("tree memory limit stops the search early", r.nodes < config.max_nodes);
  Check("tree memory limit is reported",
        r.time_decision.reason == jhbr2::TimeStopReason::kTreeFull);
}

void TestLeafTerminalDetection() {
  // Black to move, G*5b is mate (white king 5a, black gold 5c... use the
  // classic two-gold mate): king on 5a, black gold on 5c, gold in hand.
  ShogiBoard pos;
  Check("mate position parses", pos.SetFromSfen("4k4/9/4G4/9/9/9/9/9/8K b G 1"));
  SearchConfig config = BaseConfig(1);
  config.max_nodes = 4000;
  Search search(&g_nets, config);
  const SearchResult r = search.Run(pos, pos.Hash(), {});
  Check("search proves mate without the mate solvers",
        !r.best_move.is_null() && r.best_move.ToString() == "G*5b" && r.score_cp > 3000);
}

void TestAdaptiveDeadlineStopsWorkers() {
  ShogiBoard start;
  start.SetStartPos();

  SearchConfig config = BaseConfig(2);
  config.max_nodes = 10000000;

  jhbr2::TimeBudget budget;
  budget.mode = jhbr2::TimeManagementMode::kOn;
  budget.earliest_stop_ms = 80;
  budget.target_stop_ms = 80;
  budget.latest_search_ms = 80;
  budget.response_deadline_ms = 200;
  budget.root_guard_deadline_ms = 200;
  budget.mcts_time_seconds = 0.08f;

  Search search(&g_nets, config);
  search.SetTimeBudget(budget);
  const SearchResult result = search.Run(start, start.Hash(), {});

  Check("adaptive deadline returns a legal move",
        !result.best_move.is_null() &&
            IsLegalMove(start, result.best_move));
  Check("adaptive deadline performs playouts", result.nodes > 0);
  Check("adaptive deadline does not run away",
        result.time_sec >= 0.05f && result.time_sec < 0.5f);
  Check("adaptive deadline reports a timed stop",
        result.time_decision.reason == jhbr2::TimeStopReason::kTargetStable ||
            result.time_decision.reason == jhbr2::TimeStopReason::kLatest);
}

void TestStartedHookKeepsEarlyStop() {
  ShogiBoard start;
  start.SetStartPos();

  SearchConfig config = BaseConfig(2);
  config.max_nodes = 1000000;

  Search search(&g_nets, config);
  int started_calls = 0;
  const SearchResult result = search.Run(
      start, start.Hash(), {}, Search::Clock::now(), [&] {
        ++started_calls;
        search.Stop();
      });

  Check("search-start hook runs exactly once", started_calls == 1);
  Check("search-start hook returns a legal fallback move",
        !result.best_move.is_null() &&
            IsLegalMove(start, result.best_move));
  Check("early helper stop is not erased", result.nodes < config.max_nodes);
  Check("early helper stop is reported",
        result.time_decision.reason == jhbr2::TimeStopReason::kExternal);
}

void TestRootMateWorkerStopsMcts() {
  ShogiBoard mate_position;
  Check("root-mate integration position parses",
        mate_position.SetFromSfen("4k4/9/4G4/9/9/9/9/9/8K b G 1"));
  if (mate_position.GenerateLegalMoves().size() <= 1) {
    Check("root-mate integration position has alternatives", false);
    return;
  }

  SearchConfig config = BaseConfig(2);
  config.max_nodes = 10000000;
  config.max_time = 2.0f;

  Search search(&g_nets, config);
  jhbr2::MateBnsSolver mate_solver(
      /*tt_mb=*/4, std::numeric_limits<size_t>::max());
  mate_solver.set_move_cache_mb(2);
  Move mate_move;
  std::atomic<bool> proved_mate{false};
  std::thread mate_thread;

  const SearchResult result = search.Run(
      mate_position, mate_position.Hash(), {}, Search::Clock::now(), [&] {
        mate_thread = std::thread([&] {
          mate_move = mate_solver.search(
              mate_position, std::numeric_limits<size_t>::max(),
              Search::Clock::now() + std::chrono::seconds(1));
          if (!mate_move.is_null() &&
              !jhbr2::MateBnsSolver::IsNoMate(mate_move)) {
            proved_mate.store(true, std::memory_order_release);
            search.Stop();
          }
        });
      });

  mate_solver.stop();
  if (mate_thread.joinable()) mate_thread.join();
  Check("root mate worker proves the test mate",
        proved_mate.load(std::memory_order_acquire));
  Check("proved root mate stops MCTS before its node cap",
        result.nodes < config.max_nodes);
  Check("mate-stopped MCTS retains a legal fallback",
        !result.best_move.is_null() &&
            IsLegalMove(mate_position, result.best_move));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string tmp = argc > 1 ? argv[1] : "/tmp";
  lczero::ShogiTables::Init();
  jhbr5::nnue::Init();
  const std::string vpath = tmp + "/jhbr5_search_value.nn";
  const std::string ppath = tmp + "/jhbr5_search_policy.nn";
  std::string err;
  if (!jhbr5::nnue::WriteRandomValueNet(vpath, 256, 3, &err) ||
      !jhbr5::nnue::WriteRandomPolicyNet(ppath, 512, true, 3, &err) ||
      !g_nets.Load(vpath, ppath, &err)) {
    std::printf("FAIL nets: %s\n", err.c_str());
    return 1;
  }
  TestConcurrentSearchAndReuse();
  TestSingleThreadDeterminism();
  TestOptionalMontyFeatures();
  TestTreeMemoryLimitStops();
  TestLeafTerminalDetection();
  TestAdaptiveDeadlineStopsWorkers();
  TestStartedHookKeepsEarlyStop();
  TestRootMateWorkerStopsMcts();
  std::remove(vpath.c_str());
  std::remove(ppath.c_str());
  std::printf("\n=== MCTS search (NNUE): %d failed ===\n", failures);
  return failures == 0 ? 0 : 1;
}
