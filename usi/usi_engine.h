/*
  JHBR3 Shogi Engine — USI Protocol Handler

  Implements the Universal Shogi Interface protocol for communication
  with Shogi GUIs (ShogiGUI, Shogidokoro, etc.) and tournament software.

  Reference: http://shogidokoro.starfree.jp/usi.html
*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nnue/evaluator.h"
#include "book/opening_book.h"
#include "mcts/uct_search.h"
#include "shogi/board.h"
#include "usi/root_mate_state.h"
#include "usi/time_manager.h"

namespace jhbr2 {

class USIEngine {
 public:
  static constexpr const char* ENGINE_NAME = "JHBR5";
  static constexpr const char* ENGINE_AUTHOR = "JHBR5 Team";

  USIEngine();

  // Main loop: read USI commands from stdin, write responses to stdout.
  void Run();

 private:
  // --- Command handlers ---
  void CmdUsi();
  void CmdIsReady();
  void CmdSetOption(const std::vector<std::string>& parts);
  void CmdUsiNewGame();
  void CmdPosition(const std::vector<std::string>& parts);
  void CmdGo(const std::vector<std::string>& parts);
  void CmdGoMate(const std::vector<std::string>& parts);
  void CmdStop();
  void CmdGameOver(const std::vector<std::string>& parts);
  void CmdDebug();
  void CmdBench(const std::vector<std::string>& parts);

  // --- Helpers ---
  void Send(const std::string& msg);
  void Log(const std::string& msg);
  void EnsureSearch();
  // Loads the networks named by the ValueNet/PolicyNet options; returns
  // false (and logs) on failure.
  bool EnsureNetworks();

  // Option parsing. Each registered lambda implements one setoption name and
  // may throw std::invalid_argument for malformed values.
  enum class OptionSetResult {
    kSetAndLog,     // Value was set; emit the generic "Set name = value" log.
    kAlreadyLogged  // Handler already logged; skip the generic log.
  };
  void RegisterOptionParsers();
  std::unordered_map<std::string,
                     std::function<OptionSetResult(const std::string&,
                                                   const std::string&)>>
      option_parsers_;

  // "go" command decomposition.
  struct GoParameters {
    TimeControl time_control;
    int nodes_limit = 0;
    bool infinite = false;
    bool ponder = false;
  };
  GoParameters ParseGoParameters(
      const std::vector<std::string>& parts) const;
  std::optional<std::string> ProbeOpeningBook();
  struct RootMateLaunch {
    std::shared_ptr<RootMateState> state;
    std::thread search_thread;
    std::thread watchdog_thread;
  };
  RootMateLaunch LaunchRootMateSearch(
      const TimeBudget& time_budget,
      std::chrono::steady_clock::time_point move_start_time,
      dlshogi_mcts::Search* mcts_search,
      std::mutex& watchdog_mutex,
      std::condition_variable& watchdog_cv,
      bool& search_done,
      std::atomic<bool>& watchdog_fired);
  std::string FormatTimeBudgetForLog(const TimeBudget& budget,
                                     bool compact) const;
  void LogTimeResult(const dlshogi_mcts::SearchResult& result);
  void LogTimeResponse(std::chrono::steady_clock::time_point move_start_time,
                       const TimeBudget& time_budget,
                       const RootMateState& state);
  void LogRootMateResult(const RootMateState& state,
                         bool finished_before_stop,
                         bool watchdog_fired,
                         std::int64_t join_ms);
  void SelectAndReportBestMove(const dlshogi_mcts::SearchResult& result,
                               const RootMateState& root_mate);

  // --- Members ---
  lczero::ShogiBoard board_;
  std::unique_ptr<jhbr5::nnue::NetworkSet> nets_;
  std::unique_ptr<dlshogi_mcts::Search> search_;
  dlshogi_mcts::SearchConfig search_config_;
  uint64_t position_start_key_ = 0;
  std::vector<lczero::Move> position_moves_;
  bool new_game_prepared_ = false;

  // Options
  std::string value_net_path_ = "nets/value.nn";
  std::string policy_net_path_ = "nets/policy.nn";
  bool nets_are_random_ = false;
  bool root_dist_output_ = false;
  int max_nodes_ = 100000000;  // CPU search is normally time limited
  bool root_mate_solver_bns_ = true;
  int max_move_time_ms_ = 0;
  int max_move_time_1m_ms_ = 0;
  TimeManagementMode time_management_mode_ = TimeManagementMode::kShadow;
  int move_overhead_ms_ = 100;
  int time_max_extension_percent_ = 175;
  bool time_debug_ = false;
  std::string book_path_;
  std::string gote_exit_book_path_ = "user_book1_gote_exit.ybb";
  bool use_gote_exit_book_ = false;
  bool books_dirty_ = true;
  OpeningBook book_;
  OpeningBook gote_exit_book_;
};

}  // namespace jhbr2
