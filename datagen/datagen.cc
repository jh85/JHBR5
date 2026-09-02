#include "datagen/datagen.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mcts/uct_search.h"
#include "nnue/evaluator.h"
#include "nnue/record.h"
#include "nnue/types.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

namespace jhbr5::datagen {

using lczero::BLACK;
using lczero::Color;
using lczero::Move;
using lczero::MoveList;
using lczero::ShogiBoard;
using lczero::WHITE;

namespace {

struct Options {
  std::string value_path = "nets/value.nn";
  std::string policy_path = "nets/policy.nn";
  std::string out_path = "datagen.rec";
  std::string book_path;
  int games = 100;
  int threads = 1;
  int nodes = 800;
  int random_plies = 8;
  float temperature = 1.0f;
  float temp_decay = 0.9f;
  float temp_min = 0.2f;
  float dirichlet_alpha = 0.15f;
  float dirichlet_epsilon = 0.25f;
  float resign_threshold = 0.03f;
  int resign_plies = 8;
  float no_resign_fraction = 0.1f;
  int max_ply = 320;
  int leaf_mate_depth = 3;
  int root_mate_depth = 0;
  int eval_cache_mb = 16;
  int tree_memory_mb = 512;
  uint64_t seed = 0;
  int report_every = 50;
};

void Usage() {
  std::fprintf(stderr,
      "usage: jhbr5 datagen --out FILE [--value V.nn] [--policy P.nn] [--games N]\n"
      "       [--threads T] [--nodes N] [--random-plies R] [--book sfens]\n"
      "       [--temperature T] [--temp-decay D] [--temp-min M]\n"
      "       [--dirichlet-alpha A] [--dirichlet-epsilon E]\n"
      "       [--resign-threshold Q] [--resign-plies P] [--no-resign-fraction F]\n"
      "       [--max-ply N] [--leaf-mate-depth D] [--root-mate-depth D]\n"
      "       [--eval-cache-mb M] [--tree-memory-mb M] [--seed S] [--report-every N]\n");
}

bool Parse(int argc, char** argv, Options* o) {
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string* v) {
      if (i + 1 >= argc) return false;
      *v = argv[++i];
      return true;
    };
    std::string v;
#define OPT_STR(name, field) if (a == name) { if (!next(&v)) return false; o->field = v; continue; }
#define OPT_INT(name, field) if (a == name) { if (!next(&v)) return false; o->field = std::atoi(v.c_str()); continue; }
#define OPT_FLT(name, field) if (a == name) { if (!next(&v)) return false; o->field = static_cast<float>(std::atof(v.c_str())); continue; }
    OPT_STR("--value", value_path)
    OPT_STR("--policy", policy_path)
    OPT_STR("--out", out_path)
    OPT_STR("--book", book_path)
    OPT_INT("--games", games)
    OPT_INT("--threads", threads)
    OPT_INT("--nodes", nodes)
    OPT_INT("--random-plies", random_plies)
    OPT_FLT("--temperature", temperature)
    OPT_FLT("--temp-decay", temp_decay)
    OPT_FLT("--temp-min", temp_min)
    OPT_FLT("--dirichlet-alpha", dirichlet_alpha)
    OPT_FLT("--dirichlet-epsilon", dirichlet_epsilon)
    OPT_FLT("--resign-threshold", resign_threshold)
    OPT_INT("--resign-plies", resign_plies)
    OPT_FLT("--no-resign-fraction", no_resign_fraction)
    OPT_INT("--max-ply", max_ply)
    OPT_INT("--leaf-mate-depth", leaf_mate_depth)
    OPT_INT("--root-mate-depth", root_mate_depth)
    OPT_INT("--eval-cache-mb", eval_cache_mb)
    OPT_INT("--tree-memory-mb", tree_memory_mb)
    if (a == "--seed") { if (!next(&v)) return false; o->seed = std::strtoull(v.c_str(), nullptr, 10); continue; }
    OPT_INT("--report-every", report_every)
#undef OPT_STR
#undef OPT_INT
#undef OPT_FLT
    std::fprintf(stderr, "unknown option %s\n", a.c_str());
    return false;
  }
  return true;
}

int ScoreToCp(float q) {
  q = std::clamp(q, 0.001f, 0.999f);
  return static_cast<int>(-std::log(1.0f / q - 1.0f) * 756.0f);
}

struct PositionRecord {
  std::string sfen;
  int score_cp;
  Move best;
  int ply;
  Color stm;
  std::vector<std::pair<Move, int>> dist;
};

struct Shared {
  Options opts;
  const nnue::NetworkSet* nets = nullptr;
  std::vector<std::string> book;
  std::mutex out_mutex;
  data::RecordWriter writer;
  std::atomic<int> games_started{0};
  std::atomic<int> games_done{0};
  std::atomic<uint64_t> positions{0};
  std::atomic<int> results[3]{{0}, {0}, {0}};  // black win, draw, white win
  std::atomic<uint64_t> playouts{0};
  std::chrono::steady_clock::time_point t0;
};

class Worker {
 public:
  Worker(Shared* shared, int id) : shared_(shared), id_(id), rng_(shared->opts.seed * 7919 + id + 1) {
    const Options& o = shared->opts;
    dlshogi_mcts::SearchConfig cfg;
    cfg.threads = 1;
    cfg.max_nodes = o.nodes;
    cfg.max_time = 0.0f;
    cfg.leaf_mate_depth = o.leaf_mate_depth;
    cfg.root_mate_depth = o.root_mate_depth;
    cfg.eval_cache_mb = static_cast<size_t>(o.eval_cache_mb);
    cfg.tree_memory_mb = static_cast<size_t>(o.tree_memory_mb);
    cfg.dirichlet_alpha = o.dirichlet_alpha;
    cfg.dirichlet_epsilon = o.dirichlet_epsilon;
    cfg.seed = rng_();
    cfg.info_callback = nullptr;
    search_ = std::make_unique<dlshogi_mcts::Search>(shared->nets, cfg);
  }

  void Run() {
    while (shared_->games_started.fetch_add(1) < shared_->opts.games) PlayGame();
  }

 private:
  bool SetupStart(ShogiBoard* board) {
    const Options& o = shared_->opts;
    if (!shared_->book.empty()) {
      const std::string& sfen = shared_->book[rng_() % shared_->book.size()];
      if (!board->SetFromSfen(sfen)) return false;
    } else {
      board->SetStartPos();
    }
    const int plies = o.random_plies > 0 ? static_cast<int>(rng_() % (o.random_plies + 1)) : 0;
    for (int i = 0; i < plies; ++i) {
      MoveList moves = board->GenerateLegalMoves();
      if (moves.empty() || board->CanDeclareWin()) return false;
      board->DoMove(moves[static_cast<int>(rng_() % moves.size())]);
    }
    return !board->GenerateLegalMoves().empty();
  }

  Move PickMove(const std::vector<std::pair<Move, int>>& dist, float temp) {
    if (dist.empty()) return Move();
    if (temp <= 0.0f) {
      return std::max_element(dist.begin(), dist.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
    std::vector<double> w(dist.size());
    double total = 0.0;
    for (size_t i = 0; i < dist.size(); ++i) {
      w[i] = std::pow(static_cast<double>(dist[i].second), 1.0 / temp);
      total += w[i];
    }
    if (total <= 0.0) return dist[rng_() % dist.size()].first;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng_);
    for (size_t i = 0; i < dist.size(); ++i) {
      r -= w[i];
      if (r <= 0.0) return dist[i].first;
    }
    return dist.back().first;
  }

  void PlayGame() {
    const Options& o = shared_->opts;
    ShogiBoard board;
    for (int attempt = 0; attempt < 16 && !SetupStart(&board); ++attempt) {
    }
    search_->PrepareForNewGame();
    const uint64_t start_key = board.Hash();
    std::vector<Move> moves_played;
    std::vector<PositionRecord> game;
    float temp = o.temperature;
    int low_q[2] = {0, 0};
    const bool no_resign = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < o.no_resign_fraction;
    int outcome = 0;  // +1 black wins, 0 draw, -1 white wins
    bool decided = false;
    uint64_t playouts = 0;

    while (!decided) {
      const Color stm = board.side_to_move();
      const int sign = stm == BLACK ? 1 : -1;
      MoveList legal = board.GenerateLegalMoves();
      if (legal.empty()) { outcome = -sign; break; }
      if (board.CanDeclareWin()) { outcome = sign; break; }
      switch (board.CheckRepetition()) {
        case ShogiBoard::RepetitionResult::kDraw:
          if (board.RepetitionCount() >= 3) { outcome = 0; decided = true; }
          break;
        case ShogiBoard::RepetitionResult::kWin: outcome = sign; decided = true; break;
        case ShogiBoard::RepetitionResult::kLoss: outcome = -sign; decided = true; break;
        case ShogiBoard::RepetitionResult::kNone: break;
      }
      if (decided) break;
      if (board.ply() >= o.max_ply) { outcome = 0; break; }

      const auto result = search_->Run(board, start_key, moves_played);
      playouts += static_cast<uint64_t>(std::max(result.nodes, 0));
      if (result.root_visits.empty()) { outcome = -sign; break; }

      PositionRecord rec;
      rec.sfen = board.ToSfen();
      rec.score_cp = ScoreToCp(result.root_q);
      rec.best = result.best_move.is_null() ? result.root_visits.front().first : result.best_move;
      rec.ply = board.ply() + 1;
      rec.stm = stm;
      rec.dist = result.root_visits;
      game.push_back(std::move(rec));

      if (result.root_q < o.resign_threshold) {
        if (++low_q[stm] >= o.resign_plies && !no_resign) { outcome = -sign; break; }
      } else {
        low_q[stm] = 0;
      }

      const Move mv = PickMove(result.root_visits, temp);
      board.DoMove(mv);
      moves_played.push_back(mv);
      temp *= o.temp_decay;
      if (temp < o.temp_min) temp = 0.0f;
    }

    // Write the game.
    std::vector<data::Record> records;
    records.reserve(game.size());
    for (const PositionRecord& p : game) {
      data::Record r;
      ShogiBoard b;
      b.SetFromSfen(p.sfen);
      lczero::PackedSfen packed;
      if (!b.ToPackedSfen(&packed)) continue;
      std::memcpy(r.head.sfen, packed.data.data(), 32);
      r.head.score = static_cast<int16_t>(std::clamp(p.score_cp, -32000, 32000));
      r.head.move = p.best.raw();
      r.head.game_ply = static_cast<uint16_t>(p.ply);
      const int sign = p.stm == BLACK ? 1 : -1;
      r.head.result = static_cast<int8_t>(outcome * sign);
      r.head.flags = data::kSelfPlay | (no_resign ? 0 : data::kResignAdjudicated);
      int max_v = 0;
      for (const auto& [m, v] : p.dist) max_v = std::max(max_v, v);
      for (const auto& [m, v] : p.dist) {
        if (v <= 0) continue;
        data::DistEntry e;
        e.move = m.raw();
        e.visits = static_cast<uint16_t>(std::max<long long>(1, 65535LL * v / std::max(max_v, 1)));
        r.dist.push_back(e);
      }
      records.push_back(std::move(r));
    }
    {
      std::lock_guard<std::mutex> lock(shared_->out_mutex);
      for (const auto& r : records) shared_->writer.Write(r);
    }
    shared_->positions.fetch_add(records.size());
    shared_->playouts.fetch_add(playouts);
    shared_->results[outcome > 0 ? 0 : (outcome == 0 ? 1 : 2)].fetch_add(1);
    const int done = shared_->games_done.fetch_add(1) + 1;
    if (o.report_every > 0 && done % o.report_every == 0) {
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - shared_->t0).count();
      std::fprintf(stderr, "datagen: %d games, %llu positions, %.0f pos/s, %.0f playouts/s, B/D/W %d/%d/%d\n",
                   done, static_cast<unsigned long long>(shared_->positions.load()),
                   shared_->positions.load() / secs, shared_->playouts.load() / secs,
                   shared_->results[0].load(), shared_->results[1].load(), shared_->results[2].load());
    }
  }

  Shared* shared_;
  int id_;
  std::mt19937_64 rng_;
  std::unique_ptr<dlshogi_mcts::Search> search_;
};

}  // namespace

int Run(int argc, char** argv) {
  Options o;
  if (!Parse(argc, argv, &o)) {
    Usage();
    return 2;
  }
  lczero::ShogiTables::Init();
  nnue::Init();
  if (o.seed == 0) o.seed = std::random_device{}();

  nnue::NetworkSet nets;
  std::string err;
  if (!nets.Load(o.value_path, o.policy_path, &err)) {
    std::fprintf(stderr, "datagen: %s\n", err.c_str());
    return 1;
  }
  Shared shared;
  shared.opts = o;
  shared.nets = &nets;
  if (!o.book_path.empty()) {
    std::ifstream in(o.book_path);
    std::string line;
    while (std::getline(in, line)) {
      const size_t tab = line.find('\t');
      if (tab != std::string::npos) line = line.substr(0, tab);
      if (!line.empty()) shared.book.push_back(line);
    }
    std::fprintf(stderr, "datagen: %zu book positions\n", shared.book.size());
  }
  if (!shared.writer.Open(o.out_path, &err)) {
    std::fprintf(stderr, "datagen: %s\n", err.c_str());
    return 1;
  }
  shared.t0 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "datagen: %d games, %d threads, %d nodes/move, seed %llu -> %s\n", o.games,
               o.threads, o.nodes, static_cast<unsigned long long>(o.seed), o.out_path.c_str());

  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<std::thread> threads;
  for (int t = 0; t < std::max(1, o.threads); ++t) workers.push_back(std::make_unique<Worker>(&shared, t));
  for (auto& w : workers) threads.emplace_back([&w] { w->Run(); });
  for (auto& t : threads) t.join();
  shared.writer.Close();

  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - shared.t0).count();
  std::fprintf(stderr, "datagen: done %d games, %llu positions in %.0fs (%.0f pos/s), B/D/W %d/%d/%d\n",
               shared.games_done.load(), static_cast<unsigned long long>(shared.positions.load()), secs,
               shared.positions.load() / std::max(secs, 1e-9), shared.results[0].load(),
               shared.results[1].load(), shared.results[2].load());
  return 0;
}

}  // namespace jhbr5::datagen
