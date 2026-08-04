/*
  JHBR3 — Mate solver benchmark driver

  Runs one of the JHBR3 solvers over a file of SFEN mate problems and
  reports solved counts, timing percentiles, and solver statistics.
  The side to move in each SFEN is the attacker.

  Usage:
    bench_mate_solvers --solver=dfpn|bns|bns-pndn --file=problems.sfen
                       [--count=N]        problems to run       (default 1000)
                       [--skip=N]         problems to skip      (default 0)
                       [--nodes=N]        node budget/problem   (default 500000)
                       [--time-ms=N]      time budget/problem   (default 10000)
                       [--hash-mb=N]      BNS TT size           (default 256)
                       [--max-ply=N]      BNS depth cap         (default 129)
                       [--validate=0|1]   validate mate PVs     (default 1)
                       [--csv=file]       per-problem CSV output
*/

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "mate/bns.h"
#include "mate/dfpn.h"
#include "shogi/bitboard.h"
#include "shogi/board.h"

using namespace lczero;
using namespace jhbr2;
using Clock = std::chrono::steady_clock;

namespace {

bool ContainsMove(const MoveList& moves, Move target) {
  for (const Move& m : moves)
    if (m == target) return true;
  return false;
}

bool IsValidMatePv(ShogiBoard board, Move result, const std::vector<Move>& pv) {
  if (pv.empty() || pv.front() != result) return false;
  for (size_t ply = 0; ply < pv.size(); ++ply) {
    const MoveList legal = board.GenerateLegalMoves();
    if (!ContainsMove(legal, pv[ply])) return false;
    board.DoMove(pv[ply]);
    if (ply % 2 == 0 && !board.InCheck()) return false;
  }
  return board.InCheck() && board.GenerateLegalMoves().empty();
}

struct Options {
  std::string solver = "bns";
  std::string file;
  std::string csv;
  long count = 1000;
  long skip = 0;
  size_t nodes = 500000;
  long time_ms = 10000;
  size_t hash_mb = 256;
  int max_ply = 129;
  long mate3 = 0;
  long nnb = 0;
  long dom = 0;
  long order = 0;
  long mcache = 64;
  long paper_ghi = 0;
  long naive_tt = 0;
  bool validate = true;
};

long ArgLong(const char* s) { return atol(s); }

bool ParseArgs(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    auto val = [&](const char* prefix) -> const char* {
      size_t n = strlen(prefix);
      return strncmp(a, prefix, n) == 0 ? a + n : nullptr;
    };
    const char* v;
    if ((v = val("--solver="))) o->solver = v;
    else if ((v = val("--file="))) o->file = v;
    else if ((v = val("--csv="))) o->csv = v;
    else if ((v = val("--count="))) o->count = ArgLong(v);
    else if ((v = val("--skip="))) o->skip = ArgLong(v);
    else if ((v = val("--nodes="))) o->nodes = (size_t)ArgLong(v);
    else if ((v = val("--time-ms="))) o->time_ms = ArgLong(v);
    else if ((v = val("--hash-mb="))) o->hash_mb = (size_t)ArgLong(v);
    else if ((v = val("--max-ply="))) o->max_ply = (int)ArgLong(v);
    else if ((v = val("--mate3="))) o->mate3 = ArgLong(v);
    else if ((v = val("--nnb="))) o->nnb = ArgLong(v);
    else if ((v = val("--dom="))) o->dom = ArgLong(v);
    else if ((v = val("--order="))) o->order = ArgLong(v);
    else if ((v = val("--mcache="))) o->mcache = ArgLong(v);
    else if ((v = val("--paper-ghi="))) o->paper_ghi = ArgLong(v);
    else if ((v = val("--naive-tt="))) o->naive_tt = ArgLong(v);
    else if ((v = val("--validate="))) o->validate = ArgLong(v) != 0;
    else {
      fprintf(stderr, "Unknown arg: %s\n", a);
      return false;
    }
  }
  if (o->file.empty()) {
    fprintf(stderr, "Missing --file=\n");
    return false;
  }
  if (o->solver != "dfpn" && o->solver != "dfpn-fast" && o->solver != "bns" &&
      o->solver != "bns-pndn") {
    fprintf(stderr, "Unknown solver: %s\n", o->solver.c_str());
    return false;
  }
  return true;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t idx = (size_t)(p * (v.size() - 1) + 0.5);
  return v[idx];
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, &opt)) return 1;

  ShogiTables::Init();

  std::ifstream fin(opt.file);
  if (!fin.is_open()) {
    fprintf(stderr, "Cannot open %s\n", opt.file.c_str());
    return 1;
  }

  std::vector<std::string> problems;
  {
    std::string line;
    long lineno = 0;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') continue;
      if (lineno++ < opt.skip) continue;
      problems.push_back(line);
      if ((long)problems.size() >= opt.count) break;
    }
  }

  FILE* csv = nullptr;
  if (!opt.csv.empty()) {
    csv = fopen(opt.csv.c_str(), "w");
    if (!csv) {
      fprintf(stderr, "Cannot open csv %s\n", opt.csv.c_str());
      return 1;
    }
    fprintf(csv,
            "idx,solved,nomate,unsolved,valid_pv,mate_ply,ms,nodes,do_moves,"
            "tt_probes,tt_hits,tt_stores,tt_evictions,summaries,max_ply\n");
  }

  // Persistent solvers (both keep their allocations across problems).
  MateBnsSolver bns(opt.hash_mb);
  bns.set_max_ply(opt.max_ply);
  if (opt.solver == "bns") bns.set_arith(MateBnsSolver::Arith::kBns);
  if (opt.solver == "bns-pndn") bns.set_arith(MateBnsSolver::Arith::kPnDn);
  bns.set_use_mate3_probe(opt.mate3 != 0);
  bns.set_use_new_node_block(opt.nnb != 0);
  bns.set_use_dominance(opt.dom != 0);
  bns.set_move_ordering(opt.order != 0);
  bns.set_move_cache_mb((size_t)opt.mcache);
  bns.set_paper_ghi_mode(opt.paper_ghi != 0);
  if (opt.naive_tt)
    bns.set_victim_policy(BnsTT::VictimPolicy::kAlwaysFirst);
  MateDfpnSolver dfpn(opt.nodes);
  if (opt.solver == "dfpn-fast") dfpn.set_use_fast_check_movegen(true);

  long solved = 0, nomate = 0, unsolved = 0, invalid_pv = 0;
  double total_ms = 0;
  unsigned long long total_nodes = 0, total_first_visits = 0;
  std::vector<double> times_ms;
  times_ms.reserve(problems.size());

  const auto t_all0 = Clock::now();

  for (size_t i = 0; i < problems.size(); i++) {
    ShogiBoard board;
    if (!board.SetFromSfen(problems[i])) {
      fprintf(stderr, "Bad SFEN at %zu: %s\n", i, problems[i].c_str());
      continue;
    }

    Move result;
    double ms = 0;
    unsigned long long nodes = 0, do_moves = 0, tt_probes = 0, tt_hits = 0,
                       tt_stores = 0, tt_evict = 0, summaries = 0;
    int mate_ply = 0, max_ply_seen = 0;
    bool valid = true;
    std::vector<Move> pv;

    const auto deadline =
        Clock::now() + std::chrono::milliseconds(opt.time_ms);

    if (opt.solver == "dfpn" || opt.solver == "dfpn-fast") {
      const auto t0 = Clock::now();
      result = dfpn.search(board, opt.nodes, deadline);
      ms = std::chrono::duration<double, std::milli>(Clock::now() - t0)
               .count();
      nodes = dfpn.get_nodes_searched();
      mate_ply = dfpn.get_mate_ply();
      pv = dfpn.get_pv();
    } else {
      const auto t0 = Clock::now();
      result = bns.search(board, opt.nodes, deadline);
      ms = std::chrono::duration<double, std::milli>(Clock::now() - t0)
               .count();
      const auto& st = bns.stats();
      nodes = st.node_entries;
      do_moves = st.do_moves;
      tt_probes = st.tt_probes;
      tt_hits = st.tt_hits;
      tt_stores = st.tt_stores;
      tt_evict = st.tt_evictions;
      summaries = st.summaries;
      total_first_visits += st.first_visits;
      max_ply_seen = st.max_ply;
      mate_ply = bns.get_mate_ply();
      pv = bns.get_pv();
    }

    const bool is_nomate = MateBnsSolver::IsNoMate(result);
    const bool is_mate = !result.is_null() && !is_nomate;

    if (is_mate) {
      solved++;
      if (opt.validate && !IsValidMatePv(board, result, pv)) {
        invalid_pv++;
        valid = false;
      }
    } else if (is_nomate) {
      nomate++;
    } else {
      unsolved++;
    }

    total_ms += ms;
    total_nodes += nodes;
    times_ms.push_back(ms);

    if (csv) {
      fprintf(csv,
              "%zu,%d,%d,%d,%d,%d,%.3f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d\n",
              i, is_mate ? 1 : 0, is_nomate ? 1 : 0,
              (!is_mate && !is_nomate) ? 1 : 0, valid ? 1 : 0, mate_ply, ms,
              nodes, do_moves, tt_probes, tt_hits, tt_stores, tt_evict,
              summaries, max_ply_seen);
    }

    if ((i + 1) % 1000 == 0) {
      fprintf(stderr, "  ... %zu/%zu (solved %ld)\n", i + 1, problems.size(),
              solved);
    }
  }

  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t_all0).count();

  printf("solver=%s file=%s problems=%zu\n", opt.solver.c_str(),
         opt.file.c_str(), problems.size());
  printf("  solved      %ld\n", solved);
  printf("  nomate      %ld\n", nomate);
  printf("  unsolved    %ld\n", unsolved);
  printf("  invalid_pv  %ld\n", invalid_pv);
  printf("  total_ms    %.1f (wall %.1f)\n", total_ms, wall_ms);
  printf("  mean_ms     %.3f\n",
         problems.empty() ? 0 : total_ms / problems.size());
  printf("  median_ms   %.3f\n", Percentile(times_ms, 0.5));
  printf("  p90_ms      %.3f\n", Percentile(times_ms, 0.9));
  printf("  p99_ms      %.3f\n", Percentile(times_ms, 0.99));
  printf("  max_ms      %.3f\n", Percentile(times_ms, 1.0));
  printf("  total_nodes %llu\n", total_nodes);
  if (opt.solver.rfind("bns", 0) == 0) {
    const auto& st = bns.stats();
    printf("  first_visits %llu (%.1f%% of node entries)\n",
           total_first_visits,
           total_nodes ? 100.0 * total_first_visits / total_nodes : 0.0);
  }

  if (csv) fclose(csv);
  return 0;
}
