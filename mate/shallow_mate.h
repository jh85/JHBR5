// Shallow (depth-bounded) mate search for use at MCTS leaf nodes.
//
// This is a port of dlshogi's mateMoveInOddPly / mateMoveInEvenPly /
// mateMoveIn3Ply templates from DeepLearningShogi/usi/mate.h, adapted
// to jhbr2's ShogiBoard API.
//
// Cost vs. df-pn at the leaf:
//   - Per-call cost ranges from microseconds in quiet positions to a
//     few milliseconds for depth 7 in tactical positions.
//   - Coverage: mate-in-1, -3, -5, -7 (vs df-pn which can find longer
//     mates given enough budget)
//   - Trade-off: doesn't prove "no mate" (only finds mate or times out
//     by depth)
//
// Implementation notes vs. dlshogi:
//   - Checking OR nodes use `GenerateCheckingMoves`; checked AND nodes
//     call the specialized `GenerateEvasionMoves` path directly.
//   - dlshogi has a 6-state RepetitionResult; jhbr2 has 4. The
//     superior/inferior material variants don't apply here.
//   - dlshogi's `mateMoveIn1Ply()` is exposed as the shared
//     `ShogiBoard::FindMateInOneNonCheck()` bitboard routine.
//   - We omit the `gamePly() + N > draw_ply` cap (dlshogi optimization
//     for very long games) — not exposed cleanly on ShogiBoard.
//
// See:
//   - docs/architecture_improvements_research.md
//   - docs/port_5ply_mate_check_plan.md
//
#pragma once

#include <atomic>
#include <chrono>

#include "shogi/board.h"
#include "shogi/types.h"

namespace jhbr2 {

using lczero::ShogiBoard;
using lczero::Move;
using lczero::MoveList;
using lczero::UndoInfo;

namespace shallow_mate {

inline constexpr int kRepetitionLookbackPly = 16;

enum class ProbeResult {
    kMate,
    kNoMate,
    kCancelled,
};

// Optional cooperative cancellation for the post-MCTS root guard. The normal
// leaf probe passes nullptr and retains the old low-overhead bool API.
struct SearchLimits {
    using Clock = std::chrono::steady_clock;

    Clock::time_point deadline = Clock::time_point::max();
    const std::atomic<bool>* stop = nullptr;
    bool cancelled = false;
    uint32_t clock_poll_counter = 0;

    bool ShouldStop() {
        const bool stop_requested =
            stop && stop->load(std::memory_order_acquire);
        // steady_clock::now() is noticeably expensive in the depth-7 root
        // guard.  Poll it immediately, then once per 64 search checkpoints;
        // the stop flag remains responsive at every checkpoint.
        const bool deadline_passed =
            deadline != Clock::time_point::max() &&
            ((clock_poll_counter++ & 63u) == 0) && Clock::now() >= deadline;
        if (stop_requested || deadline_passed) {
            cancelled = true;
            return true;
        }
        return false;
    }
};

// =============================================================
// Helper: does playing m give check to the opponent?
// =============================================================

inline bool MoveGivesCheck(ShogiBoard& board, Move m) {
    UndoInfo undo = board.DoMove(m);
    bool gives_check = board.InCheck();
    board.UndoMove(m, undo);
    return gives_check;
}

// =============================================================
// Repetition handling
// =============================================================
//
// jhbr2's `CheckRepetition()` returns from the side-to-move's
// perspective at the position being checked:
//   kNone : no repetition
//   kDraw : 4-fold repetition without perpetual check → draw
//   kWin  : opponent was giving perpetual check → side-to-move wins
//   kLoss : we were giving perpetual check → side-to-move loses
//
// At an OR node (attacker to move), after attacker's doMove, the
// side-to-move flips to defender, and CheckRepetition returns from
// the defender's perspective:
//   kLoss → defender loses → mate found, attacker returns true
//   kWin / kDraw → defender survives → attacker continues with next move
//
// At an AND node (defender to move, in check), after defender's
// doMove, the side-to-move flips to attacker, and CheckRepetition
// returns from the attacker's perspective:
//   kWin → attacker wins → defender's evasion failed, try next evasion
//   kLoss / kDraw → defender escaped → no mate, return false
//

// =============================================================
// Forward declarations
// =============================================================

template <int depth>
bool MateInEvenPly(ShogiBoard& board, SearchLimits* limits = nullptr);
template <int depth, bool INCHECK = false>
bool MateInOddPly(ShogiBoard& board, SearchLimits* limits = nullptr);

// =============================================================
// Hand-tuned 3-ply mate (base case for MateInOddPly<3>).
// =============================================================
//
// Mirrors the structure of dlshogi's mateMoveIn3Ply: iterate checking
// moves (depth 1), iterate evasions (depth 2), inline mate-in-1
// detection (depth 3).
//
// INCHECK template param selects the safe in-check path at OR nodes.
// The common false path uses GenerateCheckingMovesNonCheck() to avoid
// the runtime in-check check and the legal-move fallback.
template <bool INCHECK = false>
inline bool MateIn3Ply(ShogiBoard& board, SearchLimits* limits = nullptr) {
    if (limits && limits->ShouldStop()) return false;
    // OR node (depth 1): try each checking move.
    // Use specialized GenerateCheckingMoves (Phase 6) — direct
    // bitboard-based enumeration of moves that cause check, avoiding
    // full legal-move generation on the common non-check path.
    MoveList checking_moves;
    if constexpr (INCHECK) {
        board.GenerateCheckingMoves(&checking_moves);
    } else {
        board.GenerateCheckingMovesNonCheck(&checking_moves);
    }
    for (int i = 0; i < checking_moves.size(); ++i) {
        if (limits && limits->ShouldStop()) return false;
        Move m1 = checking_moves[i];

        UndoInfo undo1 = board.DoMove(m1, true);

        // Repetition check from defender's perspective:
        // kWin/kDraw → defender survives → skip this attacker move.
        auto rep1 = board.CheckRepetition(kRepetitionLookbackPly);
        if (rep1 == ShogiBoard::RepetitionResult::kWin ||
            rep1 == ShogiBoard::RepetitionResult::kDraw) {
            board.UndoMove(m1, undo1);
            continue;
        }
        // kLoss → defender loses → mate found.
        if (rep1 == ShogiBoard::RepetitionResult::kLoss) {
            board.UndoMove(m1, undo1);
            return true;
        }

        // AND node (depth 2): defender tries to escape.
        MoveList evasions;
        board.GenerateEvasionMoves(&evasions);
        if (evasions.empty()) {
            // Mate-in-1 — defender has no legal evasion.
            board.UndoMove(m1, undo1);
            return true;
        }

        // For each evasion, check if attacker has mate-in-1 after.
        bool all_evasions_lose = true;
        for (int j = 0; j < evasions.size(); ++j) {
            if (limits && limits->ShouldStop()) {
                all_evasions_lose = false;
                break;
            }
            Move m2 = evasions[j];

            UndoInfo undo2 = board.DoMove(m2);

            // dlshogi simplification: if defender's evasion is itself
            // a counter-check, treat as defender escape (we don't
            // try to find mate-in-1 against an in-check attacker
            // here — would require INCHECK-aware mate-in-1).
            if (undo2.gave_check) {
                board.UndoMove(m2, undo2);
                all_evasions_lose = false;
                break;
            }

            auto rep2 = board.CheckRepetition(kRepetitionLookbackPly);
            // After defender's evasion, side-to-move = attacker.
            // kWin → attacker wins → this evasion fails, try next.
            // kLoss/kDraw → attacker doesn't mate → defender escaped.
            if (rep2 == ShogiBoard::RepetitionResult::kLoss ||
                rep2 == ShogiBoard::RepetitionResult::kDraw) {
                board.UndoMove(m2, undo2);
                all_evasions_lose = false;
                break;
            }
            if (rep2 == ShogiBoard::RepetitionResult::kWin) {
                board.UndoMove(m2, undo2);
                continue;  // this evasion fails, try next
            }

            // OR node (depth 3): use the hand-specialized bitboard
            // mate-in-1 routine. Counterchecks were rejected above, so
            // the non-check precondition is guaranteed.
            bool found_mate1 =
                !board.FindMateInOneNonCheck().is_null();

            board.UndoMove(m2, undo2);
            if (limits && limits->ShouldStop()) {
                all_evasions_lose = false;
                break;
            }
            if (!found_mate1) {
                all_evasions_lose = false;
                break;
            }
        }

        board.UndoMove(m1, undo1);
        if (limits && limits->cancelled) return false;
        if (all_evasions_lose) return true;
    }
    return false;
}

// =============================================================
// Generic OR-node template: MateInOddPly<depth>
// =============================================================
//
// Side-to-move (attacker) is searching for a forced mate within
// `depth` plies (must be odd). Returns true if at least one
// checking move leads to a position where MateInEvenPly<depth-1>
// returns true.
template <int depth, bool INCHECK>
inline bool MateInOddPly(ShogiBoard& board, SearchLimits* limits) {
    static_assert(depth >= 1 && (depth % 2) == 1,
                  "MateInOddPly: depth must be positive odd");
    if (limits && limits->ShouldStop()) return false;

    // Specialized check-only generator (Phase 6).
    MoveList moves;
    if constexpr (INCHECK) {
        board.GenerateCheckingMoves(&moves);
    } else {
        board.GenerateCheckingMovesNonCheck(&moves);
    }
    for (int i = 0; i < moves.size(); ++i) {
        if (limits && limits->ShouldStop()) return false;
        Move m = moves[i];

        UndoInfo undo = board.DoMove(m, true);

        auto rep = board.CheckRepetition(kRepetitionLookbackPly);
        if (rep == ShogiBoard::RepetitionResult::kLoss) {
            // Defender loses by repetition → mate found.
            board.UndoMove(m, undo);
            return true;
        }
        if (rep == ShogiBoard::RepetitionResult::kWin ||
            rep == ShogiBoard::RepetitionResult::kDraw) {
            // Defender survives by repetition → skip this move.
            board.UndoMove(m, undo);
            continue;
        }

        // Recurse into AND node at depth-1.
        const bool sub = MateInEvenPly<depth - 1>(board, limits);
        board.UndoMove(m, undo);
        if (limits && limits->cancelled) return false;
        if (sub) {
            return true;
        }
    }
    return false;
}

// =============================================================
// Generic AND-node template: MateInEvenPly<depth>
// =============================================================
//
// Side-to-move (defender) is in check (entered here from MateInOddPly
// by attacker's checking move). We're checking if EVERY evasion leads
// to a position where MateInOddPly<depth-1> returns true.
template <int depth>
inline bool MateInEvenPly(ShogiBoard& board, SearchLimits* limits) {
    static_assert(depth >= 0 && (depth % 2) == 0,
                  "MateInEvenPly: depth must be non-negative even");
    if (limits && limits->ShouldStop()) return false;

    if constexpr (depth == 0) {
        return !board.HasLegalEvasion();
    }

    MoveList moves;
    board.GenerateEvasionMoves(&moves);
    if (moves.empty()) {
        // No legal moves — defender is mated.
        return true;
    }

    for (int i = 0; i < moves.size(); ++i) {
        if (limits && limits->ShouldStop()) return false;
        Move m = moves[i];
        UndoInfo undo = board.DoMove(m);

        auto rep = board.CheckRepetition(kRepetitionLookbackPly);
        // After defender's doMove, side-to-move = attacker.
        if (rep == ShogiBoard::RepetitionResult::kWin) {
            // Attacker wins by repetition → this evasion fails defender.
            board.UndoMove(m, undo);
            continue;
        }
        if (rep == ShogiBoard::RepetitionResult::kLoss ||
            rep == ShogiBoard::RepetitionResult::kDraw) {
            // Attacker loses or draws → defender escaped.
            board.UndoMove(m, undo);
            return false;
        }

        // Recurse into OR node at depth-1.
        bool attacker_in_check = undo.gave_check;
        bool sub;
        if (attacker_in_check) {
            sub = MateInOddPly<depth - 1, true>(board, limits);
        } else {
            sub = MateInOddPly<depth - 1, false>(board, limits);
        }
        board.UndoMove(m, undo);
        if (limits && limits->cancelled) return false;

        if (!sub) {
            // Defender's evasion led to a non-mate position → escaped.
            return false;
        }
    }
    return true;
}

// =============================================================
// Template specializations
// =============================================================

// MateInOddPly<3> uses the hand-tuned MateIn3Ply
template <>
inline bool MateInOddPly<3, false>(ShogiBoard& board, SearchLimits* limits) {
    return MateIn3Ply<false>(board, limits);
}
template <>
inline bool MateInOddPly<3, true>(ShogiBoard& board, SearchLimits* limits) {
    return MateIn3Ply<true>(board, limits);
}

// MateInOddPly<1> — direct mate-in-1 detection
template <>
inline bool MateInOddPly<1, false>(ShogiBoard& board, SearchLimits* limits) {
    if (limits && limits->ShouldStop()) return false;
    return !board.FindMateInOneNonCheck().is_null();
}
template <>
inline bool MateInOddPly<1, true>(ShogiBoard& board, SearchLimits* limits) {
    if (limits && limits->ShouldStop()) return false;
    return !board.FindMateInOne().is_null();
}

// =============================================================
// Public interface
// =============================================================

// Check for a forced mate within `depth` plies. `depth` must be odd
// and one of {1, 3, 5, 7}. Returns true if side-to-move has a forced
// mate within `depth` plies; false if no such mate is found within
// the depth limit (which doesn't prove no mate exists — could just
// be deeper than `depth`).
inline bool HasMateWithin(ShogiBoard& board, int depth,
                          SearchLimits* limits = nullptr) {
    bool in_check = board.InCheck();
    switch (depth) {
        case 1: return in_check ? MateInOddPly<1, true>(board, limits)
                                : MateInOddPly<1, false>(board, limits);
        case 3: return in_check ? MateInOddPly<3, true>(board, limits)
                                : MateInOddPly<3, false>(board, limits);
        case 5: return in_check ? MateInOddPly<5, true>(board, limits)
                                : MateInOddPly<5, false>(board, limits);
        case 7: return in_check ? MateInOddPly<7, true>(board, limits)
                                : MateInOddPly<7, false>(board, limits);
        default: return false;
    }
}

inline ProbeResult ProbeMateWithin(ShogiBoard& board, int depth,
                                   SearchLimits* limits) {
    if (limits) {
        limits->cancelled = false;
        limits->clock_poll_counter = 0;
    }
    const bool mate = HasMateWithin(board, depth, limits);
    if (limits && limits->cancelled) return ProbeResult::kCancelled;
    return mate ? ProbeResult::kMate : ProbeResult::kNoMate;
}

}  // namespace shallow_mate
}  // namespace jhbr2
