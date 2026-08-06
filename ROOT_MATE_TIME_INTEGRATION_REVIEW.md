# Root mate checker and time-management integration review

Date: 2026-08-06

## Implementation status

Implemented on the current working tree on 2026-08-06. The independent
`DfPnMaxTime`, clock-derived node tiers, and post-MCTS grace periods were
removed. The old `DfPnMaxTime` command remains accepted but ignored for
configuration compatibility.

Verification completed:

- CPU and TensorRT `jhbr3` targets build successfully.
- `test_time_manager`, `test_lockfree_search`, `test_bns`, and
  `test_dfpn_cancellation` pass.
- The lock-free search test now includes a real BNS worker proving mate and
  stopping MCTS before its 10,000,000-node cap.
- A live TensorRT test on `4k4/9/4G4/9/9/9/9/9/8K b G 1` proved `G*5b`,
  stopped MCTS, and returned `bestmove G*5b`. The root mate record reported
  `outcome=mate`, `stop_source=mate`, and zero-millisecond join latency.

## Conclusion

The current JHBR3 root mate checker has its own time allocator on top of the
normal search time manager. This is the wrong ownership model. The root mate
checker should be a concurrent worker whose lifetime follows the main MCTS
search:

1. Start the root mate worker when MCTS starts.
2. If it proves mate, stop MCTS immediately and return the mate after the
   existing validation.
3. When MCTS stops for any reason, stop and join the root mate worker
   immediately.
4. Give both workers the same absolute response-safety deadline.

With this design, `DfPnMaxTime`, the clock-dependent mate node tiers, and the
post-MCTS grace periods are no longer search-strength parameters that need to
be tuned. The normal time manager is the single authority.

DeepLearningShogi already uses the important part of this lifecycle. JHBR3
should copy that lifecycle, but should not copy DeepLearningShogi's remaining
fixed root-df-pn node cap or minimum-search-time hyperparameter.

## DeepLearningShogi behavior

I checked official DeepLearningShogi `master` at commit
`5969e3165ab195f305940623c8a55160fc05a0a5` (2026-08-02).

Its root df-pn behavior is:

- The df-pn thread starts before `UctSearchGenmove`, so it runs concurrently
  with UCT.
- If df-pn proves mate, it immediately calls `StopUctSearch()`.
- When UCT returns, the main thread calls `dfpn_stop(true)` and joins the df-pn
  thread.
- There is no independent option equivalent to JHBR3's `DfPnMaxTime`.
- Consequently, an adaptive UCT extension also extends the concurrent df-pn
  opportunity; an early UCT stop normally stops df-pn early.

The source is here:

- Root worker lifecycle: [DeepLearningShogi `usi/main.cpp`, lines 540-572](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/5969e3165ab195f305940623c8a55160fc05a0a5/usi/main.cpp#L540-L572)
- UCT time calculation: [DeepLearningShogi `usi/UctSearch.cpp`, lines 716-740](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/5969e3165ab195f305940623c8a55160fc05a0a5/usi/UctSearch.cpp#L716-L740)

DeepLearningShogi is not completely parameter-free. It still has:

- A hidden root df-pn maximum of 2,097,152 nodes.
- `DfPn_Min_Search_Millisecs`, default 300 ms.
- `Mate_Root_Search`, default depth 33.
- `DfPn_Hash`, default 2048 MiB.

The 300 ms setting is often misunderstood. It is a minimum **total time since
the `go` command**, not an additional 300 ms after UCT. After UCT finishes,
DeepLearningShogi waits only while elapsed time is below both the current UCT
time limit and 300 ms. Therefore, if UCT already ran for more than 300 ms,
there is no extra wait. It also never uses this rule to wait beyond the UCT
time limit.

Relevant sources:

- Hidden 2,097,152-node limit: [DeepLearningShogi `usi/dfpn.h`, lines 52-89](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/5969e3165ab195f305940623c8a55160fc05a0a5/usi/dfpn.h#L52-L89)
- Root df-pn options: [DeepLearningShogi `cppshogi/usi.cpp`, lines 119-137](https://github.com/TadaoYamaoka/DeepLearningShogi/blob/5969e3165ab195f305940623c8a55160fc05a0a5/cppshogi/usi.cpp#L119-L137)

So the useful DeepLearningShogi design principle is mutual lifecycle coupling,
not its fixed 2M nodes or 300 ms constant.

## Previous JHBR3 behavior (now removed)

Before this implementation, JHBR3 had three separate mate-search limits:

1. `DfPnMaxTime`, default 4000 ms.
2. Clock-dependent node tiers from 10,000 to 2,000,000 nodes.
3. A clock-dependent post-MCTS grace period from 100 to 1,000 ms.

These are selected in `usi/time_manager.cc:20-40`. The root worker is launched
with its independent node count and deadline in `usi/usi_engine.cc:781-843`.
After MCTS returns, the engine may wait another grace period in
`usi/usi_engine.cc:939-960`.

This creates four concrete problems:

1. On a long move, the mate worker can stop at 4 seconds even while the main
   search continues for much longer.
2. On an adaptive early stop, JHBR3 can spend an extra 100-1,000 ms waiting for
   the mate worker even though the main time manager decided it was time to
   move.
3. A proved mate does not stop MCTS. JHBR3 reads the mate result only after
   `search_->Run()` returns and after the grace/join sequence.
4. The node tier is based on remaining clock, not on the duration actually
   chosen by the adaptive MCTS controller. It is therefore another time
   manager in disguise.

The hard watchdog already stops both searches at the response deadline. That
part is useful and should remain as a fail-safe.

## Recommended JHBR3 design

### One owner for search time

`TimeBudget` should decide the move lifetime. The root mate checker should not
independently decide how many milliseconds of the move it deserves.

The desired event flow is:

```text
prepare MCTS and root-mate state
              |
              +---- start BNS worker
              |
              +---- run MCTS

BNS proves mate ----------------------> request MCTS Stop()
MCTS ends for any reason -------------> request BNS stop()
external stop / hard deadline --------> stop both

join BNS -> validate mate/repetition -> choose mate or MCTS result
```

The worker must be stopped when MCTS ends due to any of these causes:

- adaptive early stop;
- target or extended time limit;
- MCTS node limit;
- external USI `stop`;
- a proved search result;
- hard response watchdog.

This also gives natural semantics to unusual USI modes:

- `go nodes`: BNS runs while the node-limited MCTS runs.
- `go infinite` or ponder: BNS runs until an external stop, unless it proves
  mate first.
- normal clock search: BNS receives exactly the opportunity allowed by the
  actual MCTS lifecycle, including adaptive extension or early termination.

### Keep a shared fail-safe deadline

BNS should still receive an absolute deadline derived from the authoritative
response/hard deadline, with the existing response reserve. This is not a
separate BNS time allocation. It protects against a delayed cancellation or a
worker that has not yet observed `stop()`.

The normal MCTS decision should stop BNS first; the absolute deadline is only
the last safety net against a timeout.

The current BNS implementation checks an explicit deadline only every 1,024
`ShouldStop()` calls, while its atomic `stop()` is checked directly. Measure
the cancellation-to-join latency. If the tail is too large for the response
reserve, make cancellation observation more frequent. This should be based on
measured join latency rather than a new playing-strength hyperparameter.

### Remove the BNS node tier from time management

For the default fixed-table BNS implementation, use an effectively unlimited
node limit and let the shared lifecycle stop it. For example, use
`std::numeric_limits<size_t>::max()` internally, or a very large nonbinding
overflow guard.

BNS uses fixed-size tables (currently 4 MiB TT and 2 MiB move cache), so its
memory does not grow linearly with searched nodes. Increasing its runtime does
consume one CPU core and can reduce MCTS throughput; that is the fundamental
tradeoff of running the worker, but a clock-derived node tier does not manage
that tradeoff coherently. It merely makes BNS disappear at unpredictable
points during longer MCTS searches.

The optional legacy tree df-pn solver is different: its node pool is allocated
from the node limit. It should retain a resource/memory cap. That cap should be
documented as an implementation resource limit, not calculated as a second
time schedule. `RootMateSolver=bns` should remain the default production path.

### Retire these active timing knobs

- `DfPnMaxTime`
- the 10K/100K/500K/2M clock-dependent root-mate node tiers
- the 100/300/500/1000 ms post-MCTS grace tiers

For configuration compatibility, JHBR3 can continue accepting
`setoption name DfPnMaxTime ...` temporarily while logging that it is retired
and has no effect. It should eventually stop advertising the option.

I do not recommend replacing the current node tier with a user-tunable
`RootMateMaxNodes` for BNS. That would preserve the same tuning problem under a
new name.

### Keep these controls and safeguards

- `RootMateSolver=bns` as the default algorithm selection.
- Fixed BNS TT/move-cache sizes as resource/performance choices.
- The global response deadline and hard watchdog.
- Solver depth/resource safety guards.
- Final mate-PV and repetition validation before replacing the MCTS result.
- A fixed memory cap for the optional tree df-pn implementation.

## Implementation ordering

The current code starts the mate thread before `EnsureSearch()` and before all
per-move MCTS configuration is installed. Because a successful mate worker
will need to call `search_->Stop()`, initialize/configure the persistent search
object first, then start both search activities.

A safe implementation sequence is:

1. Compute one `TimeBudget` and its response deadline.
2. Call `EnsureSearch()` and apply all per-move MCTS limits.
3. Construct `RootMateState` (rename the current `DfpnState`, because BNS is
   the default).
4. Start the root mate thread with no strength-related BNS node limit and the
   shared fail-safe deadline.
5. When it proves mate, publish the result and call `search_->Stop()`.
6. Run MCTS.
7. Immediately call `root_mate->Stop()` when MCTS returns; do not wait for a
   post-MCTS grace period.
8. Join the worker, validate any proved mate, and choose the result.
9. Retain the watchdog so the response deadline stops both workers.

There is a benign race to handle: BNS may finish just as MCTS finishes. The
published mate result and `done` state must use the current release/acquire
synchronization, and `Stop()` must remain idempotent.

## Logging and tests

Add one structured root-mate record per move, for example:

```text
root_mate solver=bns outcome=mate|nomate|stopped|deadline|resource
          elapsed_ms=... nodes=... stop_source=mate|mcts|external|watchdog
          join_ms=...
```

This will tell us whether the permanent extra CPU thread is worthwhile and
whether cancellation has enough timeout margin. Also compare MCTS NPS with the
root worker enabled and disabled.

Required regression tests:

1. A proved mate stops MCTS promptly.
2. Adaptive early MCTS termination stops BNS with no grace-period delay.
3. An MCTS extension lets BNS continue until the extended search ends.
4. A node-limited search stops BNS when MCTS reaches its node limit.
5. `go infinite` plus external `stop` stops and joins both workers.
6. The hard response deadline stops both and the join completes inside the
   timeout reserve.
7. Mate/repetition validation remains unchanged.
8. Cancellation before the BNS worker fully enters `search()` is not lost.

## Practical result

After this refactor, Floodgate configuration should not need a root-mate time
or node setting. If the MCTS time manager chooses 9 seconds, BNS can search for
approximately those same 9 seconds. If MCTS extends, BNS continues; if MCTS
stops early, BNS stops; and if BNS proves mate after 2 seconds, MCTS stops at
that point. Both remain bounded by the same response-safety deadline.

This eliminates the hyperparameter-tuning problem rather than merely choosing
larger values for the current independent limits.
