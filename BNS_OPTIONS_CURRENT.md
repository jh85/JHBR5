# Current BNS root-mate options

Date: 2026-08-06

## Enabling BNS

BNS is already the default root mate solver. This line is optional, but it is
safe to keep explicitly in a Floodgate configuration:

```text
setoption name RootMateSolver value bns
```

No BNS time or node option is required. The BNS worker starts with MCTS, runs
for as long as MCTS runs, stops when MCTS stops, and stops MCTS immediately if
it proves mate.

To select the old tree df-pn implementation instead:

```text
setoption name RootMateSolver value dfpn
```

## Inactive option

`DfPnMaxTime` is retired. It is no longer advertised in the USI option list.
Old configuration files may still send it, but JHBR3 logs that it is retired
and ignores the value:

```text
setoption name DfPnMaxTime value 8000
```

The old clock-dependent 10K/100K/500K/2M BNS node tiers and the post-MCTS
100/300/500/1000 ms grace periods were internal settings, not USI options.
They have been removed.

## Related options that remain active

- `TimeManagement`, `MoveOverheadMs`, `TimeMaxExtensionPercent`,
  `MaxMoveTime`, and `MaxMoveTime1m` control the main MCTS lifetime. BNS now
  automatically follows that lifetime.
- `RootMateDepth` remains active, but it is not the BNS root solver. It is the
  separate shallow post-search guard that rejects an MCTS move allowing a
  short opponent mate.
- `LeafMateMode` and `LeafMateDepth` remain active and control shallow mate
  checks inside MCTS. They do not configure BNS.

## Recommended Floodgate configuration

```text
setoption name RootMateSolver value bns
```

Keep the normal JHBR3 time-management options appropriate for the tournament.
Remove `DfPnMaxTime`; no replacement is needed.
