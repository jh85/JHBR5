# Networks

Trained networks (`value.nn`, `policy.nn`, format in `docs/NNUE_FORMAT.md`) are
not committed; `*.nn` is git-ignored. Place them here (or anywhere) and point
the `ValueNet` / `PolicyNet` USI options at them. `build/make_random_net`
produces placeholder networks for testing.

The engine accepts both format versions: each file's header (`version` field)
selects the v1 or v2 evaluation path at load time, so v1 and v2 nets can be
mixed freely (e.g. a v2 value net with a v1 policy net) via the `ValueNet` /
`PolicyNet` options.
