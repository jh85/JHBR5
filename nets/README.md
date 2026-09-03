# Networks

Trained networks (`value.nn`, `policy.nn`, format in `docs/NNUE_FORMAT.md`) are
not committed; `*.nn` is git-ignored. Place them here (or anywhere) and point
the `ValueNet` / `PolicyNet` USI options at them. `build/make_random_net`
produces placeholder networks for testing.
