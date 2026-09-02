#!/usr/bin/env python3
"""SPRT match between two USI engine configurations, in rounds of paired games
through tools/strength_test.py (one run directory per round, seeds seed+round),
aggregating the pentanomial counts and stopping when tools/sprt.py decides.

  python tools/sprt_match.py --engine-a build/jhbr5 --engine-b build/jhbr5 \
      --option-a ValueNet=nets/best/value.nn --option-a PolicyNet=nets/best/policy.nn \
      --option-b ValueNet=cand/value.nn --option-b PolicyNet=cand/policy.nn \
      --openings build-strength/openings-512.txt --nodes 5000 \
      --elo0 0 --elo1 5 --round-pairs 25 --max-pairs 1000 --output sprt-runs/cand

Engine B is the candidate: H1 (elo1) accepted -> exit 0, H0 accepted -> exit 1,
undecided at --max-pairs -> exit 2. Any strength_test.py option can be passed
through after `--`.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--option-a", action="append", default=[])
    ap.add_argument("--option-b", action="append", default=[])
    ap.add_argument("--option", action="append", default=[], help="option for both engines")
    ap.add_argument("--openings", required=True)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--byoyomi-ms", type=int)
    ap.add_argument("--main-time-ms", type=int)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--round-pairs", type=int, default=25)
    ap.add_argument("--max-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", required=True)
    ap.add_argument("extra", nargs="*", help="passed through to strength_test.py")
    args = ap.parse_args()

    pairs = 0
    rnd = 0
    penta = [0, 0, 0, 0, 0]
    os.makedirs(args.output, exist_ok=True)
    while pairs < args.max_pairs:
        rnd += 1
        n = min(args.round_pairs, args.max_pairs - pairs)
        pairs += n
        round_dir = os.path.join(args.output, f"round_{rnd:03d}")
        cmd = [sys.executable, os.path.join(HERE, "strength_test.py"), "--engine-a", args.engine_a,
               "--engine-b", args.engine_b, "--openings", args.openings, "--pairs", str(n),
               "--seed", str(args.seed + rnd), "--output", round_dir]
        if os.path.exists(os.path.join(round_dir, "config.json")):
            cmd.append("--resume")
        if args.nodes:
            cmd += ["--nodes", str(args.nodes)]
        if args.byoyomi_ms:
            cmd += ["--byoyomi-ms", str(args.byoyomi_ms)]
        if args.main_time_ms:
            cmd += ["--main-time-ms", str(args.main_time_ms)]
        for o in args.option_a + args.option:
            cmd += ["--option-a", o]
        for o in args.option_b + args.option:
            cmd += ["--option-b", o]
        cmd += args.extra
        subprocess.run(cmd, check=True)
        summary = json.load(open(os.path.join(round_dir, "summary.json")))
        penta = [x + y for x, y in zip(penta, summary["pentanomial_a_points_0_to_2"])]
        r = subprocess.run([sys.executable, os.path.join(HERE, "sprt.py"), "--pentanomial"] + [str(c) for c in penta] +
                           ["--for-b", "--elo0", str(args.elo0), "--elo1", str(args.elo1),
                            "--alpha", str(args.alpha), "--beta", str(args.beta)],
                           capture_output=True, text=True)
        print(r.stdout.strip(), flush=True)
        verdict = json.loads(r.stdout)
        json.dump({"sprt": verdict, "rounds": rnd, "pairs": pairs},
                  open(os.path.join(args.output, "sprt.json"), "w"), indent=2)
        if verdict["result"] == "H1":
            print("SPRT: candidate accepted")
            return 0
        if verdict["result"] == "H0":
            print("SPRT: candidate rejected")
            return 1
    print("SPRT: undecided at max pairs")
    return 2


if __name__ == "__main__":
    sys.exit(main())
