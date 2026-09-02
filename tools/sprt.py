#!/usr/bin/env python3
"""Sequential probability ratio test on paired-game results (pentanomial GSPRT).

Reads the `summary.json` / `games.jsonl` written by tools/strength_test.py, or
a pentanomial count vector, and reports the log-likelihood ratio against the
Elo hypotheses H0: elo = elo0 vs H1: elo = elo1 with bounds from alpha/beta.

  python tools/sprt.py --summary strength-runs/x/summary.json --elo0 0 --elo1 5
  python tools/sprt.py --pentanomial 3 20 60 25 4 --elo0 -5 --elo1 0

Exit code: 0 = H1 accepted, 1 = H0 accepted, 2 = continue testing.
The score perspective is engine A's, as in strength_test.py; pass --for-b to
test the candidate (engine B) instead.
"""
import argparse
import json
import math
import sys


def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def gsprt_llr(pentanomial, elo0, elo1):
    """Generalised SPRT LLR (Michel Van den Bergh's formulation) for
    pentanomial pair outcomes with points 0, 0.5, 1, 1.5, 2 per pair."""
    n = sum(pentanomial)
    if n == 0:
        return 0.0
    probs = [c / n for c in pentanomial]
    outcomes = [0.0, 0.25, 0.5, 0.75, 1.0]  # pair score / 2
    mean = sum(p * x for p, x in zip(probs, outcomes))
    var = sum(p * (x - mean) ** 2 for p, x in zip(probs, outcomes))
    if var <= 0:
        return 0.0
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    # LLR ≈ n/2 * [ (s1 - s0) * (2 mean - s0 - s1) ] / var  (Gaussian approximation)
    return n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", help="summary.json from strength_test.py")
    ap.add_argument("--pentanomial", nargs=5, type=int, metavar="N", help="pair counts for A points 0,0.5,1,1.5,2")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--for-b", action="store_true", help="evaluate from engine B's perspective")
    args = ap.parse_args()
    if args.summary:
        s = json.load(open(args.summary))
        penta = list(s["pentanomial_a_points_0_to_2"])
    elif args.pentanomial:
        penta = list(args.pentanomial)
    else:
        sys.exit("need --summary or --pentanomial")
    if args.for_b:
        penta = penta[::-1]
    llr = gsprt_llr(penta, args.elo0, args.elo1)
    lower = math.log(args.beta / (1.0 - args.alpha))
    upper = math.log((1.0 - args.beta) / args.alpha)
    n = sum(penta)
    score = sum(c * x for c, x in zip(penta, [0.0, 0.25, 0.5, 0.75, 1.0])) / n if n else 0.5
    status = "H1" if llr >= upper else ("H0" if llr <= lower else "continue")
    print(json.dumps({"pairs": n, "score": round(score, 4), "pentanomial": penta, "llr": round(llr, 3),
                      "lower": round(lower, 3), "upper": round(upper, 3), "elo0": args.elo0, "elo1": args.elo1,
                      "result": status}))
    return {"H1": 0, "H0": 1, "continue": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
