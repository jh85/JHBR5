#!/usr/bin/env python3
"""Export a trained checkpoint to the engine's .nn format (docs/NNUE_FORMAT.md).

  python train/export.py --checkpoint ckpt.pt --out value.nn [--calib-sfens file]
"""
import argparse
import sys

import numpy as np
import torch

import jhbr5
from model import PolicyNet, PolicyNetV2, ValueNet, ValueNetV2
from quantized import (calibrate_qb, calibrate_qb_v2, quantize_policy, quantize_policy_v2,
                       quantize_value, quantize_value_v2)

DEFAULT_CALIB = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "ln1g1g1nl/1ks2r1b1/1pppp1spp/p4pp2/9/2P1P4/PPSP1PPPP/1BG1R2S1/LN1GK2NL w - 1",
    "l2g4l/1ks1g4/2n1s1n2/pp1pppb1p/2p3ppP/P1P1PSP2/1PSP1PN2/1KGB3R1/LN1G4L w Rp 1",
    "ln4knl/2s1g2g1/p1pp1s1pp/1p2p1p2/4P1P2/2PP1P3/PPS3N1P/2GS3R1/LN1GK3L b BRbp 1",
    "l3k2nl/6g2/p1ns1p1pp/2ppp1p2/1p7/2PPPP3/PPS2SPPP/2G1K1R2/LN5NL w BGRbgs 1",
    "l1r4nl/2g1k1g2/p2pspspp/2p1p1p2/1p7/2P1P4/PP1PSPPPP/2GK2S1R/LN3G1NL b BNbp 1",
]


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    v2 = ck.get("arch", "v1") == "v2"
    if ck["net"] == "value":
        model = ValueNetV2(ck["l1"]) if v2 else ValueNet(ck["l1"], factorise=ck.get("factorise", True))
    else:
        model = PolicyNetV2(ck["l1"], see=ck.get("see", True)) if v2 else PolicyNet(ck["l1"], see=ck.get("see", True))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck, model


def export_value(model, out_path, calib_sfens, max_qb=1024):
    feats = [jhbr5.value_features(s) for s in calib_sfens]
    qb = calibrate_qb(model, feats, max_qb=max_qb)
    t = quantize_value(model, qb)
    jhbr5.write_net(out_path, jhbr5.NET_KIND_VALUE, [model.l1, model.l1, model.l1, 0],
                    jhbr5.VALUE_L2, jhbr5.VALUE_L3, jhbr5.QA, qb, jhbr5.Q_PST, False, t)
    return qb


def export_policy(model, out_path):
    t = quantize_policy(model)
    jhbr5.write_net(out_path, jhbr5.NET_KIND_POLICY, [model.l1, 0, 0, 0], 0, 0,
                    jhbr5.QA, jhbr5.POLICY_QB, jhbr5.Q_PST, model.see, t)


def export_value_v2(model, out_path, calib_sfens, max_qb=1024):
    feats = [jhbr5.value_features(s, arch=2) for s in calib_sfens]
    qb = calibrate_qb_v2(model, feats, max_qb=max_qb)
    t = quantize_value_v2(model, qb)
    jhbr5.write_net(out_path, jhbr5.NET_KIND_VALUE, [model.l1, model.l1, model.l1, 0],
                    jhbr5.VALUE2_L2, jhbr5.VALUE_L3, jhbr5.QA, qb, jhbr5.Q_PST, False, t,
                    version=2, n_phase=jhbr5.PHASE_BUCKETS)
    return qb


def export_policy_v2(model, out_path):
    t = quantize_policy_v2(model)
    jhbr5.write_net(out_path, jhbr5.NET_KIND_POLICY, [model.l1, 0, 0, 0], 0, 0,
                    jhbr5.QA, jhbr5.POLICY_QB, jhbr5.Q_PST, model.see, t, version=2, n_phase=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["v1", "v2"], default=None,
                    help="default: read from the checkpoint (v1 for older checkpoints)")
    ap.add_argument("--calib-sfens", help="file of sfens for QB calibration (value nets)")
    ap.add_argument("--max-qb", type=int, default=1024)
    args = ap.parse_args()
    ck, model = load_checkpoint(args.checkpoint)
    v2 = (args.arch or ck.get("arch", "v1")) == "v2"
    if ck["net"] == "value":
        sfens = DEFAULT_CALIB
        if args.calib_sfens:
            with open(args.calib_sfens) as f:
                sfens = [l.split("\t")[0].strip() for l in f if l.strip()][:4096]
        qb = (export_value_v2 if v2 else export_value)(model, args.out, sfens, args.max_qb)
        print(f"wrote {args.out}: value l1={model.l1} qb={qb} arch={'v2' if v2 else 'v1'}")
    else:
        (export_policy_v2 if v2 else export_policy)(model, args.out)
        print(f"wrote {args.out}: policy l1={model.l1} see={model.see} rows={model.rows} arch={'v2' if v2 else 'v1'}")


if __name__ == "__main__":
    sys.exit(main())
