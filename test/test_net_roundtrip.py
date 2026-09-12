#!/usr/bin/env python3
"""Round trip: PyTorch model -> quantised .nn -> C++ evaluator, compared against
the NumPy integer reference and against the float model.

  test_net_roundtrip.py --build <dir with jhbr5*.so> --eval <eval_positions> --sfens <file> [--tmp dir]

Exit 77 (ctest SKIP) if torch or the module are unavailable.
"""
import argparse
import os
import struct
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--build", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--sfens", required=True)
ap.add_argument("--tmp", default="/tmp")
args = ap.parse_args()

sys.path.insert(0, args.build)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
try:
    import numpy as np
    import torch
    import jhbr5
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

from export import export_policy, export_policy_v2, export_value, export_value_v2  # noqa: E402
from model import PolicyNet, PolicyNetV2, ValueNet, ValueNetV2, pairwise, screlu  # noqa: E402
from quantized import (policy_forward_q, policy_forward_q_v2, quantize_policy, quantize_policy_v2,  # noqa: E402
                       quantize_value, quantize_value_v2, value_forward_q, value_forward_q_v2)

torch.manual_seed(3)
sfens = [l.split("\t")[0].strip() for l in open(args.sfens) if l.strip()][:32]


def run_eval(vpath, ppath):
    """Runs eval_positions; success asserts the engine loaders accept the files."""
    out = subprocess.run([args.eval, vpath, ppath, args.sfens], capture_output=True, text=True, check=True).stdout
    engine = {}
    cur = None
    for line in out.splitlines():
        tag = line.split()[0]
        if tag == "S":
            cur = line[2:]
            engine[cur] = {"moves": {}}
        elif tag == "W":
            engine[cur]["wdl"] = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
        elif tag == "M":
            _, usi, logit = line.split()
            engine[cur]["moves"][usi] = float(logit)
    return engine


def header_version_nphase(path):
    with open(path, "rb") as f:
        h = f.read(82)
    return struct.unpack_from("<I", h, 8)[0], struct.unpack_from("<H", h, 80)[0]

value = ValueNet(256, factorise=True).eval()
with torch.no_grad():  # make the factoriser and PST non-trivial
    value.p.weight.uniform_(-0.05, 0.05)
    value.pst.weight.uniform_(-0.5, 0.5)
    value.l2.weight.uniform_(-2.0, 2.0)
policy = PolicyNet(512, see=True).eval()
with torch.no_grad():
    policy.out_b.weight.uniform_(-1.0, 1.0)
value.clip_weights()
policy.clip_weights()

vpath = os.path.join(args.tmp, "jhbr5_rt_value.nn")
ppath = os.path.join(args.tmp, "jhbr5_rt_policy.nn")
qb = export_value(value, vpath, sfens)
export_policy(policy, ppath)
tv = quantize_value(value, qb)
tp = quantize_policy(policy)
assert header_version_nphase(vpath) == (1, 0)
assert header_version_nphase(ppath) == (1, 0)

engine = run_eval(vpath, ppath)

fails = 0
max_wdl, max_logit, max_wdl_float, max_logit_float = 0.0, 0.0, 0.0, 0.0
for sfen in sfens:
    e = engine[sfen]
    a_us, a_them, b = jhbr5.value_features(sfen)
    ref_wdl, _ = value_forward_q(tv, qb, a_us, a_them, b)
    d = float(np.abs(ref_wdl - e["wdl"]).max())
    max_wdl = max(max_wdl, d)
    if d > 2e-5:
        fails += 1
        print(f"FAIL wdl {sfen}: engine {e['wdl']} ref {ref_wdl}")
    # float model for the quantisation-error report
    with torch.no_grad():
        batch = {"a_us_idx": torch.from_numpy(a_us), "a_us_off": torch.tensor([0, len(a_us)]),
                 "a_them_idx": torch.from_numpy(a_them), "a_them_off": torch.tensor([0, len(a_them)]),
                 "b_idx": torch.from_numpy(b), "b_off": torch.tensor([0, len(b)])}
        fl = torch.softmax(value(batch), dim=1)[0].numpy()
    max_wdl_float = max(max_wdl_float, float(np.abs(fl - e["wdl"]).max()))

    moves = jhbr5.legal_moves(sfen)
    buckets = np.array([jhbr5.move_bucket(sfen, m, True) for m in moves], dtype=np.int64)
    p_idx = jhbr5.policy_features(sfen)
    ref_logits = policy_forward_q(tp, p_idx, buckets)
    for m, r in zip(moves, ref_logits):
        d = abs(float(r) - e["moves"][m])
        max_logit = max(max_logit, d)
        if d > 2e-4:
            fails += 1
            print(f"FAIL logit {sfen} {m}: engine {e['moves'][m]} ref {r}")
    with torch.no_grad():
        hl = pairwise(policy.hidden(torch.from_numpy(p_idx), torch.tensor([0, len(p_idx)])))
        w = policy.out_w(torch.from_numpy(buckets))
        fl = ((w * hl).sum(dim=1) + policy.out_b(torch.from_numpy(buckets)).squeeze(1)).numpy()
    max_logit_float = max(max_logit_float, float(np.abs(fl - np.array([e["moves"][m] for m in moves])).max()))

os.remove(vpath)
os.remove(ppath)
print(f"test_net_roundtrip arch v1: {len(sfens)} positions, qb={qb}, engine vs integer reference: "
      f"wdl {max_wdl:.2e} logit {max_logit:.2e}; engine vs float model: wdl {max_wdl_float:.3e} "
      f"logit {max_logit_float:.3e}; {'FAILED' if fails else 'ok'}")

# --- arch v2 (docs/NNUE_V2_DESIGN.md) -----------------------------------------
value2 = ValueNetV2(256).eval()
with torch.no_grad():  # non-trivial PST and L2 (wide 3*l1 L2 input: QB calibration covers it)
    value2.pst.weight.uniform_(-0.5, 0.5)
    value2.l2.weight.uniform_(-2.0, 2.0)
policy2 = PolicyNetV2(512, see=True).eval()
with torch.no_grad():
    policy2.out_b.weight.uniform_(-1.0, 1.0)
value2.clip_weights()
policy2.clip_weights()

v2path = os.path.join(args.tmp, "jhbr5_rt_value2.nn")
p2path = os.path.join(args.tmp, "jhbr5_rt_policy2.nn")
qb2 = export_value_v2(value2, v2path, sfens)
export_policy_v2(policy2, p2path)
tv2 = quantize_value_v2(value2, qb2)
tp2 = quantize_policy_v2(policy2)
assert header_version_nphase(v2path) == (2, jhbr5.PHASE_BUCKETS)
assert header_version_nphase(p2path) == (2, 0)

engine2 = run_eval(v2path, p2path)

max_wdl, max_logit, max_wdl_float, max_logit_float = 0.0, 0.0, 0.0, 0.0
for sfen in sfens:
    e = engine2[sfen]
    a_us, a_them, b, phase = jhbr5.value_features(sfen, arch=2)
    ref_wdl, _ = value_forward_q_v2(tv2, qb2, a_us, a_them, b, phase)
    d = float(np.abs(ref_wdl - e["wdl"]).max())
    max_wdl = max(max_wdl, d)
    if d > 2e-5:
        fails += 1
        print(f"FAIL v2 wdl {sfen}: engine {e['wdl']} ref {ref_wdl}")
    # float model for the quantisation-error report
    with torch.no_grad():
        batch = {"a_us_idx": torch.from_numpy(a_us), "a_us_off": torch.tensor([0, len(a_us)]),
                 "a_them_idx": torch.from_numpy(a_them), "a_them_off": torch.tensor([0, len(a_them)]),
                 "b_idx": torch.from_numpy(b), "b_off": torch.tensor([0, len(b)]),
                 "phase": torch.tensor([phase])}
        fl = torch.softmax(value2(batch), dim=1)[0].numpy()
    max_wdl_float = max(max_wdl_float, float(np.abs(fl - e["wdl"]).max()))

    moves = jhbr5.legal_moves(sfen)
    buckets = np.array([jhbr5.move_bucket(sfen, m, True) for m in moves], dtype=np.int64)
    p_idx = jhbr5.policy_features(sfen, arch=2)
    ref_logits = policy_forward_q_v2(tp2, p_idx, buckets)
    for m, r in zip(moves, ref_logits):
        d = abs(float(r) - e["moves"][m])
        max_logit = max(max_logit, d)
        if d > 2e-4:
            fails += 1
            print(f"FAIL v2 logit {sfen} {m}: engine {e['moves'][m]} ref {r}")
    with torch.no_grad():
        hl = screlu(policy2.hidden(torch.from_numpy(p_idx), torch.tensor([0, len(p_idx)])))
        w = policy2.out_w(torch.from_numpy(buckets))
        fl = ((w * hl).sum(dim=1) + policy2.out_b(torch.from_numpy(buckets)).squeeze(1)).numpy()
    max_logit_float = max(max_logit_float, float(np.abs(fl - np.array([e["moves"][m] for m in moves])).max()))

os.remove(v2path)
os.remove(p2path)
print(f"test_net_roundtrip arch v2: {len(sfens)} positions, qb={qb2}, engine vs integer reference: "
      f"wdl {max_wdl:.2e} logit {max_logit:.2e}; engine vs float model: wdl {max_wdl_float:.3e} "
      f"logit {max_logit_float:.3e}; {'FAILED' if fails else 'ok'}")
sys.exit(1 if fails else 0)
