"""Quantisation to the engine's integer formats and a NumPy reference of the
engine's integer forward pass (nnue/value_net.cc, nnue/policy_net.cc).

The reference is used by test/test_net_roundtrip.py to check the exported
file against the C++ evaluator, and by export.py to calibrate QB.
"""
import numpy as np
import torch

import jhbr5

QA = jhbr5.QA
Q_PST = jhbr5.Q_PST
POLICY_QB = jhbr5.POLICY_QB
POLICY_SHIFT = jhbr5.POLICY_SHIFT
POLICY_FACTOR = jhbr5.POLICY_FACTOR


def _q(t, scale, dtype, lo, hi):
    return np.clip(np.rint(t.detach().cpu().numpy().astype(np.float64) * scale), lo, hi).astype(dtype)


def quantize_value(model, qb):
    l1 = model.l1
    a_w = model.folded_a_weight()
    return {
        "a_w": _q(a_w, QA, np.int8, -127, 127),
        "a_b": _q(model.a.bias, QA, np.int16, -32767, 32767),
        "b_w": _q(model.b.emb.weight, QA, np.int8, -127, 127),
        "b_b": _q(model.b.bias, QA, np.int16, -32767, 32767),
        "l2_w": _q(model.l2.weight, qb, np.int16, -32767, 32767).reshape(jhbr5.VALUE_L2, 3 * l1 // 2),
        "l2_b": _q(model.l2.bias, qb, np.int16, -32767, 32767),
        "l3_w": model.l3.weight.detach().cpu().numpy().astype(np.float32),
        "l3_b": model.l3.bias.detach().cpu().numpy().astype(np.float32),
        "l4_w": model.l4.weight.detach().cpu().numpy().astype(np.float32),
        "l4_b": model.l4.bias.detach().cpu().numpy().astype(np.float32),
        "pst": _q(model.pst.weight, Q_PST, np.int16, -32767, 32767),
    }


def quantize_policy(model):
    l1 = model.l1
    return {
        "l1_w": _q(model.hidden.emb.weight, QA, np.int8, -127, 127),
        "l1_b": _q(model.hidden.bias, QA, np.int16, -32767, 32767),
        "out_w": _q(model.out_w.weight, QA, np.int8, -127, 127).reshape(model.rows, l1 // 2),
        "out_b": _q(model.out_b.weight, QA, np.int16, -32767, 32767).reshape(model.rows),
    }


def _acc(w, b, idx):
    acc = b.astype(np.int32) + w[idx].astype(np.int32).sum(axis=0)
    assert acc.min() >= -32768 and acc.max() <= 32767, "int16 accumulator overflow"
    return acc


def _pairwise(acc, shift):
    h = acc.shape[0] // 2
    a = np.clip(acc[:h], 0, QA).astype(np.int32)
    b = np.clip(acc[h:], 0, QA).astype(np.int32)
    return (a * b) >> shift


def _screlu(x):
    return np.clip(x, 0.0, 1.0) ** 2


def value_forward_q(t, qb, a_us, a_them, b_idx):
    """Integer reference; returns (win, draw, loss) as float32 and the raw L2 sums."""
    act = np.concatenate([_pairwise(_acc(t["a_w"], t["a_b"], a_us), 0),
                          _pairwise(_acc(t["a_w"], t["a_b"], a_them), 0),
                          _pairwise(_acc(t["b_w"], t["b_b"], b_idx), 0)]).astype(np.int64)
    s = t["l2_w"].astype(np.int64) @ act  # [16]
    x = ((s.astype(np.float32) / np.float32(QA * QA)) + t["l2_b"].astype(np.float32)) / np.float32(qb)
    y = t["l3_w"] @ _screlu(x).astype(np.float32) + t["l3_b"]
    z = t["l4_w"] @ _screlu(y).astype(np.float32) + t["l4_b"]
    z = z + t["pst"][b_idx].astype(np.int32).sum(axis=0).astype(np.float32) / np.float32(Q_PST)
    e = np.exp(z - z.max())
    return (e / e.sum()).astype(np.float32), s


def value_forward_float(t_l2_input_fn):  # placeholder for symmetry; float path lives in model.py
    raise NotImplementedError


def policy_forward_q(t, p_idx, buckets):
    acc = _acc(t["l1_w"], t["l1_b"], p_idx)
    hl = _pairwise(acc, POLICY_SHIFT).astype(np.int64)
    rows = t["out_w"][buckets].astype(np.int64)  # [M, l1/2]
    dots = rows @ hl
    pre = dots.astype(np.float32) / np.float32(QA * POLICY_FACTOR) + t["out_b"][buckets].astype(np.float32)
    return pre / np.float32(POLICY_QB)


def calibrate_qb(model, feature_sets, max_qb=1024, headroom=8):
    """Largest power-of-two QB <= max_qb such that |L2 sum| * headroom < 2^31 on
    the calibration positions (docs/DESIGN.md §3.4)."""
    t = quantize_value(model, 1)  # l2 weights not needed at unit scale; recompute below
    w2 = model.l2.weight.detach().cpu().numpy().astype(np.float64)
    worst = 0.0
    for a_us, a_them, b_idx in feature_sets:
        act = np.concatenate([_pairwise(_acc(t["a_w"], t["a_b"], a_us), 0),
                              _pairwise(_acc(t["a_w"], t["a_b"], a_them), 0),
                              _pairwise(_acc(t["b_w"], t["b_b"], b_idx), 0)]).astype(np.float64)
        worst = max(worst, np.abs(w2 @ act).max())
    qb = max_qb
    while qb > 1 and worst * qb * headroom >= 2.0 ** 31:
        qb //= 2
    return qb


# --- Architecture v2 (docs/NNUE_V2_DESIGN.md §2-3) ---------------------------


def quantize_value_v2(model, qb):
    l1 = model.l1
    return {
        "a_w": _q(model.a.emb.weight, QA, np.int8, -127, 127),
        "a_b": _q(model.a.bias, QA, np.int16, -32767, 32767),
        "b_w": _q(model.b.emb.weight, QA, np.int8, -127, 127),
        "b_b": _q(model.b.bias, QA, np.int16, -32767, 32767),
        "l2_w": _q(model.l2.weight, qb, np.int16, -32767, 32767).reshape(jhbr5.VALUE2_L2, 3 * l1),
        "l2_b": _q(model.l2.bias, qb, np.int16, -32767, 32767),
        "l3_w": model.l3.weight.detach().cpu().numpy().astype(np.float32),
        "l3_b": model.l3.bias.detach().cpu().numpy().astype(np.float32),
        "l4_w": model.l4_w.detach().cpu().numpy().astype(np.float32),
        "l4_b": model.l4_b.detach().cpu().numpy().astype(np.float32),
        "pst": _q(model.pst.weight, Q_PST, np.int16, -32767, 32767),
    }


def quantize_policy_v2(model):
    l1 = model.l1
    return {
        "l1_w": _q(model.hidden.emb.weight, QA, np.int8, -127, 127),
        "l1_b": _q(model.hidden.bias, QA, np.int16, -32767, 32767),
        "out_w": _q(model.out_w.weight, QA, np.int8, -127, 127).reshape(model.rows, l1),
        "out_b": _q(model.out_b.weight, QA, np.int16, -32767, 32767).reshape(model.rows),
    }


def _screlu_full(acc, shift):
    return (np.clip(acc, 0, QA).astype(np.int32) ** 2) >> shift


def value_forward_q_v2(t, qb, a_us, a_them, b_idx, phase):
    """Integer reference of EvaluateV2 (nnue/value_net.cc); (win, draw, loss) and raw L2 sums."""
    act = np.concatenate([_screlu_full(_acc(t["a_w"], t["a_b"], a_us), 0),
                          _screlu_full(_acc(t["a_w"], t["a_b"], a_them), 0),
                          _screlu_full(_acc(t["b_w"], t["b_b"], b_idx), 0)]).astype(np.int64)
    s = t["l2_w"].astype(np.int64) @ act  # [32]
    x = ((s.astype(np.float32) / np.float32(QA * QA)) + t["l2_b"].astype(np.float32)) / np.float32(qb)
    y1 = _screlu(x).astype(np.float32)
    h = y1 + _screlu(t["l3_w"] @ y1 + t["l3_b"]).astype(np.float32)
    z = t["l4_w"][phase] @ h + t["l4_b"][phase]
    z = z + t["pst"][b_idx].astype(np.int32).sum(axis=0).astype(np.float32) / np.float32(Q_PST)
    e = np.exp(z - z.max())
    return (e / e.sum()).astype(np.float32), s


def policy_forward_q_v2(t, p_idx, buckets):
    acc = _acc(t["l1_w"], t["l1_b"], p_idx)
    hl = _screlu_full(acc, POLICY_SHIFT).astype(np.int64)
    rows = t["out_w"][buckets].astype(np.int64)  # [M, l1]
    dots = rows @ hl
    pre = dots.astype(np.float32) / np.float32(QA * POLICY_FACTOR) + t["out_b"][buckets].astype(np.float32)
    return pre / np.float32(POLICY_QB)


def calibrate_qb_v2(model, feature_sets, max_qb=1024, headroom=8):
    """v2 variant of calibrate_qb: full-width screlu activation, so the L2 dot
    spans 3*l1 terms of up to 128^2 (vs 3*l1/2 pairwise products for v1)."""
    t = quantize_value_v2(model, 1)
    w2 = model.l2.weight.detach().cpu().numpy().astype(np.float64)
    worst = 0.0
    for a_us, a_them, b_idx, _phase in feature_sets:
        act = np.concatenate([_screlu_full(_acc(t["a_w"], t["a_b"], a_us), 0),
                              _screlu_full(_acc(t["a_w"], t["a_b"], a_them), 0),
                              _screlu_full(_acc(t["b_w"], t["b_b"], b_idx), 0)]).astype(np.float64)
        worst = max(worst, np.abs(w2 @ act).max())
    qb = max_qb
    while qb > 1 and worst * qb * headroom >= 2.0 ** 31:
        qb //= 2
    return qb
