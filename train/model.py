"""JHBR5 value and policy networks in PyTorch.

Real-valued semantics that the engine's integer pipeline reproduces
(nnue/value_net.cc, nnue/policy_net.cc):

  value:  acc_g = W_g[active] summed + b_g            (three groups, A shared by two frames)
          act_g = clamp(acc_g[:h],0,1) * clamp(acc_g[h:],0,1)
          x = L2(cat(act_us, act_them, act_b)); y = L3(screlu(x)); z = L4(screlu(y)) + pst[active_b]
          softmax(z) -> (W, D, L)
  policy: hl = pairwise(W_p[active] + b_p);  logit(m) = out_w[bucket(m)] . hl + out_b[bucket(m)]

Architecture v2 (docs/NNUE_V2_DESIGN.md, selected per net by --arch v2):

  value:  group A rows are king-bucketed (no factorisers); activation is the
          full-width screlu; residual L3; L4 conditioned on the material phase
          bucket that arrives as batch['phase']:
          act_g = clamp(acc_g,0,1)^2 (full width); y1 = screlu(L2(cat(act)));
          h = y1 + screlu(L3(y1)); z = L4_w[phase] @ h + L4_b[phase] + pst[active_b]
  policy: v2 input mapping; hl = clamp(acc,0,1)^2 (full width); readout unchanged

Every feature index and move bucket comes from the engine through the `jhbr5`
module; this file contains no mapping arithmetic.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

import jhbr5

QA_CLIP = 127.0 / 128.0        # i8 weights at scale 128
BIAS_CLIP = 64.0               # i16 biases at scale 128 (headroom)
L2_CLIP = 32767.0 / 1024.0     # i16 weights at the largest calibrated QB
POLICY_OUT_BIAS_CLIP = 32767.0 / 128.0


def screlu(x):
    return torch.clamp(x, 0.0, 1.0) ** 2


def pairwise(x):
    h = x.shape[1] // 2
    return torch.clamp(x[:, :h], 0.0, 1.0) * torch.clamp(x[:, h:], 0.0, 1.0)


class SparseGroup(nn.Module):
    """EmbeddingBag(sum) + bias; `off` has n+1 entries (include_last_offset)."""

    def __init__(self, num_inputs, l1, typical_active, sparse=False):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_inputs, l1, mode="sum", include_last_offset=True, sparse=sparse)
        self.bias = nn.Parameter(torch.zeros(l1))
        bound = 1.0 / math.sqrt(typical_active)
        nn.init.uniform_(self.emb.weight, -bound, bound)

    def forward(self, idx, off):
        return self.emb(idx, off) + self.bias


class ValueNet(nn.Module):
    def __init__(self, l1=1024, factorise=True, sparse=False):
        super().__init__()
        self.l1 = l1
        self.factorise = factorise
        self.a = SparseGroup(jhbr5.GROUP_A_INPUTS, l1, 38, sparse)
        self.b = SparseGroup(jhbr5.GROUP_B_INPUTS, l1, 100, sparse)
        # Factoriser: a slot-only table added to every king square of group A,
        # folded into `a` at export (docs/DESIGN.md §9.3).
        self.p = nn.EmbeddingBag(jhbr5.NUM_SLOTS, l1, mode="sum", include_last_offset=True, sparse=sparse) if factorise else None
        # King-relative factoriser (piece type/owner x offset from the king);
        # the last row is a zero padding row for hand slots.
        self.kp = nn.EmbeddingBag(jhbr5.KPREL_INPUTS, l1, mode="sum", include_last_offset=True, sparse=sparse,
                                  padding_idx=jhbr5.KPREL_INPUTS - 1) if factorise else None
        if self.p is not None:
            nn.init.zeros_(self.p.weight)
            nn.init.zeros_(self.kp.weight)
        self.pst = nn.EmbeddingBag(jhbr5.GROUP_B_INPUTS, 3, mode="sum", include_last_offset=True, sparse=sparse)
        nn.init.zeros_(self.pst.weight)
        self.l2 = nn.Linear(3 * l1 // 2, jhbr5.VALUE_L2)
        self.l3 = nn.Linear(jhbr5.VALUE_L2, jhbr5.VALUE_L3)
        self.l4 = nn.Linear(jhbr5.VALUE_L3, 3)

    def group_a(self, idx, off, kp_idx=None):
        acc = self.a(idx, off)
        if self.p is not None:
            acc = acc + self.p(idx % jhbr5.NUM_SLOTS, off)
            if kp_idx is None:
                kp_idx = torch.from_numpy(jhbr5.kprel_index(idx.cpu().numpy())).to(idx.device)
            acc = acc + self.kp(kp_idx, off)
        return acc

    def forward(self, batch):
        acc_us = self.group_a(batch["a_us_idx"], batch["a_us_off"], batch.get("a_us_kp"))
        acc_them = self.group_a(batch["a_them_idx"], batch["a_them_off"], batch.get("a_them_kp"))
        acc_b = self.b(batch["b_idx"], batch["b_off"])
        h = torch.cat([pairwise(acc_us), pairwise(acc_them), pairwise(acc_b)], dim=1)
        x = self.l2(h)
        y = self.l3(screlu(x))
        z = self.l4(screlu(y)) + self.pst(batch["b_idx"], batch["b_off"])
        return z  # logits in order W, D, L

    @torch.no_grad()
    def clip_weights(self):
        self.a.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.b.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        if self.p is not None:
            self.p.weight.clamp_(-QA_CLIP / 2, QA_CLIP / 2)
            self.kp.weight.clamp_(-QA_CLIP / 2, QA_CLIP / 2)
        self.a.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.b.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.l2.weight.clamp_(-L2_CLIP, L2_CLIP)
        self.l2.bias.clamp_(-L2_CLIP, L2_CLIP)

    @torch.no_grad()
    def folded_a_weight(self):
        """Group A table with the factoriser folded in: [GROUP_A_INPUTS, l1]."""
        w = self.a.emb.weight.detach().clone()
        if self.p is not None:
            w = (w.view(81, jhbr5.NUM_SLOTS, self.l1) + self.p.weight.detach()[None]).reshape(-1, self.l1)
            kp_idx = torch.from_numpy(jhbr5.kprel_index(np.arange(jhbr5.GROUP_A_INPUTS, dtype=np.int64)))
            kp = self.kp.weight.detach()
            for start in range(0, w.shape[0], 16384):  # chunked gather to bound memory
                sl = slice(start, min(start + 16384, w.shape[0]))
                w[sl] += kp[kp_idx[sl].to(kp.device)].to(w.device)
        return w.clamp(-QA_CLIP, QA_CLIP)


class ValueNetV2(nn.Module):
    """v2 value net (docs/NNUE_V2_DESIGN.md §2): king-bucketed group A (no
    factorisers), full-width screlu, residual L3, phase-conditioned L4."""

    def __init__(self, l1=1024, sparse=False):
        super().__init__()
        self.l1 = l1
        self.a = SparseGroup(jhbr5.GROUP_A2_INPUTS, l1, 38, sparse)
        self.b = SparseGroup(jhbr5.GROUP_B_INPUTS, l1, 100, sparse)
        self.pst = nn.EmbeddingBag(jhbr5.GROUP_B_INPUTS, 3, mode="sum", include_last_offset=True, sparse=sparse)
        nn.init.zeros_(self.pst.weight)
        self.l2 = nn.Linear(3 * l1, jhbr5.VALUE2_L2)
        self.l3 = nn.Linear(jhbr5.VALUE2_L2, jhbr5.VALUE_L3)
        self.l4_w = nn.Parameter(torch.empty(jhbr5.PHASE_BUCKETS, 3, jhbr5.VALUE_L3))
        self.l4_b = nn.Parameter(torch.empty(jhbr5.PHASE_BUCKETS, 3))
        bound = 1.0 / math.sqrt(jhbr5.VALUE_L3)  # nn.Linear default init, per phase slice
        nn.init.uniform_(self.l4_w, -bound, bound)
        nn.init.uniform_(self.l4_b, -bound, bound)

    def forward(self, batch):
        acc_us = self.a(batch["a_us_idx"], batch["a_us_off"])
        acc_them = self.a(batch["a_them_idx"], batch["a_them_off"])
        acc_b = self.b(batch["b_idx"], batch["b_off"])
        act = torch.cat([screlu(acc_us), screlu(acc_them), screlu(acc_b)], dim=1)
        y1 = screlu(self.l2(act))
        h = y1 + screlu(self.l3(y1))
        phase = batch["phase"].long()
        z = (self.l4_w[phase] @ h.unsqueeze(2)).squeeze(2) + self.l4_b[phase]
        return z + self.pst(batch["b_idx"], batch["b_off"])  # logits in order W, D, L

    @torch.no_grad()
    def clip_weights(self):
        self.a.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.b.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.a.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.b.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.l2.weight.clamp_(-L2_CLIP, L2_CLIP)
        self.l2.bias.clamp_(-L2_CLIP, L2_CLIP)


class PolicyNet(nn.Module):
    def __init__(self, l1=4096, see=True, sparse=False):
        super().__init__()
        self.l1 = l1
        self.see = see
        self.rows = jhbr5.NUM_BUCKETS_SEE if see else jhbr5.NUM_BUCKETS
        self.hidden = SparseGroup(jhbr5.POLICY_INPUTS, l1, 55, sparse)
        self.out_w = nn.Embedding(self.rows, l1 // 2, sparse=sparse)
        self.out_b = nn.Embedding(self.rows, 1, sparse=sparse)
        nn.init.uniform_(self.out_w.weight, -1.0 / math.sqrt(l1 // 2), 1.0 / math.sqrt(l1 // 2))
        nn.init.zeros_(self.out_b.weight)

    # Moves per checkpointed chunk of the readout. The gathered rows
    # (chunk x l1/2 floats) are recomputed in backward, so peak memory is
    # bounded by the chunk instead of batch x legal moves.
    readout_chunk = 32768

    def _readout(self, buckets, hl, seg):
        return (self.out_w(buckets) * hl[seg]).sum(dim=1) + self.out_b(buckets).squeeze(1)

    def forward(self, batch):
        """Returns flat logits over all legal moves of the batch and the segment id per move."""
        hl = pairwise(self.hidden(batch["p_idx"], batch["p_off"]))
        seg = batch["mv_seg"]
        buckets = batch["mv_bucket"]
        total = buckets.shape[0]
        if total <= self.readout_chunk:
            return self._readout(buckets, hl, seg), seg
        # Always chunk (validation included); checkpoint only when gradients
        # are needed so the gathered rows are recomputed in backward.
        use_ckpt = self.training and torch.is_grad_enabled()
        parts = []
        for start in range(0, total, self.readout_chunk):
            end = min(start + self.readout_chunk, total)
            if use_ckpt:
                parts.append(torch.utils.checkpoint.checkpoint(
                    self._readout, buckets[start:end], hl, seg[start:end], use_reentrant=False))
            else:
                parts.append(self._readout(buckets[start:end], hl, seg[start:end]))
        return torch.cat(parts), seg

    @torch.no_grad()
    def clip_weights(self):
        self.hidden.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.hidden.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.out_w.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.out_b.weight.clamp_(-POLICY_OUT_BIAS_CLIP, POLICY_OUT_BIAS_CLIP)


class PolicyNetV2(nn.Module):
    """v2 policy net (docs/NNUE_V2_DESIGN.md §3): v2 input mapping, full-width
    screlu hidden layer, full-width bucket readout."""

    def __init__(self, l1=4096, see=True, sparse=False):
        super().__init__()
        self.l1 = l1
        self.see = see
        self.rows = jhbr5.NUM_BUCKETS_SEE if see else jhbr5.NUM_BUCKETS
        self.hidden = SparseGroup(jhbr5.POLICY2_INPUTS, l1, 95, sparse)
        self.out_w = nn.Embedding(self.rows, l1, sparse=sparse)
        self.out_b = nn.Embedding(self.rows, 1, sparse=sparse)
        nn.init.uniform_(self.out_w.weight, -1.0 / math.sqrt(l1), 1.0 / math.sqrt(l1))
        nn.init.zeros_(self.out_b.weight)

    # Moves per checkpointed chunk of the readout. The gathered rows
    # (chunk x l1 floats) are recomputed in backward, so peak memory is
    # bounded by the chunk instead of batch x legal moves.
    readout_chunk = 32768

    def _readout(self, buckets, hl, seg):
        return (self.out_w(buckets) * hl[seg]).sum(dim=1) + self.out_b(buckets).squeeze(1)

    def forward(self, batch):
        """Returns flat logits over all legal moves of the batch and the segment id per move."""
        hl = screlu(self.hidden(batch["p_idx"], batch["p_off"]))
        seg = batch["mv_seg"]
        buckets = batch["mv_bucket"]
        total = buckets.shape[0]
        if total <= self.readout_chunk:
            return self._readout(buckets, hl, seg), seg
        # Always chunk (validation included); checkpoint only when gradients
        # are needed so the gathered rows are recomputed in backward.
        use_ckpt = self.training and torch.is_grad_enabled()
        parts = []
        for start in range(0, total, self.readout_chunk):
            end = min(start + self.readout_chunk, total)
            if use_ckpt:
                parts.append(torch.utils.checkpoint.checkpoint(
                    self._readout, buckets[start:end], hl, seg[start:end], use_reentrant=False))
            else:
                parts.append(self._readout(buckets[start:end], hl, seg[start:end]))
        return torch.cat(parts), seg

    @torch.no_grad()
    def clip_weights(self):
        self.hidden.emb.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.hidden.bias.clamp_(-BIAS_CLIP, BIAS_CLIP)
        self.out_w.weight.clamp_(-QA_CLIP, QA_CLIP)
        self.out_b.weight.clamp_(-POLICY_OUT_BIAS_CLIP, POLICY_OUT_BIAS_CLIP)


def segment_log_softmax(logits, seg, n):
    """log-softmax of `logits` within segments (positions) given segment ids."""
    seg_max = torch.full((n,), -1e30, device=logits.device, dtype=logits.dtype)
    seg_max = seg_max.scatter_reduce(0, seg, logits, reduce="amax", include_self=True)
    shifted = logits - seg_max[seg]
    denom = torch.zeros(n, device=logits.device, dtype=logits.dtype).index_add_(0, seg, shifted.exp())
    return shifted - torch.log(denom[seg])


def wdl_from_score(score_cp, scale=340.0, offset=270.0):
    """Score (cp, stm) -> (W, D, L) with the nnue-pytorch/BulletOu win-rate model shape."""
    w = torch.sigmoid((score_cp - offset) / scale)
    l = torch.sigmoid((-score_cp - offset) / scale)
    d = torch.clamp(1.0 - w - l, min=0.0)
    t = torch.stack([w, d, l], dim=1)
    return t / t.sum(dim=1, keepdim=True)


def value_target(batch, lam, scale, offset):
    soft = wdl_from_score(batch["score"], scale, offset)
    r = batch["result"]
    hard = torch.stack([(r > 0.5).float(), (r.abs() < 0.5).float(), (r < -0.5).float()], dim=1)
    return lam * soft + (1.0 - lam) * hard


def value_loss(logits, target):
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def policy_loss(logits, seg, visits, n):
    """Cross-entropy against normalised visit counts over the legal moves."""
    total = torch.zeros(n, device=logits.device, dtype=visits.dtype).index_add_(0, seg, visits)
    target = visits / total.clamp(min=1e-9)[seg]
    logp = segment_log_softmax(logits, seg, n)
    return -(target * logp).sum() / n
