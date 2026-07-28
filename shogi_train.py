"""
Training script for ShogiBT4.

Supports supervised learning from SFEN game records (kifu).
Training data format: one line per position:
    sfen <SFEN string> moves <move1> <move2> ... result <W|D|L>

Example:
    sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 bestmove 7g7f result W

Usage:
    python shogi_train.py --data train.sfen --epochs 10 --batch 64

Multi-GPU (DDP, one process per GPU; e.g. 8x RTX 5090):
    torchrun --nproc_per_node=8 shogi_train.py \
        --data ../shards2/aggshard --epochs 1 --lr 3e-4 --warmup-steps 4000 \
        --grad-clip 0.5 --bf16 --pre-norm --gated-attention \
        --d-model 1024 --encoders 20 --heads 16 \
        --batch 256 --grad-accum 1 --workers 12 --compile \
        --save-dir checkpoints/ --save-every 1 \
        --log-csv run.csv --log-file run.log
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# cshogi is only needed for the in-check feature plane during encoding. Import
# lazily so pure shard-loading training runs don't hard-depend on it.
try:
    import cshogi as _cshogi
except Exception:
    _cshogi = None

# Add lc0 src to path for the Shogi C++ modules (via ctypes or subprocess).
# For now we do everything in Python.
from shogi_model_v2 import (ShogiBT4v2, ShogiBT4v2Config,
                            make_direction_policy_index, POLICY_SIZE,
                            NUM_DIRECTIONS, NUM_PROMO_DIRECTIONS,
                            load_state_dict_with_promotion_migration,
                            strip_state_dict_wrapper_prefix)
# unpack_planes is pure numpy (no compiled .so needed) — safe to import for
# loading packed shards on any training machine.
import jhbr2_encoder


# =====================================================================
# Shogi board encoding (pure Python, mirrors C++ encoder)
# =====================================================================

def sfen_to_planes(sfen_str, in_check=None):
    """
    Convert an SFEN string to a (148, 9, 9) input tensor, dlshogi-style.
    Encodes from the side-to-move's perspective (flipped 180° for WHITE).

    in_check: optional bool for "side-to-move is in check" (the check plane).
        If None, it is computed via cshogi (a fresh Board per call). Callers that
        already hold a live cshogi board — e.g. the pack/PSV shard generators —
        should pass board.is_check() to avoid rebuilding the board per position.

    MUST stay bit-identical to PackShogiPosition() in shogi/encoder.cc and the
    GPU unpack kernel — see docs/nyugyoku_dlshogi_features.md for the layout.
    """
    parts = sfen_str.strip().split()
    board_str = parts[0]
    side = parts[1]  # 'b' or 'w'
    hand_str = parts[2]
    flip = (side == 'w')

    planes = np.zeros((148, 9, 9), dtype=np.float32)

    # Piece type mapping. Our pieces = planes 0-13, their pieces = planes 14-27.
    PIECE_PLANE = {
        'P': 0, 'L': 1, 'N': 2, 'S': 3, 'B': 4, 'R': 5, 'G': 6, 'K': 7,
    }
    PROMOTED_PLANE = {
        'P': 8, 'L': 9, 'N': 10, 'S': 11, 'B': 12, 'R': 13,
    }

    # --- features1: piece placement (planes 0..27) ---
    r, f = 0, 8  # Start top-left: rank a (idx 0), file 9 (idx 8)
    promoted = False
    for ch in board_str:
        if ch == '/':
            r += 1
            f = 8
            continue
        if ch == '+':
            promoted = True
            continue
        if ch.isdigit():
            f -= int(ch)
            continue

        is_black = ch.isupper()
        base_ch = ch.upper()
        if promoted and base_ch in PROMOTED_PLANE:
            plane_idx = PROMOTED_PLANE[base_ch]
        else:
            plane_idx = PIECE_PLANE.get(base_ch, -1)
        promoted = False
        if plane_idx < 0:
            f -= 1
            continue

        if flip:
            is_ours = not is_black
            sq_f, sq_r = 8 - f, 8 - r  # flip 180°
        else:
            is_ours = is_black
            sq_f, sq_r = f, r

        offset = 0 if is_ours else 14
        planes[offset + plane_idx, sq_r, sq_f] = 1.0
        f -= 1

    # --- features2 sub-layout (local indices; global plane = F1 + local) ---
    F1 = 28
    F2_OUR_HAND = 0
    F2_THEIR_HAND = 28
    F2_CHECK = 56
    F2_OUR_NYU = 57
    F2_THEIR_NYU = 88
    # F2_REPETITION = 119  # unknown from a bare SFEN; left 0 (see doc)

    HAND_ORDER = ['P', 'L', 'N', 'S', 'G', 'B', 'R']
    HAND_MAX = {'P': 8, 'L': 4, 'N': 4, 'S': 4, 'G': 4, 'B': 2, 'R': 2}

    # Parse hand counts into our/their dicts.
    our_hand = {k: 0 for k in HAND_ORDER}
    their_hand = {k: 0 for k in HAND_ORDER}
    if hand_str != '-':
        count = 0
        for ch in hand_str:
            if ch.isdigit():
                count = count * 10 + int(ch)
                continue
            if count == 0:
                count = 1
            is_black = ch.isupper()
            base = ch.upper()
            if base in HAND_MAX:
                is_ours = (not is_black) if flip else is_black
                (our_hand if is_ours else their_hand)[base] += count
            count = 0

    # Hand thermometer (unary): first min(count, max) planes of each run set.
    def fill_hand(base, hand):
        off = 0
        for p in HAND_ORDER:
            n = min(hand[p], HAND_MAX[p])
            for k in range(n):
                planes[F1 + base + off + k] = 1.0
            off += HAND_MAX[p]
    fill_hand(F2_OUR_HAND, our_hand)
    fill_hand(F2_THEIR_HAND, their_hand)

    # Check (side-to-move in check). Needs real board logic. Use the caller's
    # value if given, else fall back to building a cshogi board here.
    if in_check is None and _cshogi is not None:
        try:
            in_check = _cshogi.Board(sfen_str).is_check()
        except Exception:
            in_check = False
    if in_check:
        planes[F1 + F2_CHECK] = 1.0

    # Nyugyoku one-hot blocks (mirrors ComputeEnteringKingInfo + dlshogi fill).
    MAJOR_IDX = (4, 5, 12, 13)  # B, R, +B(Horse), +R(Dragon)
    for color_is_ours in (True, False):
        camp_ranks = (0, 1, 2) if color_is_ours else (6, 7, 8)
        offset = 0 if color_is_ours else 14
        hand = our_hand if color_is_ours else their_hand

        king_in_camp = False
        pieces_in_camp = 0
        points = 0
        for rr in camp_ranks:
            for ff in range(9):
                for pt_idx in range(14):
                    if planes[offset + pt_idx, rr, ff] > 0.5:
                        if pt_idx == 7:           # King: excluded from counts
                            king_in_camp = True
                        else:
                            pieces_in_camp += 1
                            points += 5 if pt_idx in MAJOR_IDX else 1
        for p in HAND_ORDER:
            points += hand[p] * (5 if p in ('B', 'R') else 1)

        base = F2_OUR_NYU if color_is_ours else F2_THEIR_NYU
        if king_in_camp:
            planes[F1 + base + 0] = 1.0
        rest_field = 10 - pieces_in_camp
        if rest_field < 10:
            planes[F1 + base + 1 + max(0, min(rest_field, 9))] = 1.0
        # Threshold: 28 for sente (original first player), 27 for gote.
        if color_is_ours:
            threshold = 28 if side == 'b' else 27
        else:
            threshold = 27 if side == 'b' else 28
        rest_point = threshold - points
        if rest_point < 20:
            planes[F1 + base + 11 + max(0, min(rest_point, 19))] = 1.0

    return planes


def move_to_policy_index(move_str, flip):
    """
    Convert a USI move string to a direction-based policy index (0-2186).
    If flip=True, the move is from WHITE's perspective and needs 180° rotation.
    """
    return make_direction_policy_index(move_str, flip)


# =====================================================================
# Dataset
# =====================================================================

class ShardedDataset(Dataset):
    """
    Memory-efficient dataset that loads ONE shard at a time.

    Only ~4 GB in memory (one 500K-position shard) instead of ~43 GB.
    Call load_shard(idx) to switch shards between training iterations.

    Usage:
        dataset = ShardedDataset("/path/to/prefix")
        for shard_id in dataset.shard_order():
            dataset.load_shard(shard_id)
            # train on this shard

    Shards may carry either a hard policy label (`policy`, one class index per
    sample) or an aggregated sparse policy distribution (`policy_indices` /
    `policy_weights` / `policy_offsets`, written by gen_agg_shards.py). For
    the latter, __getitem__ returns a dense normalized (2187,) float vector
    and the trainer uses a soft cross-entropy.
    """

    def __init__(self, prefix, verbose=True):
        # Discover shards by globbing (handles gaps from failed workers).
        import glob
        self.shard_paths = sorted(glob.glob(f"{prefix}_*.npz"))

        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found: {prefix}_*.npz")

        self.num_shards = len(self.shard_paths)
        self.packed = False               # bit-packed format? (auto-detected)
        self.packed1 = self.packed2 = None
        self.planes = None
        self.policy = None
        self.wdl = None
        self.mlh = None
        self.sparse_policy = False        # aggregated shards (auto-detected)
        self.p_idx = self.p_wt = self.p_off = None
        self.current_shard = -1

        # Load first shard to get the count + format.
        self.load_shard(0)
        self.shard_size = len(self.policy)

        if self.packed:
            gb = self.shard_size * 299 / 1e9
            fmt = f"packed bit (~{gb:.2f} GB/shard in RAM)"
        else:
            gb = self.shard_size * 148 * 81 * 2 / 1e9
            fmt = f"float16 planes (~{gb:.1f} GB/shard in RAM)"
        if verbose:
            print(f"Found {self.num_shards} shards, ~{self.shard_size:,} positions each")
            print(f"Total: ~{self.num_shards * self.shard_size / 1e6:.0f}M positions, {fmt}")

    def load_shard(self, shard_id):
        """Load a specific shard into memory (frees the previous one)."""
        if shard_id == self.current_shard:
            return
        # Free previous shard
        self.planes = self.packed1 = self.packed2 = None
        self.policy = self.wdl = self.mlh = None
        self.p_idx = self.p_wt = self.p_off = None

        data = np.load(self.shard_paths[shard_id])
        if 'packed1' in data.files:          # bit-packed shard
            self.packed = True
            self.packed1 = data['packed1']
            self.packed2 = data['packed2']
        else:                                # float16 planes shard
            self.packed = False
            self.planes = data['planes']
        self.policy = data['policy']
        # Aggregated shards (gen_agg_shards.py): sparse policy distribution.
        self.sparse_policy = 'policy_indices' in data.files
        if self.sparse_policy:
            self.p_idx = data['policy_indices']
            self.p_wt = data['policy_weights']
            self.p_off = data['policy_offsets']
        self.wdl = data['wdl']
        # mlh = remaining plies to game end (int16). Older shards may omit it.
        self.mlh = data['mlh'] if 'mlh' in data.files else None
        self.current_shard = shard_id

    def shard_order(self, shuffle=True, seed=None):
        """Return shard indices in (optionally seeded) random order.

        Pass a seed so every DDP rank computes the SAME order (required for
        lock-step training across ranks)."""
        import random
        order = list(range(self.num_shards))
        if shuffle:
            (random.Random(seed) if seed is not None else random).shuffle(order)
        return order

    def __len__(self):
        return len(self.policy)

    def __getitem__(self, idx):
        # mlh target: remaining plies, or -1 sentinel ("no MLH target") for
        # shards without an mlh array (loss masks these out).
        mlh = float(self.mlh[idx]) if self.mlh is not None else -1.0
        if self.packed:
            # Unpack this position's bits -> (148,9,9) float32 (pure numpy).
            planes = jhbr2_encoder.unpack_planes(self.packed1[idx], self.packed2[idx])
        else:
            planes = self.planes[idx].astype(np.float32)
        if self.sparse_policy:
            # Expand the sparse distribution to a dense soft target.
            vec = np.zeros(POLICY_SIZE, dtype=np.float32)
            s, e = self.p_off[idx], self.p_off[idx + 1]
            vec[self.p_idx[s:e].astype(np.int64)] = \
                self.p_wt[s:e].astype(np.float32)
            policy = torch.from_numpy(vec)
        else:
            policy = torch.tensor(self.policy[idx], dtype=torch.long)
        return (torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)),
                policy,
                torch.tensor(self.wdl[idx], dtype=torch.float32),
                torch.tensor(mlh, dtype=torch.float32))


class ShogiDataset(Dataset):
    """
    Loads training data from a text file (slow — use PrecomputedDataset instead).
    Each line: sfen <SFEN> [bestmove <move>] [score <score>] result <W|D|L>
    """

    def __init__(self, filepath, move_index_fn):
        self.samples = []
        self.move_index_fn = move_index_fn

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    sample = self._parse_line(line)
                    if sample:
                        self.samples.append(sample)
                except Exception:
                    continue

        print(f"Loaded {len(self.samples)} positions from {filepath}")

    def _parse_line(self, line):
        # Format: sfen <SFEN> [bestmove <move>] [score <score>] result <W|D|L>
        # bestmove is optional — if missing, policy_idx = -1 (value-only training)
        parts = line.split()
        if parts[0] == 'sfen':
            sfen_parts = []
            i = 1
            while i < len(parts) and parts[i] not in ('bestmove', 'result', 'score'):
                sfen_parts.append(parts[i])
                i += 1
            sfen = ' '.join(sfen_parts)

            bestmove = None
            result = None
            score = None
            while i < len(parts):
                if parts[i] == 'bestmove' and i + 1 < len(parts):
                    bestmove = parts[i + 1]
                    i += 2
                elif parts[i] == 'score' and i + 1 < len(parts):
                    score = int(parts[i + 1])
                    i += 2
                elif parts[i] == 'result' and i + 1 < len(parts):
                    result = parts[i + 1]
                    i += 2
                else:
                    i += 1

            if result:
                # Value-only record (no bestmove) or full record
                return (sfen, bestmove, result, score)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sfen, bestmove, result, score = self.samples[idx]
        flip = sfen.split()[1] == 'w'

        # Input planes
        planes = sfen_to_planes(sfen)

        # Policy target (-1 if no bestmove available)
        if bestmove:
            move_info = move_to_policy_index(bestmove, flip)
            policy_idx = self.move_index_fn(move_info)
        else:
            policy_idx = -1  # No policy target

        # Value target: WDL
        # If we have a score, use soft WDL derived from evaluation
        if score is not None:
            import math
            win_rate = 1.0 / (1.0 + math.exp(-score / 600.0))
            # Blend with game result for a softer target
            if result == 'W':
                hard = [1.0, 0.0, 0.0]
            elif result == 'D':
                hard = [0.0, 1.0, 0.0]
            else:
                hard = [0.0, 0.0, 1.0]
            # 70% score-based, 30% game result
            soft = [win_rate, 0.0, 1.0 - win_rate]
            wdl = [0.7 * s + 0.3 * h for s, h in zip(soft, hard)]
        elif result == 'W':
            wdl = [1.0, 0.0, 0.0]
        elif result == 'D':
            wdl = [0.0, 1.0, 0.0]
        else:  # L
            wdl = [0.0, 0.0, 1.0]

        return (torch.tensor(planes),
                torch.tensor(policy_idx, dtype=torch.long),
                torch.tensor(wdl))


# =====================================================================
# Training loop
# =====================================================================

def build_move_index():
    """Direction-based policy index: move_to_policy_index already returns the index directly."""
    # With v2 encoding, move_to_policy_index returns an int (0-2186) directly.
    # No separate lookup needed.
    return lambda idx: idx


class _Tee:
    """Duplicate a stream's writes into a log file (for --log-file)."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()


FINETUNE_SCOPES = ("promotion", "policy", "last", "full")


def load_training_checkpoint(path):
    """Load a checkpoint on CPU, using mmap when the PyTorch version supports it."""
    kwargs = dict(map_location="cpu", weights_only=False)
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def restore_checkpoint_config(checkpoint):
    """Construct the exact model configuration recorded in a checkpoint."""
    cfg = ShogiBT4v2Config()
    for key, value in checkpoint.get("cfg", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def configure_finetune_scope(model, scope, last_encoders=4):
    """Select trainable parameters before DDP wraps the model.

    DDP registers trainable parameters at construction time, so stages that
    change this scope must be separate invocations.  A checkpoint produced by
    one stage can be supplied to the next with ``--finetune-from``.
    """
    if scope not in FINETUNE_SCOPES:
        raise ValueError(f"Unknown fine-tune scope: {scope}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    def unfreeze(module):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    if scope == "promotion":
        unfreeze(model.policy_head.promotion_delta)
    elif scope == "policy":
        unfreeze(model.policy_head)
    elif scope == "last":
        if not 1 <= last_encoders <= len(model.encoders):
            raise ValueError(
                f"--finetune-last-encoders must be in [1, {len(model.encoders)}], "
                f"got {last_encoders}"
            )
        for encoder in model.encoders[-last_encoders:]:
            unfreeze(encoder)
        unfreeze(model.final_norm)
        unfreeze(model.policy_head)
        unfreeze(model.value_head)
        unfreeze(model.mlh_head)
        # smolgen_global is shared by every encoder.  Keep it frozen in this
        # partial-trunk stage so earlier blocks really remain frozen.
        for parameter in model.smolgen_global.parameters():
            parameter.requires_grad = False
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(f"Fine-tune scope {scope!r} selected no parameters")
    return trainable


def build_training_optimizer(model, base_lr, weight_decay, *,
                             grouped=False, trunk_lr=None, policy_lr=None,
                             promotion_lr=None):
    """Build AdamW, optionally with reproducible differential-LR groups."""
    if not grouped:
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable parameters")
        return torch.optim.AdamW(parameters, lr=base_lr,
                                 weight_decay=weight_decay)

    trunk_lr = base_lr if trunk_lr is None else trunk_lr
    policy_lr = base_lr if policy_lr is None else policy_lr
    promotion_lr = policy_lr if promotion_lr is None else promotion_lr
    buckets = {"trunk": [], "policy": [], "promotion": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("policy_head.promotion_delta."):
            buckets["promotion"].append(parameter)
        elif name.startswith("policy_head."):
            buckets["policy"].append(parameter)
        else:
            buckets["trunk"].append(parameter)

    lrs = {
        "trunk": trunk_lr,
        "policy": policy_lr,
        "promotion": promotion_lr,
    }
    groups = [
        {"params": buckets[name], "lr": lrs[name], "group_name": name}
        for name in ("trunk", "policy", "promotion")
        if buckets[name]
    ]
    if not groups:
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def optimizer_lr_summary(optimizer):
    return ", ".join(
        f"{group.get('group_name', f'group{i}')}={group['lr']:.2e}"
        for i, group in enumerate(optimizer.param_groups)
    )


def train(args):
    # --- Distributed (DDP) setup: auto-enabled when launched via torchrun ---
    #   single GPU / DataParallel:  python shogi_train.py ...
    #   DDP (recommended, scales):  torchrun --nproc_per_node=<N> shogi_train.py ...
    is_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_dist:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    num_gpus = torch.cuda.device_count()

    # Mirror everything rank 0 prints (stdout AND stderr) into --log-file, so
    # per-shard progress lines survive the terminal. Append mode: safe across
    # resumed runs.
    if args.log_file and is_main:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fh = open(args.log_file, 'a', buffering=1)
        sys.stdout = _Tee(sys.stdout, log_fh)
        sys.stderr = _Tee(sys.stderr, log_fh)

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    log(f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"Command: {' '.join(sys.argv)}")

    # TF32 for the fp32 matmuls that remain outside autocast (free speedup on
    # Ampere+ GPUs, e.g. RTX 5090).
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    if is_dist:
        log(f"DDP enabled: world_size={world_size} (one process per GPU, NCCL)")
    log(f"Device: {device}, visible GPUs: {num_gpus}")

    # Move index function (v2: direction-based, returns index directly)
    move_info_to_idx = build_move_index()

    # --- Checkpoint and model configuration ---
    checkpoint_path = args.finetune_from or args.resume
    checkpoint = None
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        log(f"Loading checkpoint metadata from {checkpoint_path}")
        checkpoint = load_training_checkpoint(checkpoint_path)
        cfg = restore_checkpoint_config(checkpoint)
    else:
        cfg = ShogiBT4v2Config()

    def apply_architecture_option(arg_value, attr, display_name):
        if arg_value is None:
            return
        current = getattr(cfg, attr)
        if checkpoint is not None and current != arg_value:
            raise ValueError(
                f"{display_name}={arg_value} conflicts with checkpoint value "
                f"{current}; fine-tuning must keep the checkpoint architecture"
            )
        setattr(cfg, attr, arg_value)

    apply_architecture_option(args.d_model, "embedding_size", "--d-model")
    apply_architecture_option(args.d_model, "policy_d_model", "--d-model")
    apply_architecture_option(args.encoders, "num_encoders", "--encoders")
    apply_architecture_option(args.heads, "num_heads", "--heads")
    if args.gated_attention:
        apply_architecture_option(True, "gated_attention", "--gated-attention")
    if args.pre_norm:
        apply_architecture_option(True, "pre_norm", "--pre-norm")

    model = ShogiBT4v2(cfg).to(device)

    # Resolve fine-tuning scope.  A resumed fine-tuning run restores the scope
    # saved in its checkpoint unless the user explicitly supplies one.
    saved_training = checkpoint.get("training_config", {}) if checkpoint else {}
    finetune_scope = (
        args.finetune_scope
        or (saved_training.get("finetune_scope") if args.resume else None)
        or "full"
    )
    finetune_last_encoders = (
        args.finetune_last_encoders
        if args.finetune_last_encoders is not None
        else (saved_training.get("finetune_last_encoders", 4)
              if args.resume else 4)
    )

    start_epoch = 0
    restored_opt_steps = 0
    if args.finetune_from:
        legacy = load_state_dict_with_promotion_migration(
            model, checkpoint["model"])
        if legacy:
            log("  Loaded legacy epoch-3 weights; promotion delta initialized to zero")
        else:
            log("  Loaded all model weights, including a trained promotion delta")
    elif args.resume:
        if checkpoint.get("partial_epoch"):
            raise ValueError(
                "This checkpoint stopped in the middle of an epoch and has no "
                "saved data cursor. Start the next stage with --finetune-from "
                "instead of --resume."
            )
        state_dict = strip_state_dict_wrapper_prefix(checkpoint["model"])
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "An exact --resume requires a checkpoint containing the current "
                "promotion-delta parameters. Use --finetune-from to migrate an "
                "older checkpoint without restoring its optimizer."
            ) from exc
        start_epoch = checkpoint.get("epoch", 0)
        restored_opt_steps = checkpoint.get("opt_steps", 0)
        log(f"  Resuming exact training state at epoch {start_epoch}")

    # Frozen/trainable selection must happen before DDP construction.
    if args.finetune_from or args.resume and saved_training.get("is_finetune"):
        trainable_params = configure_finetune_scope(
            model, finetune_scope, finetune_last_encoders)
    elif args.finetune_scope is not None:
        trainable_params = configure_finetune_scope(
            model, finetune_scope, finetune_last_encoders)
    else:
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)

    # Unwrapped reference for checkpoint save / ONNX export: shares parameters
    # with the DDP/DataParallel/torch.compile-wrapped `model`, but its
    # state_dict keys carry no 'module.' / '_orig_mod.' prefixes.
    base_model = model
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    if args.finetune_from or saved_training.get("is_finetune"):
        log(f"Fine-tune scope: {finetune_scope}"
            + (f" (last {finetune_last_encoders} encoders)"
               if finetune_scope == "last" else ""))

    use_grad_checkpoint = args.grad_checkpoint
    if use_grad_checkpoint and finetune_scope in ("promotion", "policy"):
        use_grad_checkpoint = False
        log("Gradient checkpointing ignored: the transformer trunk is frozen")
    if use_grad_checkpoint:
        model.gradient_checkpointing = True
        log("Gradient (activation) checkpointing: ON")
    accum = max(1, args.grad_accum)
    if accum > 1:
        log(f"Grad accumulation: {accum} micro-batches "
            f"(effective batch = {args.batch * accum * world_size})")

    saved_has_groups = bool(
        checkpoint and any(
            "group_name" in group
            for group in checkpoint.get("optimizer", {}).get("param_groups", [])
        )
    )
    grouped_optimizer = bool(
        args.finetune_from or saved_has_groups
        or args.trunk_lr is not None
        or args.policy_lr is not None
        or args.promotion_lr is not None
    )
    optimizer = build_training_optimizer(
        model, args.lr, args.wd, grouped=grouped_optimizer,
        trunk_lr=args.trunk_lr, policy_lr=args.policy_lr,
        promotion_lr=args.promotion_lr)
    if args.resume and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "The optimizer in this checkpoint does not match the selected "
                "fine-tune scope. Resume with the same scope, or start a new "
                "stage with --finetune-from."
            ) from exc
    log(f"Optimizer learning rates: {optimizer_lr_summary(optimizer)}")
    del checkpoint

    # --- Mixed precision: fp16 (default) or bf16 (--bf16, better for big models) ---
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_amp = (device.type == 'cuda')
    # GradScaler is only needed for fp16; bf16 has fp32 range and needs no scaling.
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and not args.bf16))
    if use_amp:
        log(f"Autocast: {'bfloat16 (no GradScaler)' if args.bf16 else 'float16 (GradScaler on)'}")

    warmup_steps = args.warmup_steps
    # The LR scheduler (linear warmup + per-STEP cosine) is built after the
    # dataset below, so T_max can be the true total number of optimizer steps.

    # --- Multi-GPU ---
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    elif num_gpus > 1:
        log(f"Using DataParallel across {num_gpus} GPUs "
            f"(tip: `torchrun --nproc_per_node={num_gpus} shogi_train.py` is faster)")
        model = torch.nn.DataParallel(model)

    # torch.compile AFTER the DDP wrap: Dynamo's DDPOptimizer then splits the
    # graph at gradient-bucket boundaries so comm/compute overlap survives.
    if args.compile:
        model = torch.compile(model)
        log("torch.compile: ON (first few batches are slow while compiling)")

    # --- Dataset ---
    use_psv = args.psv_dir is not None
    import glob
    use_sharded = not use_psv and len(glob.glob(f"{args.data}_*.npz")) > 0
    use_text = not use_psv and not use_sharded and os.path.exists(args.data)

    if use_psv:
        from psv_dataset import PSVShardedDataset
        log(f"Using PSV dataset from {args.psv_dir}")
        sharded_dataset = PSVShardedDataset(args.psv_dir)
    elif use_sharded:
        log("Using pre-computed binary dataset (fast, memory-efficient)")
        sharded_dataset = ShardedDataset(args.data, verbose=is_main)
    elif use_text:
        log("Using text dataset (slow — consider running precompute.py first)")
        dataset = ShogiDataset(args.data, move_info_to_idx)
    else:
        print(f"Training data not found: {args.data}")
        print("Creating a small synthetic dataset for testing...")
        create_synthetic_data(args.data + ".sfen")
        dataset = ShogiDataset(args.data + ".sfen", move_info_to_idx)
        use_text = True

    # --- LR schedule: linear warmup then per-STEP cosine over ALL optimizer steps ---
    # Per-step (not per-epoch) so the LR actually glides down after warmup instead
    # of sitting pinned at the peak for a whole epoch.
    if use_psv or use_sharded:
        n_shards = getattr(sharded_dataset, 'num_shards', 1)
        shard_sz = getattr(sharded_dataset, 'shard_size', None) or 500000
        approx_positions = n_shards * shard_sz
    else:
        approx_positions = len(dataset)
    steps_per_epoch = max(1, approx_positions // max(1, world_size) // args.batch // accum)
    total_opt_steps = max(1, (args.epochs - start_epoch) * steps_per_epoch)
    if args.max_steps is not None:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    warmup_steps = min(warmup_steps, total_opt_steps - 1)
    cosine_steps = max(1, total_opt_steps - warmup_steps)
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    if warmup_steps > 0:
        _warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_steps])
    else:
        scheduler = _cosine
    log(f"LR schedule: {warmup_steps} warmup + {cosine_steps} cosine = "
        f"{total_opt_steps} optimizer steps (~{steps_per_epoch}/epoch), stepped per-step")

    # --- Training loop ---
    opt_steps = restored_opt_steps  # persistent count stored in checkpoints
    run_opt_steps = 0               # optimizer steps in this invocation
    n_skipped = 0    # optimizer steps skipped due to non-finite gradients
    stop_requested = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_mlh_loss = 0
        n_batches = 0
        t0 = time.time()

        if use_psv or use_sharded:
            # Same shard order on every rank (seeded by epoch) → DDP stays in
            # lock-step (all ranks process the same shard at the same time).
            shard_order = sharded_dataset.shard_order(shuffle=True, seed=1234 + epoch)
            for si, shard_id in enumerate(shard_order):
                sharded_dataset.load_shard(shard_id)
                if is_dist:
                    # Each rank trains on a disjoint 1/world_size slice of the shard.
                    sampler = DistributedSampler(
                        sharded_dataset, num_replicas=world_size, rank=rank,
                        shuffle=True, drop_last=True)
                    sampler.set_epoch(epoch * sharded_dataset.num_shards + si)
                    loader = DataLoader(sharded_dataset, batch_size=args.batch,
                                        sampler=sampler, num_workers=args.workers,
                                        pin_memory=True, drop_last=True)
                else:
                    loader = DataLoader(sharded_dataset, batch_size=args.batch,
                                        shuffle=True, num_workers=args.workers,
                                        pin_memory=True, drop_last=True)

                shard_loss = 0.0   # windowed: loss over just this shard
                shard_n = 0
                for planes, policy_target, wdl_target, mlh_target in loader:
                    planes = planes.to(device, non_blocking=True)
                    policy_target = policy_target.to(device, non_blocking=True)
                    wdl_target = wdl_target.to(device, non_blocking=True)
                    mlh_target = mlh_target.to(device, non_blocking=True)

                    # Optimizer step on the last micro-batch of each accum window.
                    step_now = ((n_batches + 1) % accum == 0)

                    with torch.amp.autocast('cuda', dtype=amp_dtype,
                                            enabled=(device.type == 'cuda')):
                        policy_logits, wdl_logits, mlh = model(planes)

                        if policy_target.dim() > 1:
                            # Aggregated shards: soft cross-entropy against a
                            # normalized move distribution. Samples with an
                            # empty distribution (shouldn't occur) are masked.
                            has_policy = policy_target.sum(dim=-1) > 0
                            if has_policy.any():
                                logp = F.log_softmax(
                                    policy_logits[has_policy].float(), dim=-1)
                                policy_loss = -(policy_target[has_policy]
                                                * logp).sum(dim=-1).mean()
                            else:
                                policy_loss = torch.tensor(0.0, device=device)
                        else:
                            has_policy = policy_target >= 0
                            if has_policy.any():
                                policy_loss = F.cross_entropy(
                                    policy_logits[has_policy],
                                    policy_target[has_policy])
                            else:
                                policy_loss = torch.tensor(0.0, device=device)

                        value_loss = F.cross_entropy(wdl_logits, wdl_target)

                        # MLH: Huber regression on clipped remaining plies.
                        # mlh_target < 0 means "no MLH label" (masked out).
                        mlh_mask = mlh_target >= 0
                        if args.mlh_weight > 0 and mlh_mask.any():
                            tgt = torch.clamp(mlh_target[mlh_mask], max=float(args.mlh_clip))
                            mlh_loss = F.smooth_l1_loss(mlh[mlh_mask].squeeze(-1), tgt)
                        else:
                            mlh_loss = torch.tensor(0.0, device=device)

                        loss = (policy_loss + args.value_weight * value_loss
                                + args.mlh_weight * mlh_loss)

                    # Divide by accum so grads average over the window. Under DDP,
                    # skip the all-reduce until the window's final micro-batch.
                    scaled = scaler.scale(loss / accum)
                    if is_dist and not step_now:
                        with model.no_sync():
                            scaled.backward()
                    else:
                        scaled.backward()

                    if step_now:
                        scaler.unscale_(optimizer)
                        gnorm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip)
                        # Skip the update on a non-finite gradient. Essential with
                        # --bf16 (no GradScaler to auto-skip): one NaN/Inf step can
                        # collapse a deep model to constant output.
                        if torch.isfinite(gnorm):
                            scaler.step(optimizer)
                        else:
                            n_skipped += 1
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        # Per-step LR schedule (warmup + cosine). Guard against
                        # stepping cosine past T_max (would ramp the LR back up).
                        if run_opt_steps < total_opt_steps:
                            scheduler.step()
                        opt_steps += 1
                        run_opt_steps += 1

                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_mlh_loss += mlh_loss.item()
                    shard_loss += loss.item()
                    shard_n += 1
                    n_batches += 1
                    if args.max_steps is not None and run_opt_steps >= args.max_steps:
                        stop_requested = True
                        break

                # Print after each shard: current-shard loss (live signal) plus
                # the cumulative epoch average and current LR.
                elapsed = time.time() - t0
                samples = n_batches * args.batch * world_size
                speed = samples / max(elapsed, 1)
                cur_loss = shard_loss / max(shard_n, 1)
                cum_loss = total_loss / max(n_batches, 1)
                cur_lr = scheduler.get_last_lr()[0]
                log(f"  Epoch {epoch+1} shard {shard_id:03d}: loss={cur_loss:.4f} "
                    f"(cum {cum_loss:.4f}, lr {cur_lr:.2e}, "
                    f"{samples/1e6:.1f}M, {speed:.0f}/s)")
                if stop_requested:
                    break
        else:
            # Text dataset: single loader for entire epoch
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)

            for planes, policy_target, wdl_target in loader:
                planes = planes.to(device, non_blocking=True)
                policy_target = policy_target.to(device, non_blocking=True)
                wdl_target = wdl_target.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=amp_dtype,
                                        enabled=(device.type == 'cuda')):
                    policy_logits, wdl_logits, mlh = model(planes)

                    has_policy = policy_target >= 0
                    if has_policy.any():
                        policy_loss = F.cross_entropy(
                            policy_logits[has_policy], policy_target[has_policy])
                    else:
                        policy_loss = torch.tensor(0.0, device=device)

                    value_loss = F.cross_entropy(wdl_logits, wdl_target)
                    loss = policy_loss + args.value_weight * value_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if run_opt_steps < total_opt_steps:
                    scheduler.step()
                opt_steps += 1
                run_opt_steps += 1

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_batches += 1

                if args.max_steps is not None and run_opt_steps >= args.max_steps:
                    stop_requested = True
                    break

                if n_batches % 1000 == 0:
                    avg_loss = total_loss / n_batches
                    samples = n_batches * args.batch
                    elapsed = time.time() - t0
                    speed = samples / elapsed
                    print(f"  Epoch {epoch+1} batch {n_batches}: "
                          f"loss={avg_loss:.4f} "
                          f"({samples/1e6:.1f}M samples, {speed:.0f} samples/sec)")

        elapsed = time.time() - t0
        samples_per_sec = (n_batches * args.batch * world_size) / max(elapsed, 1)
        avg_loss = total_loss / max(n_batches, 1)
        avg_policy = total_policy_loss / max(n_batches, 1)
        avg_value = total_value_loss / max(n_batches, 1)
        avg_mlh = total_mlh_loss / max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]
        log(f"Epoch {epoch+1}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"policy={avg_policy:.4f}  "
            f"value={avg_value:.4f}  "
            f"mlh={avg_mlh:.4f}  "
            f"lr={lr:.6f}  "
            f"skipped={n_skipped}  "
            f"time={elapsed:.1f}s  "
            f"speed={samples_per_sec:.0f} samples/sec")

        # Log to CSV (rank 0 only)
        if args.log_csv and is_main:
            import csv
            write_header = not os.path.exists(args.log_csv)
            with open(args.log_csv, 'a', newline='') as csvf:
                writer = csv.writer(csvf)
                if write_header:
                    writer.writerow(['epoch', 'loss', 'policy_loss', 'value_loss',
                                     'mlh_loss', 'lr', 'time_sec', 'samples_per_sec',
                                     'total_samples', 'batches'])
                writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{avg_policy:.6f}',
                                 f'{avg_value:.6f}', f'{avg_mlh:.6f}', f'{lr:.8f}',
                                 f'{elapsed:.1f}', f'{samples_per_sec:.0f}',
                                 n_batches * args.batch * world_size, n_batches])

        # Save checkpoint (rank 0 only). A step-limited run always saves its
        # final state; continue it as a new stage with --finetune-from because
        # the trainer does not persist a mid-shard data cursor.
        periodic_save = ((epoch + 1) % args.save_every == 0)
        if (periodic_save or stop_requested) and is_main:
            os.makedirs(args.save_dir, exist_ok=True)
            filename = (f"shogi_bt4_step{opt_steps}.pt" if stop_requested
                        else f"shogi_bt4_epoch{epoch+1}.pt")
            path = os.path.join(args.save_dir, filename)
            torch.save({
                'epoch': epoch + 1,
                'opt_steps': opt_steps,
                'partial_epoch': stop_requested,
                'model': base_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': vars(cfg),
                'training_config': {
                    'is_finetune': bool(
                        args.finetune_from or saved_training.get("is_finetune")),
                    'finetune_scope': finetune_scope,
                    'finetune_last_encoders': finetune_last_encoders,
                    'trunk_lr': args.trunk_lr,
                    'policy_lr': args.policy_lr,
                    'promotion_lr': args.promotion_lr,
                },
            }, path)
            log(f"  Saved {path}")

        # Keep all ranks aligned before the next epoch.
        if is_dist:
            dist.barrier()
        if stop_requested:
            log(f"Stopped after requested {run_opt_steps} optimizer steps")
            break

    # Export to ONNX (rank 0 only)
    if args.export_onnx and is_main:
        export_onnx(base_model, cfg, args.export_onnx, device)

    if is_dist:
        dist.destroy_process_group()


def export_onnx(model, cfg, path, device):
    """Export the trained model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, cfg.input_planes, 9, 9, device=device)

    torch.onnx.export(
        model, dummy, path,
        input_names=['input_planes'],
        output_names=['policy', 'wdl', 'mlh'],
        dynamic_axes={
            'input_planes': {0: 'batch'},
            'policy': {0: 'batch'},
            'wdl': {0: 'batch'},
            'mlh': {0: 'batch'},
        },
        opset_version=18,
    )
    print(f"Exported ONNX model to {path}")


def create_synthetic_data(path, n=1000):
    """Create synthetic training data for testing the pipeline."""
    import random

    PIECES = "PLNSBRGK"
    with open(path, 'w') as f:
        for _ in range(n):
            # Just use starting position with random "best moves"
            sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
            moves = ["7g7f", "2g2f", "3g3f", "6i7h", "5g5f", "4g4f",
                      "2h7h", "2h3h", "2h6h", "1g1f"]
            move = random.choice(moves)
            result = random.choice(['W', 'D', 'L'])
            f.write(f"sfen {sfen} bestmove {move} result {result}\n")
    print(f"Created synthetic data: {path} ({n} positions)")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShogiBT4")
    parser.add_argument("--data", default="train.sfen", help="Training data file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--mlh-weight", type=float, default=0.1,
                        help="Weight of the moves-left (MLH) Huber loss "
                             "(0 disables MLH training)")
    parser.add_argument("--mlh-clip", type=int, default=80,
                        help="Clip MLH target (remaining plies) to this max")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--encoders", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--gated-attention", action="store_true",
                        help="Sigmoid-gate the attention output (Qwen-style); "
                             "improves deep-training stability")
    parser.add_argument("--pre-norm", action="store_true",
                        help="Pre-norm residuals instead of post-norm; much more "
                             "stable for deep encoder stacks")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-dir", default=".", help="Directory for checkpoint files")
    parser.add_argument("--export-onnx", default=None, help="Export ONNX after training")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", default=None,
        help="Exactly resume a current-format checkpoint, including optimizer")
    checkpoint_group.add_argument(
        "--finetune-from", default=None,
        help="Initialize model weights/config from a checkpoint but start a "
             "fresh optimizer and epoch schedule; migrates legacy epoch-3 weights")
    parser.add_argument(
        "--finetune-scope", choices=FINETUNE_SCOPES, default=None,
        help="Trainable parameters: promotion=new promotion branch only; "
             "policy=complete policy head; last=all heads plus the last N "
             "encoders; full=entire model (default: full)")
    parser.add_argument(
        "--finetune-last-encoders", type=int, default=None,
        help="Number of trailing encoders trained by --finetune-scope last "
             "(default: 4)")
    parser.add_argument(
        "--trunk-lr", type=float, default=None,
        help="Differential LR for non-policy parameters (default: --lr)")
    parser.add_argument(
        "--policy-lr", type=float, default=None,
        help="Differential LR for the existing policy head (default: --lr)")
    parser.add_argument(
        "--promotion-lr", type=float, default=None,
        help="Differential LR for the promotion delta (default: --policy-lr)")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop and save after this many optimizer steps in this invocation; "
             "useful for short promotion-only stages")
    parser.add_argument("--log-csv", default=None, help="Log training stats to CSV file")
    parser.add_argument("--log-file", default=None,
                        help="Also write all console output (per-shard progress "
                             "lines etc.) to this file, appending")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Accumulate grads over N micro-batches before an "
                             "optimizer step (effective batch = batch*N*num_gpus)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (~1.3-1.6x faster; the "
                             "first few batches are slow while it compiles)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Activation checkpointing on encoder blocks "
                             "(much less GPU memory, ~30%% slower)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 autocast instead of fp16 (fp32 range, "
                             "no GradScaler; more stable for big models on Ampere+)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient norm clip (1.0=default, 10.0 for from-scratch large models)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="LR warmup steps (0=disabled, 1000=recommended for large models)")
    parser.add_argument("--psv-dir", default=None,
                        help="PSV data directory (YaneuraOu format, .bin files)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (increase for PSV: 32-64)")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.finetune_last_encoders is not None and args.finetune_last_encoders < 1:
        parser.error("--finetune-last-encoders must be at least 1")
    train(args)
