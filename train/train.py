#!/usr/bin/env python3
"""JHBR5 trainer for the value and policy networks.

  python train/train.py --net value  --shards data/*.rec --l1 1024 --steps 200000 --out runs/v1
  python train/train.py --net policy --shards data/*.rec --l1 4096 --steps 200000 --out runs/p1

Checkpoints (`ckpt_<step>.pt`, `ckpt_last.pt`) contain the float weights;
`export.py` (or `--export`) produces the engine's .nn file.
"""
import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jhbr5  # noqa: E402
from data import RecordDataset  # noqa: E402
from export import export_policy, export_policy_v2, export_value, export_value_v2, DEFAULT_CALIB  # noqa: E402
from model import PolicyNet, PolicyNetV2, ValueNet, ValueNetV2, policy_loss, value_loss, value_target  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=["value", "policy"], required=True)
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1", help="NNUE architecture (docs/NNUE_V2_DESIGN.md)")
    ap.add_argument("--shards", nargs="+", required=True, help="shard files, globs or directories")
    ap.add_argument("--l1", type=int, default=None, help="value: 1024, policy: 4096")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--schedule", choices=["exp", "cosine", "flat-cosine"], default="flat-cosine",
                    help="lr after warmup: exponential decay to lr-final, cosine to lr-final, or flat then cosine")
    ap.add_argument("--flat-frac", type=float, default=0.6, help="flat-cosine: fraction of steps at the peak lr")
    ap.add_argument("--val-shards", nargs="*", default=[], help="held-out shards for the validation loss")
    ap.add_argument("--val-batches", type=int, default=16, help="validation batches (of --batch-size) held in memory")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--wdl-lambda", type=float, default=0.7, help="weight of the score-derived WDL vs the game result")
    ap.add_argument("--score-scale", type=float, default=340.0)
    ap.add_argument("--score-offset", type=float, default=270.0)
    ap.add_argument("--no-see", action="store_true", help="policy: bucket table without SEE doubling")
    ap.add_argument("--policy-target", choices=["dist", "auto"], default="auto",
                    help="policy: 'dist' uses only records with a visit distribution; 'auto' also uses the played move of game records as a one-hot target")
    ap.add_argument("--no-factoriser", action="store_true")
    ap.add_argument("--shuffle-buffer", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume")
    ap.add_argument("--fresh-opt", action="store_true",
                    help="with --resume: keep weights but reset step to 0 and optimizer state")
    ap.add_argument("--export", help="write the .nn file here at the end")
    return ap.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    total = max(1, args.steps - args.warmup)
    t = min(1.0, (step - args.warmup) / total)
    if args.schedule == "exp":
        return args.lr * (args.lr_final / args.lr) ** t
    if args.schedule == "flat-cosine":
        if t < args.flat_frac:
            return args.lr
        t = (t - args.flat_frac) / max(1e-9, 1.0 - args.flat_frac)
    return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1.0 + math.cos(math.pi * t))


def load_validation(args, see):
    if not args.val_shards:
        return []
    ds = RecordDataset(args.val_shards, args.batch_size, max(args.batch_size, 4096), 12345,
                       require_dist=(args.net == "policy"), see=see, loop=False,
                       move_fallback=(args.net == "policy" and args.policy_target == "auto"),
                       arch=2 if args.arch == "v2" else 1)
    batches = []
    for b in ds:
        batches.append(b)
        if len(batches) >= args.val_batches:
            break
    return batches


@torch.no_grad()
def validate(model, batches, args, device):
    model.eval()
    total, n_total = 0.0, 0
    for b in batches:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        n = b["n"]
        if args.net == "value":
            loss = value_loss(model(b), value_target(b, args.wdl_lambda, args.score_scale, args.score_offset))
        else:
            logits, seg = model(b)
            loss = policy_loss(logits, seg, b["mv_visits"], n)
        total += loss.item() * n
        n_total += n
    model.train()
    return total / max(1, n_total)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    see = not args.no_see
    v2 = args.arch == "v2"
    l1 = args.l1 or (1024 if args.net == "value" else 4096)

    if args.net == "value":
        model = ValueNetV2(l1) if v2 else ValueNet(l1, factorise=not args.no_factoriser)
    else:
        model = PolicyNetV2(l1, see=see) if v2 else PolicyNet(l1, see=see)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["state_dict"])
        if not args.fresh_opt:
            opt.load_state_dict(ck["optimizer"])
            step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    ds = RecordDataset(args.shards, args.batch_size, args.shuffle_buffer, args.seed,
                       require_dist=(args.net == "policy"), see=see, loop=True,
                       move_fallback=(args.net == "policy" and args.policy_target == "auto"),
                       arch=2 if v2 else 1)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    fs_id = jhbr5.feature_set_id_v2() if v2 else jhbr5.feature_set_id()
    print(f"{args.net} net arch={args.arch} l1={l1} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"device={device} shards={len(ds.paths)} feature_set_id={fs_id}")

    def save(tag):
        ck = {"net": args.net, "arch": args.arch, "l1": l1, "see": see, "factorise": not args.no_factoriser,
              "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
              "feature_set_id": fs_id, "bucket_table_id": jhbr5.bucket_table_id(see)}
        torch.save(ck, os.path.join(args.out, f"ckpt_{tag}.pt"))

    val_batches = load_validation(args, see)
    if val_batches:
        print(f"validation: {sum(b['n'] for b in val_batches)} positions from {len(args.val_shards)} shard specs")
    best_val = float("inf")
    model.train()
    t0 = time.time()
    positions = 0
    running = 0.0
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            continue
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        n = batch["n"]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        if args.net == "value":
            logits = model(batch)
            loss = value_loss(logits, value_target(batch, args.wdl_lambda, args.score_scale, args.score_offset))
        else:
            logits, seg = model(batch)
            loss = policy_loss(logits, seg, batch["mv_visits"], n)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.clip_weights()
        step += 1
        positions += n
        running += loss.item()
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step} loss {running / args.log_every:.4f} lr {lr_at(step, args):.2e} "
                  f"pos/s {positions / dt:.0f}", flush=True)
            running = 0.0
        if val_batches and step % args.val_every == 0:
            v = validate(model, val_batches, args, device)
            print(f"step {step} val_loss {v:.4f}", flush=True)
            if v < best_val:
                best_val = v
                save("best")
        if step % args.save_every == 0:
            save(step)
            save("last")
    save("last")
    if val_batches:
        print(f"final val_loss {validate(model, val_batches, args, device):.4f} (best {best_val:.4f})")
    if args.export:
        model.eval()
        model.cpu()
        if args.net == "value":
            qb = (export_value_v2 if v2 else export_value)(model, args.export, DEFAULT_CALIB)
            print(f"exported {args.export} qb={qb}")
        else:
            (export_policy_v2 if v2 else export_policy)(model, args.export)
            print(f"exported {args.export}")


if __name__ == "__main__":
    main()
