"""Shard dataset: each DataLoader worker owns a subset of shards and a C++
BatchReader (jhbr5.BatchReader) with its own shuffle buffer."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import jhbr5


def expand_shards(specs):
    paths = []
    for spec in specs:
        if os.path.isdir(spec):
            paths += sorted(glob.glob(os.path.join(spec, "*.rec")))
        else:
            paths += sorted(glob.glob(spec)) or [spec]
    return paths


def to_tensors(b, device=None):
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            out[k] = t.to(device, non_blocking=True) if device is not None else t
        else:
            out[k] = v
    n = int(b["n"])
    n_moves = torch.from_numpy(b["n_moves"]).long()
    seg = torch.repeat_interleave(torch.arange(n), n_moves)
    out["mv_seg"] = seg.to(device) if device is not None else seg
    return out


class RecordDataset(IterableDataset):
    def __init__(self, shards, batch_size, shuffle_buffer=100000, seed=1,
                 require_dist=False, see=True, loop=True, move_fallback=False):
        super().__init__()
        self.paths = expand_shards(shards)
        if not self.paths:
            raise FileNotFoundError(f"no shards match {shards}")
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.require_dist = require_dist
        self.see = see
        self.loop = loop
        self.move_fallback = move_fallback

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        paths = self.paths[wid::nw] if nw <= len(self.paths) else self.paths
        reader = jhbr5.BatchReader(paths, self.batch_size, self.shuffle_buffer,
                                   self.seed + 1000 * wid, self.require_dist, self.see, self.loop,
                                   self.move_fallback)
        while True:
            b = reader.next()
            if b is None:
                return
            yield to_tensors(b)
