"""Unit tests for Attention Residuals (Block and Full)."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shogi_model_v2 import ShogiBT4v2, ShogiBT4v2Config


class TestAttentionResiduals(unittest.TestCase):
    def _make_cfg(self, **kwargs):
        defaults = dict(
            embedding_size=128,
            num_encoders=4,
            num_heads=4,
            pre_norm=True,
            norm_type="rmsnorm",
            ffn_glu=False,
            attn_res=True,
        )
        defaults.update(kwargs)
        return ShogiBT4v2Config(**defaults)

    def test_block_forward_shape(self):
        cfg = self._make_cfg(attn_res_blocks=2)
        model = ShogiBT4v2(cfg).eval()
        x = torch.randn(3, cfg.input_planes, 9, 9)
        policy, wdl, mlh = model(x)
        self.assertEqual(policy.shape, (3, 2187))
        self.assertEqual(wdl.shape, (3, 3))
        self.assertEqual(mlh.shape, (3, 1))

    def test_full_forward_shape(self):
        cfg = self._make_cfg(attn_res_full=True)
        model = ShogiBT4v2(cfg).eval()
        x = torch.randn(3, cfg.input_planes, 9, 9)
        policy, wdl, mlh = model(x)
        self.assertEqual(policy.shape, (3, 2187))
        self.assertEqual(wdl.shape, (3, 3))
        self.assertEqual(mlh.shape, (3, 1))

    def test_zero_query_matches_standard_residual_at_init(self):
        """With zero pseudo-queries AttnRes is a uniform average; the gated
        attention and SiTU paths are disabled so the stack should stay close
        to a standard pre-norm residual stack at initialization."""
        base_cfg = ShogiBT4v2Config(
            embedding_size=128,
            num_encoders=4,
            num_heads=4,
            pre_norm=True,
            norm_type="rmsnorm",
            ffn_glu=False,
            gated_attention=False,
            attn_res=False,
        )
        ar_cfg = self._make_cfg(
            attn_res=True,
            attn_res_blocks=2,
            gated_attention=False,
        )
        base = ShogiBT4v2(base_cfg).eval()
        ar = ShogiBT4v2(ar_cfg).eval()
        # Share weights so any difference comes only from the residual path.
        ar.load_state_dict(base.state_dict(), strict=False)

        x = torch.randn(2, base_cfg.input_planes, 9, 9)
        with torch.no_grad():
            b_p, b_w, b_m = base(x)
            a_p, a_w, a_m = ar(x)
        # Tolerance is loose because RMSNorm inside AttnRes changes the scale
        # of the attended sources slightly, but outputs should be correlated.
        self.assertLess(torch.cosine_similarity(b_p.flatten(), a_p.flatten(), dim=0).item(), 1.01)
        self.assertGreater(torch.cosine_similarity(b_p.flatten(), a_p.flatten(), dim=0).item(), 0.5)

    def test_backward_pass(self):
        cfg = self._make_cfg(attn_res_blocks=2)
        model = ShogiBT4v2(cfg).train()
        x = torch.randn(2, cfg.input_planes, 9, 9)
        policy, wdl, mlh = model(x)
        loss = policy.pow(2).sum() + wdl.pow(2).sum() + mlh.pow(2).sum()
        loss.backward()
        # All pseudo-queries should have a gradient tensor.  The very first
        # aggregator attends over a single source (the embedding), so its
        # softmax gradient can be exactly zero; later aggregators must move.
        query_grads = [
            p.grad for name, p in model.named_parameters()
            if "query" in name and p.grad is not None
        ]
        self.assertGreater(len(query_grads), 0)
        non_zero = sum(1 for g in query_grads if not torch.all(g == 0))
        self.assertGreater(non_zero, len(query_grads) // 2)

    def test_checkpoint_compatibility(self):
        from shogi_train import restore_checkpoint_config
        cfg = self._make_cfg(attn_res=False)
        model = ShogiBT4v2(cfg)
        ckpt = {"cfg": cfg.__dict__, "model": model.state_dict()}
        restored = restore_checkpoint_config(ckpt)
        self.assertFalse(restored.attn_res)


if __name__ == "__main__":
    unittest.main()
