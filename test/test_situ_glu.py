#!/usr/bin/env python3
"""Unit tests for SiTU-GLU and the GLU feed-forward network."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shogi_model_v2 import (
    ShogiBT4v2,
    ShogiBT4v2Config,
    SiTUGLU,
    GluFeedForward,
)


class TestSiTUGLU(unittest.TestCase):
    def test_output_bounded(self):
        glu = SiTUGLU(beta1=4.0, beta2=25.0)
        gate = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        up = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
        out = glu(gate, up)
        bound = 4.0 * 25.0
        self.assertTrue(
            torch.all(torch.abs(out) <= bound + 1e-5).item(),
            f"SiTU output exceeded bound {bound}"
        )

    def test_matches_swiglu_near_zero(self):
        glu = SiTUGLU(beta1=4.0, beta2=25.0)
        x = torch.randn(100) * 0.1
        swiglu = x * torch.sigmoid(x) * x
        situ = glu(x, x)
        max_diff = torch.max(torch.abs(situ - swiglu)).item()
        self.assertLess(max_diff, 1e-3,
                        f"SiTU deviated from SwiGLU by {max_diff} near zero")

    def test_zero_input_zero_output(self):
        glu = SiTUGLU(beta1=4.0, beta2=25.0)
        gate = torch.zeros(10)
        up = torch.zeros(10)
        out = glu(gate, up)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))


class TestGluFeedForward(unittest.TestCase):
    def test_forward_shape(self):
        cfg = ShogiBT4v2Config(embedding_size=128, ffn_multiplier=1.5, ffn_glu=True)
        ffn = GluFeedForward(cfg)
        x = torch.randn(2, 16, 128)
        y = ffn(x)
        self.assertEqual(y.shape, x.shape)

    def test_parameter_count_preserved(self):
        d, h = 1024, 1536
        dense_params = 2 * d * h
        glu_hidden = int(h * 2.0 / 3.0)
        glu_params = 3 * d * glu_hidden
        ratio = glu_params / dense_params
        self.assertTrue(0.99 < ratio < 1.01,
                        f"GLU params {glu_params} vs dense {dense_params} ratio {ratio}")


class TestModelWithGLU(unittest.TestCase):
    def test_default_config_uses_glu(self):
        cfg = ShogiBT4v2Config()
        self.assertTrue(cfg.ffn_glu)
        self.assertTrue(cfg.pre_norm)
        self.assertEqual(cfg.norm_type, "rmsnorm")

    def test_full_model_forward(self):
        cfg = ShogiBT4v2Config(embedding_size=256, num_encoders=4, num_heads=8)
        model = ShogiBT4v2(cfg)
        x = torch.randn(2, cfg.input_planes, 9, 9)
        policy, wdl, mlh = model(x)
        self.assertEqual(policy.shape, (2, 2187))
        self.assertEqual(wdl.shape, (2, 3))
        self.assertEqual(mlh.shape, (2, 1))
        self.assertTrue(torch.all(torch.isfinite(policy)).item())
        self.assertTrue(torch.all(torch.isfinite(wdl)).item())
        self.assertTrue(torch.all(torch.isfinite(mlh)).item())

    def test_legacy_config_still_works(self):
        cfg = ShogiBT4v2Config(
            embedding_size=256,
            num_encoders=2,
            num_heads=8,
            ffn_glu=False,
            pre_norm=False,
            norm_type="layernorm",
        )
        model = ShogiBT4v2(cfg)
        x = torch.randn(1, cfg.input_planes, 9, 9)
        policy, wdl, mlh = model(x)
        self.assertEqual(policy.shape, (1, 2187))

    def test_invalid_beta_raises(self):
        with self.assertRaises(ValueError):
            ShogiBT4v2Config(ffn_glu=True, ffn_glu_beta1=-1.0)

    def test_invalid_hidden_ratio_raises(self):
        with self.assertRaises(ValueError):
            ShogiBT4v2Config(ffn_glu=True, ffn_glu_hidden_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
