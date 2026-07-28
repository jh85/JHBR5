#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shogi_model_v2 import (
    PROMOTION_DELTA_STATE_KEYS,
    ShogiBT4v2,
    ShogiBT4v2Config,
    load_state_dict_with_promotion_migration,
    make_direction_policy_index,
)
from shogi_train import configure_finetune_scope, build_training_optimizer


def tiny_model(num_encoders=2):
    cfg = ShogiBT4v2Config()
    cfg.embedding_dense_size = 8
    cfg.embedding_size = 32
    cfg.policy_d_model = 32
    cfg.num_encoders = num_encoders
    cfg.num_heads = 2
    cfg.smolgen_hidden = 8
    cfg.smolgen_compress = 2
    cfg.smolgen_gen_size = 4
    cfg.value_hidden = 16
    cfg.mlh_hidden = 8
    return ShogiBT4v2(cfg)


class PromotionDeltaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_initialization_preserves_old_forced_pair(self):
        model = tiny_model()
        head = model.policy_head
        self.assertEqual(
            torch.count_nonzero(head.promotion_delta.weight).item(), 0)
        self.assertEqual(
            torch.count_nonzero(head.promotion_delta.bias).item(), 0)

        # 7d7c enters the promotion zone. The old gather rows for 7d7c and
        # 7d7c+ are identical, which used to force their logits to tie.
        nonpromotion = make_direction_policy_index("7d7c", False)
        promotion = make_direction_policy_index("7d7c+", False)
        x = torch.randn(2, 81, model.cfg.embedding_size)
        before = head(x)
        self.assertTrue(torch.equal(
            before[:, nonpromotion], before[:, promotion]))

        direction = promotion // 81 - 10
        with torch.no_grad():
            head.promotion_delta.bias[direction] = 1.75
        after = head(x)
        expected = torch.full((2,), 1.75)
        self.assertTrue(torch.allclose(
            after[:, promotion] - after[:, nonpromotion],
            expected, atol=1e-6, rtol=0))

    def test_promotion_delta_receives_policy_gradient(self):
        model = tiny_model()
        x = torch.randn(2, 148, 9, 9)
        target = torch.tensor([
            make_direction_policy_index("7d7c+", False),
            make_direction_policy_index("3d3c+", False),
        ])
        policy, _, _ = model(x)
        F.cross_entropy(policy, target).backward()
        weight_grad = model.policy_head.promotion_delta.weight.grad
        bias_grad = model.policy_head.promotion_delta.bias.grad
        self.assertIsNotNone(weight_grad)
        self.assertIsNotNone(bias_grad)
        self.assertGreater(torch.count_nonzero(weight_grad).item(), 0)
        self.assertGreater(torch.count_nonzero(bias_grad).item(), 0)

    def test_legacy_checkpoint_migration_is_narrow(self):
        source = tiny_model()
        legacy_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if key not in PROMOTION_DELTA_STATE_KEYS
        }
        migrated = tiny_model()
        self.assertTrue(load_state_dict_with_promotion_migration(
            migrated, legacy_state))
        self.assertEqual(torch.count_nonzero(
            migrated.policy_head.promotion_delta.weight).item(), 0)

        broken_state = dict(legacy_state)
        broken_state.pop("value_head.fc2.bias")
        with self.assertRaisesRegex(RuntimeError, "value_head.fc2.bias"):
            load_state_dict_with_promotion_migration(
                tiny_model(), broken_state)


class FinetuneConfigurationTest(unittest.TestCase):
    def test_promotion_scope_freezes_everything_else(self):
        model = tiny_model()
        count = configure_finetune_scope(model, "promotion")
        names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(names, PROMOTION_DELTA_STATE_KEYS)
        self.assertEqual(count, sum(
            parameter.numel()
            for parameter in model.policy_head.promotion_delta.parameters()))

    def test_last_scope_trains_last_encoder_and_heads(self):
        model = tiny_model(num_encoders=3)
        configure_finetune_scope(model, "last", last_encoders=1)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertFalse(any(name.startswith("encoders.0.") for name in trainable))
        self.assertFalse(any(name.startswith("encoders.1.") for name in trainable))
        self.assertTrue(any(name.startswith("encoders.2.") for name in trainable))
        self.assertTrue(any(name.startswith("policy_head.") for name in trainable))
        self.assertTrue(any(name.startswith("value_head.") for name in trainable))
        self.assertTrue(any(name.startswith("mlh_head.") for name in trainable))
        self.assertNotIn("smolgen_global.weight", trainable)

    def test_differential_optimizer_groups(self):
        model = tiny_model()
        configure_finetune_scope(model, "full")
        optimizer = build_training_optimizer(
            model, 1e-5, 0.01, grouped=True,
            trunk_lr=1e-5, policy_lr=1e-4, promotion_lr=3e-4)
        got = {
            group["group_name"]: group["lr"]
            for group in optimizer.param_groups
        }
        self.assertEqual(got, {
            "trunk": 1e-5,
            "policy": 1e-4,
            "promotion": 3e-4,
        })

    def test_last_scope_backward_with_gradient_checkpointing(self):
        model = tiny_model(num_encoders=3)
        configure_finetune_scope(model, "last", last_encoders=1)
        model.gradient_checkpointing = True
        model.train()
        policy, wdl, mlh = model(torch.randn(2, 148, 9, 9))
        (policy.mean() + wdl.mean() + mlh.mean()).backward()
        self.assertIsNone(model.encoders[0].q_proj.weight.grad)
        self.assertIsNone(model.encoders[1].q_proj.weight.grad)
        self.assertIsNotNone(model.encoders[2].q_proj.weight.grad)
        self.assertIsNotNone(model.policy_head.promotion_delta.weight.grad)
        self.assertIsNotNone(model.value_head.fc2.weight.grad)

    def test_promotion_only_step_changes_no_old_weight(self):
        model = tiny_model()
        configure_finetune_scope(model, "promotion")
        optimizer = build_training_optimizer(
            model, 3e-4, 0.01, grouped=True,
            promotion_lr=3e-4)
        old_policy_weight = model.policy_head.wq.weight.detach().clone()
        old_delta = model.policy_head.promotion_delta.weight.detach().clone()

        x = torch.randn(2, 148, 9, 9)
        target = torch.tensor([
            make_direction_policy_index("7d7c+", False),
            make_direction_policy_index("3d3c+", False),
        ])
        policy, _, _ = model(x)
        F.cross_entropy(policy, target).backward()
        optimizer.step()

        self.assertTrue(torch.equal(
            old_policy_weight, model.policy_head.wq.weight))
        self.assertFalse(torch.equal(
            old_delta, model.policy_head.promotion_delta.weight))


if __name__ == "__main__":
    unittest.main()
