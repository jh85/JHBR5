#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_spsa_tuning import (
    Parameter,
    make_candidates,
    merge_parameters,
    negate_elo,
    parse_parameter,
    perturbation,
    update_center,
)


class ParameterTest(unittest.TestCase):
    def test_linear_round_trip(self):
        parameter = Parameter("CInit", 1.25, 0.75, 2.0)
        for value in (0.75, 1.25, 2.0):
            self.assertAlmostEqual(parameter.from_unit(parameter.to_unit(value)), value)

    def test_log_round_trip(self):
        parameter = Parameter("CBase", 19652.0, 1000.0, 100000.0, "log")
        for value in (1000.0, 19652.0, 100000.0):
            self.assertAlmostEqual(
                parameter.from_unit(parameter.to_unit(value)),
                value,
                places=8,
            )
        self.assertAlmostEqual(parameter.from_unit(0.5), 10000.0)

    def test_integer_quantization(self):
        parameter = Parameter("Depth", 5.0, 1.0, 7.0, "linear", "int")
        self.assertEqual(parameter.from_unit(0.5), 4.0)
        self.assertEqual(parameter.format(5.0), "5")

    def test_negate_elo(self):
        self.assertEqual(negate_elo(42.0), -42.0)
        self.assertEqual(negate_elo("-inf"), "+inf")
        self.assertEqual(negate_elo("+inf"), "-inf")

    def test_parse_and_preset_override(self):
        override = parse_parameter("CInit=1.1:0.5:1.8:linear:float")
        parameters = merge_parameters("nonroot-puct", [override])
        self.assertEqual(parameters[0], override)
        self.assertEqual(len(parameters), 3)


class SpsaMathTest(unittest.TestCase):
    def test_perturbation_is_deterministic(self):
        self.assertEqual(
            perturbation(123, 7, 8),
            perturbation(123, 7, 8),
        )
        self.assertTrue(all(value in (-1, 1) for value in perturbation(123, 7, 8)))

    def test_candidate_bounds_and_boundary_distance(self):
        parameters = [
            Parameter("x", 0.0, 0.0, 1.0),
            Parameter("y", 0.5, 0.0, 1.0),
        ]
        plus, minus = make_candidates(parameters, [0.0, 0.5], [1, -1], 0.2)
        self.assertEqual(plus, [0.2, 0.3])
        self.assertEqual(minus, [0.0, 0.7])

    def test_better_plus_moves_toward_plus(self):
        center = [0.5, 0.5]
        plus = [0.6, 0.4]
        minus = [0.4, 0.6]
        updated, gradient, steps, signal = update_center(
            center,
            plus,
            minus,
            plus_score=0.75,
            ak=0.05,
            gradient_clip=5.0,
            max_step=0.15,
        )
        self.assertGreater(signal, 0)
        self.assertGreater(gradient[0], 0)
        self.assertLess(gradient[1], 0)
        self.assertGreater(steps[0], 0)
        self.assertLess(steps[1], 0)
        self.assertGreater(updated[0], center[0])
        self.assertLess(updated[1], center[1])

    def test_even_match_has_no_update(self):
        center = [0.3]
        updated, gradient, steps, signal = update_center(
            center,
            [0.4],
            [0.2],
            plus_score=0.5,
            ak=0.1,
            gradient_clip=5.0,
            max_step=0.15,
        )
        self.assertEqual(signal, 0.0)
        self.assertEqual(gradient, [0.0])
        self.assertEqual(steps, [0.0])
        self.assertEqual(updated, center)

    def test_updates_stay_finite_and_bounded(self):
        updated, _, _, _ = update_center(
            [0.99],
            [1.0],
            [0.98],
            plus_score=1.0,
            ak=1.0,
            gradient_clip=5.0,
            max_step=0.15,
        )
        self.assertTrue(math.isfinite(updated[0]))
        self.assertEqual(updated, [1.0])


if __name__ == "__main__":
    unittest.main()
