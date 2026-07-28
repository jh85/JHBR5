#!/usr/bin/env python3
"""Resumable SPSA tuning of continuous JHBR3 USI search parameters.

Each SPSA iteration creates two simultaneous perturbations around the current
parameter vector and compares them directly in one color-reversed paired
match.  The match score supplies a noisy directional gradient.  Hardware
topology and all non-tuned USI options stay fixed.

Use tools/run_topology_tuning.py before this script when moving to a new GPU
type.  Use this script for continuous search parameters such as PUCT and FPU,
not discrete topology choices such as worker count or minibatch size.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = 1
CONTROLLED_OPTIONS = {"onnxmodel", "numgpus"}
DISCRETE_TOPOLOGY_PARAMETERS = {
    "threads",
    "workerspergpu",
    "minibatchsize",
    "numgpus",
    "maxnodes",
}


@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    initial: float
    minimum: float
    maximum: float
    scale: str = "linear"
    kind: str = "float"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name cannot be empty")
        values = (self.initial, self.minimum, self.maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{self.name}: values must be finite")
        if self.minimum >= self.maximum:
            raise ValueError(f"{self.name}: minimum must be below maximum")
        if not self.minimum <= self.initial <= self.maximum:
            raise ValueError(f"{self.name}: initial value must be within its bounds")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name}: scale must be linear or log")
        if self.kind not in {"float", "int"}:
            raise ValueError(f"{self.name}: kind must be float or int")
        if self.scale == "log" and self.minimum <= 0:
            raise ValueError(f"{self.name}: logarithmic bounds must be positive")
        if self.kind == "int" and (
            not float(self.initial).is_integer()
            or not float(self.minimum).is_integer()
            or not float(self.maximum).is_integer()
        ):
            raise ValueError(f"{self.name}: integer parameter values must be integers")

    def to_unit(self, value: float) -> float:
        value = min(max(value, self.minimum), self.maximum)
        if self.scale == "log":
            low = math.log(self.minimum)
            return (math.log(value) - low) / (math.log(self.maximum) - low)
        return (value - self.minimum) / (self.maximum - self.minimum)

    def from_unit(self, unit: float) -> float:
        unit = min(max(unit, 0.0), 1.0)
        if self.scale == "log":
            low = math.log(self.minimum)
            value = math.exp(low + unit * (math.log(self.maximum) - low))
        else:
            value = self.minimum + unit * (self.maximum - self.minimum)
        if self.kind == "int":
            value = float(round(value))
        return min(max(value, self.minimum), self.maximum)

    def format(self, value: float) -> str:
        if self.kind == "int":
            return str(int(round(value)))
        return f"{value:.10g}"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


NONROOT_PUCT = (
    Parameter("CInit", 1.25, 0.75, 2.0),
    Parameter("CBase", 19652.0, 1000.0, 100000.0, "log"),
    Parameter("FpuReduction", 0.27, 0.0, 0.6),
)

ROOT_PUCT = (
    Parameter("CInitRoot", 1.25, 0.75, 2.0),
    Parameter("CBaseRoot", 19652.0, 1000.0, 100000.0, "log"),
    Parameter("FpuReductionRoot", 0.0, 0.0, 0.4),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_output() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("spsa-runs") / stamp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_parameter(text: str) -> Parameter:
    """Parse NAME=INITIAL:MIN:MAX[:linear|log[:float|int]]."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "parameter must be NAME=INITIAL:MIN:MAX[:SCALE[:KIND]]"
        )
    name, fields_text = text.split("=", 1)
    fields = [field.strip() for field in fields_text.split(":")]
    if len(fields) < 3 or len(fields) > 5:
        raise argparse.ArgumentTypeError(
            "parameter must be NAME=INITIAL:MIN:MAX[:SCALE[:KIND]]"
        )
    try:
        initial, minimum, maximum = (float(field) for field in fields[:3])
        scale = fields[3].casefold() if len(fields) >= 4 else "linear"
        kind = fields[4].casefold() if len(fields) >= 5 else "float"
        return Parameter(name.strip(), initial, minimum, maximum, scale, kind)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_engine_options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    folded: set[str] = set()
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--engine-option must be NAME=VALUE: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("--engine-option has an empty name")
        key = name.casefold()
        if key in CONTROLLED_OPTIONS:
            raise SystemExit(f"{name} is controlled by the SPSA driver/match runner")
        if key in folded:
            raise SystemExit(f"duplicate --engine-option: {name}")
        folded.add(key)
        result[name] = value
    return result


def merge_parameters(preset: str, overrides: list[Parameter]) -> list[Parameter]:
    if preset == "none":
        parameters: list[Parameter] = []
    elif preset == "nonroot-puct":
        parameters = list(NONROOT_PUCT)
    elif preset == "puct":
        parameters = [*NONROOT_PUCT, *ROOT_PUCT]
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown preset: {preset}")

    positions = {
        parameter.name.casefold(): index for index, parameter in enumerate(parameters)
    }
    for parameter in overrides:
        key = parameter.name.casefold()
        if key in positions:
            parameters[positions[key]] = parameter
        else:
            positions[key] = len(parameters)
            parameters.append(parameter)

    if not parameters:
        raise SystemExit("no parameters selected")
    names: set[str] = set()
    for parameter in parameters:
        key = parameter.name.casefold()
        if key in names:
            raise SystemExit(f"duplicate parameter: {parameter.name}")
        if key in CONTROLLED_OPTIONS:
            raise SystemExit(
                f"{parameter.name} is controlled by the SPSA driver/match runner"
            )
        if key in DISCRETE_TOPOLOGY_PARAMETERS:
            raise SystemExit(
                f"{parameter.name} is a topology/budget parameter; tune it "
                "with tools/run_topology_tuning.py or an explicit grid"
            )
        names.add(key)
    return parameters


def parameter_values(
    parameters: list[Parameter], units: list[float]
) -> dict[str, float]:
    return {
        parameter.name: parameter.from_unit(unit)
        for parameter, unit in zip(parameters, units)
    }


def formatted_parameter_options(
    parameters: list[Parameter], units: list[float]
) -> dict[str, str]:
    return {
        parameter.name: parameter.format(parameter.from_unit(unit))
        for parameter, unit in zip(parameters, units)
    }


def actual_units(
    parameters: list[Parameter], requested_units: list[float]
) -> list[float]:
    return [
        parameter.to_unit(parameter.from_unit(unit))
        for parameter, unit in zip(parameters, requested_units)
    ]


def perturbation(seed: int, iteration: int, dimensions: int) -> list[int]:
    # A separate deterministic RNG per iteration makes a resumed run identical
    # even if random calls are added elsewhere later.
    rng = random.Random(seed + iteration * 0x9E3779B1)
    return [rng.choice((-1, 1)) for _ in range(dimensions)]


def make_candidates(
    parameters: list[Parameter],
    center: list[float],
    delta: list[int],
    ck: float,
) -> tuple[list[float], list[float]]:
    plus_requested = [
        min(max(value + ck * direction, 0.0), 1.0)
        for value, direction in zip(center, delta)
    ]
    minus_requested = [
        min(max(value - ck * direction, 0.0), 1.0)
        for value, direction in zip(center, delta)
    ]
    return (
        actual_units(parameters, plus_requested),
        actual_units(parameters, minus_requested),
    )


def update_center(
    center: list[float],
    plus: list[float],
    minus: list[float],
    plus_score: float,
    ak: float,
    gradient_clip: float,
    max_step: float,
) -> tuple[list[float], list[float], list[float], float]:
    # A direct plus-vs-minus match estimates which perturbation is stronger.
    # The centered score is bounded, robust to 0%/100% short-match outcomes,
    # and its unknown Elo scaling is absorbed into the normalized gain `a`.
    signal = min(max(2.0 * (plus_score - 0.5), -1.0), 1.0)
    raw_gradient: list[float] = []
    applied_steps: list[float] = []
    updated: list[float] = []
    for value, plus_value, minus_value in zip(center, plus, minus):
        distance = plus_value - minus_value
        gradient = signal / distance if abs(distance) > 1.0e-12 else 0.0
        raw_gradient.append(gradient)
        clipped = min(max(gradient, -gradient_clip), gradient_clip)
        step = min(max(ak * clipped, -max_step), max_step)
        applied_steps.append(step)
        updated.append(min(max(value + step, 0.0), 1.0))
    return updated, raw_gradient, applied_steps, signal


def mean_units(samples: list[list[float]]) -> list[float]:
    if not samples:
        raise ValueError("cannot average an empty sample")
    return [
        sum(sample[index] for sample in samples) / len(samples)
        for index in range(len(samples[0]))
    ]


def negate_elo(value: float | str | None) -> float | str | None:
    if isinstance(value, (int, float)):
        return -float(value)
    if value == "+inf":
        return "-inf"
    if value == "-inf":
        return "+inf"
    return value


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def add_option_arguments(
    command: list[str], label: str, options: dict[str, str]
) -> None:
    flag = f"--option-{label.casefold()}"
    for name, value in sorted(options.items(), key=lambda item: item[0].casefold()):
        command.extend([flag, f"{name}={value}"])


def add_match_control(
    command: list[str],
    *,
    nodes: int | None,
    main_time_ms: int,
    byoyomi_ms: int,
    increment_ms: int,
) -> None:
    if nodes is not None:
        command.extend(["--nodes", str(nodes)])
    else:
        command.extend(["--main-time-ms", str(main_time_ms)])
        command.extend(["--byoyomi-ms", str(byoyomi_ms)])
        command.extend(["--increment-ms", str(increment_ms)])


class SpsaRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(__file__).resolve().parent.parent
        self.wrapper = self.repo / "tools" / "run_strength_test.sh"
        self.engine = args.engine.expanduser().resolve()
        self.model = args.model.expanduser().resolve()
        self.openings = args.openings.expanduser().resolve()
        self.confirmation_openings = (
            args.confirmation_openings.expanduser().resolve()
            if args.confirmation_openings
            else self.openings
        )
        self.output = args.output.expanduser().resolve()
        self.parameters = merge_parameters(args.preset, args.parameter)
        self.fixed_options = parse_engine_options(args.engine_option)
        parameter_names = {parameter.name.casefold() for parameter in self.parameters}
        overlap = [
            name for name in self.fixed_options if name.casefold() in parameter_names
        ]
        if overlap:
            raise SystemExit(
                "tuned parameters cannot also be fixed engine options: "
                + ", ".join(overlap)
            )

        for path, label in (
            (self.engine, "engine"),
            (self.model, "model"),
            (self.openings, "openings"),
            (self.confirmation_openings, "confirmation openings"),
            (self.wrapper, "strength-test wrapper"),
        ):
            if not path.is_file():
                raise SystemExit(f"{label} does not exist: {path}")

        self.config_path = self.output / "config.json"
        self.history_path = self.output / "history.jsonl"
        self.state_path = self.output / "state.json"
        self.result_path = self.output / "result.json"
        self.config = self.make_config()
        self.history: list[dict[str, Any]] = []

    def make_config(self) -> dict[str, Any]:
        stable = {
            "schema": SCHEMA,
            "engine": artifact(self.engine),
            "model": artifact(self.model),
            "openings": artifact(self.openings),
            "confirmation_openings": artifact(self.confirmation_openings),
            "parameters": [parameter.as_dict() for parameter in self.parameters],
            "fixed_options": self.fixed_options,
            "spsa": {
                "iterations": self.args.iterations,
                "pairs_per_iteration": self.args.pairs_per_iteration,
                "a": self.args.a,
                "c": self.args.c,
                "alpha": self.args.alpha,
                "gamma": self.args.gamma,
                "stability": self.args.stability,
                "gradient_clip": self.args.gradient_clip,
                "max_step": self.args.max_step,
                "average_start": self.args.average_start,
                "seed": self.args.seed,
            },
            "screening_match": self.match_config(confirmation=False),
            "confirmation_match": self.match_config(confirmation=True),
        }
        fingerprint = hashlib.sha256(
            json.dumps(stable, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            **stable,
            "fingerprint": fingerprint,
            "created_utc": utc_now(),
        }

    def match_config(self, *, confirmation: bool) -> dict[str, Any]:
        if confirmation:
            control = self.confirmation_control()
            return {
                "pairs": self.args.confirmation_pairs,
                "nodes": control[0],
                "main_time_ms": control[1],
                "byoyomi_ms": control[2],
                "increment_ms": control[3],
                "gpu_devices": (
                    self.args.confirmation_gpu_devices
                    if self.args.confirmation_gpu_devices is not None
                    else self.args.gpu_devices
                ),
                "gpus_per_worker": (
                    self.args.confirmation_gpus_per_worker or self.args.gpus_per_worker
                ),
                "workers": self.args.confirmation_workers,
                "seed": self.args.confirmation_seed,
            }
        return {
            "pairs": self.args.pairs_per_iteration,
            "nodes": self.args.nodes,
            "main_time_ms": self.args.main_time_ms,
            "byoyomi_ms": self.args.byoyomi_ms,
            "increment_ms": self.args.increment_ms,
            "gpu_devices": self.args.gpu_devices,
            "gpus_per_worker": self.args.gpus_per_worker,
            "workers": self.args.workers,
            "seed": self.args.seed,
        }

    def confirmation_control(self) -> tuple[int | None, int, int, int]:
        clock_override = any(
            value is not None
            for value in (
                self.args.confirmation_main_time_ms,
                self.args.confirmation_byoyomi_ms,
                self.args.confirmation_increment_ms,
            )
        )
        if clock_override:
            return (
                None,
                self.args.confirmation_main_time_ms or 0,
                self.args.confirmation_byoyomi_ms or 0,
                self.args.confirmation_increment_ms or 0,
            )
        if self.args.confirmation_nodes is not None:
            return (self.args.confirmation_nodes, 0, 0, 0)
        return (
            self.args.nodes,
            self.args.main_time_ms,
            self.args.byoyomi_ms,
            self.args.increment_ms,
        )

    def prepare(self) -> None:
        if self.output.exists() and not self.args.resume:
            raise SystemExit(
                f"{self.output} already exists; choose another --output "
                "or pass --resume"
            )
        self.output.mkdir(parents=True, exist_ok=True)
        if self.args.resume:
            if not self.config_path.is_file():
                raise SystemExit(f"{self.output}: cannot resume without config.json")
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != self.config["fingerprint"]:
                raise SystemExit(f"{self.output}: configuration differs from saved run")
        else:
            atomic_write_json(self.config_path, self.config)

        if self.history_path.is_file():
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.history.append(json.loads(line))
        for expected, record in enumerate(self.history):
            if record.get("iteration") != expected:
                raise SystemExit(
                    f"{self.history_path}: expected iteration {expected}, "
                    f"found {record.get('iteration')}"
                )

    def current_units(self) -> list[float]:
        if self.history:
            return list(self.history[-1]["center_after_unit"])
        return [parameter.to_unit(parameter.initial) for parameter in self.parameters]

    def recommendation_units(self) -> list[float]:
        samples = [
            list(record["center_after_unit"])
            for record in self.history
            if record["iteration"] >= self.args.average_start
        ]
        return mean_units(samples) if samples else self.current_units()

    def base_match_command(
        self,
        *,
        output: Path,
        openings: Path,
        pairs: int,
        seed: int,
        nodes: int | None,
        main_time_ms: int,
        byoyomi_ms: int,
        increment_ms: int,
        gpu_devices: str | None,
        gpus_per_worker: int,
        workers: int | None,
        options_a: dict[str, str],
        options_b: dict[str, str],
    ) -> list[str]:
        command = [
            str(self.wrapper),
            "match",
            "--engine-a",
            str(self.engine),
            "--engine-b",
            str(self.engine),
            "--openings",
            str(openings),
            "--pairs",
            str(pairs),
            "--seed",
            str(seed),
            "--gpus-per-worker",
            str(gpus_per_worker),
            "--max-plies",
            str(self.args.max_plies),
            "--move-timeout",
            str(self.args.move_timeout),
            "--startup-timeout",
            str(self.args.startup_timeout),
            "--retries",
            str(self.args.retries),
            "--clock-grace-ms",
            str(self.args.clock_grace_ms),
            "--output",
            str(output),
        ]
        add_match_control(
            command,
            nodes=nodes,
            main_time_ms=main_time_ms,
            byoyomi_ms=byoyomi_ms,
            increment_ms=increment_ms,
        )
        if gpu_devices is not None:
            command.extend(["--gpu-devices", gpu_devices])
        if workers is not None:
            command.extend(["--workers", str(workers)])
        if self.args.protocol_log:
            command.append("--protocol-log")
        add_option_arguments(command, "A", options_a)
        add_option_arguments(command, "B", options_b)
        if output.exists():
            command.append("--resume")
        return command

    def candidate_options(self, units: list[float]) -> dict[str, str]:
        return {
            "OnnxModel": str(self.model),
            **self.fixed_options,
            **formatted_parameter_options(self.parameters, units),
        }

    def run_match(self, command: list[str], output: Path) -> dict[str, Any]:
        print("+ " + shlex.join(command), flush=True)
        completed = subprocess.run(command, cwd=self.repo, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"strength match failed with exit code {completed.returncode}: "
                f"{output}"
            )
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"strength match produced no summary: {output}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("failed_pairs") != 0 or summary.get(
            "completed_pairs"
        ) != summary.get("requested_pairs"):
            raise SystemExit(f"incomplete strength match: {summary_path}")
        return summary

    def save_iteration(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.history.append(record)
        self.write_state()

    def write_state(self) -> None:
        current = self.current_units()
        recommendation = self.recommendation_units()
        atomic_write_json(
            self.state_path,
            {
                "schema": SCHEMA,
                "completed_iterations": len(self.history),
                "current_unit": current,
                "current": parameter_values(self.parameters, current),
                "recommendation_unit": recommendation,
                "recommendation": parameter_values(self.parameters, recommendation),
                "average_start": self.args.average_start,
                "updated_utc": utc_now(),
            },
        )

    def run_iteration(self, iteration: int, center: list[float]) -> list[float]:
        ak = self.args.a / ((self.args.stability + iteration + 1.0) ** self.args.alpha)
        ck = self.args.c / ((iteration + 1.0) ** self.args.gamma)
        delta = perturbation(self.args.seed, iteration, len(self.parameters))
        plus, minus = make_candidates(self.parameters, center, delta, ck)
        plus_options = self.candidate_options(plus)
        minus_options = self.candidate_options(minus)

        # Alternate process labels to cancel any persistent A/B startup bias.
        plus_is_a = iteration % 2 == 0
        options_a = plus_options if plus_is_a else minus_options
        options_b = minus_options if plus_is_a else plus_options
        match_output = self.output / "iterations" / f"iteration-{iteration:04d}"
        command = self.base_match_command(
            output=match_output,
            openings=self.openings,
            pairs=self.args.pairs_per_iteration,
            seed=self.args.seed + iteration,
            nodes=self.args.nodes,
            main_time_ms=self.args.main_time_ms,
            byoyomi_ms=self.args.byoyomi_ms,
            increment_ms=self.args.increment_ms,
            gpu_devices=self.args.gpu_devices,
            gpus_per_worker=self.args.gpus_per_worker,
            workers=self.args.workers,
            options_a=options_a,
            options_b=options_b,
        )

        print(
            f"\nSPSA iteration {iteration + 1}/{self.args.iterations}: "
            f"a_k={ak:.6g}, c_k={ck:.6g}, plus_is_A={plus_is_a}",
            flush=True,
        )
        print(
            "plus  " + json.dumps(parameter_values(self.parameters, plus)),
            flush=True,
        )
        print(
            "minus " + json.dumps(parameter_values(self.parameters, minus)),
            flush=True,
        )
        summary = self.run_match(command, match_output)
        score_a = float(summary["score"])
        plus_score = score_a if plus_is_a else 1.0 - score_a
        updated, raw_gradient, steps, signal = update_center(
            center,
            plus,
            minus,
            plus_score,
            ak,
            self.args.gradient_clip,
            self.args.max_step,
        )
        record = {
            "schema": SCHEMA,
            "iteration": iteration,
            "completed_utc": utc_now(),
            "a_k": ak,
            "c_k": ck,
            "delta": delta,
            "plus_is_a": plus_is_a,
            "center_before_unit": center,
            "plus_unit": plus,
            "minus_unit": minus,
            "plus": parameter_values(self.parameters, plus),
            "minus": parameter_values(self.parameters, minus),
            "match_score_a": score_a,
            "plus_score": plus_score,
            "signal": signal,
            "raw_gradient": raw_gradient,
            "applied_steps": steps,
            "center_after_unit": updated,
            "center_after": parameter_values(self.parameters, updated),
            "match_output": str(match_output),
            "match_summary": summary,
        }
        self.save_iteration(record)
        print(
            "updated " + json.dumps(parameter_values(self.parameters, updated)),
            flush=True,
        )
        return updated

    def run_confirmation(self, recommendation: list[float]) -> dict[str, Any] | None:
        if self.args.confirmation_pairs <= 0:
            return None
        initial = [
            parameter.to_unit(parameter.initial) for parameter in self.parameters
        ]
        control = self.confirmation_control()
        gpu_devices = (
            self.args.confirmation_gpu_devices
            if self.args.confirmation_gpu_devices is not None
            else self.args.gpu_devices
        )
        gpus_per_worker = (
            self.args.confirmation_gpus_per_worker or self.args.gpus_per_worker
        )
        output = self.output / "confirmation"
        command = self.base_match_command(
            output=output,
            openings=self.confirmation_openings,
            pairs=self.args.confirmation_pairs,
            seed=self.args.confirmation_seed,
            nodes=control[0],
            main_time_ms=control[1],
            byoyomi_ms=control[2],
            increment_ms=control[3],
            gpu_devices=gpu_devices,
            gpus_per_worker=gpus_per_worker,
            workers=self.args.confirmation_workers,
            options_a=self.candidate_options(initial),
            options_b=self.candidate_options(recommendation),
        )
        print(
            "\nConfirmation: A=initial baseline, B=SPSA recommendation",
            flush=True,
        )
        summary = self.run_match(command, output)
        return {
            "output": str(output),
            "summary": summary,
            "candidate_score": 1.0 - float(summary["score"]),
            "candidate_elo": negate_elo(summary.get("elo")),
        }

    def write_result(
        self, recommendation: list[float], confirmation: dict[str, Any] | None
    ) -> None:
        initial = {parameter.name: parameter.initial for parameter in self.parameters}
        result = {
            "schema": SCHEMA,
            "status": "complete",
            "completed_utc": utc_now(),
            "iterations": len(self.history),
            "initial": initial,
            "final_center": parameter_values(self.parameters, self.current_units()),
            "tail_average_start": self.args.average_start,
            "recommendation": parameter_values(self.parameters, recommendation),
            "recommendation_options": formatted_parameter_options(
                self.parameters, recommendation
            ),
            "confirmation": confirmation,
            "config": str(self.config_path),
            "history": str(self.history_path),
        }
        atomic_write_json(self.result_path, result)

    def dry_run(self) -> int:
        center = [parameter.to_unit(parameter.initial) for parameter in self.parameters]
        ck = self.args.c
        delta = perturbation(self.args.seed, 0, len(self.parameters))
        plus, minus = make_candidates(self.parameters, center, delta, ck)
        command = self.base_match_command(
            output=self.output / "iterations" / "iteration-0000",
            openings=self.openings,
            pairs=self.args.pairs_per_iteration,
            seed=self.args.seed,
            nodes=self.args.nodes,
            main_time_ms=self.args.main_time_ms,
            byoyomi_ms=self.args.byoyomi_ms,
            increment_ms=self.args.increment_ms,
            gpu_devices=self.args.gpu_devices,
            gpus_per_worker=self.args.gpus_per_worker,
            workers=self.args.workers,
            options_a=self.candidate_options(plus),
            options_b=self.candidate_options(minus),
        )
        print(json.dumps(self.config, indent=2, sort_keys=True))
        print("\nFirst plus point:")
        print(json.dumps(parameter_values(self.parameters, plus), indent=2))
        print("First minus point:")
        print(json.dumps(parameter_values(self.parameters, minus), indent=2))
        print("\nFirst match command:")
        print(shlex.join(command))
        return 0

    def run(self) -> int:
        if self.args.dry_run:
            return self.dry_run()
        self.prepare()
        center = self.current_units()
        self.write_state()
        for iteration in range(len(self.history), self.args.iterations):
            center = self.run_iteration(iteration, center)
        recommendation = self.recommendation_units()
        confirmation = self.run_confirmation(recommendation)
        self.write_result(recommendation, confirmation)
        print(f"\nSPSA complete: {self.result_path}", flush=True)
        print(
            "recommendation "
            + json.dumps(parameter_values(self.parameters, recommendation)),
            flush=True,
        )
        if confirmation:
            print(
                f"confirmation candidate score "
                f"{confirmation['candidate_score'] * 100:.2f}%",
                flush=True,
            )
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--openings", required=True, type=Path)
    parser.add_argument(
        "--preset",
        choices=("nonroot-puct", "puct", "none"),
        default="nonroot-puct",
        help="built-in parameter set (default: nonroot-puct)",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        type=parse_parameter,
        metavar="NAME=INITIAL:MIN:MAX[:SCALE[:KIND]]",
        help="add a parameter or override one from the preset",
    )
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="fixed USI option applied to all candidates",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--pairs-per-iteration", type=int, default=16)
    parser.add_argument(
        "--nodes",
        type=int,
        help="fixed nodes per move (default: 4000 when no clock is given)",
    )
    parser.add_argument("--main-time-ms", type=int, default=0)
    parser.add_argument("--byoyomi-ms", type=int, default=0)
    parser.add_argument("--increment-ms", type=int, default=0)
    parser.add_argument("--a", type=float, default=0.08)
    parser.add_argument("--c", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.602)
    parser.add_argument("--gamma", type=float, default=0.101)
    parser.add_argument(
        "--stability",
        type=float,
        help="SPSA stability constant A (default: 10%% of iterations)",
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.15,
        help="maximum normalized update per parameter and iteration",
    )
    parser.add_argument(
        "--average-start",
        type=int,
        help="first iteration included in tail-average recommendation",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--gpu-devices")
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--move-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--clock-grace-ms", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--protocol-log", action="store_true")

    parser.add_argument("--confirmation-pairs", type=int, default=0)
    parser.add_argument("--confirmation-openings", type=Path)
    parser.add_argument("--confirmation-nodes", type=int)
    parser.add_argument("--confirmation-main-time-ms", type=int)
    parser.add_argument("--confirmation-byoyomi-ms", type=int)
    parser.add_argument("--confirmation-increment-ms", type=int)
    parser.add_argument("--confirmation-gpu-devices")
    parser.add_argument("--confirmation-gpus-per-worker", type=int)
    parser.add_argument("--confirmation-workers", type=int)
    parser.add_argument("--confirmation-seed", type=int, default=21260725)

    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the first iteration without creating a run",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.pairs_per_iteration <= 0:
        parser.error("--pairs-per-iteration must be positive")
    for name in (
        "main_time_ms",
        "byoyomi_ms",
        "increment_ms",
        "clock_grace_ms",
        "confirmation_pairs",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    screening_clock = (
        args.main_time_ms > 0 or args.byoyomi_ms > 0 or args.increment_ms > 0
    )
    if args.nodes is not None and screening_clock:
        parser.error("--nodes cannot be combined with screening clock options")
    if args.nodes is None and not screening_clock:
        args.nodes = 4000
    if args.nodes is not None and args.nodes <= 0:
        parser.error("--nodes must be positive")
    if (
        args.nodes is None
        and args.main_time_ms == 0
        and args.byoyomi_ms == 0
        and args.increment_ms == 0
    ):
        parser.error("screening time control must allocate positive time")

    confirmation_clock = any(
        value is not None
        for value in (
            args.confirmation_main_time_ms,
            args.confirmation_byoyomi_ms,
            args.confirmation_increment_ms,
        )
    )
    if args.confirmation_nodes is not None and confirmation_clock:
        parser.error("--confirmation-nodes cannot be combined with confirmation clocks")
    if args.confirmation_nodes is not None and args.confirmation_nodes <= 0:
        parser.error("--confirmation-nodes must be positive")
    for name in (
        "confirmation_main_time_ms",
        "confirmation_byoyomi_ms",
        "confirmation_increment_ms",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if confirmation_clock and not any(
        (
            (args.confirmation_main_time_ms or 0) > 0,
            (args.confirmation_byoyomi_ms or 0) > 0,
            (args.confirmation_increment_ms or 0) > 0,
        )
    ):
        parser.error("confirmation time control must allocate positive time")

    for name in ("gpus_per_worker", "max_plies", "retries"):
        value = getattr(args, name)
        if name == "retries":
            if value < 0:
                parser.error("--retries must be non-negative")
        elif value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("workers", "confirmation_gpus_per_worker", "confirmation_workers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "a",
        "c",
        "alpha",
        "gamma",
        "gradient_clip",
        "max_step",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")

    if args.stability is None:
        args.stability = max(1.0, args.iterations * 0.1)
    if not math.isfinite(args.stability) or args.stability < 0:
        parser.error("--stability must be non-negative and finite")
    if args.average_start is None:
        args.average_start = args.iterations // 2
    if not 0 <= args.average_start < args.iterations:
        parser.error("--average-start must be within the iteration range")
    return args


def main() -> int:
    return SpsaRun(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
