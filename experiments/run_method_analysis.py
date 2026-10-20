from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

# Ensure imports work when running: python experiments/run_method_analysis.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nas_framework.benchmark_api import CSVBenchmarkAPI
from nas_framework.crossover import UniformCrossover
from nas_framework.evaluator import Evaluator, DvolverEvaluator
from nas_framework.mutation import SinglePointMutation
from nas_framework.population import Individual, Population
from nas_framework.replacement import ElitistReplacement
from nas_framework.search_space import CSVSearchSpace, CSVGenotypeDvolverSearchSpace
from nas_framework.search_strategy import (
    BruteForceParetoSearch,
    DvolverSearchStrategy,
    RandomSearch,
    SkylineSearch,
    MOWSOSearch,
    MOSHOSearch,
)
from nas_framework.selection import TournamentSelection
from nas_framework.termination import TerminationCriteria
from utilities.metrics import (
    c_metric,
    igd_plus,
    non_dominated,
    normalized_hypervolume_2d,
)
from utilities.plotting import (
    save_context_metric_heatmap,
    save_hv_boxplot,
    save_pareto_scatter,
    save_runtime_boxplot,
)
import matplotlib.pyplot as plt

DEFAULT_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
DEFAULT_DEVICES = ("edgegpu", "edgetpu", "eyeriss", "fpga", "pixel3", "raspi4")
OBJECTIVE_DIRECTIONS = (1, -1)
ALL_METHODS = ("random", "bruteforce", "skyline", "mowso", "mosho", "dvolver")


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.exists():
        return p
    alt = ROOT / path_like
    if alt.exists():
        return alt
    return p


def _extract_front(strategy: Any, result: Any) -> list[Individual]:
    if isinstance(result, Population):
        return result.pareto_front()
    if isinstance(result, list) and (not result or isinstance(result[0], Individual)):
        return result

    history = getattr(strategy, "history", None)
    if history and getattr(history, "pareto_archive", None):
        archive = history.pareto_archive
        if archive:
            return archive[-1]

    raise TypeError(
        "Unable to get Pareto front from strategy result. "
        "Expected Population, list[Individual], or strategy.history.pareto_archive."
    )


def _individual_objectives(
    ind: Individual,
    method: str,
    benchmark: CSVBenchmarkAPI,
    dataset: str,
    device: str,
) -> tuple[float, float] | None:
    if ind.fitness is None:
        return None

    if method != "dvolver":
        acc, lat = ind.fitness
        if acc != acc or lat != lat:
            return None
        return float(acc), float(lat)

    # Dvolver fitness is (accuracy, speed). Convert to comparable latency metric.
    acc = float(ind.fitness[0])
    speed = float(ind.fitness[1])
    if acc != acc or speed != speed:
        return None

    architecture = getattr(ind, "architecture", None)
    if isinstance(architecture, dict):
        genotype = architecture.get("benchmark_genotype")
        if isinstance(genotype, list):
            try:
                lat = float(benchmark.query_latency(genotype, dataset, device))
                if lat == lat:
                    return acc, lat
            except Exception:
                pass

    # Fallback when benchmark genotype is unavailable.
    lat_proxy = 1.0 / max(speed, 1e-12)
    return acc, float(lat_proxy)


def _build_strategy(
    method: str,
    csv_path: Path,
    dataset: str,
    device: str,
    pop_size: int,
    budget: int,
):
    benchmark = CSVBenchmarkAPI(str(csv_path))

    if method == "skyline":
        search_space = CSVSearchSpace(str(csv_path))
        evaluator = Evaluator(benchmark, dataset=dataset, device=device)
        return SkylineSearch(search_space=search_space, evaluator=evaluator)
    if method == "bruteforce":
        search_space = CSVSearchSpace(str(csv_path))
        evaluator = Evaluator(benchmark, dataset=dataset, device=device)
        return BruteForceParetoSearch(search_space=search_space, evaluator=evaluator)
    if method == "random":
        search_space = CSVSearchSpace(str(csv_path))
        evaluator = Evaluator(benchmark, dataset=dataset, device=device)
        population = Population(search_space, evaluator, size=pop_size)
        return RandomSearch(
            population=population,
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=ElitistReplacement(),
            evaluator=evaluator,
            budget=budget,
        )
    if method == "mowso":
        search_space = CSVSearchSpace(str(csv_path))
        evaluator = Evaluator(benchmark, dataset=dataset, device=device)
        return MOWSOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=50,
            max_iterations=300,
            archive_size=50,
        )
    if method == "mosho":
        search_space = CSVSearchSpace(str(csv_path))
        evaluator = Evaluator(benchmark, dataset=dataset, device=device)
        return MOSHOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=50,
            max_iterations=300,
            archive_size=50,
        )
    if method == "dvolver":
        search_space = CSVGenotypeDvolverSearchSpace(str(csv_path))
        evaluator = DvolverEvaluator(
            benchmark=benchmark,
            dataset=dataset,
            device=device,
        )
        termination = TerminationCriteria(
            max_generations=max(1, budget),
            max_evaluations=None,
            hypervolume_patience=10,
        )
        return DvolverSearchStrategy(
            population_size=pop_size,
            crossover_prob=0.1,
            mutation_prob=0.1,
            search_space=search_space,
            evaluator=evaluator,
            termination=termination,
        )
    raise ValueError(f"Unknown method: {method}")


def _to_points(
    front: list[Individual],
    method: str,
    benchmark: CSVBenchmarkAPI,
    dataset: str,
    device: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for ind in front:
        parsed = _individual_objectives(ind, method, benchmark, dataset, device)
        if parsed is None:
            continue
        acc, lat = parsed
        points.append((float(acc), float(lat)))
    return points


def _save_method_comparison_plot(comparison_rows: list[dict[str, float]], output_path: Path) -> None:
    if not comparison_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in comparison_rows]
    hv_vals = [row["hv_mean"] for row in comparison_rows]
    igd_vals = [row["igd_plus_mean"] for row in comparison_rows]
    runtime_vals = [row["runtime_mean"] for row in comparison_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(methods, hv_vals)
    axes[0].set_title("Mean Normalized HV")
    axes[0].set_ylabel("higher is better")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(methods, igd_vals)
    axes[1].set_title("Mean IGD+")
    axes[1].set_ylabel("lower is better")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(methods, runtime_vals)
    axes[2].set_title("Mean Runtime (s)")
    axes[2].set_ylabel("lower is better")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _contexts(datasets: tuple[str, ...], devices: tuple[str, ...]):
    context_id = 0
    for dataset in datasets:
        for device in devices:
            context_id += 1
            yield context_id, dataset, device


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.10g}"


def _load_optimal_fronts(path: Path) -> dict[int, list[tuple[float, float]]]:
    fronts: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required_columns = {"context_id", "accuracy", "latency"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Optimal-front CSV {path} is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            fronts[int(row["context_id"])].append(
                (float(row["accuracy"]), float(row["latency"]))
            )

    if not fronts:
        raise ValueError(f"Optimal-front CSV contains no Pareto points: {path}")
    return dict(fronts)


def _reference_point(front: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        min(point[0] for point in front) - 1e-9,
        max(point[1] for point in front) + 1e-9,
    )


def _write_dict_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    method: str,
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    results_root: Path,
    optimal_front_csv: Path | None = None,
) -> dict[str, float]:
    out_dir = results_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    optimal_path = (
        optimal_front_csv
        or ROOT / "experiments" / "results" / "optimal_pareto_fronts.csv"
    )
    optimal_fronts = _load_optimal_fronts(optimal_path)

    pareto_rows: list[list[str]] = []
    run_context_metrics: list[dict[str, Any]] = []
    fronts_by_context: dict[int, list[list[tuple[float, float]]]] = defaultdict(list)

    for run_id in range(1, runs + 1):
        if seed is not None:
            random.seed(seed + run_id)

        for context_id, dataset, device in _contexts(datasets, devices):
            benchmark = CSVBenchmarkAPI(str(csv_path))
            strategy = _build_strategy(
                method=method,
                csv_path=csv_path,
                dataset=dataset,
                device=device,
                pop_size=pop_size,
                budget=budget,
            )

            t0 = time.perf_counter()
            result = strategy.run()
            runtime_s = time.perf_counter() - t0
            front = _extract_front(strategy, result)
            points = _to_points(front, method, benchmark, dataset, device)
            fronts_by_context[context_id].append(points)

            best_acc = max((p[0] for p in points), default=float("nan"))
            best_lat = min((p[1] for p in points), default=float("nan"))
            evals = getattr(strategy, "evaluations", 0)
            if method == "dvolver" and hasattr(strategy, "history"):
                evals = getattr(strategy.history, "evaluation_count", evals)

            for ind in front:
                parsed = _individual_objectives(ind, method, benchmark, dataset, device)
                if parsed is None:
                    continue
                acc, lat = parsed
                arch_id = None
                if ind.metadata:
                    arch_id = ind.metadata.get("arch_id")
                architecture = getattr(ind, "architecture", None)
                if arch_id is None and isinstance(architecture, dict):
                    arch_id = architecture.get("arch_id")
                pareto_rows.append([
                    str(run_id),
                    str(context_id),
                    str(arch_id),
                    device,
                    dataset,
                    f"{acc:.6f}",
                    f"{lat:.6f}",
                    f"{runtime_s:.6f}",
                    str(evals),
                ])

            run_context_metrics.append(
                {
                    "run_id": run_id,
                    "context_id": context_id,
                    "dataset": dataset,
                    "device": device,
                    "best_accuracy": best_acc,
                    "best_latency": best_lat,
                    "runtime": runtime_s,
                    "points": points,
                }
            )

    # Build reference fronts and reference points per context from all runs.
    reference_front_by_context: dict[int, list[tuple[float, float]]] = {}
    for context_id, runs_fronts in fronts_by_context.items():
        union_points = [p for front in runs_fronts for p in front]
        reference_front = non_dominated(union_points, OBJECTIVE_DIRECTIONS)
        reference_front_by_context[context_id] = reference_front

        if context_id not in optimal_fronts:
            raise ValueError(
                f"Optimal-front CSV does not contain context {context_id}."
            )

    # Compute metrics per run/context.
    metrics_rows: list[list[str]] = []
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    hv_by_context: dict[int, list[float]] = defaultdict(list)
    runtime_by_context: dict[int, list[float]] = defaultdict(list)

    for rec in run_context_metrics:
        cid = rec["context_id"]
        points = rec["points"]
        reference_front = reference_front_by_context[cid]
        optimal_front = optimal_fronts[cid]
        ref_point = _reference_point(optimal_front)

        hv = normalized_hypervolume_2d(
            points, OBJECTIVE_DIRECTIONS, ref_point, optimal_front
        )
        igd_p = igd_plus(points, reference_front, OBJECTIVE_DIRECTIONS)
        c_val = c_metric(points, reference_front, OBJECTIVE_DIRECTIONS)
        runtime_s = rec["runtime"]

        grouped[cid].append(
            {
                "best_accuracy": rec["best_accuracy"],
                "best_latency": rec["best_latency"],
                "hv": hv,
                "igd_plus": igd_p,
                "c_metric": c_val,
                "runtime": runtime_s,
            }
        )
        hv_by_context[cid].append(hv)
        runtime_by_context[cid].append(runtime_s)

        metrics_rows.append(
            [
                str(rec["run_id"]),
                str(cid),
                rec["device"],
                rec["dataset"],
                _fmt(rec["best_accuracy"]),
                _fmt(rec["best_latency"]),
                _fmt(hv),
                _fmt(igd_p),
                _fmt(c_val),
                _fmt(runtime_s),
            ]
        )

    # Aggregate mean/std per context.
    summary_rows: list[list[str]] = []
    context_meta = {cid: (ds, dv) for cid, ds, dv in _contexts(datasets, devices)}

    for cid in sorted(grouped.keys()):
        vals = grouped[cid]
        dataset, device = context_meta[cid]

        def _m(key: str) -> float:
            return mean(v[key] for v in vals)

        def _s(key: str) -> float:
            if len(vals) <= 1:
                return 0.0
            return pstdev(v[key] for v in vals)

        summary_rows.append(
            [
                str(cid),
                device,
                dataset,
                _fmt(_m("best_accuracy")),
                _fmt(_s("best_accuracy")),
                _fmt(_m("best_latency")),
                _fmt(_s("best_latency")),
                _fmt(_m("hv")),
                _fmt(median(v["hv"] for v in vals)),
                _fmt(_s("hv")),
                _fmt(_m("igd_plus")),
                _fmt(_s("igd_plus")),
                _fmt(_m("c_metric")),
                _fmt(_s("c_metric")),
                _fmt(_m("runtime")),
                _fmt(_s("runtime")),
            ]
        )

    pareto_csv = out_dir / f"{method}_pareto_front.csv"
    metrics_csv = out_dir / f"{method}_context_metrics.csv"
    per_run_csv = out_dir / f"{method}_metrics_by_run.csv"

    _write_csv(
        pareto_csv,
        [
            "run_id",
            "context_id",
            "archid",
            "device",
            "dataset",
            "accuracy",
            "latency",
            "runtime_sec",
            "evaluations",
        ],
        pareto_rows,
    )

    _write_csv(
        metrics_csv,
        [
            "context_id",
            "device",
            "dataset",
            "best_accuracy_mean",
            "best_accuracy_std",
            "best_latency_mean",
            "best_latency_std",
            "hv_mean",
            "hv_median",
            "hv_std",
            "igd_plus_mean",
            "igd_plus_std",
            "c_metric_mean",
            "c_metric_std",
            "runtime_mean",
            "runtime_std",
        ],
        summary_rows,
    )

    _write_csv(
        per_run_csv,
        [
            "run_id",
            "context_id",
            "device",
            "dataset",
            "best_accuracy",
            "best_latency",
            "hv",
            "igd_plus",
            "c_metric",
            "runtime_sec",
        ],
        metrics_rows,
    )

    # Plots
    save_hv_boxplot(
        hv_by_context,
        out_dir / "hv_boxplot.png",
        title="Normalized Hypervolume Distribution by Context",
    )
    save_runtime_boxplot(runtime_by_context, out_dir / "runtime_boxplot.png")

    hv_means = {cid: mean(vals) for cid, vals in hv_by_context.items() if vals}
    save_context_metric_heatmap(
        hv_means,
        out_dir / "hv_mean_heatmap.png",
        title=f"Normalized HV Mean by Context ({method})",
    )

    # Pareto scatter per context from all runs + reference front.
    for cid, fronts in fronts_by_context.items():
        merged = [p for front in fronts for p in front]
        reference = reference_front_by_context.get(cid, [])
        dataset, device = context_meta[cid]
        save_pareto_scatter(
            points=merged,
            reference_points=reference,
            title=f"Pareto Scatter context={cid} ({dataset}, {device})",
            output_path=out_dir / f"pareto_scatter_context_{cid}.png",
        )

    print("Analysis completed successfully.")
    print(f"Method: {method}")
    print(f"Runs: {runs}")
    print(f"Output folder: {out_dir}")
    print(f"Pareto CSV: {pareto_csv}")
    print(f"Context summary CSV: {metrics_csv}")
    print(f"Per-run metrics CSV: {per_run_csv}")

    flat_vals = [v for vals in grouped.values() for v in vals]
    if not flat_vals:
        return {
            "method": method,
            "best_accuracy_mean": float("nan"),
            "best_latency_mean": float("nan"),
            "hv_mean": float("nan"),
            "igd_plus_mean": float("nan"),
            "c_metric_mean": float("nan"),
            "runtime_mean": float("nan"),
        }

    return {
        "method": method,
        "best_accuracy_mean": mean(v["best_accuracy"] for v in flat_vals),
        "best_latency_mean": mean(v["best_latency"] for v in flat_vals),
        "hv_mean": mean(v["hv"] for v in flat_vals),
        "igd_plus_mean": mean(v["igd_plus"] for v in flat_vals),
        "c_metric_mean": mean(v["c_metric"] for v in flat_vals),
        "runtime_mean": mean(v["runtime"] for v in flat_vals),
    }


def recalculate_saved_hypervolumes(
    results_root: Path,
    optimal_front_csv: Path,
) -> None:
    optimal_fronts = _load_optimal_fronts(optimal_front_csv)
    pareto_csvs = sorted(results_root.glob("*/*_pareto_front.csv"))
    if not pareto_csvs:
        raise FileNotFoundError(
            f"No saved method Pareto fronts found under {results_root}."
        )

    all_hv_by_method: dict[str, list[float]] = {}
    for pareto_csv in pareto_csvs:
        method = pareto_csv.name[: -len("_pareto_front.csv")]
        out_dir = pareto_csv.parent
        metrics_csv = out_dir / f"{method}_metrics_by_run.csv"
        summary_csv = out_dir / f"{method}_context_metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Per-run metrics CSV not found: {metrics_csv}")
        if not summary_csv.exists():
            raise FileNotFoundError(f"Context summary CSV not found: {summary_csv}")

        with pareto_csv.open("r", encoding="utf-8", newline="") as fh:
            front_reader = csv.DictReader(fh)
            required_front_columns = {"run_id", "context_id", "accuracy", "latency"}
            missing = required_front_columns.difference(front_reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Pareto-front CSV {pareto_csv} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )
            points_by_run_context: dict[
                tuple[int, int], list[tuple[float, float]]
            ] = defaultdict(list)
            for row in front_reader:
                key = (int(row["run_id"]), int(row["context_id"]))
                points_by_run_context[key].append(
                    (float(row["accuracy"]), float(row["latency"]))
                )

        normalized_hv_by_run_context: dict[tuple[int, int], float] = {}
        for (run_id, context_id), points in points_by_run_context.items():
            if context_id not in optimal_fronts:
                raise ValueError(
                    f"Optimal-front CSV does not contain context {context_id}."
                )
            optimal_front = optimal_fronts[context_id]
            normalized_hv_by_run_context[(run_id, context_id)] = (
                normalized_hypervolume_2d(
                    points,
                    OBJECTIVE_DIRECTIONS,
                    _reference_point(optimal_front),
                    optimal_front,
                )
            )

        with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
            metrics_reader = csv.DictReader(fh)
            metric_fieldnames = list(metrics_reader.fieldnames or [])
            if "hv" not in metric_fieldnames:
                raise ValueError(f"Per-run metrics CSV has no 'hv' column: {metrics_csv}")
            metric_rows = list(metrics_reader)

        hv_by_context: dict[int, list[float]] = defaultdict(list)
        for row in metric_rows:
            key = (int(row["run_id"]), int(row["context_id"]))
            if key not in normalized_hv_by_run_context:
                raise ValueError(
                    f"Saved Pareto front {pareto_csv} has no points for "
                    f"run {key[0]}, context {key[1]}."
                )
            hv = normalized_hv_by_run_context[key]
            row["hv"] = _fmt(hv)
            hv_by_context[key[1]].append(hv)
        _write_dict_rows(metrics_csv, metric_fieldnames, metric_rows)

        with summary_csv.open("r", encoding="utf-8", newline="") as fh:
            summary_reader = csv.DictReader(fh)
            summary_fieldnames = list(summary_reader.fieldnames or [])
            summary_rows = list(summary_reader)
        if not {"context_id", "hv_mean", "hv_std"}.issubset(summary_fieldnames):
            raise ValueError(
                f"Context summary CSV is missing required HV columns: {summary_csv}"
            )
        if "hv_median" not in summary_fieldnames:
            summary_fieldnames.insert(summary_fieldnames.index("hv_mean") + 1, "hv_median")

        for row in summary_rows:
            context_id = int(row["context_id"])
            context_hvs = hv_by_context.get(context_id)
            if not context_hvs:
                raise ValueError(
                    f"No recalculated HV values for context {context_id} "
                    f"in method {method}."
                )
            row["hv_mean"] = _fmt(mean(context_hvs))
            row["hv_median"] = _fmt(median(context_hvs))
            row["hv_std"] = _fmt(pstdev(context_hvs) if len(context_hvs) > 1 else 0.0)
        _write_dict_rows(summary_csv, summary_fieldnames, summary_rows)

        save_hv_boxplot(
            dict(hv_by_context),
            out_dir / "hv_boxplot.png",
            title="Normalized Hypervolume Distribution by Context",
        )
        save_context_metric_heatmap(
            {
                context_id: mean(values)
                for context_id, values in hv_by_context.items()
                if values
            },
            out_dir / "hv_mean_heatmap.png",
            title=f"Normalized HV Mean by Context ({method})",
        )
        all_hv_by_method[method] = [
            hv for values in hv_by_context.values() for hv in values
        ]
        print(
            f"Recalculated {method}: {len(normalized_hv_by_run_context)} "
            f"run/context HV values."
        )

    comparison_csv = results_root / "method_comparison_summary.csv"
    if comparison_csv.exists():
        with comparison_csv.open("r", encoding="utf-8", newline="") as fh:
            comparison_reader = csv.DictReader(fh)
            comparison_fieldnames = list(comparison_reader.fieldnames or [])
            comparison_rows = list(comparison_reader)
        if "method" not in comparison_fieldnames or "hv_mean" not in comparison_fieldnames:
            raise ValueError(
                f"Method comparison CSV is missing method or hv_mean columns: "
                f"{comparison_csv}"
            )
        for row in comparison_rows:
            values = all_hv_by_method.get(row["method"])
            if values:
                row["hv_mean"] = _fmt(mean(values))
        _write_dict_rows(comparison_csv, comparison_fieldnames, comparison_rows)


def run_all_methods(
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    results_root: Path,
    optimal_front_csv: Path | None = None,
) -> None:
    comparison_rows: list[dict[str, float]] = []

    for method in ALL_METHODS:
        print(f"\n=== Running method: {method} ===")
        summary = run_analysis(
            method=method,
            csv_path=csv_path,
            runs=runs,
            pop_size=pop_size,
            budget=budget,
            seed=seed,
            datasets=datasets,
            devices=devices,
            results_root=results_root,
            optimal_front_csv=optimal_front_csv,
        )
        comparison_rows.append(summary)

    comparison_rows.sort(key=lambda row: row["hv_mean"], reverse=True)

    comparison_csv = results_root / "method_comparison_summary.csv"
    _write_csv(
        comparison_csv,
        [
            "method",
            "best_accuracy_mean",
            "best_latency_mean",
            "hv_mean",
            "igd_plus_mean",
            "c_metric_mean",
            "runtime_mean",
        ],
        [
            [
                str(row["method"]),
                _fmt(row["best_accuracy_mean"]),
                _fmt(row["best_latency_mean"]),
                _fmt(row["hv_mean"]),
                _fmt(row["igd_plus_mean"]),
                _fmt(row["c_metric_mean"]),
                _fmt(row["runtime_mean"]),
            ]
            for row in comparison_rows
        ],
    )

    _save_method_comparison_plot(
        comparison_rows,
        results_root / "method_comparison.png",
    )

    print("\nAll methods completed.")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Comparison plot: {results_root / 'method_comparison.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one or all methods on all contexts and export metrics + comparison plots."
    )
    parser.add_argument(
        "--method",
        choices=[*ALL_METHODS, "all"],
        default="random",
        help="Search strategy method to analyze, or 'all' to run the full benchmark suite.",
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input benchmark CSV.",
    )
    parser.add_argument("--runs", type=int, default=20, help="Number of repetitions.")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (random method).")
    parser.add_argument("--budget", type=int, default=60, help="Evaluation budget (random method).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root output directory where method folder will be created.",
    )
    parser.add_argument(
        "--optimal-front",
        default="experiments/results/optimal_pareto_fronts.csv",
        help="CSV containing the fixed optimal Pareto front for each context.",
    )
    parser.add_argument(
        "--recalculate-hv",
        action="store_true",
        help="Recalculate normalized HV in existing method reports from saved Pareto fronts.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    results_root = _resolve_path(args.results_root)
    optimal_front_csv = _resolve_path(args.optimal_front)
    if args.recalculate_hv:
        recalculate_saved_hypervolumes(results_root, optimal_front_csv)
    else:
        csv_path = _resolve_path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if args.method == "all":
            run_all_methods(
                csv_path=csv_path,
                runs=args.runs,
                pop_size=args.pop_size,
                budget=args.budget,
                seed=args.seed,
                datasets=DEFAULT_DATASETS,
                devices=DEFAULT_DEVICES,
                results_root=results_root,
                optimal_front_csv=optimal_front_csv,
            )
        else:
            run_analysis(
                method=args.method,
                csv_path=csv_path,
                runs=args.runs,
                pop_size=args.pop_size,
                budget=args.budget,
                seed=args.seed,
                datasets=DEFAULT_DATASETS,
                devices=DEFAULT_DEVICES,
                results_root=results_root,
                optimal_front_csv=optimal_front_csv,
            )
