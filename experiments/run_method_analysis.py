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
from nas_framework.evaluator import Evaluator
from nas_framework.mutation import SinglePointMutation, GaussianMutation, ABCNeighborSampler
from nas_framework.population import Individual, Population, PSOPopulation, ABCPopulation
from nas_framework.replacement import ElitistReplacement, CrowdingReplacement, RankBasedReplacement
from nas_framework.search_space import CSVSearchSpace
from nas_framework.search_strategy import ABCFireflyStrategy,BruteForceParetoSearch, RandomSearch, SkylineSearch, MOWSOSearch, MOSHOSearch, MOSHOEnhancedSearch, PSOSearchStrategy, ABCSearchStrategy, FireflySearchStrategy, HybridMBOStrategy,  APSOESearch, NSGA2SearchStrategy, NSGA2EnhancedSearchStrategy
from nas_framework.selection import TournamentSelection, RouletteWheelSelection
from utilities.metrics import c_metric, igd_plus, non_dominated, normalized_hypervolume_to_reference_2d
from utilities.plotting import (
    save_context_metric_heatmap,
    save_hv_boxplot,
    save_pareto_scatter,
    save_runtime_boxplot,
)

DEFAULT_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
DEFAULT_DEVICES = ("edgegpu", "edgetpu", "eyeriss", "fpga", "pixel3", "raspi4")
OBJECTIVE_DIRECTIONS = (1, -1)
REPRODUCTION_METHODS = (
    "random",
    "bruteforce",
    "skyline",
    "mowso",
    "mosho",
    "mosho_enhanced",
    "abc_firefly",
    "pso",
    "abc",
    "firefly",
    "hybrid_mbo",
    "apso_e",
    "nsga2",
    "nsga2-kumar",
)

MOSHO_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "mosho_enhanced": {"method": "mosho_enhanced", "tex_label": "MOSHO", "disabled_units": set(), "units_removed": ""},
    "mosho": {"method": "mosho", "tex_label": "MOSHO-Base", "disabled_units": set(), "units_removed": ""},
    "mowso": {"method": "mowso", "tex_label": "MOWSO", "disabled_units": set(), "units_removed": ""},
    "abl_u01": {"method": "mosho_enhanced", "tex_label": "ABL-U01", "disabled_units": {"U01"}, "units_removed": "U01"},
    "abl_u02": {"method": "mosho_enhanced", "tex_label": "ABL-U02", "disabled_units": {"U02"}, "units_removed": "U02"},
    "abl_u03": {"method": "mosho_enhanced", "tex_label": "ABL-U03", "disabled_units": {"U03"}, "units_removed": "U03"},
    "abl_u04": {"method": "mosho_enhanced", "tex_label": "ABL-U04", "disabled_units": {"U04"}, "units_removed": "U04"},
    "abl_u05": {"method": "mosho_enhanced", "tex_label": "ABL-U05", "disabled_units": {"U05"}, "units_removed": "U05"},
    "abl_u06": {"method": "mosho_enhanced", "tex_label": "ABL-U06", "disabled_units": {"U06"}, "units_removed": "U06"},
    "abl_u07": {"method": "mosho_enhanced", "tex_label": "ABL-U07", "disabled_units": {"U07"}, "units_removed": "U07"},
    "abl_u08": {"method": "mosho_enhanced", "tex_label": "ABL-U08", "disabled_units": {"U08"}, "units_removed": "U08"},
    "abl_u10": {"method": "mosho_enhanced", "tex_label": "ABL-U10", "disabled_units": {"U10"}, "units_removed": "U10"},
    "abl_u11": {"method": "mosho_enhanced", "tex_label": "ABL-U11", "disabled_units": {"U11"}, "units_removed": "U11"},
    "g_search": {"method": "mosho_enhanced", "tex_label": "G-SEARCH", "disabled_units": {"U01", "U02"}, "units_removed": "U01,U02"},
    "g_adapt": {"method": "mosho_enhanced", "tex_label": "G-ADAPT", "disabled_units": {"U05", "U06", "U07"}, "units_removed": "U05,U06,U07"},
    "g_archive": {"method": "mosho_enhanced", "tex_label": "G-ARCHIVE", "disabled_units": {"U08", "U10"}, "units_removed": "U08,U10"},
    "g_noadv": {"method": "mosho_enhanced", "tex_label": "G-NOADV", "disabled_units": {"U08", "U10", "U11"}, "units_removed": "U08,U10,U11"},
    "g_nobase": {"method": "mosho_enhanced", "tex_label": "G-NOBASE", "disabled_units": {"U01", "U02", "U03", "U04", "U05", "U06", "U07"}, "units_removed": "U01,U02,U03,U04,U05,U06,U07"},
    "g_core": {"method": "mosho_enhanced", "tex_label": "G-CORE", "disabled_units": {"U02", "U08", "U10", "U11"}, "units_removed": "U02,U08,U10,U11"},
}


def _parse_list_arg(raw: str) -> tuple[str, ...]:
    raw = (raw or "").strip()
    if not raw:
        return ()
    if "," in raw:
        items = [x.strip() for x in raw.split(",")]
    else:
        items = [x.strip() for x in raw.split()]  # allow space-separated
    return tuple(x for x in items if x)


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


def _build_strategy(
    method: str,
    search_space: CSVSearchSpace,
    evaluator: Evaluator,
    pop_size: int,
    budget: int,
    disabled_units: set[str] | None = None,
):
    if method == "skyline":
        return SkylineSearch(search_space=search_space, evaluator=evaluator)
    if method == "bruteforce":
        return BruteForceParetoSearch(search_space=search_space, evaluator=evaluator)
    if method == "random":
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
        return MOWSOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
        )
    if method == "mosho":
        return MOSHOSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
            disabled_units=disabled_units,
        )
    if method == "mosho_enhanced":
        return MOSHOEnhancedSearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
            disabled_units=disabled_units,
        )
    if method == "pso":
        population = PSOPopulation(search_space, evaluator, size=pop_size, w=0.4)
        return PSOSearchStrategy(
            population=population,
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(search_space),
            replacement=CrowdingReplacement(),
            evaluator=evaluator,
            budget=budget,
            w=0.4,
        )
    if method == "abc_firefly":
        return ABCFireflyStrategy(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            budget=budget,
            fa_prob=0.5,
            archive_size=pop_size,
            exh_fraction=4.0,
            dmt_fraction=0.67,
        )
    if method == "abc":
        limit = max(5, budget // 25)
        population = ABCPopulation(search_space, evaluator, size=pop_size, abandonment_limit=limit)
        return ABCSearchStrategy(
            population=population,
            neighbor_sampler=ABCNeighborSampler(search_space),
            selection=RouletteWheelSelection(),
            evaluator=evaluator,
            budget=budget,
        )
    if method == "firefly":
        return FireflySearchStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
            w_perf=0.6,
            gamma=1.0,
            beta0=1.0,
            max_chances=5,
            use_fap=True,
            fa_prob=0.5,
        )
    if method == "hybrid_mbo":
        return HybridMBOStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=3),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
            w_perf=0.6,
            rank_fraction=0.5,
            fa_prob=0.5,
        )
    

    if method == "apso_e":
        return APSOESearch(
            search_space=search_space,
            evaluator=evaluator,
            pop_size=pop_size,
            max_iterations=max(1, budget // pop_size),
            archive_size=pop_size,
            w=0.4,
            c1=1.5,
            c2=1.5,
            w_decay=0.99,
            max_hop=3,
            div_threshold=0.4,
            n_inject=4,
        )
    if method == "nsga2":
        return NSGA2SearchStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=2),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=RankBasedReplacement(w_perf=0.6),
            evaluator=evaluator,
            budget=budget,
        )
    if method == "nsga2-kumar":
        return NSGA2EnhancedSearchStrategy(
            population=Population(search_space, evaluator, size=pop_size),
            selection=TournamentSelection(k=2),
            crossover=UniformCrossover(),
            mutation=SinglePointMutation(search_space),
            replacement=CrowdingReplacement(),
            evaluator=evaluator,
            budget=budget,
            archive_size=pop_size,
        )
 
    raise ValueError(f"Unknown method: {method}")


def _to_points(front: list[Individual]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for ind in front:
        if ind.fitness is None:
            continue
        acc, lat = ind.fitness
        if acc != acc or lat != lat:
            continue
        points.append((float(acc), float(lat)))
    return points


def _load_reference_fronts(path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Reference fronts file not found: {path}")

    fronts: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"device", "dataset", "accuracy", "latency"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV missing required columns: {sorted(missing)}")

        for row in reader:
            device = (row.get("device") or "").strip()
            dataset = (row.get("dataset") or "").strip()
            if not device or not dataset:
                continue

            try:
                acc = float(row.get("accuracy") or "nan")
                lat = float(row.get("latency") or "nan")
            except ValueError:
                continue

            if acc != acc or lat != lat:
                continue

            fronts.setdefault((dataset, device), []).append((float(acc), float(lat)))

    return fronts


def _reference_point(reference_front: list[tuple[float, float]]) -> tuple[float, float]:
    if not reference_front:
        return (0.0, 0.0)
    min_acc = min(p[0] for p in reference_front)
    max_lat = max(p[1] for p in reference_front)
    return (min_acc - 1e-9, max_lat + 1e-9)


def _spacing(points: list[tuple[float, float]]) -> float:
    """Spacing (SP): std-dev of distances between consecutive points along the front.

    We sort by accuracy (desc) then latency (asc) to approximate the front order.
    """
    if len(points) < 3:
        return 0.0

    ordered = sorted(points, key=lambda p: (-p[0], p[1]))
    dists: list[float] = []
    for a, b in zip(ordered, ordered[1:]):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dists.append((dx * dx + dy * dy) ** 0.5)
    if len(dists) <= 1:
        return 0.0
    return float(pstdev(dists))


def _tukey_iqr(values: list[float]) -> float:
    """IQR using Tukey's hinges (median of lower/upper halves)."""
    xs = sorted(values)
    n = len(xs)
    if n < 4:
        return 0.0
    half = n // 2
    lower = xs[:half]
    upper = xs[-half:]
    return float(median(upper) - median(lower))


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


def _summary_from_per_run_csv(csv_path: Path) -> dict[str, float]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_id = int(row["run_id"])
            grouped[run_id].append(
                {
                    "hv": float(row["hv"]),
                    "igd_plus": float(row["igd_plus"]),
                    "spacing": float(row["spacing"]),
                    "runtime_sec": float(row["runtime_sec"]),
                }
            )

    aggregated: dict[str, list[float]] = {
        "hv": [],
        "igd_plus": [],
        "spacing": [],
        "runtime_sec": [],
    }
    for run_rows in grouped.values():
        for key in aggregated:
            aggregated[key].append(mean(item[key] for item in run_rows))

    return {
        "hv_median": float(median(aggregated["hv"])),
        "hv_iqr": _tukey_iqr(aggregated["hv"]),
        "igd_plus_median": float(median(aggregated["igd_plus"])),
        "igd_plus_iqr": _tukey_iqr(aggregated["igd_plus"]),
        "spacing_median": float(median(aggregated["spacing"])),
        "spacing_iqr": _tukey_iqr(aggregated["spacing"]),
        "runtime_sec_median": float(median(aggregated["runtime_sec"])),
        "runtime_sec_iqr": _tukey_iqr(aggregated["runtime_sec"]),
        "n_runs": float(len(aggregated["hv"])),
    }


def run_analysis(
    method: str,
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    seed_step: int,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    reference_fronts_csv: Path | None,
    results_root: Path,
    disabled_units: set[str] | None = None,
    output_name: str | None = None,
) -> Path:
    out_dir = results_root / "All_Methods" / (output_name or method)
    out_dir.mkdir(parents=True, exist_ok=True)

    pareto_rows: list[list[str]] = []
    run_context_metrics: list[dict[str, Any]] = []
    fronts_by_context: dict[int, list[list[tuple[float, float]]]] = defaultdict(list)

    for run_id in range(1, runs + 1):
        if seed is not None:
            if seed_step and seed_step > 0:
                random.seed(seed + (run_id - 1) * seed_step)
            else:
                # Backwards-compatible default (legacy behaviour).
                random.seed(seed + run_id)

        for context_id, dataset, device in _contexts(datasets, devices):
            search_space = CSVSearchSpace(str(csv_path))
            benchmark = CSVBenchmarkAPI(str(csv_path))
            evaluator = Evaluator(benchmark, dataset=dataset, device=device)
            strategy = _build_strategy(
                method,
                search_space,
                evaluator,
                pop_size,
                budget,
                disabled_units=disabled_units,
            )

            t0 = time.perf_counter()
            result = strategy.run()
            runtime_s = time.perf_counter() - t0
            front = _extract_front(strategy, result)
            points = _to_points(front)
            fronts_by_context[context_id].append(points)

            best_acc = max((p[0] for p in points), default=float("nan"))
            best_lat = min((p[1] for p in points), default=float("nan"))
            evals = getattr(strategy, "evaluations", 0)

            sp = _spacing(points)

            for ind in front:
                if ind.fitness is None:
                    continue
                acc, lat = ind.fitness
                if acc != acc or lat != lat:
                    continue
                arch_id = ind.metadata.get("arch_id") if ind.metadata else None
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
                    "evaluations": float(evals),
                    "spacing": float(sp),
                    "points": points,
                }
            )

    # Reference fronts/points per context.
    reference_front_by_context: dict[int, list[tuple[float, float]]] = {}
    ref_point_by_context: dict[int, tuple[float, float]] = {}

    context_meta = {cid: (ds, dv) for cid, ds, dv in _contexts(datasets, devices)}

    if reference_fronts_csv is not None:
        ref_map = _load_reference_fronts(reference_fronts_csv)
        for cid in fronts_by_context.keys():
            dataset, device = context_meta[cid]
            ref_front = ref_map.get((dataset, device), [])
            if not ref_front:
                raise KeyError(
                    f"No reference front found for dataset={dataset}, device={device} in {reference_fronts_csv}"
                )
            ref_front_nd = non_dominated(ref_front, OBJECTIVE_DIRECTIONS)
            reference_front_by_context[cid] = ref_front_nd
            ref_point_by_context[cid] = _reference_point(ref_front_nd)
    else:
        # Build reference fronts and reference points per context from all runs.
        for cid, runs_fronts in fronts_by_context.items():
            union_points = [p for front in runs_fronts for p in front]
            reference_front = non_dominated(union_points, OBJECTIVE_DIRECTIONS)
            reference_front_by_context[cid] = reference_front
            ref_point_by_context[cid] = _reference_point(union_points)

    # Compute metrics per run/context.
    metrics_rows: list[list[str]] = []
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    hv_by_context: dict[int, list[float]] = defaultdict(list)
    runtime_by_context: dict[int, list[float]] = defaultdict(list)

    for rec in run_context_metrics:
        cid = rec["context_id"]
        points = rec["points"]
        reference_front = reference_front_by_context[cid]
        ref_point = ref_point_by_context[cid]

        hv = normalized_hypervolume_to_reference_2d(
            points,
            reference_front,
            OBJECTIVE_DIRECTIONS,
            ref_point,
        )
        igd_p = igd_plus(points, reference_front, OBJECTIVE_DIRECTIONS)
        c_val = c_metric(points, reference_front, OBJECTIVE_DIRECTIONS)
        runtime_s = rec["runtime"]
        sp = rec["spacing"]
        evals = rec["evaluations"]

        grouped[cid].append(
            {
                "best_accuracy": rec["best_accuracy"],
                "best_latency": rec["best_latency"],
                "hv": hv,
                "igd_plus": igd_p,
                "spacing": sp,
                "c_metric": c_val,
                "runtime": runtime_s,
                "evaluations": evals,
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
                _fmt(sp),
                _fmt(c_val),
                _fmt(runtime_s),
                _fmt(evals),
            ]
        )

    # Aggregate mean/std + median/IQR per context.
    summary_rows: list[list[str]] = []

    for cid in sorted(grouped.keys()):
        vals = grouped[cid]
        dataset, device = context_meta[cid]

        def _m(key: str) -> float:
            return mean(v[key] for v in vals)

        def _s(key: str) -> float:
            if len(vals) <= 1:
                return 0.0
            return pstdev(v[key] for v in vals)

        def _med(key: str) -> float:
            return float(median(v[key] for v in vals))

        def _iqr(key: str) -> float:
            return _tukey_iqr([float(v[key]) for v in vals])

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
                _fmt(_s("hv")),
                _fmt(_m("igd_plus")),
                _fmt(_s("igd_plus")),
                _fmt(_m("spacing")),
                _fmt(_s("spacing")),
                _fmt(_m("c_metric")),
                _fmt(_s("c_metric")),
                _fmt(_m("runtime")),
                _fmt(_s("runtime")),
                _fmt(_m("evaluations")),
                _fmt(_s("evaluations")),
                _fmt(_med("hv")),
                _fmt(_iqr("hv")),
                _fmt(_med("igd_plus")),
                _fmt(_iqr("igd_plus")),
                _fmt(_med("spacing")),
                _fmt(_iqr("spacing")),
                _fmt(_med("runtime")),
                _fmt(_iqr("runtime")),
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
            "hv_std",
            "igd_plus_mean",
            "igd_plus_std",
            "spacing_mean",
            "spacing_std",
            "c_metric_mean",
            "c_metric_std",
            "runtime_mean",
            "runtime_std",
            "evaluations_mean",
            "evaluations_std",
            "hv_median",
            "hv_iqr",
            "igd_plus_median",
            "igd_plus_iqr",
            "spacing_median",
            "spacing_iqr",
            "runtime_median",
            "runtime_iqr",
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
            "spacing",
            "c_metric",
            "runtime_sec",
            "evaluations",
        ],
        metrics_rows,
    )

    # Plots
    save_hv_boxplot(hv_by_context, out_dir / "hv_boxplot.png")
    save_runtime_boxplot(runtime_by_context, out_dir / "runtime_boxplot.png")

    hv_means = {cid: mean(vals) for cid, vals in hv_by_context.items() if vals}
    save_context_metric_heatmap(
        hv_means,
        out_dir / "hv_mean_heatmap.png",
        title=f"HV Mean by Context ({method})",
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
    return per_run_csv


def run_ablation_suite(
    *,
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    seed_step: int,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    reference_fronts_csv: Path | None,
    results_root: Path,
) -> Path:
    summary_rows: list[list[str]] = []

    for variant_name, spec in MOSHO_ABLATION_VARIANTS.items():
        per_run_csv = run_analysis(
            method=spec["method"],
            csv_path=csv_path,
            runs=runs,
            pop_size=pop_size,
            budget=budget,
            seed=seed,
            seed_step=seed_step,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=reference_fronts_csv,
            results_root=results_root,
            disabled_units=set(spec["disabled_units"]),
            output_name=variant_name,
        )
        stats = _summary_from_per_run_csv(per_run_csv)
        summary_rows.append(
            [
                variant_name,
                spec["tex_label"],
                spec["method"],
                ",".join(sorted(spec["disabled_units"])),
                spec["units_removed"],
                _fmt(stats["igd_plus_median"]),
                _fmt(stats["igd_plus_iqr"]),
                _fmt(stats["hv_median"]),
                _fmt(stats["hv_iqr"]),
                _fmt(stats["spacing_median"]),
                _fmt(stats["spacing_iqr"]),
                _fmt(stats["runtime_sec_median"]),
                _fmt(stats["runtime_sec_iqr"]),
                str(int(stats["n_runs"])),
            ]
        )

    summary_csv = results_root / "ablation_suite_summary.csv"
    _write_csv(
        summary_csv,
        [
            "variant",
            "tex_label",
            "base_method",
            "disabled_units",
            "units_removed",
            "igd_plus_median",
            "igd_plus_iqr",
            "hv_median",
            "hv_iqr",
            "spacing_median",
            "spacing_iqr",
            "runtime_sec_median",
            "runtime_sec_iqr",
            "n_runs",
        ],
        summary_rows,
    )
    print(f"Ablation suite summary CSV: {summary_csv}")
    return summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one method on all contexts multiple times and export metrics + plots."
    )
    parser.add_argument(
        "--method",
        choices=list(REPRODUCTION_METHODS),
        default="random",
        help="Search strategy method to analyze.",
    )
    parser.add_argument(
        "--variant",
        default="",
        help=(
            "Optional MOSHO ablation variant name. "
            f"Supported: {', '.join(sorted(MOSHO_ABLATION_VARIANTS))}"
        ),
    )
    parser.add_argument(
        "--suite",
        choices=["", "mosho_ablation"],
        default="",
        help="Run a predefined suite of experiments and emit a consolidated summary CSV.",
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input benchmark CSV.",
    )
    parser.add_argument("--runs", type=int, default=30, help="Number of independent repetitions.")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (random method).")
    parser.add_argument("--budget", type=int, default=60, help="Evaluation budget (random method).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help=(
            "Seed schedule step between runs. "
            "Set to 7 to match ablation_study.tex (seed = base + (run-1)*7). "
            "Default 0 keeps legacy behaviour."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Datasets to run (comma or space separated).",
    )
    parser.add_argument(
        "--devices",
        default=",".join(DEFAULT_DEVICES),
        help="Devices to run (comma or space separated).",
    )
    parser.add_argument(
        "--reference-fronts",
        default="",
        help=(
            "Optional CSV containing fixed reference Pareto fronts (device,dataset,accuracy,latency). "
            "If provided, IGD+/nHV are computed against this fixed reference (recommended for ablation)."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root output directory where method folder will be created.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    datasets = _parse_list_arg(args.datasets) or DEFAULT_DATASETS
    devices = _parse_list_arg(args.devices) or DEFAULT_DEVICES
    reference = _resolve_path(args.reference_fronts) if (args.reference_fronts or "").strip() else None

    results_root = _resolve_path(args.results_root)
    if args.suite == "mosho_ablation":
        run_ablation_suite(
            csv_path=csv_path,
            runs=args.runs,
            pop_size=args.pop_size,
            budget=args.budget,
            seed=args.seed,
            seed_step=args.seed_step,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=reference,
            results_root=results_root,
        )
        raise SystemExit(0)

    disabled_units: set[str] | None = None
    output_name: str | None = None
    method = args.method
    if (args.variant or "").strip():
        key = args.variant.strip().lower()
        if key not in MOSHO_ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown variant {args.variant!r}. Supported values: {sorted(MOSHO_ABLATION_VARIANTS)}"
            )
        spec = MOSHO_ABLATION_VARIANTS[key]
        method = str(spec["method"])
        disabled_units = set(spec["disabled_units"])
        output_name = key

    run_analysis(
        method=method,
        csv_path=csv_path,
        runs=args.runs,
        pop_size=args.pop_size,
        budget=args.budget,
        seed=args.seed,
        seed_step=args.seed_step,
        datasets=datasets,
        devices=devices,
        reference_fronts_csv=reference,
        results_root=results_root,
        disabled_units=disabled_units,
        output_name=output_name,
    )
