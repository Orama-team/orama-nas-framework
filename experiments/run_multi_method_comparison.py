from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
import pandas as pd

# Ensure imports work when running: python experiments/run_multi_method_comparison.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_method_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_DEVICES,
    _resolve_path,
    _write_csv,
    run_analysis,
)

ALLOWED_METHODS = (
    "random",
    "bruteforce",
    "skyline",
    "mowso",
    "mosho",
    "mosho_enhanced",
    "nsga2",
    "nsga2-kumar",
)


def _parse_float(raw: str) -> float:
    if raw == "inf":
        return float("inf")
    return float(raw)


def _load_per_run_metrics(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "run_id": int(row["run_id"]),
                    "context_id": int(row["context_id"]),
                    "device": row["device"],
                    "dataset": row["dataset"],
                    "best_accuracy": _parse_float(row["best_accuracy"]),
                    "best_latency": _parse_float(row["best_latency"]),
                    "hv": _parse_float(row["hv"]),
                    "igd_plus": _parse_float(row["igd_plus"]),
                    "spacing": _parse_float(row["spacing"]),
                    "c_metric": _parse_float(row["c_metric"]),
                    "runtime_sec": _parse_float(row["runtime_sec"]),
                }
            )
    return rows


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.10g}"


def compare_many_methods(
    methods: list[str],
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int | None,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    results_root: Path,
) -> None:
    unique_methods: list[str] = []
    for m in methods:
        if m not in ALLOWED_METHODS:
            raise ValueError(f"Unknown method: {m}. Allowed: {', '.join(ALLOWED_METHODS)}")
        if m not in unique_methods:
            unique_methods.append(m)

    if len(unique_methods) < 2:
        raise ValueError("You must provide at least two distinct methods.")

    # Run each method analysis with exactly the same configuration.
    for method in unique_methods:
        run_analysis(
            method=method,
            csv_path=csv_path,
            runs=runs,
            pop_size=pop_size,
            budget=budget,
            seed=seed,
            seed_step=0,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=None,
            results_root=results_root,
        )

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    by_context_method: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for method in unique_methods:
        per_run_csv = results_root / "All_Methods" / method / f"{method}_metrics_by_run.csv"
        if not per_run_csv.exists():
            raise FileNotFoundError(f"Missing per-run metrics file: {per_run_csv}")

        rows = _load_per_run_metrics(per_run_csv)
        rows_by_method[method] = rows

        for row in rows:
            by_context_method[row["context_id"]][method].append(row)

    context_ids = sorted(by_context_method.keys())
    if not context_ids:
        raise ValueError("No contexts found in method outputs.")

    eps = 1e-12

    def _avg(records: list[dict[str, Any]], key: str) -> float:
        return mean(r[key] for r in records)

    # Per-context ranking using lexicographic priority:
    # higher hv, lower igd_plus, lower runtime_sec.
    context_rows: list[list[str]] = []
    rank_points = {m: 0 for m in unique_methods}
    win_counts = {m: 0 for m in unique_methods}

    for cid in context_ids:
        methods_here = [m for m in unique_methods if by_context_method[cid].get(m)]
        if len(methods_here) < 2:
            continue

        dataset = by_context_method[cid][methods_here[0]][0]["dataset"]
        device = by_context_method[cid][methods_here[0]][0]["device"]

        # Build sortable records with deterministic tie-break by method name.
        scored: list[tuple[str, float, float, float]] = []
        for method in methods_here:
            recs = by_context_method[cid][method]
            hv_mean = _avg(recs, "hv")
            igd_mean = _avg(recs, "igd_plus")
            runtime_mean = _avg(recs, "runtime_sec")
            scored.append((method, hv_mean, igd_mean, runtime_mean))

        scored.sort(
            key=lambda x: (
                -round(x[1], 12),
                round(x[2], 12),
                round(x[3], 12),
                x[0],
            )
        )

        best_method = scored[0][0]
        # Handle strict tie at top by hv/igd/runtime.
        top = scored[0]
        tied_top = [
            s for s in scored
            if abs(s[1] - top[1]) <= eps and abs(s[2] - top[2]) <= eps and abs(s[3] - top[3]) <= eps
        ]
        if len(tied_top) == 1:
            win_counts[best_method] += 1

        # Rank points: n for first, n-1 for second, ...
        n = len(scored)
        for idx, (method, _, _, _) in enumerate(scored):
            rank_points[method] += n - idx

        context_rows.append(
            [
                str(cid),
                device,
                dataset,
                "; ".join(m for m, _, _, _ in scored),
                best_method if len(tied_top) == 1 else "tie",
            ]
        )

    # Global aggregates per method.
    global_rows: list[list[str]] = []
    for method in unique_methods:
        rows = rows_by_method[method]
        global_rows.append(
            [
                method,
                _fmt(mean(r["best_accuracy"] for r in rows)),
                _fmt(mean(r["best_latency"] for r in rows)),
                _fmt(mean(r["hv"] for r in rows)),
                _fmt(mean(r["igd_plus"] for r in rows)),
                _fmt(mean(r["spacing"] for r in rows)),
                _fmt(mean(r["runtime_sec"] for r in rows)),
                str(win_counts[method]),
                str(rank_points[method]),
            ]
        )

    # Sort by rank points desc then wins desc.
    global_rows.sort(key=lambda r: (-int(r[8]), -int(r[7]), r[0]))

    out_dir = results_root / "comparisons"
    methods_label = "_vs_".join(unique_methods)

    contexts_csv = out_dir / f"{methods_label}_contexts.csv"
    summary_csv = out_dir / f"{methods_label}_summary.csv"

    _write_csv(
        contexts_csv,
        ["context_id", "device", "dataset", "ranking_best_to_worst", "winner"],
        context_rows,
    )
    _write_csv(
        summary_csv,
        [
            "method",
            "best_accuracy_mean",
            "best_latency_mean",
            "nhv_mean",
            "igd_plus_mean",
            "spacing_mean",
            "exec_time_mean_sec",
            "context_wins",
            "rank_points",
        ],
        global_rows,
    )

    # Keep a machine-readable workbook alongside the CSV summaries.  The
    # per-run sheet preserves the values used for aggregation; the other
    # sheets are directly suitable for tables/plots in a report.
    workbook_path = out_dir / f"{methods_label}_metrics.xlsx"
    per_run_rows = [
        {"method": method, **row}
        for method in unique_methods
        for row in rows_by_method[method]
    ]
    summary_frame = pd.DataFrame(
        global_rows,
        columns=[
            "method", "best_accuracy_mean", "best_latency_mean", "nhv_mean",
            "igd_plus_mean", "spacing_mean", "exec_time_mean_sec",
            "context_wins", "rank_points",
        ],
    )
    per_run_frame = pd.DataFrame(per_run_rows).rename(
        columns={"hv": "nhv", "runtime_sec": "exec_time_sec"}
    )
    context_frame = pd.DataFrame(
        context_rows,
        columns=["context_id", "device", "dataset", "ranking_best_to_worst", "winner"],
    )
    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            summary_frame.to_excel(writer, sheet_name="summary", index=False)
            per_run_frame[
                ["method", "run_id", "context_id", "device", "dataset",
                 "nhv", "igd_plus", "spacing", "exec_time_sec"]
            ].to_excel(writer, sheet_name="per_run", index=False)
            context_frame.to_excel(writer, sheet_name="contexts", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires openpyxl. Install dependencies from requirements.txt."
        ) from exc

    print("Multi-method comparison completed successfully.")
    print(f"Methods: {', '.join(unique_methods)}")
    print(f"Runs: {runs}")
    print(f"Contexts compared: {len(context_rows)}")
    print(f"Context ranking CSV: {contexts_csv}")
    print(f"Global summary CSV: {summary_csv}")
    print(f"Excel metrics workbook: {workbook_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare multiple search methods under identical settings."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="List of methods to compare (e.g. random mowso skyline).",
    )
    parser.add_argument(
        "--csv",
        default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv",
        help="Input benchmark CSV.",
    )
    parser.add_argument("--runs", type=int, default=30, help="Number of independent repetitions.")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size for population-based methods.")
    parser.add_argument("--budget", type=int, default=60, help="Evaluation budget for budgeted methods.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root output directory.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    compare_many_methods(
        methods=list(args.methods),
        csv_path=csv_path,
        runs=args.runs,
        pop_size=args.pop_size,
        budget=args.budget,
        seed=args.seed,
        datasets=DEFAULT_DATASETS,
        devices=DEFAULT_DEVICES,
        results_root=_resolve_path(args.results_root),
    )
