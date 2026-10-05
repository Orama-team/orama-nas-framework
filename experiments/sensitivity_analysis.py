from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.run_method_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_DEVICES,
    _parse_list_arg,
    _resolve_path,
    run_analysis,
)


def _write_summary(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["budget", "population", "hv_median", "igd_plus_median", "spacing_median", "runtime_median"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_context_metrics(root: Path, budgets: tuple[int, ...], populations: tuple[int, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for budget in budgets:
        for population in populations:
            metrics_path = (
                root
                / f"budget_{budget}_pop_{population}"
                / "mosho_context_metrics.csv"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing sensitivity output: {metrics_path}")
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            if not records:
                raise ValueError(f"Sensitivity output is empty: {metrics_path}")

            def median(field: str) -> str:
                values = sorted(float(record[field]) for record in records)
                midpoint = len(values) // 2
                if len(values) % 2:
                    value = values[midpoint]
                else:
                    value = (values[midpoint - 1] + values[midpoint]) / 2
                return f"{value:.10g}"

            rows.append({
                "budget": str(budget),
                "population": str(population),
                "hv_median": median("hv_median"),
                "igd_plus_median": median("igd_plus_median"),
                "spacing_median": median("spacing_median"),
                "runtime_median": median("runtime_median"),
            })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MOSHO budget and population sensitivity experiments.")
    parser.add_argument("--csv", default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=7)
    parser.add_argument("--budgets", default="1000,2000,5000,7000,10000,15000")
    parser.add_argument("--populations", default="10,20,50")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--results-root", default="experiments/results/_sensitivity results")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_path = _resolve_path(args.csv)
    results_root = _resolve_path(args.results_root)
    budgets = tuple(int(value) for value in _parse_list_arg(args.budgets))
    populations = tuple(int(value) for value in _parse_list_arg(args.populations))
    datasets = _parse_list_arg(args.datasets) or DEFAULT_DATASETS
    devices = _parse_list_arg(args.devices) or DEFAULT_DEVICES

    for budget in budgets:
        for population in populations:
            run_analysis(
                method="mosho",
                csv_path=csv_path,
                runs=args.runs,
                pop_size=population,
                budget=budget,
                seed=args.seed,
                seed_step=args.seed_step,
                datasets=datasets,
                devices=devices,
                reference_fronts_csv=None,
                results_root=results_root / "budget_variation",
                output_name=f"budget_{budget}_pop_{population}",
            )

    budget_results = results_root / "budget_variation" / "All_Methods"
    rows = _aggregate_context_metrics(budget_results, budgets, populations)
    _write_summary(rows, results_root / "budget_variation" / "sensitivity_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
