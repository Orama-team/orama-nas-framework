from __future__ import annotations

import argparse
from pathlib import Path

from experiments.run_method_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_DEVICES,
    REPRODUCTION_METHODS,
    _parse_list_arg,
    _resolve_path,
    run_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every registered method with one reproducible configuration."
    )
    parser.add_argument("--csv", default="nas_benchmarks/datasets/nas_hw_search_space_bench.csv")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--budget", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=7)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--results-root", default="experiments/results")
    return parser


def run_all_methods(
    *,
    csv_path: Path,
    runs: int,
    pop_size: int,
    budget: int,
    seed: int,
    seed_step: int,
    datasets: tuple[str, ...],
    devices: tuple[str, ...],
    results_root: Path,
) -> None:
    for method in REPRODUCTION_METHODS:
        run_analysis(
            method=method,
            csv_path=csv_path,
            runs=runs,
            pop_size=pop_size,
            budget=budget,
            seed=seed,
            seed_step=seed_step,
            datasets=datasets,
            devices=devices,
            reference_fronts_csv=None,
            results_root=results_root,
        )


def main() -> int:
    args = build_parser().parse_args()
    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    run_all_methods(
        csv_path=csv_path,
        runs=args.runs,
        pop_size=args.pop_size,
        budget=args.budget,
        seed=args.seed,
        seed_step=args.seed_step,
        datasets=_parse_list_arg(args.datasets) or DEFAULT_DATASETS,
        devices=_parse_list_arg(args.devices) or DEFAULT_DEVICES,
        results_root=_resolve_path(args.results_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
