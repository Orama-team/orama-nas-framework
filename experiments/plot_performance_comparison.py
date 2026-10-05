from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_method_frames(results_root: Path) -> pd.DataFrame:
    method_root = results_root / "All_Methods"
    frames: list[pd.DataFrame] = []
    for metrics_path in sorted(method_root.glob("*/*_context_metrics.csv")):
        frame = pd.read_csv(metrics_path)
        frame["method"] = metrics_path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No method context metrics found under {method_root}")
    return pd.concat(frames, ignore_index=True)


def _bar_figure(frame: pd.DataFrame, columns: list[tuple[str, str]], output: Path, title: str) -> None:
    summary = frame.groupby("method", sort=True)[[key for key, _ in columns]].mean().sort_index()
    axes = summary.plot.bar(subplots=True, figsize=(13, 7), layout=(1, len(columns)), legend=False, rot=55)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for axis, (key, label) in zip(axes, columns):
        axis.set_title(label)
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25)
    plt.suptitle(title)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def _green_metrics(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby("method", sort=True)[["best_accuracy_mean", "best_latency_mean"]].mean()
    normalized = summary.copy()
    normalized["best_accuracy_mean"] = (summary["best_accuracy_mean"] - summary["best_accuracy_mean"].min()) / (summary["best_accuracy_mean"].max() - summary["best_accuracy_mean"].min() or 1)
    normalized["best_latency_mean"] = (summary["best_latency_mean"].max() - summary["best_latency_mean"]) / (summary["best_latency_mean"].max() - summary["best_latency_mean"].min() or 1)
    normalized.columns = ["Accuracy (higher is better)", "Latency (lower is better)"]
    axis = normalized.plot.bar(figsize=(13, 7), color=["#2e8b57", "#66cdaa"], rot=55)
    axis.set_ylabel("Normalized score")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def _performance_assessment(frame: pd.DataFrame, output: Path) -> None:
    metrics = ["hv_mean", "igd_plus_mean", "spacing_mean", "runtime_mean"]
    summary = frame.groupby("method", sort=True)[metrics].mean()
    score = pd.DataFrame(index=summary.index)
    score["nHV"] = summary["hv_mean"]
    score["IGD+"] = 1 - (summary["igd_plus_mean"] - summary["igd_plus_mean"].min()) / (summary["igd_plus_mean"].max() - summary["igd_plus_mean"].min() or 1)
    score["Spacing"] = 1 - (summary["spacing_mean"] - summary["spacing_mean"].min()) / (summary["spacing_mean"].max() - summary["spacing_mean"].min() or 1)
    score["Runtime"] = 1 - (summary["runtime_mean"] - summary["runtime_mean"].min()) / (summary["runtime_mean"].max() - summary["runtime_mean"].min() or 1)
    axis = score.plot.bar(figsize=(14, 7), rot=55, ylim=(0, 1), colormap="viridis")
    axis.set_ylabel("Normalized performance score")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the reproducibility performance-comparison figures.")
    parser.add_argument("--results-root", default="experiments/results")
    args = parser.parse_args()
    root = Path(args.results_root)
    frame = _load_method_frames(root)
    output = root / "Performance_Comparison"
    _bar_figure(frame, [("hv_mean", "nHV"), ("igd_plus_mean", "IGD+")], output / "fig_comparative_nhv_igd.png", "Comparative convergence and diversity")
    _bar_figure(frame, [("runtime_mean", "Runtime"), ("spacing_mean", "Spacing")], output / "fig_comparative_time_spacing.png", "Comparative cost and spread")
    _green_metrics(frame, output / "fig_green_metrics.png")
    _performance_assessment(frame, output / "fig_performance_assessment.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
