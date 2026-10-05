<p align="center">
  <h1 align="center">Orama NAS Codebase</h1>
  <p align="center">
    A modular framework for composing and evaluating multi-objective Neural Architecture Search strategies from interchangeable components.
  </p>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework-architecture">Architecture</a> •
  <a href="#search-strategies">Strategies</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**Orama NAS** is a research framework for multi-objective Neural Architecture Search (NAS) on the [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) search space. It operates on a unified dataset produced by merging **NAS-Bench-201** (accuracy) and **HW-NAS-Bench** (hardware latency) to jointly optimize **accuracy–latency trade-offs** across diverse hardware targets.

The framework decomposes every search strategy into pluggable, interchangeable components — selection, crossover, mutation, variation, replacement, termination, and evaluation — making it straightforward to prototype new methods, run ablations, and produce fair comparisons under identical budgets.

This codebase accompanies the paper on **MOSHO** (Multi-Objective Shark Hunting Optimization), our proposed bio-inspired NAS method. All results reported in the paper are fully reproducible from the artifacts in `experiments/results/`.

### Key Features

- **Modular component design** — swap any operator (selection, mutation, crossover, replacement, termination) without changing the search strategy logic.
- **17 search strategies** — from baselines (Random, Brute-Force) to state-of-the-art metaheuristics (NSGA-II, PSO, ABC, Firefly, MOWSO) and our proposed MOSHO variants.
- **Multi-objective evaluation** — built-in Normalized Hypervolume (NHV), IGD⁺, and C-metric computation against optimal reference fronts.
- **18 hardware × dataset contexts** — 3 datasets (CIFAR-10, CIFAR-100, ImageNet16-120) × 6 devices (EdgeGPU, EdgeTPU, Eyeriss, FPGA, Pixel3, RasPi4).
- **Full ablation support** — unit-level ablation infrastructure for MOSHO with 11 individually disableable algorithm components.
- **Hyperparameter tuning** — successive-halving random search tuner for MOSHO energy parameters.

---

## Framework Architecture

```
orama-nas-codebase/
├── nas_framework/            # Core framework
│   ├── search_space.py       # SearchSpace, CSVSearchSpace (NAS-Bench-201 encoding)
│   ├── benchmark_api.py      # BenchmarkAPI, NASBench201BenchmarkAPI, CSVBenchmarkAPI
│   ├── evaluator.py          # Bi-objective evaluator (accuracy ↑, latency ↓)
│   ├── population.py         # Individual, Population, ABCPopulation, PSOPopulation
│   ├── selection.py          # Tournament, RouletteWheel, Neighbor, Guidance
│   ├── crossover.py          # Uniform, SinglePoint
│   ├── mutation.py           # SinglePoint, BitFlip, ABCNeighborSampler, Gaussian
│   ├── variation.py          # CrossoverMutation, MutationOnly
│   ├── replacement.py        # Elitist, Generational, RankBased, Crowding
│   ├── termination.py        # MaxEvaluations, MaxGenerations, Composite
│   ├── mo_utils.py           # Pareto sorting, crowding distance, dominance
│   ├── history.py            # Run history tracking and Pareto archive snapshots
│   └── search_strategy.py    # All search strategy implementations
│
├── nas_benchmarks/           # Benchmark data & preprocessing
│   ├── datasets/             # Merged CSV benchmark (NAS-Bench-201 + HW-NAS-Bench)
│   ├── benchmarks_preprocessing_pipeline.py
│   └── impute_worst_case_latency.py
│
├── experiments/              # Experiment runners & results
│   ├── run_method_analysis.py          # Single-method multi-context evaluation
│   ├── run_multi_method_comparison.py  # Head-to-head method comparison
│   ├── run_all_methods.py              # Reproducible all-method experiment suite
│   ├── sensitivity_analysis.py         # Reproducible MOSHO sensitivity sweeps
│   ├── plot_performance_comparison.py  # Rebuilds the four performance figures
│   ├── tune_mosho_hyperparams.py       # MOSHO hyperparameter tuning
│   ├── context_runner.py               # Context-aware experiment runner
│   ├── make_ablation_overview_png.py   # Ablation figure generation
│   └── results/                        # Pre-computed results for reproducibility
│
├── utilities/                # Shared utilities
│   ├── metrics.py            # HV, NHV, IGD⁺, C-metric implementations
│   └── plotting.py           # Pareto scatter, boxplot, heatmap plots
│
└── requirements.txt
```

### Component Interfaces

Every component exposes a minimal abstract interface, enabling plug-and-play composition:

| Component       | Base Class     | Implementations                                                  |
|-----------------|----------------|-------------------------------------------------------------------|
| **Selection**   | `Selection`    | `TournamentSelection`, `RouletteWheelSelection`, `NeighborSelection`, `GuidanceSelection` |
| **Crossover**   | `Crossover`    | `UniformCrossover`, `SinglePointCrossover`                        |
| **Mutation**    | `Mutation`     | `SinglePointMutation`, `BitFlipMutation`, `ABCNeighborSampler`, `GaussianMutation` |
| **Variation**   | `Variation`    | `CrossoverMutationVariation`, `MutationOnlyVariation`             |
| **Replacement** | `Replacement`  | `ElitistReplacement`, `GenerationalReplacement`, `RankBasedReplacement`, `CrowdingReplacement` |
| **Termination** | `Termination`  | `MaxEvaluationsTermination`, `MaxGenerationsTermination`, `CompositeTermination` |

---

## Search Strategies

The framework implements the following 17 search strategies:

### Baselines
| Strategy | Class | Description |
|----------|-------|-------------|
| Random Search | `RandomSearch` | Uniform random sampling baseline |
| Brute-Force Pareto | `BruteForceParetoSearch` | Exhaustive enumeration of the full search space |
| Skyline | `SkylineSearch` | Exact 2-objective Pareto front via sort-and-sweep |

### Evolutionary Algorithms
| Strategy | Class | Description |
|----------|-------|-------------|
| Genetic Algorithm | `GeneticAlgorithm` | Pareto-based GA with selection → variation → replacement loop |
| (μ, λ)-ES | `EvolutionStrategy` | Evolution strategy with mutation-only variation |
| NSGA-II | `NSGA2SearchStrategy` | Non-dominated sorting GA with crowding distance |

### Swarm Intelligence
| Strategy | Class | Description |
|----------|-------|-------------|
| PSO | `PSOSearchStrategy` | Multi-objective Particle Swarm Optimization (MOIPSO) |
| ABC | `ABCSearchStrategy` | Artificial Bee Colony (HiveNAS) with employed/onlooker/scout phases |
| Firefly | `FireflySearchStrategy` | Firefly Algorithm for discrete NAS |
| ABC-Firefly | `ABCFireflyStrategy` | Hybrid ABC + Firefly search |
| APSOE | `APSOESearch` | Adaptive Particle Swarm Optimization with Evolution |

### Bio-Inspired Optimizers
| Strategy | Class | Description |
|----------|-------|-------------|
| MOWSO | `MOWSOSearch` | Multi-Objective White Shark Optimizer with archive true-distance pruning |
| MBO | `MBOSearchStrategy` | Monarch Butterfly Optimization |
| Hybrid MBO | `HybridMBOStrategy` | Hybrid Monarch Butterfly Optimization |

### **MOSHO** (Proposed)
| Strategy | Class | Description |
|----------|-------|-------------|
| **MOSHO** | `MOSHOSearch` | Multi-Objective Shark Hunting Optimization — our proposed method |
| **MOSHO Enhanced** | `MOSHOEnhancedSearch` | MOSHO with clustered archive sampling, mono-objective safeguards, and robustness enhancements |

MOSHO features six bio-inspired hunting operators — **patrol**, **scent tracking**, **circle tightening**, **burst attack**, **crossover**, and **scout** — with adaptive operator probability credit assignment. It uses an energy model with parameters `e0`, `e_min`, `e_max`, and `delta` to control exploration–exploitation balance across iterations.

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/<your-org>/orama-nas-codebase.git
cd orama-nas-codebase
pip install -r requirements.txt
```

### Dataset

The framework operates on a merged CSV benchmark that combines accuracy metrics from NAS-Bench-201 and hardware latency measurements from HW-NAS-Bench. A pre-processed version is provided at:

```
nas_benchmarks/datasets/nas_hw_search_space_bench.csv
```

To regenerate from the raw benchmark files:

```bash
python nas_benchmarks/benchmarks_preprocessing_pipeline.py \
    --nas path/to/NAS-Bench-201-v1_0-e61699.pth \
    --hw  path/to/HW-NAS-Bench-v1_0.pickle
```

### Quick Start

```python
from nas_framework import (
    CSVSearchSpace, CSVBenchmarkAPI, Evaluator,
    MOSHOEnhancedSearch, History,
)

# Load the benchmark
csv_path = "nas_benchmarks/datasets/nas_hw_search_space_bench.csv"
search_space = CSVSearchSpace(csv_path)
benchmark = CSVBenchmarkAPI(csv_path)
evaluator = Evaluator(benchmark, dataset="cifar100", device="edgegpu")

# Run MOSHO
history = History()
strategy = MOSHOEnhancedSearch(
    search_space=search_space,
    evaluator=evaluator,
    pop_size=50,
    max_iterations=300,
    history=history,
)
pareto_front = strategy.run()

# Inspect results
for ind in pareto_front:
    acc, lat = ind.fitness
    print(f"Accuracy: {acc:.2f}%  Latency: {lat:.4f} ms")
```

---

## Experiments

### Single-Method Evaluation

Run a method across all 18 dataset × device contexts with the evaluation
protocol used for the reported results:

```bash
python experiments/run_method_analysis.py \
    --method mosho_enhanced \
    --csv nas_benchmarks/datasets/nas_hw_search_space_bench.csv \
    --runs 30 \
    --budget 5000 \
    --pop-size 50
```

### Reproducing all results

The root `Makefile` is the single reproducibility entry point. It runs the
registered methods, reruns the MOSHO sensitivity sweeps, and rebuilds every
figure in `experiments/results/Performance_Comparison/`:

```bash
make reproduce
```

The evaluation protocol uses **30 independent runs per method and context**.
The random seed for run `n` is `42 + (n - 1) * 7`, giving the sequence
`42, 49, 56, ...`. Evaluation budgets are
`B ∈ {1,000, 2,000, 5,000, 7,000, 10,000, 15,000}`. All methods use identical
contexts, run counts, seed schedule, and budget when being compared.

Before evaluation, each method's hyperparameters are tuned with **Optuna**.
The selected configuration is then fixed for the independent evaluation runs;
evaluation runs do not retune parameters or share state. The enhanced NSGA-II
implementation is named `nsga2-kumar` in all new experiment outputs and
commands.

### Multi-Method Comparison

Compare multiple methods under identical experimental conditions:

```bash
python experiments/run_multi_method_comparison.py \
    --methods random skyline mowso nsga2-kumar \
    --csv nas_benchmarks/datasets/nas_hw_search_space_bench.csv \
    --runs 30 \
    --budget 5000
```

### MOSHO Ablation Study

Run ablation variants that disable individual MOSHO units (U01–U11):

```bash
python experiments/run_method_analysis.py \
    --method abl_u01 \
    --csv nas_benchmarks/datasets/nas_hw_search_space_bench.csv \
    --runs 30
```

Available ablation identifiers: `abl_u01` through `abl_u11`, `g_search`, `g_adapt`, `g_archive`, `g_noadv`, `g_nobase`, `g_core`.

### Hyperparameter Tuning

Tune MOSHO energy parameters with successive-halving random search:

```bash
python experiments/tune_mosho_hyperparams.py \
    --budget 5000 \
    --device edgegpu
```

### Evaluation Metrics

All experiments are evaluated using the following multi-objective performance indicators:

| Metric | Description |
|--------|-------------|
| **NHV** | Normalized Hypervolume — fraction of the reference front's hypervolume covered |
| **IGD⁺** | Inverted Generational Distance Plus — convergence and spread to the reference front |
| **C-metric** | Coverage metric — fraction of the reference front dominated by the method |

---

## Reproducibility

All results reported in the paper are available under `experiments/results/` for full reproducibility:

```
experiments/results/
├── All_Methods/                  # Per-method run results (17 methods)
│   ├── mosho/                    #   └── metrics CSVs, Pareto front CSVs, plots
│   ├── mosho_enhanced/
│   ├── nsga2/
│   ├── nsga2-kumar/
│   ├── pso/
│   ├── mowso/
│   ├── random/
│   ├── ...
│   └── skyline/
├── Ablation_Results/             # MOSHO ablation study (unit-level & group-level)
│   ├── abl_u01/ ... abl_u11/    # Individual unit ablations
│   ├── g_search/, g_adapt/, ... # Group ablations
│   └── ablation_suite_summary.csv
├── Performance_Comparison/       # Comparative figures
│   ├── fig_comparative_nhv_igd.png
│   ├── fig_comparative_time_spacing.png
│   ├── fig_green_metrics.png
│   └── fig_performance_assessment.png
├── _sensitivity results/         # Budget & population sensitivity analysis
│   ├── budget_variation/
│   └── population_variation/
└── optimal_pareto_fronts.csv     # Reference optimal Pareto fronts (brute-force)
```

Each method directory contains:
- `<method>_metrics_by_run.csv` — per-run metrics (HV, IGD⁺, C-metric, runtime) across all seeds
- `<method>_context_metrics.csv` — aggregated statistics per dataset × device context
- `<method>_pareto_front.csv` — discovered Pareto front architectures
- `pareto_scatter_context_*.png` — Pareto front visualizations per context
- `hv_boxplot.png` — hypervolume distribution across contexts
- `runtime_boxplot.png` — runtime distribution across contexts

---

## Search Space

The NAS-Bench-201 search space encodes architectures as a 6-gene integer vector, where each gene selects one of 5 operations:

| Index | Operation |
|-------|-----------|
| 0 | `none` (zero) |
| 1 | `skip_connect` (identity) |
| 2 | `nor_conv_1x1` |
| 3 | `nor_conv_3x3` |
| 4 | `avg_pool_3x3` |

Total search space size: 5⁶ = **15,625 architectures**, each with pre-evaluated accuracy and latency across all contexts.

---

## Hardware Targets

| Device | Description |
|--------|-------------|
| EdgeGPU | Edge GPU accelerator |
| EdgeTPU | Google Edge TPU |
| Eyeriss | MIT Eyeriss accelerator |
| FPGA | Field-Programmable Gate Array |
| Pixel3 | Google Pixel 3 mobile CPU |
| RasPi4 | Raspberry Pi 4 |

---

## Requirements

```
numpy>=1.24
pandas>=2.0
torch>=2.0
matplotlib>=3.7
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{orama2025mosho,
  title   = {MOSHO: Multi-Objective Shark Hunting Optimization for 
             Hardware-Aware Neural Architecture Search},
  author  = {<Authors>},
  journal = {<Journal/Conference>},
  year    = {2025}
}
```
