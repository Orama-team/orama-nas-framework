PYTHON ?= python

.PHONY: reproduce run-all-methods sensitivity-analysis plot-performance-comparison

reproduce: run-all-methods sensitivity-analysis plot-performance-comparison

run-all-methods:
	$(PYTHON) experiments/run_all_methods.py

sensitivity-analysis:
	$(PYTHON) experiments/sensitivity_analysis.py

plot-performance-comparison:
	$(PYTHON) experiments/plot_performance_comparison.py
