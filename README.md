# Hybrid BHA for Distribution Network Restoration

Research prototype for service restoration in the IEEE 33-bus distribution
network using a discrete Black Hole Algorithm (BHA).

This project started as a school project: implement and understand the Black Hole Algorithm for distribution-system restoration. It
then became a larger challenge: design a hybrid algorithm that respects the
discrete structure of switch configurations and converges faster than the
paper-faithful implementation.

The result is a discrete, diversity-preserving BHA that searches directly over
open-switch sets, preserves radial network configurations, and combines
black-hole attraction with mutation and diversity recovery.

> **Research status:** This is an experimental research prototype, not a
> production grid-control system. The benchmark results are reproducible
> observations for this implementation and test setup, not a universal proof
> that the hybrid method is faster for every network or parameter choice.

## What this project does

Given a faulted distribution network, the optimizer searches for a set of open
lines that:

- restores service while keeping the network radial;
- minimizes unserved load;
- reduces switching operations;
- limits active power losses; and
- avoids voltage-limit violations.

The fitness function is minimized. Invalid or non-convergent configurations are
penalized, and every candidate evaluated by the optimizers is tracked so that
comparisons can use the same unique-evaluation budget.

## Algorithms

### Paper-faithful discrete BHA

[`core/bha_paper.py`](core/bha_paper.py) adapts the original BHA equations to
the discrete switch-set search space. Stars move toward the current black hole,
the event horizon is represented using Hamming distance, and swallowed stars are
replaced by new radial configurations.

### Hybrid/discrete BHA

[`core/bha.py`](core/bha.py) is the main contribution of this project. It:

- represents each star directly as a tuple of open line indices;
- generates radial candidates instead of repairing arbitrary continuous
  positions after the fact;
- moves stars toward the best switch set through discrete exchanges;
- applies valid local mutations to maintain exploration; and
- replaces duplicate or stagnant worst candidates to recover diversity.

This makes the search space and the algorithm's operators match the actual
decision variables: discrete switch configurations.

### Baseline

[`core/baseline.py`](core/baseline.py) provides random search with the same
maximum number of distinct fitness evaluations.

## Results

The checked-in CSV files under [`results/`](results/) contain seeded benchmark
runs comparing `BHA_discrete`, `BHA_paper`, and `random_search`. The recorded
benchmark uses 20 runs per fault location and a budget of 750 unique
evaluations.

The summary in
[`results/benchmark_summary.csv`](results/benchmark_summary.csv) shows that:

- for the fault locations with matching exhaustive references, the hybrid BHA
  reached the exact reference optimum in every recorded run;
- the paper-faithful BHA also performed strongly, but had a non-zero average
  error for fault line 11;
- random search had substantially higher mean fitness on most fault locations;
- all methods returned feasible solutions in the recorded benchmark; and
- runtime was broadly similar, so the main improvement is convergence quality
  and evaluation efficiency rather than raw wall-clock runtime.

The convergence histories in
[`results/convergence.csv`](results/convergence.csv) show the hybrid BHA
reaching its final best score after an average of approximately **155 unique
evaluations**, compared with approximately **224** for the paper-faithful BHA
and **387** for random search, when averaged across the recorded fault
locations. These are averages derived from the checked-in histories and should
not be interpreted as guarantees for other networks or budgets.

The exact reference values and open-line configurations are recorded in
[`results/exact_optima.csv`](results/exact_optima.csv). The raw per-run data,
including fitness, runtime, feasibility, losses, minimum voltage, and selected
switches, is available in
[`results/benchmark.csv`](results/benchmark.csv).

### Recreate the benchmark

Run from the project root:

```powershell
python -m experiments.run_benchmark --fault-lines 3 7 8 9 11 32 36 --seeds 20 --budget 750
```

This writes:

| File | Contents |
| --- | --- |
| `results/benchmark.csv` | One row per method, seed, and fault location |
| `results/benchmark_summary.csv` | Mean, standard deviation, median, best score, runtime, and feasibility |
| `results/convergence.csv` | Best-so-far convergence histories |
| `results/exact_optima.csv` | Exhaustive reference optima where enabled |

For a faster run without exhaustive reference calculations, add
`--skip-exact`.

## Installation

This project currently targets Python 3.12 on Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current
terminal, or call `.venv\Scripts\python.exe` directly.

## Tests

```powershell
python -m pytest -q
```

The tests verify radial solutions, evaluation-budget limits, reproducibility,
and monotonic best-so-far histories.

## Project structure

```text
core/
  bha.py             Hybrid discrete BHA
  bha_paper.py       Paper-faithful discrete BHA
  baseline.py        Random-search baseline
  fitness.py         Power-flow evaluation and objective function
  network.py         IEEE 33-bus grid model
experiments/
  run_benchmark.py   Reproducible benchmark and CSV export
results/
  *.csv              Checked-in benchmark and convergence data
tests/
  test_bha.py
  test_network.py
app.py               Streamlit visualisation
```

## Important limitations

- Results depend on the IEEE 33-bus model, objective weights, fault locations,
  random seeds, and evaluation budget.
- The current comparison is based on unique fitness evaluations, not a claim
  that every implementation detail has identical computational cost.
- The repository preserves both the paper-faithful and hybrid implementations
  for an apples-to-apples ablation study.
- More networks, fault scenarios, statistical tests, and parameter studies are
  needed before making broad claims about generalisation.

## License and attribution

The algorithmic design is based on the Black Hole Algorithm literature. Add the
project's chosen license and the full paper citation here before publishing
the repository publicly.
