# Revised BHA audit

## Scope

The revised discrete BHA was compared with the existing random-search baseline
for fault indices 7, 32, and 36. Each method used seeds 0 through 19 and a
maximum of 750 distinct fitness evaluations. Lower fitness is better.

## Implementation audit

- **Random seeds:** `numpy.random.default_rng(seed)` is created inside each
  optimizer run. Repeating seed 0 produced the same result and initial
  population; seed 1 produced a different initial population.
- **Population initialization:** BHA initializes 30 distinct radial
  configurations. The direct audit measured 30/30 unique candidates for seeds 0
  and 1. Candidates are checked for radiality before evaluation.
- **Optimum tracking:** BHA maintains a global best independently of the
  current population. The run-level result is re-evaluated from that global
  best. The recorded convergence history improved from 5.213073 at the first
  iteration to 2.034933 at the final iteration for fault 7.
- **Measurement changes:** BHA and random search now expose evaluation/sample
  checkpoints and unique-configuration counts. Benchmark rows also include
  runtime, feasibility, and optimality gap.

## Exact comparison

The exact references enumerate every radial configuration for each tested fault:

| Fault | True optimum | Optimal open lines | Radial configurations | Exact runtime (s) |
|---:|---:|---|---:|---:|
| 7 | 2.034933 | (32, 33, 35, 36) | 10,914 | 548.72 |
| 32 | 2.026771 | (33, 34, 35, 36) | 12,729 | 427.78 |
| 36 | 2.026771 | (32, 33, 34, 35) | 5,889 | 242.64 |

## 20-seed summary

| Fault | Method | Mean fitness | Fitness std. dev. | Mean runtime (s) | Mean unique evaluations | Mean optimality gap |
|---:|---|---:|---:|---:|---:|---:|
| 7 | BHA | 2.034933 | 0.000000 | 37.35 | 750 | 0.000000 |
| 7 | Random search | 3.183486 | 0.489276 | 35.12 | 750 | 1.148553 |
| 32 | BHA | 2.026771 | 0.000000 | 36.94 | 750 | 0.000000 |
| 32 | Random search | 2.830509 | 0.399040 | 34.34 | 750 | 0.803738 |
| 36 | BHA | 2.026771 | 0.000000 | 38.62 | 750 | 0.000000 |
| 36 | Random search | 2.791384 | 0.414757 | 36.29 | 750 | 0.764613 |

All 120 comparison runs returned feasible configurations. The revised BHA used
about 2-6% more runtime than random search in these measurements, but reached
the exact enumerated optimum for all 60 BHA runs; random search reached it only
occasionally. Complete per-run and convergence data are in the CSV files next
to this report.
