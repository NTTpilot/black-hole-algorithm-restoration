"""Reproducible BHA comparison, convergence capture, and exact references.

Run from the project root with:
    python -m experiments.run_benchmark --fault-lines 7 32 34 36
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from core.bha import black_hole_algorithm
from core.bha_paper import black_hole_algorithm_paper
from core.baseline import random_search
from core.fitness import Evaluation, Weights, evaluate
from core.network import GridModel


def exact_optimum(grid: GridModel, weights: Weights) -> tuple[tuple[int, ...], Evaluation, int]:
    """Enumerate every valid radial configuration and return the true minimum."""
    best_open: tuple[int, ...] | None = None
    best_eval: Evaluation | None = None
    evaluated = 0
    for candidate in grid.all_configurations():
        if not grid.topology(candidate).is_radial:
            continue
        current = evaluate(grid, candidate, weights)
        evaluated += 1
        if best_eval is None or current.fitness < best_eval.fitness:
            best_open, best_eval = tuple(candidate), current
    if best_open is None or best_eval is None:
        raise RuntimeError("No valid radial configuration exists for this fault.")
    return best_open, best_eval, evaluated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-lines", type=int, nargs="+", default=[7],
                        help="0-based fault indices to test")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=750)
    parser.add_argument("--stars", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-exact", action="store_true",
                        help="skip brute-force references for large search spaces")
    args = parser.parse_args()

    weights = Weights()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []

    for fault_line in args.fault_lines:
        reference_fitness: float | None = None
        reference_open: tuple[int, ...] | None = None
        if not args.skip_exact:
            exact_grid = GridModel()
            exact_grid.set_fault(fault_line)
            started = time.perf_counter()
            reference_open, reference_eval, n_radial = exact_optimum(exact_grid, weights)
            reference_fitness = reference_eval.fitness
            reference_rows.append({
                "fault_line_0_based": fault_line,
                "true_best_fitness": reference_fitness,
                "true_best_open_lines": str(reference_open),
                "radial_configurations_evaluated": n_radial,
                "exact_runtime_seconds": time.perf_counter() - started,
            })
            print(f"fault={fault_line} exact={reference_fitness:.6f} open={reference_open}")

        for seed in range(args.seeds):
            for method in ("BHA_paper", "BHA_discrete", "random_search"):
                grid = GridModel()
                grid.set_fault(fault_line)
                started = time.perf_counter()
                if method == "BHA_paper":
                    result = black_hole_algorithm_paper(
                        grid, weights, n_stars=args.stars, max_iter=10_000,
                        seed=seed, max_evals=args.budget,
                    )
                elif method == "BHA_discrete":
                    result = black_hole_algorithm(
                        grid, weights, n_stars=args.stars, max_iter=10_000,
                        seed=seed, max_evals=args.budget,
                    )
                else:
                    result = random_search(
                        grid, weights, max_evals=args.budget, seed=seed
                    )
                runtime = time.perf_counter() - started
                history_best = result.history_best
                history_evals = result.history_evals
                history_unique = result.history_unique_configurations
                initial_unique = (
                    len(set(result.initial_population))
                    if method.startswith("BHA") else None
                )
                gap = (
                    result.best_fitness - reference_fitness
                    if reference_fitness is not None else None
                )
                run_rows.append({
                    "method": method,
                    "seed": seed,
                    "fault_line_0_based": fault_line,
                    "best_fitness": result.best_fitness,
                    "optimality_gap": gap,
                    "best_open_lines": str(result.best_open),
                    "unique_evaluations": result.n_unique_evals,
                    "initial_unique_configurations": initial_unique,
                    "runtime_seconds": runtime,
                    "history_points": len(history_best),
                    "radial": result.best_eval.radial,
                    "converged": result.best_eval.converged,
                    "feasible": result.best_eval.feasible,
                    "unserved_mw": result.best_eval.unserved_mw,
                    "switch_ops": result.best_eval.n_switch_ops,
                    "losses_kw": result.best_eval.losses_kw,
                    "v_min": result.best_eval.v_min,
                })
                for point, (best, evaluations, unique) in enumerate(
                    zip(history_best, history_evals, history_unique), start=1
                ):
                    convergence_rows.append({
                        "method": method,
                        "seed": seed,
                        "fault_line_0_based": fault_line,
                        "iteration_or_sample": point,
                        "evaluations": evaluations,
                        "unique_configurations": unique,
                        "best_fitness": best,
                    })
                print(f"{method:14s} fault={fault_line:02d} seed={seed:02d} "
                      f"fitness={result.best_fitness:.6f} "
                      f"evals={result.n_unique_evals} time={runtime:.3f}s")

    runs = pd.DataFrame(run_rows)
    convergence = pd.DataFrame(convergence_rows)
    references = pd.DataFrame(reference_rows)
    runs.to_csv(output_dir / "benchmark.csv", index=False)
    convergence.to_csv(output_dir / "convergence.csv", index=False)
    if reference_rows:
        references.to_csv(output_dir / "exact_optima.csv", index=False)
    if run_rows:
        summary = runs.groupby(["fault_line_0_based", "method"]).agg(
            runs=("best_fitness", "count"),
            mean_fitness=("best_fitness", "mean"),
            std_fitness=("best_fitness", "std"),
            median_fitness=("best_fitness", "median"),
            best_fitness=("best_fitness", "min"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_unique_configurations=("unique_evaluations", "mean"),
            feasible_rate=("feasible", "mean"),
        ).reset_index()
        if reference_rows:
            summary = summary.merge(
                references[["fault_line_0_based", "true_best_fitness"]],
                on="fault_line_0_based",
                how="left",
            )
        summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
        print("\nSummary (lower fitness and gap are better):")
        print(summary.to_string(index=False))
    else:
        print("\nNo seeded comparison runs requested.")


if __name__ == "__main__":
    main()