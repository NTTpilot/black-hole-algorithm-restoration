"""Pre-compute the TRUE optimum (brute force) for every meaningful fault.

The app reads results/exact_optima.csv and shows "true optimum" next to what the
algorithm found. Run once, from the project root:

    python -m experiments.precompute_optima --workers 4

Takes about 4 minutes per fault on one core, ~30 faults: use --workers.
Results are written after EACH fault, and faults already in the CSV are skipped,
so you can stop (Ctrl+C) and restart without losing work.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from core.fitness import Weights, evaluate
from core.network import GridModel


def meaningful_faults() -> list[int]:
    """Same list as the app dropdown: every line except the trunk."""
    grid = GridModel()
    out = []
    for line in range(grid.n_line):
        grid.set_fault(line)
        now = grid.topology(grid.initial_open)
        best_case = grid.topology(())  # every available line closed = pure connectivity
        lost = grid.load_p_mw[~now.energized].sum()
        unavoidable = grid.load_p_mw[~best_case.energized].sum()
        if lost - unavoidable > 1e-9 or lost < 1e-9:   # restorable outage, or a tie switch
            out.append(line)
    return out


def compute_row(args: tuple[int, int | None]) -> dict:
    fault, limit = args
    weights = Weights()
    grid = GridModel()
    grid.set_fault(fault)
    started = time.perf_counter()
    best_open, best_eval, n_radial, n_feasible = None, None, 0, 0
    for candidate in grid.all_configurations():
        if not grid.topology(candidate).is_radial:
            continue
        ev = evaluate(grid, candidate, weights)
        n_radial += 1
        n_feasible += int(ev.feasible)
        if best_eval is None or ev.fitness < best_eval.fitness:
            best_open, best_eval = tuple(candidate), ev
        if limit is not None and n_radial >= limit:
            break
    return {
        "fault_line_0_based": fault,
        "true_best_fitness": best_eval.fitness,
        "true_best_open_lines": str(best_open),
        "radial_configurations_evaluated": n_radial,
        "exact_runtime_seconds": time.perf_counter() - started,
        "feasible_configurations": n_feasible,
        "best_is_feasible": bool(best_eval.feasible),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", type=int, nargs="+", default=None,
                    help="0-based fault lines (default: all meaningful faults)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", type=str, default="results/exact_optima.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="TESTING ONLY: stop after this many radial configs (result is NOT the optimum)")
    ap.add_argument("--force", action="store_true", help="recompute faults already in the CSV")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set() if args.force or existing.empty else set(existing.fault_line_0_based)
    todo = [f for f in (args.faults if args.faults else meaningful_faults()) if f not in done]
    print(f"{len(todo)} fault(s) to compute: {todo}")

    def save(row: dict) -> None:
        nonlocal existing
        if not existing.empty:
            existing = existing[existing.fault_line_0_based != row["fault_line_0_based"]]
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.sort_values("fault_line_0_based").to_csv(out, index=False)
        print(f"fault {row['fault_line_0_based']:2d}: optimum {row['true_best_fitness']:.4f} "
              f"open={row['true_best_open_lines']} feasible={row['best_is_feasible']} "
              f"({row['exact_runtime_seconds']:.0f}s)", flush=True)

    jobs = [(f, args.limit) for f in todo]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for row in pool.imap_unordered(compute_row, jobs):
                save(row)
    else:
        for job in jobs:
            save(compute_row(job))
    print("done ->", out)


if __name__ == "__main__":
    main()