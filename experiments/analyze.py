"""Compare methods from a benchmark run.

Usage (from the project root):
    python -m experiments.analyze results_200
    python -m experiments.analyze results_750 results_200     # several folders
Prints a comparison table and saves convergence_<folder>.png in the folder.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    has_gap = "optimality_gap" in runs and runs["optimality_gap"].notna().any()
    g = runs.groupby(["fault_line_0_based", "method"])
    out = g.agg(
        runs=("best_fitness", "count"),
        mean_fitness=("best_fitness", "mean"),
        std=("best_fitness", "std"),
        worst=("best_fitness", "max"),
        evals=("unique_evaluations", "mean"),
        seconds=("runtime_seconds", "mean"),
    )
    if has_gap:
        out["hit_optimum_%"] = g["optimality_gap"].apply(lambda s: 100 * (s.abs() < 1e-6).mean())
        out["mean_gap"] = g["optimality_gap"].mean()
    return out.round(3).reset_index()


def convergence_plot(conv: pd.DataFrame, path: Path, title: str) -> None:
    faults = sorted(conv.fault_line_0_based.unique())
    fig, axes = plt.subplots(1, len(faults), figsize=(5 * len(faults), 4), squeeze=False)
    max_e = int(conv.evaluations.max())
    # start at 30 evals: the first points are just the random initial population,
    # and BHA only records a point after each full iteration
    grid = np.arange(min(30, max_e), max_e + 1)
    for ax, fault in zip(axes[0], faults):
        for method, d in conv[conv.fault_line_0_based == fault].groupby("method"):
            curves = []
            for _, run in d.groupby("seed"):
                run = run.sort_values("evaluations")
                idx = np.searchsorted(run.evaluations.to_numpy(), grid, side="right") - 1
                best = run.best_fitness.to_numpy()
                c = np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan)
                curves.append(c)
            with np.errstate(all="ignore"):
                ax.plot(grid, np.nanmean(curves, axis=0), label=method)
        ax.set_title(f"fault line {fault} (0-based)")
        ax.set_xlabel("distinct configurations evaluated")
        ax.set_ylabel("best fitness so far (mean over seeds)")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    for folder in map(Path, sys.argv[1:] or ["results"]):
        runs = pd.read_csv(folder / "benchmark.csv")
        print(f"\n=== {folder} ===")
        print(summary(runs).to_string(index=False))
        out = folder / f"convergence_{folder.name}.png"
        convergence_plot(pd.read_csv(folder / "convergence.csv"), out, folder.name)
        print("saved", out)


if __name__ == "__main__":
    main()
