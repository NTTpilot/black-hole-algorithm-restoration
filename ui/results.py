"""ui/results.py -- reads pre-computed brute-force optima and benchmark tables (if present)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Optimum:
    fitness: float
    open_lines: str
    feasible: Optional[bool]


def load_optima() -> dict[int, Optimum]:
    """True (brute-force) optimum per fault. exact_optima.csv wins; the
    true_best_fitness column of benchmark_summary.csv is the fallback."""
    out: dict[int, Optimum] = {}
    summary = RESULTS / "benchmark_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        if "true_best_fitness" in df:
            for fault, grp in df.dropna(subset=["true_best_fitness"]).groupby("fault_line_0_based"):
                out[int(fault)] = Optimum(float(grp.true_best_fitness.iloc[0]), "", None)
    exact = RESULTS / "exact_optima.csv"
    if exact.exists():
        df = pd.read_csv(exact)
        for _, r in df.iterrows():
            feas = r.get("best_is_feasible")
            out[int(r.fault_line_0_based)] = Optimum(
                float(r.true_best_fitness), str(r.get("true_best_open_lines", "")),
                None if pd.isna(feas) else bool(feas),
            )
    return out


def load_benchmark_summary() -> Optional[pd.DataFrame]:
    path = RESULTS / "benchmark_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "true_best_fitness" in df:
        df["mean_gap"] = df["mean_fitness"] - df["true_best_fitness"]
    return df
