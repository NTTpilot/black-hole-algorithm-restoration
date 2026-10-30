"""ui/runner.py -- runs one algorithm on one fault and returns a uniform result."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.baseline import random_search
from core.bha import black_hole_algorithm
from core.bha_paper import black_hole_algorithm_paper
from core.fitness import Evaluation, Weights, evaluate
from core.network import GridModel

METHOD_LABELS = {
    "paper": "Black Hole Algorithm (Hatamlou)",
    "hybrid": "Hybrid BHA (greedy + mutation)",
    "random": "Random search",
}
METHOD_SHORT = {"paper": "BHA (paper)", "hybrid": "Hybrid BHA", "random": "Random search"}
FEASIBLE_TOL = 1e-3     # p.u. of total voltage violation tolerated when saying "within limits"

ProgressFn = Callable[[int, float], None]       # (evaluations used, best fitness so far)


@dataclass
class PostFault:
    """State of the grid right after the fault, before any switching."""
    initial_open: tuple[int, ...]
    energized: np.ndarray
    vm_pu: np.ndarray            # NaN where there is no power
    lost_mw: float
    lost_frac: float


@dataclass
class RunResult:
    method: str
    fault: int
    seed: int
    budget: int
    best_open: tuple[int, ...]
    fitness: float
    ev: Evaluation
    curve_evals: list[int] = field(default_factory=list)
    curve_best: list[float] = field(default_factory=list)
    runtime_s: float = 0.0
    n_evals: int = 0
    energized: Optional[np.ndarray] = None

    @property
    def label(self) -> str:
        return METHOD_LABELS[self.method]

    @property
    def short(self) -> str:
        return METHOD_SHORT[self.method]

    @property
    def evals_to_best(self) -> int:
        """Evaluations used when the final best score was first reached."""
        for e, b in zip(self.curve_evals, self.curve_best):
            if b <= self.fitness + 1e-12:
                return e
        return self.n_evals

    @property
    def voltage_ok(self) -> bool:
        return self.ev.converged and self.ev.v_violation < FEASIBLE_TOL

    @property
    def fully_restored(self) -> bool:
        return self.ev.unserved_mw < 1e-9


def new_seed() -> int:
    return int(np.random.default_rng().integers(0, 1_000_000))


def warm_up() -> None:
    """One throw-away power flow so the first live run is not slower than the rest."""
    g = GridModel()
    g.set_fault(7)
    evaluate(g, g.initial_open, strict=False)


def post_fault_state(fault: int) -> PostFault:
    g = GridModel()
    g.set_fault(fault)
    topo = g.topology(g.initial_open)
    ev = evaluate(g, g.initial_open, Weights(), strict=False)
    vm = ev.vm_pu if ev.vm_pu is not None else np.full(g.n_bus, np.nan)
    lost = float(g.load_p_mw[~topo.energized].sum())
    return PostFault(g.initial_open, topo.energized, vm, lost, lost / g.total_load_mw)


def run_method(method: str, fault: int, budget: int, n_stars: int, seed: int,
               on_progress: Optional[ProgressFn] = None) -> RunResult:
    grid = GridModel()
    grid.set_fault(fault)
    w = Weights()
    started = time.perf_counter()

    if method == "paper":
        r = black_hole_algorithm_paper(
            grid, w, n_stars=n_stars, max_iter=100_000, seed=seed, max_evals=budget,
            progress_callback=(lambda it, ev, best: on_progress(ev, best)) if on_progress else None)
    elif method == "hybrid":
        r = black_hole_algorithm(
            grid, w, n_stars=n_stars, max_iter=100_000, seed=seed, max_evals=budget,
            progress_callback=(lambda it, ev, best: on_progress(ev, best)) if on_progress else None)
    elif method == "random":
        r = random_search(
            grid, w, max_evals=budget, seed=seed,
            progress_callback=(lambda n, ev, best: on_progress(ev, best)) if on_progress else None)
    else:
        raise ValueError(f"unknown method {method!r}")

    runtime = time.perf_counter() - started
    topo = grid.topology(r.best_open)
    scores = list(r.eval_scores)
    if scores:      # best score after EVERY evaluation, so all algorithms start at evaluation 1
        curve_evals = list(range(1, len(scores) + 1))
        curve_best = [float(v) for v in np.minimum.accumulate(scores)]
    else:
        curve_evals, curve_best = list(r.history_evals), list(r.history_best)
    return RunResult(
        method=method, fault=fault, seed=seed, budget=budget,
        best_open=tuple(r.best_open), fitness=float(r.best_fitness), ev=r.best_eval,
        curve_evals=curve_evals, curve_best=curve_best,
        runtime_s=runtime, n_evals=int(r.n_unique_evals), energized=topo.energized,
    )