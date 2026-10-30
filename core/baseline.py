"""Reproducible random-search baseline using the same radial candidates/fitness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .fitness import Evaluation, Weights, evaluate, make_fitness
from .network import GridModel


@dataclass
class BaselineResult:
    best_open: tuple[int, ...]
    best_fitness: float
    best_eval: Evaluation
    n_unique_evals: int
    history_best: list[float]
    history_evals: list[int]
    history_unique_configurations: list[int]
    eval_scores: list[float] = field(default_factory=list)  # score of every distinct configuration, in evaluation order


def random_search(
    grid: GridModel,
    weights: Weights = Weights(),
    max_evals: int = 750,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> BaselineResult:
    """Sample unique radial configurations until the distinct-evaluation budget."""
    if max_evals < 1:
        raise ValueError("max_evals must be positive")
    rng = np.random.default_rng(seed)
    available = tuple(grid.available)
    k = grid.n_open_required
    fit_fn = make_fitness(grid, weights)
    cache = fit_fn.cache  # type: ignore[attr-defined]
    best_open: tuple[int, ...] | None = None
    best_fitness = float("inf")
    history: list[float] = []
    history_evals: list[int] = []
    history_unique_configurations: list[int] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0

    while len(cache) < max_evals and attempts < max_evals * 1000:
        attempts += 1
        candidate = tuple(sorted(rng.choice(available, size=k, replace=False).tolist()))
        if candidate in seen or not grid.topology(candidate).is_radial:
            continue
        seen.add(candidate)
        value = fit_fn(candidate)
        if value < best_fitness:
            best_open, best_fitness = candidate, float(value)
        history.append(best_fitness)
        history_evals.append(len(cache))
        history_unique_configurations.append(len(seen))
        if progress_callback is not None and (len(seen) % 5 == 0 or len(cache) >= max_evals):
            progress_callback(len(seen), len(cache), best_fitness)  # (samples, evaluations, best)

    if best_open is None:
        raise RuntimeError("Random search did not sample a radial configuration.")
    return BaselineResult(
        best_open=best_open,
        best_fitness=best_fitness,
        best_eval=evaluate(grid, best_open, weights),
        n_unique_evals=len(cache),
        history_best=history,
        history_evals=history_evals,
        history_unique_configurations=history_unique_configurations,
        eval_scores=list(cache.values()),
    )