"""Discrete, diversity-preserving Black Hole Algorithm for switch restoration.

Unlike the original random-key implementation, stars are represented directly
as tuples of open line indices. Every candidate evaluated by the optimizer is
radial, and attraction/mutation operations are performed on switch sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .fitness import Evaluation, Weights, evaluate, make_fitness
from .network import GridModel


@dataclass
class BHAResult:
    best_open: tuple[int, ...]
    best_fitness: float
    best_eval: Evaluation
    history_best: list[float] = field(default_factory=list)
    history_mean: list[float] = field(default_factory=list)
    history_swallowed: list[int] = field(default_factory=list)
    history_evals: list[int] = field(default_factory=list)
    history_unique_configurations: list[int] = field(default_factory=list)
    n_unique_evals: int = 0
    n_swallowed: int = 0
    positions: list[np.ndarray] = field(default_factory=list)
    bh_index: list[int] = field(default_factory=list)
    initial_population: list[tuple[int, ...]] = field(default_factory=list)
    eval_scores: list[float] = field(default_factory=list)  # score of every distinct configuration, in evaluation order


def _is_radial(grid: GridModel, candidate: tuple[int, ...]) -> bool:
    return grid.topology(candidate).is_radial


def black_hole_algorithm(
    grid: GridModel,
    weights: Weights = Weights(),
    n_stars: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    horizon_scale: float = 1.0,  # kept for backwards compatibility; see note below
    record_positions: bool = False,
    callback: Optional[Callable[[int, float], None]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    max_evals: Optional[int] = None,
    mutation_rate: float = 0.30,
    restart_fraction: float = 0.20,
) -> BHAResult:
    """Run a discrete BHA/hybrid search.

    `max_evals` limits distinct fitness evaluations, allowing fair budget-matched
    comparisons. If omitted, the legacy `n_stars`/`max_iter` style is used.
    `horizon_scale` remains accepted so older callers do not break, but the old
    continuous-distance event horizon is replaced by duplicate/stagnation
    restarts because its units were inappropriate for discrete switch sets.
    """
    del horizon_scale
    if n_stars < 2:
        raise ValueError("n_stars must be at least 2")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not 0.0 <= restart_fraction <= 1.0:
        raise ValueError("restart_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    k = grid.n_open_required
    available = tuple(grid.available)
    fit_fn = make_fitness(grid, weights)
    cache = fit_fn.cache  # type: ignore[attr-defined]

    def score(candidate: tuple[int, ...]) -> float:
        key = tuple(sorted(candidate))
        if key not in cache:
            if max_evals is not None and len(cache) >= max_evals:
                return float("inf")
            cache[key] = evaluate(grid, key, weights).fitness
        return float(cache[key])

    def random_radial(exclude: set[tuple[int, ...]] | None = None) -> tuple[int, ...]:
        """Sample a unique radial configuration without evaluating power flow."""
        exclude = exclude or set()
        for _ in range(5000):
            candidate = tuple(sorted(rng.choice(available, size=k, replace=False).tolist()))
            if candidate not in exclude and _is_radial(grid, candidate):
                return candidate
        # Rare fallback: enumerate the small combination space to find a valid one.
        for raw in grid.all_configurations():
            candidate = tuple(sorted(raw))
            if candidate not in exclude and _is_radial(grid, candidate):
                return candidate
        raise RuntimeError("Could not find another radial configuration.")

    def attract(star: tuple[int, ...], black_hole: tuple[int, ...]) -> tuple[int, ...]:
        """Move a star part-way toward the black hole by swapping open lines."""
        star_only = sorted(set(star) - set(black_hole))
        bh_only = sorted(set(black_hole) - set(star))
        if not star_only:
            return star
        n_changes = int(rng.integers(1, len(star_only) + 1))
        remove = set(rng.choice(star_only, size=n_changes, replace=False).tolist())
        add_pool = [line for line in bh_only if line not in set()]
        add = set(rng.choice(add_pool, size=min(n_changes, len(add_pool)), replace=False).tolist())
        candidate = (set(star) - remove) | add
        # Keep the fixed number of open switches even if sets were unusual.
        while len(candidate) < k:
            remaining = [line for line in available if line not in candidate]
            candidate.add(int(rng.choice(remaining)))
        while len(candidate) > k:
            removable = [line for line in candidate if line not in black_hole]
            candidate.remove(int(rng.choice(removable or list(candidate))))
        return tuple(sorted(candidate))

    def mutate(star: tuple[int, ...], swaps: int = 1) -> tuple[int, ...]:
        candidate = set(star)
        for _ in range(swaps):
            opened = list(candidate)
            closed = [line for line in available if line not in candidate]
            if not opened or not closed:
                break
            candidate.remove(int(rng.choice(opened)))
            candidate.add(int(rng.choice(closed)))
        return tuple(sorted(candidate))

    # Initial population: unique, radial configurations only.
    population: list[tuple[int, ...]] = []
    target_size = min(n_stars, max_evals) if max_evals is not None else n_stars
    while len(population) < target_size:
        candidate = random_radial(set(population))
        population.append(candidate)
        score(candidate)
        if max_evals is not None and len(cache) >= max_evals:
            break
    if not population:
        raise RuntimeError("The evaluation budget is too small to initialize BHA.")

    fitness = np.array([score(c) for c in population], dtype=float)
    best_idx = int(np.argmin(fitness))
    global_best = population[best_idx]
    global_best_fitness = float(fitness[best_idx])
    result = BHAResult(best_open=global_best, best_fitness=global_best_fitness,
                       best_eval=evaluate(grid, global_best, weights),
                       initial_population=population.copy())
    stale = 0

    for iteration in range(max_iter):
        if max_evals is not None and len(cache) >= max_evals:
            break
        bh_idx = int(np.argmin(fitness))
        black_hole = population[bh_idx]
        next_population = population.copy()
        swallowed = 0

        for i, star in enumerate(population):
            if i == bh_idx or (max_evals is not None and len(cache) >= max_evals):
                continue

            # Discrete attraction; if it produces a loop, use a local mutation.
            proposal = attract(star, black_hole)
            if not _is_radial(grid, proposal):
                proposal = mutate(star, swaps=1 if rng.random() < 0.75 else 2)
            if not _is_radial(grid, proposal):
                proposal = random_radial(set(population))

            # Mutation prevents deterministic collapse into identical switch sets.
            if rng.random() < mutation_rate:
                mutated = mutate(proposal, swaps=1 if rng.random() < 0.8 else 2)
                if _is_radial(grid, mutated):
                    proposal = mutated

            proposal_fit = score(proposal)
            if proposal_fit < fitness[i] or proposal == star:
                next_population[i] = proposal

        # Diversity recovery: replace duplicates/worst stars after stagnation.
        next_fitness = np.array([score(c) for c in next_population], dtype=float)
        new_best_idx = int(np.argmin(next_fitness))
        if next_fitness[new_best_idx] < global_best_fitness:
            global_best = next_population[new_best_idx]
            global_best_fitness = float(next_fitness[new_best_idx])
            stale = 0
        else:
            stale += 1

        if stale >= 5 and max_evals is not None and len(cache) < max_evals:
            n_restart = max(1, int(len(next_population) * restart_fraction))
            worst = np.argsort(next_fitness)[-n_restart:]
            seen = set(next_population)
            for idx in worst:
                if len(cache) >= max_evals:
                    break
                candidate = random_radial(seen)
                seen.add(candidate)
                next_population[int(idx)] = candidate
                next_fitness[int(idx)] = score(candidate)
                swallowed += 1
            stale = 0
        elif stale >= 5 and max_evals is None:
            n_restart = max(1, int(len(next_population) * restart_fraction))
            worst = np.argsort(next_fitness)[-n_restart:]
            seen = set(next_population)
            for idx in worst:
                candidate = random_radial(seen)
                seen.add(candidate)
                next_population[int(idx)] = candidate
                next_fitness[int(idx)] = score(candidate)
                swallowed += 1
            stale = 0

        population = next_population
        fitness = np.array([score(c) for c in population], dtype=float)
        current_idx = int(np.argmin(fitness))
        if fitness[current_idx] < global_best_fitness:
            global_best = population[current_idx]
            global_best_fitness = float(fitness[current_idx])

        result.history_best.append(global_best_fitness)
        result.history_mean.append(float(np.mean(fitness[np.isfinite(fitness)])))
        result.history_evals.append(len(cache))
        result.history_unique_configurations.append(len(cache))
        result.history_swallowed.append(swallowed)
        result.n_swallowed += swallowed
        result.bh_index.append(current_idx)
        if record_positions:
            result.positions.append(np.asarray(population, dtype=float))
        if callback is not None:
            callback(iteration, global_best_fitness)
        if progress_callback is not None:  # (iteration, evaluations used, best fitness)
            progress_callback(iteration, len(cache), global_best_fitness)

    result.best_open = global_best
    result.best_fitness = global_best_fitness
    result.best_eval = evaluate(grid, global_best, weights)
    result.n_unique_evals = len(cache)
    result.eval_scores = list(cache.values())
    return result