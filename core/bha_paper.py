"""
core/bha_paper.py
-----------------
Black Hole Algorithm following Hatamlou (2013) as closely as a DISCRETE
search space allows. Every step of the paper has a direct counterpart:

  paper (continuous)                        this file (switch sets)
  ------------------------------------      ------------------------------------
  star = point in R^d                       star = radial configuration (k open lines)
  x_i += rand * (x_BH - x_i)                move a fraction `rand` of the way to the
                                            BH by exchanging open lines, one at a
                                            time, keeping the network radial
  stars ALWAYS move (no acceptance test)    same (greedy acceptance is optional)
  best star becomes the black hole          same
  R = f_BH / sum_i f_i                      same formula, see `_horizon_lines`
  swallowed if distance(star, BH) < R       distance = number of lines in which the
                                            star differs from the BH (Hamming)
  swallowed star -> new random star         new random RADIAL configuration

Two switches turn this into the hybrid that is documented in the report:
  greedy=True        a star only moves if the new position is better
  mutation_rate>0    random line swap applied after the attraction step
Both default to OFF, i.e. the paper's behaviour.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bha import BHAResult
from .fitness import Weights, evaluate, make_fitness
from .network import GridModel


def black_hole_algorithm_paper(
    grid: GridModel,
    weights: Weights = Weights(),
    n_stars: int = 30,
    max_iter: int = 100,
    seed: Optional[int] = None,
    max_evals: Optional[int] = None,
    horizon_scale: float = 0.5,
    horizon_ref: str = "all",
    greedy: bool = False,
    mutation_rate: float = 0.0,
    record_positions: bool = False,
    callback: Optional[Callable[[int, float], None]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> BHAResult:
    """
    horizon_ref    "all": R uses the mean fitness of every star (paper formula).
                   "feasible": mean only over stars scoring below
                   weights.non_converged (see diagnosis in the report).
    horizon_scale  Event-horizon radius in "paper units" is R = f_BH / sum(f).
                   Because R <= 1/N, a distance measured in lines (0..k) would
                   almost never fall below it, so R is converted to lines with
                       radius_lines = k * min(1, horizon_scale * N * R)
                                    = k * min(1, horizon_scale * f_BH / mean(f)).
                   Early on (BH much better than the average star) the radius
                   is ~0, so only exact duplicates are swallowed. Once the
                   population has converged (f_BH ~ mean f) the radius grows to
                   horizon_scale * k lines and stars sitting next to the black
                   hole are swallowed and replaced: exploration is restored
                   exactly when it is needed.
    """
    if n_stars < 2:
        raise ValueError("n_stars must be at least 2")
    rng = np.random.default_rng(seed)
    k = grid.n_open_required
    available = tuple(grid.available)
    fit_fn = make_fitness(grid, weights)
    cache = fit_fn.cache  # type: ignore[attr-defined]

    def out_of_budget() -> bool:
        return max_evals is not None and len(cache) >= max_evals

    def score(c: tuple[int, ...]) -> float:
        if c not in cache and out_of_budget():
            return float("inf")
        return float(fit_fn(c))

    def is_radial(c) -> bool:
        return grid.topology(c).is_radial

    def random_radial(exclude: set) -> tuple[int, ...]:
        for _ in range(5000):
            c = tuple(sorted(rng.choice(available, size=k, replace=False).tolist()))
            if c not in exclude and is_radial(c):
                return c
        raise RuntimeError("could not sample another radial configuration")

    def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return len(set(a) - set(b))

    def move_towards(star: tuple[int, ...], bh: tuple[int, ...], r: float) -> tuple[int, ...]:
        """x + r*(BH - x): adopt round(r * d) of the d differing lines, one
        radial-preserving exchange at a time (always possible by the basis
        exchange property of spanning trees)."""
        star_only = [l for l in star if l not in bh]
        bh_only = [l for l in bh if l not in star]
        n_moves = int(round(r * len(star_only)))
        current = set(star)
        for _ in range(n_moves):
            a = star_only.pop(int(rng.integers(len(star_only))))
            order = rng.permutation(len(bh_only))
            for j in order:
                b = bh_only[int(j)]
                cand = (current - {a}) | {b}
                if is_radial(tuple(sorted(cand))):
                    current = cand
                    bh_only.pop(int(j))
                    break
        return tuple(sorted(current))

    def mutate(star: tuple[int, ...]) -> tuple[int, ...]:
        for _ in range(20):
            opened = list(star)
            closed = [l for l in available if l not in star]
            cand = (set(star) - {int(rng.choice(opened))}) | {int(rng.choice(closed))}
            cand = tuple(sorted(cand))
            if is_radial(cand):
                return cand
        return star

    # ---- 1. initial population: distinct radial configurations -------------
    pop: list[tuple[int, ...]] = []
    while len(pop) < n_stars and not (max_evals is not None and len(pop) >= max_evals):
        pop.append(random_radial(set(pop)))
    fit = np.array([score(c) for c in pop])
    bh = int(np.argmin(fit))

    res = BHAResult(best_open=pop[bh], best_fitness=float(fit[bh]),
                    best_eval=None,  # type: ignore[arg-type]
                    initial_population=list(pop))
    n_stars = len(pop)

    for it in range(max_iter):
        if out_of_budget():
            break

        # ---- 3. every star moves towards the black hole -------------------
        r = float(rng.random())
        for i in range(n_stars):
            if i == bh or out_of_budget():
                continue
            new = move_towards(pop[i], pop[bh], r)
            if mutation_rate > 0 and rng.random() < mutation_rate:
                new = mutate(new)
            new_fit = score(new)
            if (not greedy) or new_fit < fit[i]:
                pop[i], fit[i] = new, new_fit

        # ---- 4. a star better than the black hole takes its place ----------
        best_i = int(np.argmin(fit))
        if fit[best_i] < fit[bh]:
            bh = best_i

        # ---- 5. event horizon -----------------------------------------------
        finite = np.isfinite(fit)
        if horizon_ref == "feasible":
            # mean over stars that are not catastrophically penalised, so that a few
            # diverging configurations (score in the thousands) do not make R ~ 0
            ok = finite & (fit < weights.non_converged)
            pool = fit[ok] if ok.sum() >= 2 else fit[finite]
        else:
            pool = fit[finite]
        mean_f = float(pool.mean())
        radius_lines = k * min(1.0, horizon_scale * fit[bh] / max(mean_f, 1e-12))
        swallowed = 0
        seen = set(pop)
        for i in range(n_stars):
            if i == bh or out_of_budget():
                continue
            if hamming(pop[i], pop[bh]) < radius_lines or hamming(pop[i], pop[bh]) == 0:
                new = random_radial(seen)
                seen.add(new)
                pop[i], fit[i] = new, score(new)
                swallowed += 1
        best_i = int(np.argmin(fit))
        if fit[best_i] < fit[bh]:
            bh = best_i

        res.n_swallowed += swallowed
        res.history_best.append(float(fit[bh]))
        res.history_mean.append(float(fit[np.isfinite(fit)].mean()))
        res.history_evals.append(len(cache))
        res.history_unique_configurations.append(len(cache))
        res.history_swallowed.append(swallowed)
        res.bh_index.append(bh)
        if record_positions:
            res.positions.append(np.asarray(pop, dtype=float))
        if callback is not None:
            callback(it, float(fit[bh]))
        if progress_callback is not None:  # (iteration, evaluations used, best fitness)
            progress_callback(it, len(cache), float(fit[bh]))

    res.best_open = pop[bh]
    res.best_fitness = float(fit[bh])
    res.best_eval = evaluate(grid, pop[bh], weights)
    res.n_unique_evals = len(cache)
    res.eval_scores = list(cache.values())
    return res