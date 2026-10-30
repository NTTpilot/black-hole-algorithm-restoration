"""Tests for discrete search behavior. Requires the project's dependencies."""
import pytest

pytest.importorskip("pandapower")

from core.bha import black_hole_algorithm
from core.bha_paper import black_hole_algorithm_paper
from core.baseline import random_search
from core.network import GridModel


def test_bha_returns_valid_radial_configuration_within_budget():
    grid = GridModel()
    grid.set_fault(7)
    result = black_hole_algorithm(grid, n_stars=8, max_iter=100,
                                  seed=123, max_evals=25)
    assert len(result.best_open) == grid.n_open_required
    assert len(set(result.best_open)) == len(result.best_open)
    assert grid.topology(result.best_open).is_radial
    assert result.n_unique_evals <= 25
    assert result.best_eval.radial


def test_random_search_respects_unique_evaluation_budget():
    grid = GridModel()
    grid.set_fault(7)
    result = random_search(grid, max_evals=20, seed=123)
    assert result.n_unique_evals <= 20
    assert grid.topology(result.best_open).is_radial


def test_paper_bha_returns_valid_radial_configuration_within_budget():
    grid = GridModel()
    grid.set_fault(7)
    result = black_hole_algorithm_paper(grid, n_stars=8, max_iter=100,
                                        seed=123, max_evals=25)
    assert len(result.best_open) == grid.n_open_required
    assert len(set(result.best_open)) == len(result.best_open)
    assert grid.topology(result.best_open).is_radial
    assert result.n_unique_evals <= 25
    assert result.best_eval.radial


def test_paper_bha_is_reproducible_with_a_seed():
    outcomes = []
    for _ in range(2):
        grid = GridModel()
        grid.set_fault(7)
        r = black_hole_algorithm_paper(grid, n_stars=8, max_iter=100,
                                       seed=7, max_evals=40)
        outcomes.append((r.best_open, r.best_fitness))
    assert outcomes[0] == outcomes[1]


def test_paper_bha_history_never_gets_worse():
    grid = GridModel()
    grid.set_fault(7)
    r = black_hole_algorithm_paper(grid, n_stars=8, max_iter=100,
                                   seed=5, max_evals=60)
    hist = r.history_best
    assert len(hist) > 0
    assert all(b <= a + 1e-12 for a, b in zip(hist, hist[1:]))


def test_every_algorithm_exposes_one_score_per_evaluated_configuration():
    for run in (
        lambda g: black_hole_algorithm(g, n_stars=8, max_iter=100, seed=1, max_evals=30),
        lambda g: black_hole_algorithm_paper(g, n_stars=8, max_iter=100, seed=1, max_evals=30),
        lambda g: random_search(g, max_evals=30, seed=1),
    ):
        grid = GridModel()
        grid.set_fault(7)
        result = run(grid)
        assert len(result.eval_scores) == result.n_unique_evals
        assert min(result.eval_scores) == result.best_fitness