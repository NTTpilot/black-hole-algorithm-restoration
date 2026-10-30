"""
core/fitness.py
---------------
Fitness function = "the judge". Takes a switch configuration, returns a
score (LOWER = BETTER) plus a detailed breakdown.

Score (converged power flow) =
        w_unserved  * (unserved load fraction)
      + w_switching * (number of switching operations)
      + w_losses    * (active losses in kW)
      + w_voltage   * (sum of voltage-limit violations in p.u.)

Non-radial configuration (loops): w_nonradial * loops + w_switching * ops + w_unserved
    -> no power flow is run.

Diverging configuration (power flow fails, or hopeless voltage estimate):
        w_non_converged
      + w_unserved  * unserved fraction
      + w_switching * ops
      + w_voltage   * (violation of the ESTIMATED voltages)
    The last term is what removes the flat plateau: a configuration that
    would collapse at 0.4 p.u. scores worse than one collapsing at 0.65 p.u.,
    so the BHA has a direction to follow even where the power flow is useless.
    The estimate comes from GridModel.estimate_voltages (no power flow needed).

Notes:
  * The raw 33-bus data has NO current limits, so only voltage limits are
    checked. The base-case min voltage is ~0.913 p.u., hence the default
    lower limit of 0.90 and not 0.95.
  * prescreen_v: if the estimated min voltage is below this value, the
    (40 ms) power flow is skipped and the config is scored as diverging.
    On fault line 7, every config that actually converged had an estimate
    above 0.67, so 0.60 is a safe cut. Set to None to always run the power flow.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandapower as pp

from .network import GridModel


@dataclass
class Weights:
    unserved: float = 1000.0
    switching: float = 0.5
    losses: float = 0.01          # per kW
    voltage: float = 1000.0       # per p.u. of violation (summed over buses)
    nonradial: float = 500.0      # per loop
    non_converged: float = 2000.0 # base penalty for a diverging power flow
    v_min: float = 0.90
    v_max: float = 1.05
    prescreen_v: Optional[float] = 0.60


@dataclass
class Evaluation:
    fitness: float
    radial: bool
    converged: bool = False
    unserved_mw: float = 0.0
    unserved_frac: float = 0.0
    n_switch_ops: int = 0
    losses_kw: float = float("nan")
    v_min: float = float("nan")          # from the real power flow
    v_max: float = float("nan")
    v_violation: float = 0.0
    v_min_est: float = float("nan")      # from the cheap estimate (always filled if radial)
    vm_pu: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def feasible(self) -> bool:
        return self.radial and self.converged and self.v_violation == 0.0


def _violation(vm: np.ndarray, w: Weights) -> float:
    return float(
        np.sum(np.clip(w.v_min - vm, 0, None)) + np.sum(np.clip(vm - w.v_max, 0, None))
    )


def evaluate(grid: GridModel, open_lines: Sequence[int], w: Weights = Weights(),
             strict: bool = True) -> Evaluation:
    """strict=False lets you score reference states (e.g. g.initial_open right
    after a fault) that are not full candidate configurations."""
    open_lines = tuple(sorted(open_lines))
    grid.validate(open_lines, strict)
    topo = grid.topology(open_lines)
    ops = grid.switching_operations(open_lines)

    # ---- loops: reject without power flow ---------------------------------
    if not topo.is_radial:
        return Evaluation(
            fitness=w.nonradial * topo.n_cycles + w.switching * ops + w.unserved,
            radial=False,
            n_switch_ops=ops,
        )

    # ---- unserved load (buses cut off from the substation) ----------------
    served = topo.energized
    unserved_mw = float(grid.load_p_mw[~served].sum())
    unserved_frac = unserved_mw / grid.total_load_mw

    # ---- cheap voltage estimate (no power flow) ---------------------------
    vm_est = grid.estimate_voltages(open_lines)[served]
    v_min_est = float(np.nanmin(vm_est))

    def diverged() -> Evaluation:
        fitness = (
            w.non_converged
            + w.unserved * unserved_frac
            + w.switching * ops
            + w.voltage * _violation(vm_est, w)
        )
        return Evaluation(
            fitness=fitness, radial=True, converged=False,
            unserved_mw=unserved_mw, unserved_frac=unserved_frac,
            n_switch_ops=ops, v_min_est=v_min_est,
        )

    if w.prescreen_v is not None and v_min_est < w.prescreen_v:
        return diverged()

    # ---- power flow -------------------------------------------------------
    grid.apply_configuration(open_lines, strict)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp(grid.net, numba=False, init="flat")
    except Exception:  # pandapower.LoadflowNotConverged and friends
        return diverged()

    vm = grid.net.res_bus.vm_pu.to_numpy()
    vm_served = vm[served]
    losses_kw = float(grid.net.res_line.pl_mw.fillna(0.0).sum() * 1000.0)
    violation = _violation(vm_served, w)

    fitness = (
        w.unserved * unserved_frac
        + w.switching * ops
        + w.losses * losses_kw
        + w.voltage * violation
    )
    return Evaluation(
        fitness=fitness, radial=True, converged=True,
        unserved_mw=unserved_mw, unserved_frac=unserved_frac, n_switch_ops=ops,
        losses_kw=losses_kw,
        v_min=float(np.nanmin(vm_served)), v_max=float(np.nanmax(vm_served)),
        v_violation=violation, v_min_est=v_min_est,
        vm_pu=np.where(served, vm, np.nan),
    )


def make_fitness(grid: GridModel, w: Weights = Weights()) -> Callable[[Sequence[int]], float]:
    """Cached scalar fitness function: open_lines -> float. This is what the
    BHA will call. The cache matters: the search space is small, so the same
    configuration is evaluated many times."""
    cache: dict[tuple[int, ...], float] = {}

    def f(open_lines: Sequence[int]) -> float:
        key = tuple(sorted(open_lines))
        if key not in cache:
            cache[key] = evaluate(grid, key, w).fitness
        return cache[key]

    f.cache = cache  # type: ignore[attr-defined]
    return f