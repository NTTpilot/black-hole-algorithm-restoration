"""ui/components.py -- reusable Streamlit pieces (banners, metrics, tables)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .results import Optimum
from .runner import PostFault, RunResult


def switch_changes(res: RunResult, post: PostFault) -> tuple[list[int], list[int]]:
    """(lines the algorithm closed, lines the algorithm opened) relative to the post-fault state."""
    before, after = set(post.initial_open), set(res.best_open)
    return sorted(before - after), sorted(after - before)


def status_banner(res: RunResult, opt: Optional[Optimum], post: PostFault) -> None:
    if post.lost_mw < 1e-9:
        if res.ev.n_switch_ops == 0:
            st.success("No outage: this tie switch carries no load, so the grid keeps running as it was. "
                       "The best plan is to change nothing.")
        else:
            st.info(f"No outage. The algorithm still found a setting that lowers the score using "
                    f"{res.ev.n_switch_ops} switch operation(s).")
    elif res.fully_restored and res.voltage_ok:
        st.success(f"Service restored: 100% of the load is supplied and every voltage is within limits "
                   f"(lowest {res.ev.v_min:.3f} p.u.).")
    elif not res.fully_restored:
        st.warning(f"{res.ev.unserved_mw * 1000:.0f} kW of load could not be reconnected.")
    else:
        st.warning(f"All load is reconnected, but no configuration keeps every voltage within limits. "
                   f"This is the best compromise found (lowest voltage {res.ev.v_min:.3f} p.u.).")
    if opt is not None:
        gap = res.fitness - opt.fitness
        if abs(gap) < 1e-6:
            st.info(f"This matches the true optimum found by testing every possible configuration "
                    f"(score {opt.fitness:.4f}).")
        else:
            st.info(f"True optimum (every configuration tested in advance): {opt.fitness:.4f}. "
                    f"This run: {res.fitness:.4f} (gap {gap:+.4f}).")


def metrics_block(res: RunResult, post: PostFault) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Load restored", f"{(1 - res.ev.unserved_frac) * 100:.1f}%",
              delta=f"{post.lost_frac * 100:.1f}% was out", delta_color="off")
    c2.metric("Switch operations", f"{res.ev.n_switch_ops}")
    c3.metric("Power losses", f"{res.ev.losses_kw:.1f} kW" if res.ev.converged else "n/a")
    c4.metric("Lowest voltage", f"{res.ev.v_min:.3f} p.u." if res.ev.converged else "n/a")


def switching_plan(res: RunResult, post: PostFault, grid) -> None:
    closed, opened = switch_changes(res, post)
    if not closed and not opened:
        st.markdown("**Switching plan:** no switch needs to change.")
        return
    lines = ["**Switching plan** (what the operator has to do):"]
    step = 1
    for l in opened:
        lines.append(f"{step}. Open branch {l + 1} (bus {int(grid.from_bus[l]) + 1} - bus {int(grid.to_bus[l]) + 1})")
        step += 1
    for l in closed:
        lines.append(f"{step}. Close branch {l + 1} (bus {int(grid.from_bus[l]) + 1} - bus {int(grid.to_bus[l]) + 1})")
        step += 1
    st.markdown("\n".join(lines))


def results_table(results: list[RunResult], opt: Optional[Optimum]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Algorithm": r.label,
            "Score (lower = better)": round(r.fitness, 4),
            "Load restored": f"{(1 - r.ev.unserved_frac) * 100:.1f}%",
            "Switch ops": r.ev.n_switch_ops,
            "Losses (kW)": round(r.ev.losses_kw, 1) if r.ev.converged else None,
            "Lowest voltage (p.u.)": round(r.ev.v_min, 3) if r.ev.converged else None,
            "Evaluations used": r.n_evals,
            "Evaluations to reach its best": r.evals_to_best,
            "Time (s)": round(r.runtime_s, 1),
        }
        if opt is not None:
            row["Gap to true optimum"] = round(r.fitness - opt.fitness, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_verdict(results: list[RunResult], opt: Optional[Optimum]) -> str:
    best = min(r.fitness for r in results)
    tied = [r for r in results if r.fitness <= best + 1e-9]
    if len(tied) == len(results):
        fastest = min(tied, key=lambda r: r.evals_to_best)
        return (f"All three reached the same score ({best:.4f}). "
                f"{fastest.label} got there fastest ({fastest.evals_to_best} evaluations).")
    names = " and ".join(r.label for r in tied)
    others = [r for r in results if r not in tied]
    text = f"Best score {best:.4f}: {names}. "
    text += "Behind: " + ", ".join(f"{r.label} ({r.fitness:.4f})" for r in others) + "."
    if len(tied) > 1:
        fastest = min(tied, key=lambda r: r.evals_to_best)
        text += f" Of the leaders, {fastest.label} was fastest ({fastest.evals_to_best} evaluations)."
    return text