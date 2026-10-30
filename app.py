"""
Power restoration after a fault, with the Black Hole Algorithm.
Run from the project root:   streamlit run app.py
"""
import time

import streamlit as st

from core.network import GridModel
from ui import plots
from ui.components import (compare_verdict, metrics_block, results_table, status_banner,
                           switch_changes, switching_plan)
from ui.faults import fault_label, offered_faults
from ui.results import load_benchmark_summary, load_optima
from ui.runner import METHOD_SHORT, new_seed, post_fault_state, run_method, warm_up

st.set_page_config(page_title="Power restoration - Black Hole Algorithm", layout="wide")


# ----------------------------------------------------------------- cached helpers
@st.cache_resource
def static_grid() -> GridModel:
    """Read-only copy used for geometry and loads. Never call set_fault on it."""
    warm_up()                      # first power flow is slow: pay for it at start-up
    return GridModel()


@st.cache_data
def fault_list():
    return offered_faults()


@st.cache_data
def post_state(fault: int):
    return post_fault_state(fault)


@st.cache_data
def optima():
    return load_optima()


grid = static_grid()
infos = fault_list()
known = optima()

# ----------------------------------------------------------------------- sidebar
st.sidebar.header("1. Choose the fault")
lines = [i.line for i in infos]
labels = {i.line: fault_label(i, known.get(i.line) is not None and known[i.line].feasible is False)
          for i in infos}
fault = st.sidebar.selectbox("Line that fails", lines, index=lines.index(7),
                             format_func=lambda l: labels[l])
st.sidebar.caption("Not offered: branch 1 (the trunk). Everything is lost and nothing can be rerouted, "
                   "so there is nothing to optimise. Branches 33-37 are tie switches: they carry no load, "
                   "so a fault there causes no outage.")

st.sidebar.header("2. Settings")
random_seed = st.sidebar.checkbox("New random seed every run", value=True)
seed_input = st.sidebar.number_input("Seed", min_value=0, max_value=999_999, value=42,
                                     disabled=random_seed)
with st.sidebar.expander("Advanced"):
    n_stars = st.slider("Stars (population size)", 10, 60, 30, 5)


def current_seed() -> int:
    return new_seed() if random_seed else int(seed_input)


post = post_state(fault)
opt = known.get(fault)

# ------------------------------------------------------------------------ header
st.title("Restoring power after a fault")
st.caption("An outage hits a distribution network. The Black Hole Algorithm searches for the switch "
           "settings that bring power back to as many customers as possible, within voltage limits, "
           "with as few switch operations as possible.")

tab_main, tab_cmp = st.tabs(["Restore service", "Compare algorithms"])

# ======================================================================= TAB 1
with tab_main:
    budget = st.slider("Search budget (configurations tested)", 100, 1500, 750, 50, key="budget_main")
    sec_per_eval = st.session_state.get("sec_per_eval", 0.027)
    st.caption(f"Estimated run time: about {budget * sec_per_eval:.0f} s on this computer.")
    go_main = st.button("Cause the fault and restore service", type="primary", key="go_main")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            plots.network_figure(grid, fault, post.initial_open, post.energized, vm=post.vm_pu,
                                 title="After the fault"),
            width="stretch", key="net_pre")
        if post.lost_mw < 1e-9:
            st.caption("Nothing loses power: this is a tie switch that is normally open, so the grid keeps running.")
        else:
            st.caption(f"{post.lost_mw * 1000:.0f} kW ({post.lost_frac * 100:.0f}% of all load) has no power.")

    if go_main:
        seed = current_seed()
        with col_right:
            bar = st.progress(0.0, text="Starting...")
            live = st.empty()
        ev_hist: list[int] = []
        best_hist: list[float] = []
        state = {"last": 0.0, "n": 0}

        def on_progress(evals: int, best: float) -> None:
            ev_hist.append(evals)
            best_hist.append(best)
            bar.progress(min(1.0, evals / budget),
                         text=f"Tested {evals} of {budget} configurations - best score so far {best:.3f}")
            if time.time() - state["last"] > 0.6:
                state["last"] = time.time()
                state["n"] += 1
                live.plotly_chart(
                    plots.convergence_figure(
                        {METHOD_SHORT["paper"]: (ev_hist, best_hist, "paper")}, opt.fitness if opt else None,
                        height=300, title="Searching..."),
                    width="stretch", key=f"live_{state['n']}")

        res = run_method("paper", fault, budget, n_stars, seed, on_progress)
        bar.empty()
        live.empty()
        st.session_state["sec_per_eval"] = res.runtime_s / max(res.n_evals, 1)
        st.session_state["main"] = res

    res = st.session_state.get("main")
    shown = res is not None and res.fault == fault
    with col_right:
        if shown:
            closed, opened = switch_changes(res, post)
            st.plotly_chart(
                plots.network_figure(grid, fault, res.best_open, res.energized, vm=res.ev.vm_pu,
                                     closed_by_alg=closed, opened_by_alg=opened,
                                     title="After restoration"),
                width="stretch", key="net_post")
        elif not go_main:
            st.info("Press the button to let the algorithm find the switch settings.")

    if shown:
        status_banner(res, opt, post)
        metrics_block(res, post)
        switching_plan(res, post, grid)
        st.caption(f"Run details: seed {res.seed}, {res.n_evals} configurations tested, "
                   f"{res.runtime_s:.1f} s.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            plots.convergence_figure(
                {METHOD_SHORT["paper"]: (res.curve_evals, res.curve_best, "paper")},
                opt.fitness if opt else None),
            width="stretch", key="conv_main")
        c2.plotly_chart(plots.voltage_figure(post.vm_pu, res.ev.vm_pu, 0.90),
                        width="stretch", key="volt_main")

# ======================================================================= TAB 2
with tab_cmp:
    budget_c = st.slider("Search budget per algorithm", 100, 1500, 400, 50, key="budget_cmp")
    st.caption(f"Estimated run time for all three: about "
               f"{3 * budget_c * st.session_state.get('sec_per_eval', 0.027):.0f} s. "
               "All three get the same fault, the same budget and the same seed.")
    go_cmp = st.button("Cause the fault and run all three algorithms", type="primary", key="go_cmp")

    if go_cmp:
        seed = current_seed()
        order = ["paper", "hybrid", "random"]
        bars = {m: st.progress(0.0, text=f"{METHOD_SHORT[m]}: waiting...") for m in order}
        results = []
        for m in order:
            def cb(evals: int, best: float, m=m) -> None:
                bars[m].progress(min(1.0, evals / budget_c),
                                 text=f"{METHOD_SHORT[m]}: {evals}/{budget_c} - best score {best:.3f}")
            bars[m].progress(0.0, text=f"{METHOD_SHORT[m]}: running...")
            r = run_method(m, fault, budget_c, n_stars, seed, cb)
            bars[m].progress(1.0, text=f"{METHOD_SHORT[m]}: done in {r.runtime_s:.1f} s, score {r.fitness:.4f}")
            results.append(r)
        st.session_state["cmp"] = results
        st.session_state["sec_per_eval"] = sum(r.runtime_s for r in results) / max(
            sum(r.n_evals for r in results), 1)

    results = st.session_state.get("cmp")
    if results and results[0].fault == fault:
        st.subheader("Result")
        st.write(compare_verdict(results, opt))
        st.caption(f"Seed {results[0].seed}, budget {results[0].budget} configurations each.")
        st.dataframe(results_table(results, opt), hide_index=True, width="stretch")
        st.plotly_chart(
            plots.convergence_figure({r.short: (r.curve_evals, r.curve_best, r.method) for r in results},
                                     opt.fitness if opt else None, height=340,
                                     title="Best score found so far (lower is better)"),
            width="stretch", key="conv_cmp")
        cols = st.columns(3)
        for col, r in zip(cols, results):
            closed, opened = switch_changes(r, post)
            with col:
                st.plotly_chart(
                    plots.network_figure(grid, fault, r.best_open, r.energized, vm=r.ev.vm_pu,
                                         closed_by_alg=closed, opened_by_alg=opened,
                                         title=f"{r.short}: score {r.fitness:.3f}", height=330,
                                         show_legend=False),
                    width="stretch", key=f"net_cmp_{r.method}")
        st.caption("In the grids: green = switch closed by the algorithm, orange dashed = switch opened by it, "
                   "red X = the fault, dotted grey = open switch, red dots = buses without power.")
    elif not go_cmp:
        st.info("Press the button to run the three algorithms one after the other.")

    with st.expander("Offline benchmark: 20 seeds per method, 750 configurations each"):
        bench = load_benchmark_summary()
        if bench is None:
            st.caption("No benchmark file found. Run `python -m experiments.run_benchmark` first.")
        else:
            sub = bench[bench.fault_line_0_based == fault]
            if sub.empty:
                st.caption("This fault was not part of the benchmark. Showing the faults that were.")
                sub = bench
            cols_wanted = [c for c in ["fault_line_0_based", "method", "runs", "mean_fitness", "std_fitness",
                                       "true_best_fitness", "mean_gap"] if c in sub]
            table = sub[cols_wanted].copy()
            table["fault_line_0_based"] = table["fault_line_0_based"] + 1   # show branch numbers
            table = table.rename(columns={"fault_line_0_based": "branch"}).round(4)
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("A single live run is one random sample. This table is the fair comparison.")