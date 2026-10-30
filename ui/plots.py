"""ui/plots.py -- all the figures (plotly)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from .layout import LABEL_POS, TIE_BEND, bus_xy, line_path

METHOD_COLORS = {"paper": "#1f77b4", "hybrid": "#ff7f0e", "random": "#8c8c8c"}

C_CLOSED, C_OPEN = "#7a8896", "#a0aab5"
C_FAULT, C_NEWCLOSED, C_NEWOPEN = "#d62728", "#2ca02c", "#ff7f0e"
C_POWER, C_NOPOWER = "#2e9e5b", "#c0392b"


def _transparent(fig: go.Figure) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _mid(grid, l: int) -> tuple[float, float, bool]:
    """Midpoint of a line as drawn, and whether it is drawn horizontally."""
    x, y = line_path(int(grid.from_bus[l]), int(grid.to_bus[l]), TIE_BEND.get(l, 0.0))
    k = len(x) // 2
    mx = float(x[k] if len(x) > 2 else (x[0] + x[1]) / 2)
    my = float(y[k] if len(y) > 2 else (y[0] + y[1]) / 2)
    horizontal = abs(float(y[0]) - float(y[-1])) < 1e-9 and len(x) == 2
    return mx, my, horizontal


def network_figure(grid, fault: int, open_lines: Sequence[int], energized: np.ndarray,
                   vm: Optional[np.ndarray] = None, closed_by_alg: Sequence[int] = (),
                   opened_by_alg: Sequence[int] = (), title: str = "", height: int = 400,
                   show_legend: bool = True) -> go.Figure:
    """grid: any object with from_bus, to_bus, n_line, n_bus, load_p_mw, slack_bus, normal_open."""
    open_set, newc, newo = set(open_lines), set(closed_by_alg), set(opened_by_alg)
    cats = {
        "Closed line":            dict(lines=[], color=C_CLOSED, width=3, dash="solid"),
        "Open switch":            dict(lines=[], color=C_OPEN, width=2.5, dash="dash"),
        "Faulted line":           dict(lines=[], color=C_FAULT, width=5, dash="solid"),
        "Closed by algorithm":    dict(lines=[], color=C_NEWCLOSED, width=5, dash="solid"),
        "Opened by algorithm":    dict(lines=[], color=C_NEWOPEN, width=4, dash="dash"),
    }
    for l in range(grid.n_line):
        if l == fault:
            cats["Faulted line"]["lines"].append(l)
        elif l in newc:
            cats["Closed by algorithm"]["lines"].append(l)
        elif l in newo:
            cats["Opened by algorithm"]["lines"].append(l)
        elif l in open_set:
            cats["Open switch"]["lines"].append(l)
        else:
            cats["Closed line"]["lines"].append(l)

    fig = go.Figure()
    for name, c in cats.items():
        if not c["lines"]:
            continue
        xs: list = []
        ys: list = []
        for l in c["lines"]:
            x, y = line_path(int(grid.from_bus[l]), int(grid.to_bus[l]), TIE_BEND.get(l, 0.0))
            xs += list(x) + [None]
            ys += list(y) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, hoverinfo="skip",
                                 line=dict(color=c["color"], width=c["width"], dash=c["dash"])))

    # invisible mid-line markers carry the hover text for every line
    mids = [_mid(grid, l) for l in range(grid.n_line)]
    hover = []
    for l in range(grid.n_line):
        state = ("FAULTED" if l == fault else "newly closed" if l in newc else
                 "newly opened" if l in newo else "open" if l in open_set else "closed")
        hover.append(f"Branch {l + 1}: bus {int(grid.from_bus[l]) + 1} - bus {int(grid.to_bus[l]) + 1}<br>{state}")
    fig.add_trace(go.Scatter(x=[m[0] for m in mids], y=[m[1] for m in mids], mode="markers",
                             showlegend=False, hoverinfo="text", text=hover,
                             marker=dict(size=14, opacity=0)))

    # tie-switch numbers (small, grey)
    ties = [l for l in getattr(grid, "normal_open", ()) if l != fault]
    if ties:
        tie_label_offset = 0.15
        fig.add_trace(go.Scatter(
            x=[mids[l][0] + tie_label_offset for l in ties],
            y=[mids[l][1] + tie_label_offset for l in ties], mode="text",
            text=[f"tie {l + 1}" for l in ties], textposition="top right", showlegend=False,
            hoverinfo="skip", textfont=dict(size=9, color=C_OPEN)))

    # what the algorithm changed, written on the line itself
    for lines, word, col in ((sorted(newc), "CLOSE", C_NEWCLOSED), (sorted(newo), "OPEN", C_NEWOPEN)):
        if lines:
            fig.add_trace(go.Scatter(
                x=[mids[l][0] for l in lines], y=[mids[l][1] for l in lines], mode="text",
                text=[f"<b>{word} {l + 1}</b>" for l in lines], showlegend=False, hoverinfo="skip",
                textposition=["bottom center" if mids[l][2] else "middle right" for l in lines],
                textfont=dict(size=12, color=col)))

    # the fault itself
    fx, fy, fh = mids[fault]
    fig.add_trace(go.Scatter(
        x=[fx], y=[fy], mode="markers+text", text=[f"<b>FAULT: branch {fault + 1}</b>"],
        textposition="bottom center" if fh else "middle right", showlegend=False, hoverinfo="skip",
        textfont=dict(size=12, color=C_FAULT),
        marker=dict(symbol="x", size=17, color=C_FAULT, line=dict(color=C_FAULT, width=3))))

    # buses
    xs = [bus_xy(b)[0] for b in range(grid.n_bus)]
    ys = [bus_xy(b)[1] for b in range(grid.n_bus)]
    colors = [C_POWER if energized[b] else C_NOPOWER for b in range(grid.n_bus)]
    btxt = []
    for b in range(grid.n_bus):
        t = f"Bus {b + 1}<br>load {grid.load_p_mw[b] * 1000:.0f} kW"
        if not energized[b]:
            t += "<br><b>NO POWER</b>"
        elif vm is not None and not np.isnan(vm[b]):
            t += f"<br>voltage {vm[b]:.3f} p.u."
        btxt.append(t)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", showlegend=False, hoverinfo="text", text=btxt,
        marker=dict(size=[16 if b == grid.slack_bus else 11 for b in range(grid.n_bus)],
                    symbol=["square" if b == grid.slack_bus else "circle" for b in range(grid.n_bus)],
                    color=colors, line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="text", text=[str(b + 1) for b in range(grid.n_bus)], showlegend=False,
        hoverinfo="skip", textfont=dict(size=10),
        textposition=[LABEL_POS.get(b + 1, "top center") for b in range(grid.n_bus)]))
    sx, sy = bus_xy(grid.slack_bus)
    fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="text", text=["<b>Substation</b>"],
                             textposition="bottom center", showlegend=False, hoverinfo="skip",
                             textfont=dict(size=10)))

    # legend entries for bus colours
    for name, col in (("Bus with power", C_POWER), ("Bus without power", C_NOPOWER)):
        if name == "Bus without power" and energized.all():
            continue
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, color=col)))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0, xanchor="left"), height=height,
        margin=dict(l=5, r=5, t=40 if title else 10, b=5), showlegend=show_legend,
        legend=dict(orientation="h", yanchor="top", y=-0.02, x=0, font=dict(size=11)),
        xaxis=dict(visible=False, range=[-0.8, 17.8]),
        yaxis=dict(visible=False, range=[-4.1, 1.0]), hovermode="closest")
    return _transparent(fig)


def convergence_figure(curves: dict[str, tuple[Sequence[int], Sequence[float], str]],
                       optimum: Optional[float] = None, height: int = 340,
                       title: str = "Best score found so far") -> go.Figure:
    """
    curves: label -> (evaluations, best-so-far score, method key).
    Linear axis, zoomed on the interesting range: the huge scores of the first few
    random guesses are cut off at the top instead of squashing everything else.
    """
    series = {k: (np.asarray(e, dtype=float), np.asarray(b, dtype=float), key)
              for k, (e, b, key) in curves.items() if len(e)}
    finals = [b[-1] for _, b, _ in series.values()]
    low = min(finals + ([optimum] if optimum is not None else []))
    # top of the chart = the worst score any curve has after its first 30 evaluations
    tops = [b[min(29, len(b) - 1)] for _, b, _ in series.values()]
    high = max(tops)
    if high - low < 1e-9:
        high = low + max(abs(low) * 0.1, 0.1)
    pad = (high - low) * 0.12
    y_range = [low - pad, high + pad]

    fig = go.Figure()
    for label, (e, b, key) in series.items():
        color = METHOD_COLORS.get(key, "#1f77b4")
        fig.add_trace(go.Scatter(x=e, y=b, mode="lines", name=label,
                                 line=dict(color=color, width=3, shape="hv")))
        hit = int(np.argmax(b <= b[-1] + 1e-12))     # first evaluation that reached its final score
        fig.add_trace(go.Scatter(
            x=[e[hit]], y=[b[-1]], mode="markers", showlegend=False,
            hovertext=f"{label}: best score {b[-1]:.4f}, reached at evaluation {int(e[hit])}",
            hoverinfo="text", marker=dict(size=12, color=color, line=dict(color="white", width=2))))
    if optimum is not None:
        fig.add_hline(y=optimum, line=dict(color="#2ca02c", width=2, dash="dash"),
                      annotation_text="true optimum", annotation_position="top left",
                      annotation_font=dict(color="#2ca02c", size=12))
    if any(b[0] > y_range[1] for _, b, _ in series.values()):
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.0, xanchor="right", yanchor="bottom",
                           showarrow=False, font=dict(size=10, color="#8c8c8c"),
                           text=f"first guesses above {y_range[1]:.1f} are cut off")
    x_max = max(float(e[-1]) for e, _, _ in series.values())
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0, xanchor="left"), height=height,
        margin=dict(l=10, r=10, t=50, b=85),
        xaxis=dict(title="configurations evaluated", range=[0, x_max * 1.02]),
        yaxis=dict(title="score (lower is better)", range=y_range),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0))
    return _transparent(fig)


def voltage_figure(before: Optional[np.ndarray], after: Optional[np.ndarray], v_limit: float,
                   height: int = 340) -> go.Figure:
    buses = list(range(1, len(after if after is not None else before) + 1))
    fig = go.Figure()
    if before is not None:
        fig.add_trace(go.Scatter(x=buses, y=before, mode="lines+markers", name="after fault, before switching",
                                 line=dict(color=C_FAULT, dash="dot"), connectgaps=False))
    if after is not None:
        fig.add_trace(go.Scatter(x=buses, y=after, mode="lines+markers", name="after restoration",
                                 line=dict(color=C_POWER, width=3), connectgaps=False))
    fig.add_hline(y=v_limit, line=dict(color="#888", dash="dash"),
                  annotation_text=f"limit {v_limit:.2f}", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text="Voltage at every bus (gaps = no power)", font=dict(size=14), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=50, b=85),
        xaxis_title="bus number", yaxis_title="voltage (p.u.)",
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0))
    return _transparent(fig)