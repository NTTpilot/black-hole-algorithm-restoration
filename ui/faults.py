"""ui/faults.py -- which faults the dropdown offers, and how they are labelled."""
from __future__ import annotations

from dataclasses import dataclass

from core.network import GridModel


@dataclass(frozen=True)
class FaultInfo:
    line: int          # 0-based line index (used by the code)
    branch: int        # 1-based branch number (shown to people)
    bus_from: int      # 1-based bus numbers
    bus_to: int
    lost_mw: float     # load without power right after the fault
    lost_frac: float

    @property
    def no_outage(self) -> bool:
        return self.lost_mw < 1e-9


def offered_faults() -> list[FaultInfo]:
    """
    Every line is offered except the trunk (branch 1): when it fails everything
    is lost and nothing can be rerouted, so there is nothing to optimise.
    The 5 tie switches (branches 33-37) are offered too, but they carry no load,
    so a fault there causes no outage (the label says so).
    """
    grid = GridModel()
    out: list[FaultInfo] = []
    for line in range(grid.n_line):
        grid.set_fault(line)
        now = grid.topology(grid.initial_open)
        best_case = grid.topology(())          # every available line closed
        lost = float(grid.load_p_mw[~now.energized].sum())
        unavoidable = float(grid.load_p_mw[~best_case.energized].sum())
        if lost - unavoidable > 1e-9 or lost < 1e-9:
            out.append(FaultInfo(
                line=line, branch=line + 1,
                bus_from=int(grid.from_bus[line]) + 1, bus_to=int(grid.to_bus[line]) + 1,
                lost_mw=lost, lost_frac=lost / grid.total_load_mw,
            ))
    return out


def fault_label(info: FaultInfo, known_infeasible: bool = False) -> str:
    text = (f"Branch {info.branch}: bus {info.bus_from} - bus {info.bus_to}"
            f"   ({info.lost_mw * 1000:.0f} kW out)")
    if info.no_outage:
        text = (f"Branch {info.branch}: bus {info.bus_from} - bus {info.bus_to}"
                f"   (tie switch, no outage)")
    if known_infeasible:
        text += "   [!] no fully valid fix exists"
    return text


meaningful_faults = offered_faults  # old name, kept so nothing breaks