"""
core/network.py
---------------
IEEE 33-bus distribution network (Baran & Wu, 1989) + fault handling +
switch-configuration logic.

Conventions (IMPORTANT, read once):
  * Everything is 0-based internally (pandapower style).
      line index i  <->  "branch i+1" in Baran & Wu
      bus  index i  <->  "bus i+1"
  * Lines 0..31  are normally CLOSED (sectionalizing switches).
  * Lines 32..36 are normally OPEN   (tie switches).
  * A "configuration" = the set of OPEN lines. Everything else is closed.
    A faulted line is permanently out of service and is NOT part of the
    search space.
  * For a radial network the number of open lines is fixed:
        k = (#available lines) - (#buses - #components)
    With no fault k = 5, with one fault on a loop k = 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional, Sequence

import numpy as np
import pandapower.networks as pn


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Return True if a and b were in different sets (merged),
        False if they were already connected (=> this edge closes a loop)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


@dataclass
class Topology:
    n_cycles: int            # number of independent loops among closed lines
    energized: np.ndarray    # bool[n_bus]: connected to the substation?
    n_islands: int           # components among closed lines

    @property
    def is_radial(self) -> bool:
        return self.n_cycles == 0


class GridModel:
    """Holds the network, the current fault, and all topology logic."""

    def __init__(self) -> None:
        self.net = pn.case33bw()
        self.n_bus = len(self.net.bus)
        self.n_line = len(self.net.line)
        self.slack_bus = int(self.net.ext_grid.bus.iloc[0])
        self.from_bus = self.net.line.from_bus.to_numpy(dtype=int)
        self.to_bus = self.net.line.to_bus.to_numpy(dtype=int)

        # Normal (pre-fault) state
        self.normal_open: tuple[int, ...] = tuple(
            int(i) for i in self.net.line.index[~self.net.line.in_service]
        )

        # Load per bus (MW) -> used for "unserved load"
        self.load_p_mw = np.zeros(self.n_bus)
        for _, row in self.net.load.iterrows():
            self.load_p_mw[int(row.bus)] += float(row.p_mw)
        self.total_load_mw = float(self.load_p_mw.sum())
        self.load_q_mvar = np.zeros(self.n_bus)
        for _, row in self.net.load.iterrows():
            self.load_q_mvar[int(row.bus)] += float(row.q_mvar)

        # Line impedances in per-unit (Sbase = 1 MVA), for the cheap voltage estimate
        vn_kv = float(self.net.bus.vn_kv.iloc[0])
        z_base = vn_kv ** 2 / 1.0
        length = self.net.line.length_km.to_numpy(dtype=float)
        self.r_pu = self.net.line.r_ohm_per_km.to_numpy(dtype=float) * length / z_base
        self.x_pu = self.net.line.x_ohm_per_km.to_numpy(dtype=float) * length / z_base

        self.faulted_line: Optional[int] = None
        self._refresh()

    # ------------------------------------------------------------------ fault
    def set_fault(self, line: Optional[int]) -> None:
        """Mark one line as faulted (None = no fault)."""
        if line is not None and not (0 <= line < self.n_line):
            raise ValueError(f"line must be in [0, {self.n_line - 1}]")
        self.faulted_line = line
        self._refresh()

    def _refresh(self) -> None:
        f = self.faulted_line
        self.available: list[int] = [i for i in range(self.n_line) if i != f]

        # components of the graph made of ALL available lines
        uf = _UnionFind(self.n_bus)
        for l in self.available:
            uf.union(self.from_bus[l], self.to_bus[l])
        self.n_components = len({uf.find(b) for b in range(self.n_bus)})

        # how many lines must be open for a radial (forest) configuration
        self.n_open_required = len(self.available) - (self.n_bus - self.n_components)

        # state right after the fault, before any restoration action
        self.initial_open: tuple[int, ...] = tuple(
            l for l in self.normal_open if l != f
        )

    # --------------------------------------------------------------- topology
    def topology(self, open_lines: Iterable[int]) -> Topology:
        open_set = set(open_lines)
        uf = _UnionFind(self.n_bus)
        cycles = 0
        for l in self.available:
            if l in open_set:
                continue
            if not uf.union(self.from_bus[l], self.to_bus[l]):
                cycles += 1
        root = uf.find(self.slack_bus)
        energized = np.array([uf.find(b) == root for b in range(self.n_bus)])
        n_islands = len({uf.find(b) for b in range(self.n_bus)})
        return Topology(cycles, energized, n_islands)

    def switching_operations(self, open_lines: Iterable[int]) -> int:
        """Number of switches whose state differs from the post-fault state.
        (Opening the faulted line itself is done by protection: not counted.)"""
        new_open = set(open_lines)
        old_open = set(self.initial_open)
        return len(new_open ^ old_open)

    # ---------------------------------------------------------- configuration
    def validate(self, open_lines: Sequence[int], strict: bool = True) -> None:
        """strict=True: must be a full candidate (exactly k open lines).
        strict=False: only check lines are valid (used to score reference
        states such as the post-fault state, which has k+1 open lines)."""
        s = set(open_lines)
        if len(s) != len(open_lines):
            raise ValueError("duplicate lines in configuration")
        if not s.issubset(self.available):
            raise ValueError("configuration contains the faulted/unknown line")
        if strict and len(s) != self.n_open_required:
            raise ValueError(
                f"need exactly {self.n_open_required} open lines, got {len(s)}"
            )

    def apply_configuration(self, open_lines: Sequence[int], strict: bool = True) -> None:
        """Push a configuration into the pandapower network."""
        self.validate(open_lines, strict)
        in_service = np.ones(self.n_line, dtype=bool)
        in_service[list(open_lines)] = False
        if self.faulted_line is not None:
            in_service[self.faulted_line] = False
        self.net.line["in_service"] = in_service

    # ------------------------------------------------- cheap voltage estimate
    def estimate_voltages(self, open_lines: Iterable[int]) -> np.ndarray:
        """
        Fast, power-flow-free voltage estimate (per-unit, NaN for buses not
        connected to the substation). Walks the tree from the substation and
        uses   drop(line) ~ (r*P_down + x*Q_down) / V_parent,
        where P_down/Q_down is the load below that line. It ignores losses,
        so it is optimistic, but it still ranks configurations correctly
        even when the real power flow diverges. Assumes no loops.
        """
        open_set = set(open_lines)
        adj: list[list[tuple[int, int]]] = [[] for _ in range(self.n_bus)]
        for l in self.available:
            if l in open_set:
                continue
            a, b = int(self.from_bus[l]), int(self.to_bus[l])
            adj[a].append((b, l))
            adj[b].append((a, l))

        parent = {self.slack_bus: (-1, -1)}
        order = [self.slack_bus]
        for u in order:                       # BFS (list grows while iterating)
            for v, l in adj[u]:
                if v not in parent:
                    parent[v] = (u, l)
                    order.append(v)

        p_down = self.load_p_mw.copy()
        q_down = self.load_q_mvar.copy()
        for v in reversed(order[1:]):         # accumulate loads towards the root
            u, _ = parent[v]
            p_down[u] += p_down[v]
            q_down[u] += q_down[v]

        vm = np.full(self.n_bus, np.nan)
        vm[self.slack_bus] = 1.0
        for v in order[1:]:
            u, l = parent[v]
            drop = (self.r_pu[l] * p_down[v] + self.x_pu[l] * q_down[v]) / max(vm[u], 0.1)
            vm[v] = vm[u] - drop
        return vm

    # ------------------------------------------------------ search-space tools
    def all_configurations(self) -> Iterable[tuple[int, ...]]:
        """Every possible set of k open lines (used for brute-force reference)."""
        return combinations(self.available, self.n_open_required)

    def decode_position(self, position: np.ndarray) -> tuple[int, ...]:
        """
        Random-key decoding for the BHA: a star is a continuous vector of
        length k with values in [0, len(available)). Each value is rounded
        down to an index into `available`; duplicates are bumped to the next
        free index so we always return k DISTINCT lines.
        """
        n = len(self.available)
        chosen: list[int] = []
        used: set[int] = set()
        for x in position:
            idx = int(np.clip(np.floor(x), 0, n - 1))
            while idx in used:
                idx = (idx + 1) % n
            used.add(idx)
            chosen.append(self.available[idx])
        return tuple(sorted(chosen))