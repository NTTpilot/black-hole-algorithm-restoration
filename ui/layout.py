"""
ui/layout.py
------------
Hand-placed drawing of the IEEE 33-bus feeder. EDIT THIS FILE to move buses.

BUS_XY maps the 1-based bus number (as in Baran & Wu) to an (x, y) position.
TIE_BEND controls how much each tie-switch line is curved (0 = straight).
Positive = bends one way, negative = the other way. Line index is 0-based.
"""
from __future__ import annotations

import numpy as np

# Main feeder: buses 1..18 along the top
BUS_XY: dict[int, tuple[float, float]] = {b: (float(b - 1), 0.0) for b in range(1, 19)}

# Lateral A (buses 19-22) hangs from bus 2, drawn on the lowest row
BUS_XY.update({19: (1.0, -3.2), 20: (3.0, -3.2), 21: (5.0, -3.2), 22: (7.0, -3.2)})

# Lateral B (buses 23-25) hangs from bus 3, drawn on the upper row at the left
BUS_XY.update({23: (2.0, -1.6), 24: (3.4, -1.6), 25: (4.8, -1.6)})

# Lateral C (buses 26-33) hangs from bus 6, drawn on the upper row at the right
BUS_XY.update({26: (7.2, -1.6), 27: (8.6, -1.6), 28: (10.0, -1.6), 29: (11.4, -1.6),
               30: (12.8, -1.6), 31: (14.2, -1.6), 32: (15.6, -1.6), 33: (17.0, -1.6)})

# Curvature of the tie lines (0-based line index -> bend)
TIE_BEND: dict[int, float] = {32: 0.0, 33: -0.25, 34: 0.0, 35: 0.0, 36: -0.3}

# Where the bus number is written relative to the bus (default: above). 1-based bus numbers.
LABEL_POS: dict[int, str] = {19: "bottom center", 20: "bottom center", 21: "bottom center",
                             22: "bottom center", 33: "bottom center"}


def bus_xy(bus0: int) -> tuple[float, float]:
    """Position of a 0-based bus index."""
    return BUS_XY[bus0 + 1]


def line_path(a0: int, b0: int, bend: float = 0.0, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Points along the line between two 0-based buses (a Bezier curve if bend != 0)."""
    (x1, y1), (x2, y2) = bus_xy(a0), bus_xy(b0)
    if bend == 0.0:
        return np.array([x1, x2]), np.array([y1, y2])
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    cx, cy = mx - dy * bend, my + dx * bend       # control point, perpendicular to the chord
    t = np.linspace(0, 1, n)
    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
    return x, y