"""Initial smoke tests for the IEEE 33-bus network.

These tests require the dependencies in requirements.txt to be installed.
"""

import pytest

pandapower = pytest.importorskip("pandapower")

from core.network import GridModel


def test_ieee33_model_loads():
    grid = GridModel()
    assert grid.n_bus == 33
    assert grid.n_line == 37
    assert grid.total_load_mw > 0


def test_normal_open_switch_count():
    grid = GridModel()
    assert len(grid.normal_open) == 5


def test_base_case_power_flow_converges():
    import pandapower as pp

    grid = GridModel()
    grid.apply_configuration(grid.normal_open, strict=False)
    pp.runpp(grid.net, numba=False, init="flat")
    assert grid.net.res_bus.vm_pu.notna().all()
