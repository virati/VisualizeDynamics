"""The GPU flow viewer's Python side: flows, brackets, screening, the page."""

import json
import re

import numpy as np
import pytest
import sympy as sym

from vizdyn import gpu

HOPF = ["fc * win * (mu * x - y - x * (x**2 + y**2))", "fc * (1 - win) * (x + mu * y - y * (x**2 + y**2))"]
HOPF_PARAMS = {"mu": 1.0, "fc": 1.0, "win": 0.5}


def test_planar_equations_lift_with_a_contracting_z():
    f = gpu.equation_flow("Hopf", HOPF, HOPF_PARAMS)
    assert f.planar
    assert f.exprs[2] == -gpu.X[2]
    with pytest.raises(ValueError, match="no value for"):
        gpu.equation_flow("Hopf", HOPF, {"mu": 1.0})


def test_hopf_limit_cycle_is_bounded_and_regular():
    e = gpu.to_attractor(gpu.equation_flow("Hopf", HOPF, HOPF_PARAMS))
    assert e["vizdyn"]["bounded"]
    assert e["vizdyn"]["lyapunov"] < 0.02
    assert e["vizdyn"]["autolieError"] < 1e-9


def test_bracket_matches_autolie_and_hand_algebra():
    # Linear fields: (df/dx) g - (dg/dx) f = (A C - C A) x.
    f = gpu.equation_flow("f", ["-y", "x", "0"])
    g = gpu.equation_flow("g", ["x", "0", "z"])
    b = gpu.bracket_flow(f, g)
    A = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    C = np.diag([1, 0, 1])
    expected = (A @ C - C @ A) @ np.array(gpu.X)
    assert [sym.expand(e - x) for e, x in zip(b.exprs, expected)] == [0, 0, 0]
    ok, err = gpu.check_against_autolie(b, np.random.default_rng(0).normal(size=(5, 3)), parts=(f, g))
    assert ok and err < 1e-12


@pytest.mark.slow
def test_dysts_bracket_builds():
    f, g = gpu.dysts_flow("Rossler"), gpu.dysts_flow("SprottA")
    e = gpu.to_attractor(gpu.bracket_flow(f, g), parts=(f, g))
    assert e["name"] == "[Rossler, SprottA]"
    assert e["vizdyn"]["bounded"]


def test_render_html_embeds_the_config():
    e = gpu.to_attractor(gpu.equation_flow("Hopf", HOPF, HOPF_PARAMS))
    html = gpu.render_html([e], look="notebook", flat="xy")
    blob = re.search(r"window\.VIZDYN_CONFIG = (\{.*?\});\n", html).group(1)
    config = json.loads(blob.replace("<\\/", "</"))
    assert config["flat"] == "xy" and config["palette"] == "Reusser"
    assert config["systems"][0]["deriv"] == e["deriv"]
    assert "/*CONFIG*/" not in html
