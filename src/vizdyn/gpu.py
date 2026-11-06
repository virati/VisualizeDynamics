"""GPU flow viewer: any vector field, rendered like attractor-screensaver.

A `Flow` is a vector field on R^3 held as sympy expressions in x, y, z. It can
come from a dysts system, from equation strings (a 2-D system gets z' = -z, so
it settles into the plane and the viewer can open flat on it), or from the Lie
bracket of two other flows, computed with autoLie:

    [f, g](x) = (df/dx) g(x) - (dg/dx) f(x)          (autoLie's convention)

`to_attractor` prints the field as GLSL (autolie.symbolics.codegen), checks the
closed form against autoLie's autograd operators on the trajectory, and runs
it through the same fixed-step RK4 the shader uses to pick a time step, a seed
and a view. `render_html` puts the result into the WebGL viewer page
(web/viewer.html, built into frontend/webgl/viewer.html by web/build.mjs),
which warps between 3-D and a 2-D plot of any two state axes, and slices.
"""

from __future__ import annotations

import json
import math
import types
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as onp
import sympy as sym

warnings.filterwarnings("ignore", message="Numba not installed")

X = sym.symbols("x y z")
VIEWER = Path(__file__).parent / "frontend" / "webgl" / "viewer.html"

# The renderer takes 60 RK4 steps a second. Reusser's Lorenz (dt 0.005) moves
# about this fraction of its attractor's width per step; matching it keeps
# every flow at a pace the eye can follow.
PACE = 0.012
ESCAPE = 1e3   # the shader starts pulling particles back at |p| = 1000
PARTICLES = 96


@dataclass
class Flow:
    """A vector field on R^3 as sympy expressions in x, y, z."""

    name: str
    exprs: list
    description: str = ""
    # Where to seed particles: (origin, scale) candidates, best guess first.
    seeds: list = field(default_factory=lambda: [(onp.zeros(3), 1.0)])
    planar: bool = False   # a 2-D system lifted with z' = -z

    @property
    def equations(self) -> list[str]:
        return [f"d{v}/dt = {sym.sstr(e)}" for v, e in zip("xyz", self.exprs)]

    def numeric(self) -> Callable:
        """Vectorised numpy field: (3, N) -> (3, N)."""
        fn = sym.lambdify([X], self.exprs, "numpy")

        def vec(P):
            return onp.stack([onp.broadcast_to(onp.asarray(c, dtype=float), P.shape[1:]) for c in fn(P)])

        return vec

    def autograd(self):
        """The same field as an autoLie operable, for autoLie's own operators."""
        import autograd.numpy as anp
        from autolie.op_plus import operable

        fn = sym.lambdify([X], self.exprs, modules=[{"array": anp.array}, anp])
        return operable(lambda p, args: anp.array([anp.reshape(c, ()) + 0.0 * p[0] for c in fn(p)]))


# --- where flows come from -------------------------------------------------------

@lru_cache(maxsize=None)
def dysts_names() -> tuple[str, ...]:
    """The 3-D autonomous dysts flows whose right-hand sides trace to closed form."""
    from dysts.systems import get_attractor_list

    names = []
    for n in get_attractor_list():
        try:
            dysts_flow(n)
            names.append(n)
        except Exception:
            pass
    return tuple(names)


@lru_cache(maxsize=None)
def dysts_flow(name: str) -> Flow:
    import dysts.flows
    from autolie.symbolics.codegen import symbolize

    cls = getattr(dysts.flows, name)
    inst = cls()
    md = inst.metadata
    if inst.dimension != 3:
        raise ValueError(f"{name} is {inst.dimension}-dimensional")
    if md.get("nonautonomous") or md.get("delay"):
        raise ValueError(f"{name} is not an autonomous ODE")
    params = dict(inst.params)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in params.values()):
        raise ValueError(f"{name} has non-scalar parameters")
    rhs = getattr(cls._rhs, "py_func", cls._rhs)
    exprs = [sym.sympify(e) for e in symbolize(rhs)(*X, sym.Integer(0), **params)]
    if any(e.free_symbols - set(X) for e in exprs):
        raise ValueError(f"{name} depends on something other than x, y, z")
    mean = onp.asarray(md.get("mean", [0, 0, 0]), dtype=float)
    spread = float(onp.linalg.norm(md.get("std", [1, 1, 1])))
    ic = onp.asarray(inst.ic, dtype=float)
    return Flow(
        name=name,
        exprs=exprs,
        description=f"{md.get('description', '')} {md.get('citation', '')}".strip(),
        seeds=[(mean, max(0.1, spread / 2)), (ic, max(0.1, spread / 4)), (onp.zeros(3), 1.0)],
    )


def equation_flow(name: str, equations: Sequence[str], params: Optional[dict] = None,
                  extent: float = 3.0) -> Flow:
    """A flow from equation strings in x, y (and z), with parameter values substituted.

    Two equations make a planar system, lifted to 3-D with z' = -z. `extent` is
    the half-width of the region to seed in, e.g. the 2-D plot's axis limit.
    """
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    local = {"x": X[0], "y": X[1], "z": X[2]}
    subs = {sym.Symbol(k): v for k, v in (params or {}).items()}
    exprs = []
    for eq in equations:
        e = parse_expr(eq.replace("^", "**"), local_dict=local, transformations=standard_transformations)
        exprs.append(sym.sympify(e).subs(subs))
    if len(exprs) == 2:
        exprs.append(-X[2])
    if len(exprs) != 3:
        raise ValueError("give two equations (x, y) or three (x, y, z)")
    free = set().union(*(e.free_symbols for e in exprs)) - set(X)
    if free:
        raise ValueError(f"no value for {', '.join(sorted(map(str, free)))}")
    planar = len(equations) == 2
    return Flow(name=name, exprs=exprs, planar=planar,
                seeds=[(onp.zeros(3), extent if not planar else extent)])


def bracket_flow(f: Flow, g: Flow) -> Flow:
    """[f, g] = (df/dx) g - (dg/dx) f, in closed form via autolie.symbolics.codegen."""
    from autolie.symbolics.codegen import bracket

    return Flow(
        name=f"[{f.name}, {g.name}]",
        exprs=bracket(f.exprs, g.exprs, X),
        description=(f"The Lie bracket of {f.name} and {g.name}: how far their flows fail to "
                     "commute. Follow f then g for a moment, then each backwards, and the "
                     "miss is proportional to [f, g]."),
        seeds=f.seeds + g.seeds,
        planar=f.planar and g.planar,
    )


def check_against_autolie(flow: Flow, points, parts: Optional[tuple[Flow, Flow]] = None, rtol=1e-6):
    """Worst relative disagreement between the closed form and autoLie at `points`.

    For a bracket, pass `parts=(f, g)` and the comparison is against
    lie_lib.bracket_field on the two autograd fields, i.e. an independent
    computation of the bracket; otherwise against the field's own autograd copy.
    """
    import autograd.numpy as anp
    from autolie.lie_lib import bracket_field

    ref = bracket_field(parts[0].autograd(), parts[1].autograd()) if parts else flow.autograd()
    vec = flow.numeric()
    worst = 0.0
    for p in points:
        a = onp.asarray(ref(anp.array(p, dtype=float), None), dtype=float)
        b = vec(onp.asarray(p, dtype=float).reshape(3, 1))[:, 0]
        worst = max(worst, onp.linalg.norm(a - b) / max(onp.linalg.norm(a), 1e-12))
    return worst <= rtol, worst


# --- screening ----------------------------------------------------------------------

def _sphere(n, rng):
    v = rng.normal(size=(3, n))
    v /= onp.linalg.norm(v, axis=0)
    return v * rng.uniform(size=n) ** (1 / 3)


def _rk4(V, P, dt):
    k1 = V(P)
    k2 = V(P + 0.5 * dt * k1)
    k3 = V(P + 0.5 * dt * k2)
    k4 = V(P + dt * k3)
    return P + dt / 6 * (k1 + 2 * (k2 + k3) + k4)


def _spread(P):
    lo, hi = onp.percentile(P, [5, 95], axis=1)
    return float(onp.max(hi - lo))


def screen(V, origin, scale, rng, steps=4000):
    """Run a field the way the shader will and measure what it settles into.

    Returns None if it escapes or every particle stalls at a fixed point.
    """
    with onp.errstate(all="ignore"):
        P = origin[:, None] + scale * _sphere(PARTICLES, rng)
        speed = onp.median(onp.linalg.norm(V(P), axis=0))
        if not (speed > 0 and onp.isfinite(speed)):
            return None
        dt = PACE * max(scale, 1e-3) / speed
        for _ in range(8):
            for _ in range(steps // 8):
                P = _rk4(V, P, dt)
            alive = onp.all(onp.isfinite(P), axis=0) & (onp.linalg.norm(P, axis=0) < ESCAPE)
            if alive.mean() < 0.5:
                return None
            Q = P[:, alive]
            ext, sp = _spread(Q), onp.median(onp.linalg.norm(V(Q), axis=0))
            if not (ext > 1e-6 and sp > 1e-12):
                return None
            dt = float(onp.clip(PACE * ext / sp, dt / 4, dt * 4))

        P = P[:, alive]
        T = P + 1e-7 * ext * _sphere(P.shape[1], rng)
        trail, logs = [], 0.0
        for i in range(steps):
            P, T = _rk4(V, P, dt), _rk4(V, T, dt)
            if i % 10 == 9:
                d, d0 = onp.linalg.norm(T - P, axis=0), 1e-7 * ext
                ok = d > 0
                logs += onp.mean(onp.log(d[ok] / d0)) if ok.any() else 0.0
                T = P + (T - P) * (d0 / onp.where(ok, d, 1))
            if i % 4 == 0:
                trail.append(P.copy())
        trail = onp.stack(trail)
        final = onp.all(onp.isfinite(P), axis=0) & (onp.linalg.norm(P, axis=0) < ESCAPE)
        if final.mean() < 0.8:
            return None
        trail = trail[:, :, final]
        cloud = trail.transpose(1, 0, 2).reshape(3, -1)
        ext = _spread(cloud)
        if not ext > 1e-6:
            return None
        if (onp.std(trail, axis=0).max(axis=0) > 0.02 * ext).mean() < 0.5:
            return None
        sp = onp.median(onp.linalg.norm(V(cloud), axis=0))
        lo, hi = onp.percentile(cloud, [2, 98], axis=1)
        return {
            "dt": float(dt),
            "lyapunov": float(logs / (steps * dt) * (ext / sp)),
            "center": ((lo + hi) / 2).tolist(),
            "size": float(onp.max(hi - lo)),
            "origin": [float(v) for v in origin],
            "scale": float(scale),
            "samples": cloud[:, rng.choice(cloud.shape[1], 12, replace=False)].T,
            "bounded": True,
        }


def _fallback(V, origin, scale, rng):
    """No bounded, moving attractor from any seed: show the flow near the seed anyway."""
    P = origin[:, None] + scale * _sphere(PARTICLES, rng)
    with onp.errstate(all="ignore"):
        speed = onp.median(onp.linalg.norm(V(P), axis=0))
    if not (speed > 0 and onp.isfinite(speed)):
        speed = 1.0
    return {
        "dt": float(PACE * scale / speed),
        "lyapunov": float("nan"),
        "center": origin.tolist(),
        "size": 4 * scale,
        "origin": origin.tolist(),
        "scale": scale,
        "samples": P[:, :12].T,
        "bounded": False,
        "respawn": 4 * scale,
    }


# --- into the renderer -------------------------------------------------------------

def to_attractor(flow: Flow, parts: Optional[tuple[Flow, Flow]] = None, seed: int = 0) -> dict:
    """The flow in the renderer's attractor format, screened and checked.

    `parts=(f, g)` marks a bracket, so the GLSL is checked against autoLie's
    bracket of the two rather than against itself.
    """
    from autolie.symbolics.codegen import glsl

    rng = onp.random.default_rng(seed)
    V = flow.numeric()
    runs = [r for o, s in flow.seeds if (r := screen(V, onp.asarray(o, float), float(s), rng))]
    run = max(runs, key=lambda r: r["lyapunov"]) if runs else _fallback(
        V, onp.asarray(flow.seeds[0][0], float), float(flow.seeds[0][1]), rng)
    ok, err = check_against_autolie(flow, run["samples"], parts)
    if not ok:
        raise ValueError(f"{flow.name}: closed form disagrees with autoLie (relative error {err:.2e})")

    lyap = run["lyapunov"]
    if not run["bounded"]:
        behaviour = ("No bounded, moving attractor from any seed tried: particles escape or stall. "
                     "What you see is the flow near the seed, re-seeded when a particle runs off.")
    elif lyap > 0.02:
        behaviour = f"Chaotic: nearby trajectories separate by about e^{lyap:.2f} per crossing."
    else:
        behaviour = "Regular: trajectories settle onto closed or quasi-periodic loops."

    entry = {
        "name": flow.name,
        "deriv": ",\n".join("    " + glsl(e) for e in flow.exprs),
        "params": {},
        "dt": float(f"{run['dt']:.4g}"),
        "initScale": float(f"{run['scale']:.4g}"),
        "initOrigin": [float(f"{v:.4g}") for v in run["origin"]],
        "view": {
            "center": [float(f"{v:.4g}") for v in run["center"]],
            "size": [float(f"{run['size']:.4g}")] * 3,
            "rotation": [1.5707963268, 0, 0],
        },
        "history": {"origin": flow.name, "text": [t for t in (flow.description, behaviour) if t]},
        "vizdyn": {
            "equations": flow.equations,
            "bounded": run["bounded"],
            "lyapunov": None if math.isnan(lyap) else float(f"{lyap:.4g}"),
            "autolieError": float(f"{err:.2e}"),
            "planar": flow.planar,
        },
    }
    if run.get("respawn"):
        entry["respawn"] = float(f"{run['respawn']:.4g}")
    return entry


LOOKS = {
    # The screensaver's own rendering on black.
    "dark": {"theme": "dark"},
    # Part 1 of Reusser's notebook: thin flat strokes on white, white rims.
    "notebook": {
        "theme": "paper", "palette": "Reusser", "particles": 160, "steps": 140,
        "trail": {"width": 4, "fade": 0, "tailFade": 0, "colorBy": 1, "shading": 0,
                  "borderOpacity": 1, "borderWidth": 0.9, "gridOpacity": 0, "shadow": False},
    },
}


def render_html(systems: Sequence[dict], pinned: Optional[str] = None, look: str = "dark",
                flat: Optional[str] = None, **overrides) -> str:
    """The viewer page with `systems` loaded, ready for st.components.v1.html.

    `flat` opens on a 2-D plot of that plane ('xy', 'xz', 'yz') instead of 3-D.
    """
    config = {**LOOKS.get(look, {}), **overrides,
              "systems": list(systems), "pinned": pinned or (systems[0]["name"] if systems else None),
              "flat": flat}
    page = VIEWER.read_text()
    marker = "window.VIZDYN_CONFIG = /*CONFIG*/null;"
    if marker not in page:
        raise RuntimeError("viewer.html has no config marker; rebuild it with node web/build.mjs")
    # </script> inside a string would end the script element early.
    blob = json.dumps(config).replace("</", "<\\/")
    return page.replace(marker, f"window.VIZDYN_CONFIG = {blob};")


def system_context(entry: dict) -> str:
    """A compact description of a rendered system, for the chat agent's prompt."""
    v = entry.get("vizdyn", {})
    parts = [f"3-D view: {entry['name']}"]
    parts += v.get("equations", [])
    if v.get("bounded") is False:
        parts.append("no bounded attractor (particles escape or stall)")
    elif v.get("lyapunov") is not None:
        parts.append(f"largest Lyapunov exponent ~ {v['lyapunov']} per attractor crossing")
    return " | ".join(parts)
