#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Flow Field Visualization — Streamlit frontend.

Run with:
    uv run streamlit run app/streamlit_app.py

Layout
------
Sidebar        : system selector · parameter sliders · initial-condition inputs
                 · custom equation editor · reset button
Main area left : 2D phase portrait (top) | 3D energy surface (top-right)
                 time series (below)
Main area right: persistent chat panel (full height, scrollable history)
"""

import sys
from pathlib import Path

# Make the src package importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # headless — Streamlit owns the display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vizdyn.systems import get_system
from vizdyn.equation_parser import EquationParser
from vizdyn.analysis import compute_trajectory, compute_flow_field, compute_field_magnitude
from vizdyn.visualization import (
    plot_vector_field,
    plot_trajectory,
    plot_start_marker,
    plot_phase_surface,
    plot_contour_projections,
)
from vizdyn.agent import DynSysAgent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VisualizeDynamics",
    page_icon="🌀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_NAMES = ["Hopf", "VDPol", "SN", "global", "Custom"]
MESH_LIM = 3
MESH_RES = 50

DEFAULT_EQUATIONS = {
    "Hopf": (
        "fc * win * (mu * x - y - x * (x**2 + y**2))",
        "fc * (1 - win) * (x + mu * y - y * (x**2 + y**2))",
    ),
    "VDPol": (
        "fc * (win * mu * x - y**2 * x - y)",
        "fc * x",
    ),
    "SN": (
        "fc * (mu - x**2)",
        "fc * (-y)",
    ),
    "global": (
        "fc * y",
        "fc * (mu * y + x - x**2 + x * y)",
    ),
    "Custom": (
        "mu * x - y",
        "x + mu * y",
    ),
}

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = dict(
        system_type="Hopf",
        mu=0.0,
        cfreq=3.0,
        w=0.5,
        cx=4.0,
        cy=-2.0,
        dx_eq=DEFAULT_EQUATIONS["Hopf"][0],
        dy_eq=DEFAULT_EQUATIONS["Hopf"][1],
        custom_dx_eq=DEFAULT_EQUATIONS["Custom"][0],
        custom_dy_eq=DEFAULT_EQUATIONS["Custom"][1],
        eq_error="",
        chat_history=[],   # list of {"role": "user"|"assistant", "content": str}
        agent=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Agent — lazy singleton
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising agent...")
def _get_agent() -> DynSysAgent:
    return DynSysAgent()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_current_system():
    """Return the currently configured DynamicalSystem."""
    stype = st.session_state.system_type
    mu    = st.session_state.mu
    fc    = st.session_state.cfreq
    win   = st.session_state.w

    if stype == "Custom":
        parser = EquationParser()
        try:
            return parser.create_system_from_equations(
                st.session_state.custom_dx_eq,
                st.session_state.custom_dy_eq,
                mu=mu, fc=fc, win=win,
            )
        except Exception as exc:
            st.session_state.eq_error = str(exc)
            # Fall back to Hopf
            return get_system("Hopf", mu=mu, fc=fc, win=win)
    else:
        st.session_state.eq_error = ""
        return get_system(stype, mu=mu, fc=fc, win=win)


def _system_context() -> str:
    stype = st.session_state.system_type
    dx = st.session_state.dx_eq
    dy = st.session_state.dy_eq
    return (
        f"System={stype} | "
        f"mu={st.session_state.mu:.3f} | "
        f"cfreq={st.session_state.cfreq:.3f} | "
        f"w={st.session_state.w:.3f} | "
        f"initial=({st.session_state.cx:.2f}, {st.session_state.cy:.2f}) | "
        f"dx/dt={dx} | dy/dt={dy}"
    )


# ---------------------------------------------------------------------------
# Plot builders (return matplotlib Figure objects)
# ---------------------------------------------------------------------------

def _build_phase_portrait(system, cx, cy) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    X, Y, Z = compute_flow_field(
        system,
        xlim=(-MESH_LIM, MESH_LIM),
        ylim=(-MESH_LIM, MESH_LIM),
        resolution=MESH_RES,
        normalize=True,
    )
    plot_vector_field(ax, X, Y, Z)
    ax.axhline(y=0, color="r", linewidth=0.8)
    ax.axis([-MESH_LIM, MESH_LIM, -MESH_LIM, MESH_LIM])

    _, traj = compute_trajectory(system, (cx, cy))
    plot_trajectory(ax, traj)
    plot_start_marker(ax, cx, cy)

    ax.set_title("Phase portrait")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def _build_phase_surface(system) -> plt.Figure:
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection="3d")

    X2, Y2, Zmag = compute_field_magnitude(
        system,
        xlim=(-MESH_LIM, MESH_LIM),
        ylim=(-MESH_LIM, MESH_LIM),
        resolution=20,
    )
    plot_phase_surface(ax, X2, Y2, Zmag)
    ax.plot_surface(X2, Y2, Zmag, alpha=0.2)
    plot_contour_projections(ax, X2, Y2, Zmag, offset=10)

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((0, 20))
    ax.set_axis_off()
    ax.set_title("Energy landscape")
    fig.tight_layout()
    return fig


def _build_timeseries(system, cx, cy) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.0))
    t, traj = compute_trajectory(system, (cx, cy))
    ax.plot(t, traj)
    ax.set_xlabel("t")
    ax.set_title("Time series")
    ax.legend(["x(t)", "y(t)"], loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.title("🌀 VisualizeDynamics")

        # ── System selector ──────────────────────────────────────────────
        st.subheader("System")
        new_system = st.selectbox(
            "Type",
            SYSTEM_NAMES,
            index=SYSTEM_NAMES.index(st.session_state.system_type),
            key="_sb_system",
        )
        if new_system != st.session_state.system_type:
            st.session_state.system_type = new_system
            if new_system != "Custom":
                dx, dy = DEFAULT_EQUATIONS[new_system]
                st.session_state.dx_eq = dx
                st.session_state.dy_eq = dy

        # ── Parameters ──────────────────────────────────────────────────
        st.subheader("Parameters")
        st.session_state.mu = st.slider(
            "μ (bifurcation)", -10.0, 8.0, float(st.session_state.mu), 0.1, key="_sb_mu"
        )
        st.session_state.cfreq = st.slider(
            "CFreq", 0.0, 15.0, float(st.session_state.cfreq), 0.1, key="_sb_cfreq"
        )
        st.session_state.w = st.slider(
            "W factor", 0.0, 1.0, float(st.session_state.w), 0.01, key="_sb_w"
        )

        # ── Initial condition ────────────────────────────────────────────
        st.subheader("Initial condition")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.cx = st.number_input(
                "x₀", value=float(st.session_state.cx), step=0.1, key="_sb_cx"
            )
        with col2:
            st.session_state.cy = st.number_input(
                "y₀", value=float(st.session_state.cy), step=0.1, key="_sb_cy"
            )

        # ── Custom equations ─────────────────────────────────────────────
        st.subheader("Equations")
        eq_dx = st.text_input(
            "dx/dt =",
            value=st.session_state.dx_eq,
            key="_sb_dx",
        )
        eq_dy = st.text_input(
            "dy/dt =",
            value=st.session_state.dy_eq,
            key="_sb_dy",
        )

        if st.button("Apply equations", use_container_width=True):
            parser = EquationParser()
            try:
                parser.parse_equation(eq_dx)
                parser.parse_equation(eq_dy)
                st.session_state.custom_dx_eq = eq_dx
                st.session_state.custom_dy_eq = eq_dy
                st.session_state.dx_eq = eq_dx
                st.session_state.dy_eq = eq_dy
                st.session_state.system_type = "Custom"
                st.session_state.eq_error = ""
                st.success("Equations applied.")
            except Exception as exc:
                st.session_state.eq_error = str(exc)
                st.error(f"Parse error: {exc}")

        if st.session_state.eq_error:
            st.error(st.session_state.eq_error)

        # ── Reset ────────────────────────────────────────────────────────
        st.divider()
        if st.button("Reset to defaults", use_container_width=True):
            for k in ("system_type", "mu", "cfreq", "w", "cx", "cy",
                      "dx_eq", "dy_eq", "custom_dx_eq", "custom_dy_eq", "eq_error"):
                del st.session_state[k]
            st.rerun()


# ---------------------------------------------------------------------------
# Chat panel
# ---------------------------------------------------------------------------

def _render_chat():
    """Render the chat panel inside whatever container the caller provides."""
    agent = _get_agent()

    if not agent.available:
        st.warning("Agent unavailable — set `GEMINI_KEY` in `.env`.", icon="⚠️")

    # Scrollable history area
    history_container = st.container(height=520, border=False)
    with history_container:
        if not st.session_state.chat_history:
            st.caption("Ask anything about the current system…")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input pinned below history
    if prompt := st.chat_input("Ask about the dynamics…", key="chat_input"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if agent.available:
            with st.spinner("Thinking…"):
                response = agent.chat(prompt, _system_context())
        else:
            response = "_Agent unavailable. Check `.env` for `GEMINI_KEY`._"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear", use_container_width=True, key="chat_clear"):
            st.session_state.chat_history = []
            _get_agent().reset_history()
            st.rerun()


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def main():
    _render_sidebar()

    system = _get_current_system()
    cx = st.session_state.cx
    cy = st.session_state.cy

    # Two-column layout: plots (left, 60%) | chat (right, 40%)
    col_plots, col_chat = st.columns([3, 2], gap="medium")

    with col_plots:
        # ── Row 1: phase portrait + energy surface side by side ──────────
        sub_phase, sub_surf = st.columns(2, gap="small")

        with sub_phase:
            with st.spinner("Computing phase portrait…"):
                fig_phase = _build_phase_portrait(system, cx, cy)
            st.pyplot(fig_phase, use_container_width=True)
            plt.close(fig_phase)

        with sub_surf:
            with st.spinner("Computing energy landscape…"):
                fig_surf = _build_phase_surface(system)
            st.pyplot(fig_surf, use_container_width=True)
            plt.close(fig_surf)

        # ── Row 2: time series ───────────────────────────────────────────
        with st.spinner("Computing time series…"):
            fig_ts = _build_timeseries(system, cx, cy)
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

    with col_chat:
        st.subheader("Agent chat")
        _render_chat()


if __name__ == "__main__":
    main()
