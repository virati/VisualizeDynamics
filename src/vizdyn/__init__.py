"""VisualizeDynamics - A library for visualizing dynamical systems."""

from .systems import (
    DynamicalSystem,
    HopfSystem,
    SaddleNodeSystem,
    VanDerPolSystem,
    GlobalSystem,
)
from .analysis import (
    compute_trajectory,
    find_critical_points,
    compute_flow_field,
)

__all__ = [
    "DynamicalSystem",
    "HopfSystem",
    "SaddleNodeSystem",
    "VanDerPolSystem",
    "GlobalSystem",
    "compute_trajectory",
    "find_critical_points",
    "compute_flow_field",
]
