"""Analysis tools for dynamical systems."""

import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
from sklearn import preprocessing as pproc
from typing import Tuple, Optional

from .systems import DynamicalSystem


def compute_trajectory(
    system: DynamicalSystem,
    initial_state: Tuple[float, float],
    t_span: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute trajectory for a dynamical system.

    Parameters
    ----------
    system : DynamicalSystem
        The dynamical system to integrate
    initial_state : tuple
        Initial condition (x0, y0)
    t_span : tuple
        Time span (t_start, t_end)
    n_points : int
        Number of time points

    Returns
    -------
    t : np.ndarray
        Time array
    trajectory : np.ndarray
        Trajectory array of shape (n_points, 2)
    """
    t = np.linspace(t_span[0], t_span[1], n_points)
    trajectory = odeint(system, initial_state, t)
    return t, trajectory


def compute_flow_field(
    system: DynamicalSystem,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 50,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the flow field for a dynamical system.

    Parameters
    ----------
    system : DynamicalSystem
        The dynamical system
    xlim : tuple
        x-axis limits
    ylim : tuple
        y-axis limits
    resolution : int
        Grid resolution
    normalize : bool
        Whether to normalize vectors

    Returns
    -------
    X : np.ndarray
        X meshgrid
    Y : np.ndarray
        Y meshgrid
    Z : np.ndarray
        Vector field of shape (2, resolution*resolution)
    """
    xd = np.linspace(xlim[0], xlim[1], resolution)
    yd = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xd, yd)
    XX = np.array([X.ravel(), Y.ravel()])

    Z = np.array(system(XX, []))

    if normalize:
        Z = pproc.normalize(Z.T, norm='l2').T

    return X, Y, Z


def find_critical_points(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find critical points in 1D data.

    Parameters
    ----------
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y values

    Returns
    -------
    critical_indices : np.ndarray
        Indices of critical points
    stability : np.ndarray
        Stability of critical points (sign of derivative)
    """
    critical_indices = sig.argrelextrema(np.abs(y), np.less_equal)[0]

    # Get derivative
    ydiff = np.diff(y)
    stability = np.zeros(shape=critical_indices.shape)

    for idx, i in enumerate(critical_indices):
        # Handle boundary case where critical point is at last index
        if i < len(ydiff):
            stability[idx] = np.sign(ydiff[i])
        else:
            stability[idx] = 0  # Unknown stability at boundary

    return critical_indices, stability


def find_critical_points_2d(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, list]:
    """
    Find critical points in 2D data (simplified version).

    Parameters
    ----------
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y field values

    Returns
    -------
    critical_indices : np.ndarray
        Indices of critical points
    stability : list
        Empty list (stability not computed for 2D)
    """
    critical_indices = sig.argrelextrema(np.abs(y), np.less_equal)[0]
    return critical_indices, []


def compute_field_magnitude(
    system: DynamicalSystem,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the magnitude of the vector field.

    Parameters
    ----------
    system : DynamicalSystem
        The dynamical system
    xlim : tuple
        x-axis limits
    ylim : tuple
        y-axis limits
    resolution : int
        Grid resolution

    Returns
    -------
    X : np.ndarray
        X meshgrid
    Y : np.ndarray
        Y meshgrid
    magnitude : np.ndarray
        Magnitude of vector field
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()])

    Z = np.array(system(XX, []))
    magnitude = np.linalg.norm(Z, axis=0).reshape((X.T.shape[0], Y.T.shape[0]))

    return X, Y, magnitude


def compute_trajectory_dynamics(
    system: DynamicalSystem,
    trajectory_points: np.ndarray,
    n_samples: int = 40
) -> dict:
    """
    Compute dynamics along a manually drawn trajectory.

    Parameters
    ----------
    system : DynamicalSystem
        The dynamical system
    trajectory_points : np.ndarray
        Trajectory points of shape (n_points, 2)
    n_samples : int
        Number of samples along each segment

    Returns
    -------
    dict
        Dictionary containing:
        - 'field': Vector field along trajectory
        - 'difference': Difference from trajectory direction
        - 'alignment': Dot product with trajectory direction
        - 'magnitude': Magnitude of difference
    """
    n_points = trajectory_points.shape[0]

    field_values = []
    differences = []
    alignments = []
    magnitudes = []

    # Overall trajectory direction
    traj_vector = np.array([
        trajectory_points[-1, 0] - trajectory_points[0, 0],
        trajectory_points[-1, 1] - trajectory_points[0, 1]
    ])
    repeated_traj = np.tile(traj_vector.reshape(-1, 1), n_samples)

    for i in range(n_points - 1):
        # Sample along segment
        x_range = np.linspace(trajectory_points[i, 0], trajectory_points[i+1, 0], n_samples)
        y_range = np.linspace(trajectory_points[i, 1], trajectory_points[i+1, 1], n_samples)

        XY = np.vstack((x_range, y_range))

        # Compute field
        Z = system(XY, [])
        field_values.append(Z)

        # Compute differences
        diff = repeated_traj - Z
        differences.append(diff)

        # Compute alignment
        alignment = np.dot(traj_vector, Z)
        alignments.append(alignment)

        # Compute magnitude
        mag = np.linalg.norm(diff)
        magnitudes.append(mag)

    return {
        'field': np.array(field_values),
        'difference': np.array(differences),
        'alignment': np.array(alignments),
        'magnitude': np.array(magnitudes)
    }
