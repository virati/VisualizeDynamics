"""Visualization utilities for dynamical systems."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple


def add_arrow(line, position=None, direction='right', size=50, color=None):
    """
    Add an arrow to a line.

    Helper function from: https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib

    Parameters
    ----------
    line : Line2D object
        The line to add arrow to
    position : float, optional
        x-position of the arrow. If None, mean of xdata is taken
    direction : str
        'left' or 'right'
    size : int
        Size of the arrow in fontsize points
    color : str, optional
        If None, line color is taken
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def plot_vector_field(ax, X, Y, Z, **kwargs):
    """
    Plot a vector field using quiver.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    X : np.ndarray
        X meshgrid
    Y : np.ndarray
        Y meshgrid
    Z : np.ndarray
        Vector field of shape (2, n)
    **kwargs
        Additional arguments to pass to quiver

    Returns
    -------
    quiver : matplotlib.quiver.Quiver
        The quiver plot object
    """
    default_kwargs = {'width': 0.01, 'alpha': 0.4}
    default_kwargs.update(kwargs)

    return ax.quiver(X[:], Y[:], Z[0, :], Z[1, :], **default_kwargs)


def plot_trajectory(ax, trajectory, colormap='rainbow', **kwargs):
    """
    Plot a trajectory with color gradient.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    trajectory : np.ndarray
        Trajectory array of shape (n_points, 2)
    colormap : str
        Name of the colormap to use
    **kwargs
        Additional arguments to pass to scatter

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        The scatter plot object
    """
    n_points = trajectory.shape[0]
    z = np.linspace(0.0, 30.0, n_points)
    traj_cmap = getattr(cm, colormap)(z / 30)

    default_kwargs = {'alpha': 0.8, 's': 20}
    default_kwargs.update(kwargs)

    return ax.scatter(trajectory[:, 0], trajectory[:, 1], color=traj_cmap, **default_kwargs)


def plot_start_marker(ax, x, y, **kwargs):
    """
    Plot a start location marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    x : float
        x coordinate
    y : float
        y coordinate
    **kwargs
        Additional arguments to pass to scatter

    Returns
    -------
    scatter : matplotlib.collections.PathCollection
        The scatter plot object
    """
    default_kwargs = {'color': 'r', 'marker': '>', 's': 300}
    default_kwargs.update(kwargs)

    return ax.scatter(x, y, **default_kwargs)


def plot_phase_surface(ax, X, Y, magnitude, **kwargs):
    """
    Plot a 3D phase surface with gradient coloring.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axes to plot on
    X : np.ndarray
        X meshgrid
    Y : np.ndarray
        Y meshgrid
    magnitude : np.ndarray
        Magnitude of vector field
    **kwargs
        Additional arguments to pass to plot_surface

    Returns
    -------
    surface : mpl_toolkits.mplot3d.art3d.Poly3DCollection
        The surface plot object
    """
    Gx, Gy = np.gradient(magnitude)
    G = (Gx**2 + Gy**2)**0.5
    N = 2 * G / G.max()

    default_kwargs = {'alpha': 0.2, 'facecolors': cm.jet(N)}
    default_kwargs.update(kwargs)

    return ax.plot_surface(X, Y, 1/10 * magnitude, **default_kwargs)


def plot_contour_projections(ax, X, Y, magnitude, offset=10):
    """
    Plot contour projections on 3D plot walls.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axes to plot on
    X : np.ndarray
        X meshgrid
    Y : np.ndarray
        Y meshgrid
    magnitude : np.ndarray
        Magnitude of vector field
    offset : float
        Offset for contour projections

    Returns
    -------
    list
        List of contour plot objects
    """
    contours = []

    # Z projection
    contours.append(ax.contourf(X, Y, magnitude, zdir='z', offset=-offset,
                                 cmap=cm.winter, alpha=0.1))
    # X projection
    contours.append(ax.contourf(X, Y, magnitude, zdir='x', offset=-offset,
                                 cmap=cm.winter, alpha=0.1))
    # Y projection
    contours.append(ax.contourf(X, Y, magnitude, zdir='y', offset=-offset,
                                 cmap=cm.winter, alpha=0.1))

    return contours
