"""Plotting logic for the interactive flow field application."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vizdyn.systems import DynamicalSystem
from vizdyn.analysis import (
    compute_trajectory,
    compute_flow_field,
    compute_field_magnitude,
)
from vizdyn.visualization import (
    plot_vector_field,
    plot_trajectory,
    plot_start_marker,
    plot_phase_surface,
    plot_contour_projections,
)


class FlowFieldPlotter:
    """Handles all plotting operations for flow field visualization."""

    def __init__(self, mesh_lim: float = 3, mesh_res: int = 50):
        """
        Initialize the plotter.

        Parameters
        ----------
        mesh_lim : float
            Spatial limits for the mesh
        mesh_res : int
            Resolution of the flow field mesh
        """
        self.mesh_lim = mesh_lim
        self.mesh_res = mesh_res

    def setup_2d_phase_plot(
        self,
        ax,
        system: DynamicalSystem,
        initial_state: Tuple[float, float]
    ) -> Tuple:
        """
        Setup the 2D phase space plot with vector field and trajectory.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        system : DynamicalSystem
            The dynamical system
        initial_state : tuple
            Initial condition (x, y)

        Returns
        -------
        tuple
            (vector_field, trajectory_scatter, start_marker)
        """
        # Compute and plot flow field
        X, Y, Z = compute_flow_field(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=self.mesh_res,
            normalize=True
        )

        vector_field = plot_vector_field(ax, X, Y, Z)
        ax.axhline(y=0, color='r')
        ax.axis([-self.mesh_lim, self.mesh_lim, -self.mesh_lim, self.mesh_lim])

        # Compute and plot trajectory
        t, traj = compute_trajectory(system, initial_state)
        trajectory_scatter = plot_trajectory(ax, traj)
        start_marker = plot_start_marker(ax, initial_state[0], initial_state[1])

        return vector_field, trajectory_scatter, start_marker

    def update_2d_phase_plot(
        self,
        ax,
        system: DynamicalSystem,
        initial_state: Tuple[float, float],
        vector_field,
        old_trajectory_scatter,
        old_start_marker
    ) -> Tuple:
        """
        Update the 2D phase space plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        system : DynamicalSystem
            The dynamical system
        initial_state : tuple
            Initial condition (x, y)
        vector_field : matplotlib.quiver.Quiver
            Existing vector field to update
        old_trajectory_scatter : matplotlib.collections.PathCollection
            Old trajectory to remove
        old_start_marker : matplotlib.collections.PathCollection
            Old marker to remove

        Returns
        -------
        tuple
            (new_trajectory_scatter, new_start_marker)
        """
        # Update flow field
        X, Y, Z = compute_flow_field(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=self.mesh_res,
            normalize=True
        )
        vector_field.set_UVC(Z[0, :], Z[1, :])

        # Update trajectory
        t, traj = compute_trajectory(system, initial_state)

        old_trajectory_scatter.remove()
        new_trajectory_scatter = plot_trajectory(ax, traj)

        old_start_marker.remove()
        new_start_marker = plot_start_marker(ax, initial_state[0], initial_state[1])

        return new_trajectory_scatter, new_start_marker

    def setup_phase_surface(self, ax) -> None:
        """
        Setup the 3D phase space surface plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes3D
            The 3D axes to plot on
        system : DynamicalSystem
            The dynamical system
        """
        # Will be populated in update_phase_surface
        ax.set_zlim((0, 20))

    def update_phase_surface(
        self,
        ax,
        system: DynamicalSystem
    ) -> None:
        """
        Update the 3D phase space surface plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes3D
            The 3D axes to plot on
        system : DynamicalSystem
            The dynamical system
        """
        ax.cla()

        X2, Y2, Zmag = compute_field_magnitude(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=20
        )

        # Plot surface with gradient coloring
        plot_phase_surface(ax, X2, Y2, Zmag)

        # Add plain surface overlay
        ax.plot_surface(X2, Y2, Zmag, alpha=0.2)

        # Plot contour projections
        plot_contour_projections(ax, X2, Y2, Zmag, offset=10)

        # Set limits and styling
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        ax.set_zlim((0, 20))
        ax.set_axis_off()

    def plot_timeseries(
        self,
        ax,
        system: DynamicalSystem,
        initial_state: Tuple[float, float]
    ) -> None:
        """
        Plot time series of the trajectory.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        system : DynamicalSystem
            The dynamical system
        initial_state : tuple
            Initial condition (x, y)
        """
        ax.cla()
        t, traj = compute_trajectory(system, initial_state)
        ax.plot(t, traj)

    def update_all_plots(
        self,
        main_ax,
        tser_ax,
        phslice_ax,
        system: DynamicalSystem,
        initial_state: Tuple[float, float],
        vector_field,
        old_trajectory_scatter,
        old_start_marker
    ) -> Tuple:
        """
        Update all plots at once.

        Parameters
        ----------
        main_ax : matplotlib.axes.Axes
            Main 2D phase space axes
        tser_ax : matplotlib.axes.Axes
            Time series axes
        phslice_ax : matplotlib.axes.Axes3D
            3D phase surface axes
        system : DynamicalSystem
            The dynamical system
        initial_state : tuple
            Initial condition (x, y)
        vector_field : matplotlib.quiver.Quiver
            Existing vector field
        old_trajectory_scatter : matplotlib.collections.PathCollection
            Old trajectory scatter
        old_start_marker : matplotlib.collections.PathCollection
            Old start marker

        Returns
        -------
        tuple
            (new_trajectory_scatter, new_start_marker)
        """
        # Update 2D phase plot
        new_traj, new_marker = self.update_2d_phase_plot(
            main_ax,
            system,
            initial_state,
            vector_field,
            old_trajectory_scatter,
            old_start_marker
        )

        # Update time series
        self.plot_timeseries(tser_ax, system, initial_state)

        # Update phase surface
        self.update_phase_surface(phslice_ax, system)

        return new_traj, new_marker
