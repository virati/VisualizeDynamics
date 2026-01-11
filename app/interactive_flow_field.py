#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Flow Field Visualization

Created on Wed Dec 28 14:00:03 2016
@author: virati
Inspired/based on code from: http://matplotlib.org/examples/widgets/slider_demo.html
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from vizdyn.systems import get_system
from vizdyn.analysis import (
    compute_trajectory,
    compute_flow_field,
    compute_field_magnitude,
    compute_trajectory_dynamics,
)
from vizdyn.visualization import (
    add_arrow,
    plot_vector_field,
    plot_trajectory,
    plot_start_marker,
    plot_phase_surface,
    plot_contour_projections,
)


class InteractiveFlowField:
    """Interactive visualization of dynamical systems flow fields."""

    def __init__(self):
        # Configuration
        self.mesh_lim = 3
        self.mesh_res = 50
        self.system_type = 'Hopf'

        # Initial parameters
        self.mu = 0
        self.cfreq = 3
        self.w = 0.5

        # Initial state
        self.cx = 4
        self.cy = -2

        # Trajectory tracking
        self.tidx = 0
        self.trajectory_points = []

        # Create figure and axes
        self._setup_figure()

        # Create initial plots
        self._initialize_plots()

        # Setup GUI controls
        self._setup_controls()

        # Connect events
        self._connect_events()

    def _setup_figure(self):
        """Setup the figure and axes layout."""
        self.fig = plt.figure()

        # Time series plot
        self.tser_ax = plt.axes([0.05, 0.25, 0.90, 0.20], facecolor='white')

        # Phase space 3D plot
        self.phslice_ax = plt.axes([0.5, 0.50, 0.45, 0.45],
                                    facecolor='white', projection='3d')

        # Main 2D phase plot
        self.main_ax = plt.axes([0.05, 0.50, 0.45, 0.45], facecolor='white')

    def _initialize_plots(self):
        """Create initial plots."""
        # Get current system
        system = get_system(self.system_type, mu=self.mu, fc=self.cfreq, win=self.w)

        # Compute flow field
        X, Y, Z = compute_flow_field(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=self.mesh_res,
            normalize=True
        )

        # Plot vector field
        self.vector_field = plot_vector_field(self.main_ax, X, Y, Z)
        self.main_ax.axhline(y=0, color='r')
        self.main_ax.axis([-self.mesh_lim, self.mesh_lim,
                           -self.mesh_lim, self.mesh_lim])

        # Compute and plot trajectory
        t, traj = compute_trajectory(system, (self.cx, self.cy))
        self.traj_scatter = plot_trajectory(self.main_ax, traj)
        self.start_marker = plot_start_marker(self.main_ax, self.cx, self.cy)

        # Plot phase space surface
        self._update_phase_surface(system)

        # Plot time series
        self.tser_ax.plot(t, traj)

    def _update_phase_surface(self, system):
        """Update the 3D phase space surface plot."""
        self.phslice_ax.cla()

        X2, Y2, Zmag = compute_field_magnitude(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=20
        )

        # Plot surface
        plot_phase_surface(self.phslice_ax, X2, Y2, Zmag)

        # Add plain surface overlay
        self.phslice_ax.plot_surface(X2, Y2, Zmag, alpha=0.2)

        # Plot contour projections
        plot_contour_projections(self.phslice_ax, X2, Y2, Zmag, offset=10)

        # Set limits and styling
        self.phslice_ax.set_zlim((0, 20))

    def _setup_controls(self):
        """Setup GUI controls (sliders, buttons, radio)."""
        axcolor = 'lightgoldenrodyellow'

        # Sliders
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axw = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

        self.sfreq = Slider(axfreq, 'CFreq', 0, 15.0, valinit=self.cfreq)
        self.samp = Slider(axamp, 'Mu', -10, 8, valinit=self.mu)
        self.sw = Slider(axw, 'W factor', 0, 1.0, valinit=self.w)

        # Reset button
        resetax = plt.axes([0, self.mesh_lim, 0, self.mesh_lim])
        self.button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # Radio buttons for system selection
        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('Hopf', 'VDPol', 'SN', 'global'), active=0)

    def _connect_events(self):
        """Connect event handlers."""
        self.sfreq.on_changed(self.update)
        self.samp.on_changed(self.update)
        self.sw.on_changed(self.update)
        self.button.on_clicked(self.reset)
        self.radio.on_clicked(self.set_system_type)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update(self, val):
        """Update callback for sliders."""
        self.tidx += 1
        print(self.tidx)

        # Get current parameter values
        self.mu = self.samp.val
        self.cfreq = self.sfreq.val
        self.w = self.sw.val

        # Get current system
        system = get_system(self.system_type, mu=self.mu, fc=self.cfreq, win=self.w)

        # Update flow field
        X, Y, Z = compute_flow_field(
            system,
            xlim=(-self.mesh_lim, self.mesh_lim),
            ylim=(-self.mesh_lim, self.mesh_lim),
            resolution=self.mesh_res,
            normalize=True
        )

        self.vector_field.set_UVC(Z[0, :], Z[1, :])

        # Update trajectory
        t, traj = compute_trajectory(system, (self.cx, self.cy))

        self.traj_scatter.remove()
        self.traj_scatter = plot_trajectory(self.main_ax, traj)

        self.start_marker.remove()
        self.start_marker = plot_start_marker(self.main_ax, self.cx, self.cy)

        # Update time series
        self.tser_ax.cla()
        self.tser_ax.plot(t, traj)

        # Update phase surface
        self._update_phase_surface(system)
        self.phslice_ax.set_xlim((-2, 2))
        self.phslice_ax.set_ylim((-2, 2))
        self.phslice_ax.set_zlim((0, 20))
        self.phslice_ax.set_axis_off()

        plt.draw()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.main_ax:
            return

        system = get_system(self.system_type, mu=self.mu, fc=self.cfreq, win=self.w)

        if event.button == 1:  # Left click - new trajectory
            self.cx, self.cy = event.xdata, event.ydata

            t, traj = compute_trajectory(system, (self.cx, self.cy))

            self.traj_scatter.remove()
            self.traj_scatter = plot_trajectory(self.main_ax, traj)

            self.tser_ax.cla()
            self.tser_ax.plot(t, traj)

            self.start_marker.remove()
            self.start_marker = plot_start_marker(self.main_ax, self.cx, self.cy)

            plt.draw()
            self.fig.canvas.draw_idle()

        elif event.button == 3:  # Right click - add to manual trajectory
            print('Adding Trajectory Point')
            self.trajectory_points.append([event.xdata, event.ydata])
            traj = np.array(self.trajectory_points)

            # Draw trajectory segments
            for i in range(traj.shape[0] - 1):
                line = self.main_ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color='k')
                add_arrow(line[0])

            self.main_ax.scatter(traj[:, 0], traj[:, 1], s=200)

            plt.draw()
            self.fig.canvas.draw_idle()
            print('trajectory plot')

            # Analyze dynamics along trajectory
            if traj.shape[0] > 1:
                dynamics = compute_trajectory_dynamics(system, traj, n_samples=40)

                self.tser_ax.cla()

                # Plot field along trajectory
                field_flat = dynamics['field'].swapaxes(1, 2).reshape(-1, 2, order='C')
                self.tser_ax.plot(field_flat)

                # Plot alignment
                alignment_flat = dynamics['alignment'].reshape(-1, 1)
                self.tser_ax.plot(alignment_flat, linestyle='--', linewidth=10)

    def reset(self, event):
        """Reset button callback."""
        self.sfreq.reset()
        self.samp.reset()

    def set_system_type(self, label):
        """Radio button callback to change system type."""
        self.system_type = label
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive plot."""
        plt.show()


def main():
    """Main entry point for the application."""
    app = InteractiveFlowField()
    app.show()


if __name__ == '__main__':
    main()
