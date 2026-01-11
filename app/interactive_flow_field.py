#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Flow Field Visualization

Created on Wed Dec 28 14:00:03 2016
@author: virati
Inspired/based on code from: http://matplotlib.org/examples/widgets/slider_demo.html

Refactored to separate concerns:
- plotting.py: All visualization logic
- gui_components.py: GUI widgets and event handling
- interactive_flow_field.py: Main application orchestration
"""

import matplotlib.pyplot as plt

from vizdyn.frontend.plotting import FlowFieldPlotter
from vizdyn.frontend.gui_components import GUIManager, EventHandler, StateManager


class InteractiveFFGui:
    """
    Interactive visualization of dynamical systems flow fields.

    This is the main orchestrator that coordinates between:
    - StateManager: Application state
    - FlowFieldPlotter: Visualization logic
    - GUIManager: GUI widgets
    - EventHandler: User interactions
    """

    def __init__(self):
        """Initialize the interactive flow field application."""
        # Configuration
        self.mesh_lim = 3
        self.mesh_res = 50

        # Initialize components
        self.state = StateManager(system_type="Hopf", mu=0, cfreq=3, w=0.5, cx=4, cy=-2)

        self.plotter = FlowFieldPlotter(mesh_lim=self.mesh_lim, mesh_res=self.mesh_res)

        # Setup figure and axes
        self._setup_figure()

        # Initialize GUI components
        self.gui = GUIManager(
            self.fig,
            initial_params={
                "mu": self.state.mu,
                "cfreq": self.state.cfreq,
                "w": self.state.w,
            },
        )

        # Initialize event handler
        self.event_handler = EventHandler(
            self.main_ax, self.tser_ax, self.plotter, mesh_lim=self.mesh_lim
        )

        # Create initial plots
        self._initialize_plots()

        # Connect all callbacks
        self._connect_callbacks()

    def _setup_figure(self):
        """Setup the figure and axes layout."""
        self.fig = plt.figure()

        # Time series plot
        self.tser_ax = plt.axes([0.05, 0.25, 0.90, 0.20], facecolor="white")

        # Phase space 3D plot
        self.phslice_ax = plt.axes(
            [0.5, 0.50, 0.45, 0.45], facecolor="white", projection="3d"
        )

        # Main 2D phase plot
        self.main_ax = plt.axes([0.05, 0.50, 0.45, 0.45], facecolor="white")

    def _initialize_plots(self):
        """Create initial plots using the plotter."""
        system = self.state.get_current_system()
        initial_state = self.state.get_initial_state()

        # Setup 2D phase plot
        self.vector_field, self.traj_scatter, self.start_marker = (
            self.plotter.setup_2d_phase_plot(self.main_ax, system, initial_state)
        )

        # Setup phase surface
        self.plotter.update_phase_surface(self.phslice_ax, system)

        # Plot time series
        self.plotter.plot_timeseries(self.tser_ax, system, initial_state)

    def _connect_callbacks(self):
        """Connect all GUI callbacks and event handlers."""
        # Connect GUI widget callbacks
        self.gui.connect_callbacks(
            update_callback=self.on_parameter_update,
            reset_callback=self.on_reset,
            system_callback=self.on_system_change,
        )

        # Connect mouse events
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_click)

    def on_parameter_update(self, val):
        """
        Callback for parameter slider updates.

        Parameters
        ----------
        val : float
            Slider value (not used, we query all sliders)
        """
        # Get current parameters from GUI
        params = self.gui.get_parameter_values()

        # Update state
        self.state.update_parameters(params["mu"], params["cfreq"], params["w"])

        print(self.state.update_counter)

        # Get updated system
        system = self.state.get_current_system()
        initial_state = self.state.get_initial_state()

        # Update all plots
        self.traj_scatter, self.start_marker = self.plotter.update_all_plots(
            self.main_ax,
            self.tser_ax,
            self.phslice_ax,
            system,
            initial_state,
            self.vector_field,
            self.traj_scatter,
            self.start_marker,
        )

        # Redraw
        plt.draw()
        self.fig.canvas.draw_idle()

    def on_reset(self, event):
        """
        Callback for reset button.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            Button click event
        """
        self.gui.reset_sliders()

    def on_system_change(self, label: str):
        """
        Callback for system type radio button.

        Parameters
        ----------
        label : str
            Selected system type
        """
        self.state.set_system_type(label)
        self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        """
        Callback for mouse click events.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event
        """
        # Only handle clicks in the main axes
        if event.inaxes != self.main_ax:
            return

        system = self.state.get_current_system()

        if event.button == 1:  # Left click - new initial condition
            self.traj_scatter, self.start_marker, new_state = (
                self.event_handler.handle_left_click(
                    event.xdata,
                    event.ydata,
                    system,
                    self.traj_scatter,
                    self.start_marker,
                )
            )

            # Update state
            self.state.update_initial_state(new_state[0], new_state[1])

            plt.draw()
            self.fig.canvas.draw_idle()

        elif event.button == 3:  # Right click - manual trajectory
            self.event_handler.handle_right_click(
                event.xdata,
                event.ydata,
                system,
                self.state.mu,
                self.state.cfreq,
                self.state.w,
            )

            plt.draw()
            self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive plot."""
        plt.show()

    # ========== Compatibility properties for backward compatibility ==========
    # These properties maintain the old API for existing tests

    @property
    def system_type(self):
        """Get current system type (compatibility property)."""
        return self.state.system_type

    @property
    def mu(self):
        """Get current mu parameter (compatibility property)."""
        return self.state.mu

    @property
    def cfreq(self):
        """Get current cfreq parameter (compatibility property)."""
        return self.state.cfreq

    @property
    def w(self):
        """Get current w parameter (compatibility property)."""
        return self.state.w

    @property
    def cx(self):
        """Get current x initial state (compatibility property)."""
        return self.state.cx

    @property
    def cy(self):
        """Get current y initial state (compatibility property)."""
        return self.state.cy

    @property
    def tidx(self):
        """Get update counter (compatibility property)."""
        return self.state.update_counter

    @property
    def trajectory_points(self):
        """Get manual trajectory points (compatibility property)."""
        return self.event_handler.manual_trajectory_points

    @property
    def sfreq(self):
        """Get frequency slider (compatibility property)."""
        return self.gui.sfreq

    @property
    def samp(self):
        """Get amplitude slider (compatibility property)."""
        return self.gui.samp

    @property
    def sw(self):
        """Get w slider (compatibility property)."""
        return self.gui.sw

    @property
    def button(self):
        """Get reset button (compatibility property)."""
        return self.gui.button

    @property
    def radio(self):
        """Get radio buttons (compatibility property)."""
        return self.gui.radio

    def update(self, val):
        """Update method (compatibility wrapper)."""
        self.on_parameter_update(val)

    def reset(self, event):
        """Reset method (compatibility wrapper)."""
        self.on_reset(event)

    def set_system_type(self, label: str):
        """Set system type method (compatibility wrapper)."""
        self.on_system_change(label)

    def on_click(self, event):
        """Click handler method (compatibility wrapper)."""
        self.on_mouse_click(event)


def main():
    """Main entry point for the application."""
    app = InteractiveFFGui()
    app.show()


if __name__ == "__main__":
    main()
