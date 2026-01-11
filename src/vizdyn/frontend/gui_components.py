"""GUI components and interaction handling for the flow field application."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import Callable, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vizdyn.systems import get_system, DynamicalSystem
from vizdyn.analysis import compute_trajectory, compute_trajectory_dynamics
from vizdyn.visualization import add_arrow


class GUIManager:
    """Manages GUI widgets and their callbacks."""

    def __init__(
        self,
        fig,
        initial_params: dict,
        axcolor: str = 'lightgoldenrodyellow'
    ):
        """
        Initialize the GUI manager.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to add widgets to
        initial_params : dict
            Initial parameter values (mu, cfreq, w)
        axcolor : str
            Color for widget backgrounds
        """
        self.fig = fig
        self.axcolor = axcolor
        self.initial_params = initial_params

        # Create widgets
        self._create_sliders()
        self._create_buttons()
        self._create_radio_buttons()

    def _create_sliders(self):
        """Create parameter sliders."""
        # Frequency slider
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=self.axcolor)
        self.sfreq = Slider(
            axfreq,
            'CFreq',
            0,
            15.0,
            valinit=self.initial_params['cfreq']
        )

        # Mu (bifurcation parameter) slider
        axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=self.axcolor)
        self.samp = Slider(
            axamp,
            'Mu',
            -10,
            8,
            valinit=self.initial_params['mu']
        )

        # W factor slider
        axw = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=self.axcolor)
        self.sw = Slider(
            axw,
            'W factor',
            0,
            1.0,
            valinit=self.initial_params['w']
        )

    def _create_buttons(self):
        """Create control buttons."""
        # Note: Reset button position uses mesh_lim which should be passed
        # For now using a default position
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(
            resetax,
            'Reset',
            color=self.axcolor,
            hovercolor='0.975'
        )

    def _create_radio_buttons(self):
        """Create system selection radio buttons."""
        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=self.axcolor)
        self.radio = RadioButtons(
            rax,
            ('Hopf', 'VDPol', 'SN', 'global'),
            active=0
        )

    def connect_callbacks(
        self,
        update_callback: Callable,
        reset_callback: Callable,
        system_callback: Callable
    ):
        """
        Connect callbacks to widgets.

        Parameters
        ----------
        update_callback : callable
            Callback for parameter updates
        reset_callback : callable
            Callback for reset button
        system_callback : callable
            Callback for system type changes
        """
        self.sfreq.on_changed(update_callback)
        self.samp.on_changed(update_callback)
        self.sw.on_changed(update_callback)
        self.button.on_clicked(reset_callback)
        self.radio.on_clicked(system_callback)

    def get_parameter_values(self) -> dict:
        """
        Get current parameter values from sliders.

        Returns
        -------
        dict
            Current parameter values
        """
        return {
            'mu': self.samp.val,
            'cfreq': self.sfreq.val,
            'w': self.sw.val
        }

    def reset_sliders(self):
        """Reset sliders to initial values."""
        self.sfreq.reset()
        self.samp.reset()
        self.sw.reset()


class EventHandler:
    """Handles mouse and keyboard events."""

    def __init__(
        self,
        main_ax,
        tser_ax,
        plotter,
        mesh_lim: float = 3
    ):
        """
        Initialize the event handler.

        Parameters
        ----------
        main_ax : matplotlib.axes.Axes
            Main 2D phase space axes
        tser_ax : matplotlib.axes.Axes
            Time series axes
        plotter : FlowFieldPlotter
            Plotter instance for visualization
        mesh_lim : float
            Spatial limits for the mesh
        """
        self.main_ax = main_ax
        self.tser_ax = tser_ax
        self.plotter = plotter
        self.mesh_lim = mesh_lim

        # State for manual trajectory drawing
        self.manual_trajectory_points = []

    def handle_left_click(
        self,
        x: float,
        y: float,
        system: DynamicalSystem,
        traj_scatter,
        start_marker
    ) -> Tuple:
        """
        Handle left click event (select new initial condition).

        Parameters
        ----------
        x : float
            Click x-coordinate
        y : float
            Click y-coordinate
        system : DynamicalSystem
            Current dynamical system
        traj_scatter : matplotlib.collections.PathCollection
            Current trajectory scatter
        start_marker : matplotlib.collections.PathCollection
            Current start marker

        Returns
        -------
        tuple
            (new_trajectory_scatter, new_start_marker, new_initial_state)
        """
        from vizdyn.analysis import compute_trajectory
        from vizdyn.visualization import plot_trajectory, plot_start_marker

        # Compute new trajectory
        t, traj = compute_trajectory(system, (x, y))

        # Remove old plots
        traj_scatter.remove()
        start_marker.remove()

        # Plot new trajectory
        new_traj_scatter = plot_trajectory(self.main_ax, traj)
        new_start_marker = plot_start_marker(self.main_ax, x, y)

        # Update time series
        self.tser_ax.cla()
        self.tser_ax.plot(t, traj)

        return new_traj_scatter, new_start_marker, (x, y)

    def handle_right_click(
        self,
        x: float,
        y: float,
        system: DynamicalSystem,
        mu: float,
        cfreq: float,
        w: float
    ):
        """
        Handle right click event (add point to manual trajectory).

        Parameters
        ----------
        x : float
            Click x-coordinate
        y : float
            Click y-coordinate
        system : DynamicalSystem
            Current dynamical system
        mu : float
            Current mu parameter
        cfreq : float
            Current frequency parameter
        w : float
            Current w parameter
        """
        print('Adding Trajectory Point')
        self.manual_trajectory_points.append([x, y])
        traj = np.array(self.manual_trajectory_points)

        # Draw trajectory segments with arrows
        for i in range(traj.shape[0] - 1):
            line = self.main_ax.plot(
                traj[i:i+2, 0],
                traj[i:i+2, 1],
                color='k'
            )
            add_arrow(line[0])

        self.main_ax.scatter(traj[:, 0], traj[:, 1], s=200)

        # Analyze dynamics along trajectory if we have enough points
        if traj.shape[0] > 1:
            self._analyze_manual_trajectory(traj, system)

    def _analyze_manual_trajectory(
        self,
        trajectory: np.ndarray,
        system: DynamicalSystem
    ):
        """
        Analyze and plot dynamics along manual trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory points
        system : DynamicalSystem
            Current dynamical system
        """
        dynamics = compute_trajectory_dynamics(system, trajectory, n_samples=40)

        self.tser_ax.cla()

        # Plot field along trajectory
        field_flat = dynamics['field'].swapaxes(1, 2).reshape(-1, 2, order='C')
        self.tser_ax.plot(field_flat)

        # Plot alignment
        alignment_flat = dynamics['alignment'].reshape(-1, 1)
        self.tser_ax.plot(alignment_flat, linestyle='--', linewidth=10)

    def clear_manual_trajectory(self):
        """Clear the manual trajectory points."""
        self.manual_trajectory_points = []


class StateManager:
    """Manages application state."""

    def __init__(
        self,
        system_type: str = 'Hopf',
        mu: float = 0,
        cfreq: float = 3,
        w: float = 0.5,
        cx: float = 4,
        cy: float = -2
    ):
        """
        Initialize state manager.

        Parameters
        ----------
        system_type : str
            Initial system type
        mu : float
            Initial mu parameter
        cfreq : float
            Initial frequency parameter
        w : float
            Initial w parameter
        cx : float
            Initial x position
        cy : float
            Initial y position
        """
        self.system_type = system_type
        self.mu = mu
        self.cfreq = cfreq
        self.w = w
        self.cx = cx
        self.cy = cy
        self.update_counter = 0

    def get_current_system(self) -> DynamicalSystem:
        """
        Get the current dynamical system.

        Returns
        -------
        DynamicalSystem
            Current system instance
        """
        return get_system(
            self.system_type,
            mu=self.mu,
            fc=self.cfreq,
            win=self.w
        )

    def update_parameters(self, mu: float, cfreq: float, w: float):
        """
        Update system parameters.

        Parameters
        ----------
        mu : float
            Bifurcation parameter
        cfreq : float
            Frequency parameter
        w : float
            W factor parameter
        """
        self.mu = mu
        self.cfreq = cfreq
        self.w = w
        self.update_counter += 1

    def update_initial_state(self, cx: float, cy: float):
        """
        Update initial state.

        Parameters
        ----------
        cx : float
            Initial x position
        cy : float
            Initial y position
        """
        self.cx = cx
        self.cy = cy

    def set_system_type(self, system_type: str):
        """
        Set the system type.

        Parameters
        ----------
        system_type : str
            System type name
        """
        self.system_type = system_type

    def get_initial_state(self) -> Tuple[float, float]:
        """
        Get the current initial state.

        Returns
        -------
        tuple
            (cx, cy)
        """
        return (self.cx, self.cy)
