"""GUI components and interaction handling for the flow field application."""

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from typing import Callable, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vizdyn.systems import get_system, DynamicalSystem
from vizdyn.analysis import compute_trajectory, compute_trajectory_dynamics
from vizdyn.visualization import add_arrow
from vizdyn.equation_parser import EquationParser


class GUIManager:
    """Manages GUI widgets and their callbacks."""

    def __init__(
        self,
        fig,
        initial_params: dict,
        axcolor: str = 'lightgoldenrodyellow',
        y_offset: float = 0.20,
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
        y_offset : float
            Vertical offset applied to all widget positions (figure fraction).
            Use this to push controls above a bottom panel (e.g. the chat strip).
        """
        self.fig = fig
        self.axcolor = axcolor
        self.initial_params = initial_params
        self.y_offset = y_offset

        # Create widgets
        self._create_sliders()
        self._create_buttons()
        self._create_radio_buttons()
        self._create_equation_inputs()

    def _create_sliders(self):
        """Create parameter sliders."""
        slider_left = 0.05
        slider_width = 0.35
        yo = self.y_offset

        # W factor slider
        axw = plt.axes([slider_left, yo + 0.01, slider_width, 0.03], facecolor=self.axcolor)
        self.sw = Slider(axw, 'W factor', 0, 1.0, valinit=self.initial_params['w'])

        # Mu (bifurcation parameter) slider
        axamp = plt.axes([slider_left, yo + 0.05, slider_width, 0.03], facecolor=self.axcolor)
        self.samp = Slider(axamp, 'Mu', -10, 8, valinit=self.initial_params['mu'])

        # Frequency slider
        axfreq = plt.axes([slider_left, yo + 0.09, slider_width, 0.03], facecolor=self.axcolor)
        self.sfreq = Slider(axfreq, 'CFreq', 0, 15.0, valinit=self.initial_params['cfreq'])

    def _create_buttons(self):
        """Create control buttons."""
        yo = self.y_offset
        resetax = plt.axes([0.05, yo + 0.13, 0.1, 0.03])
        self.button = Button(resetax, 'Reset', color=self.axcolor, hovercolor='0.975')

    def _create_radio_buttons(self):
        """Create system selection radio buttons."""
        rax = plt.axes([0.025, 0.66, 0.10, 0.12], facecolor=self.axcolor)
        self.radio = RadioButtons(
            rax,
            ('Hopf', 'VDPol', 'SN', 'global', 'Custom'),
            active=0
        )

    def _create_equation_inputs(self):
        """Create text input boxes for custom equations."""
        eq_left = 0.50
        eq_width = 0.45
        yo = self.y_offset

        # dx/dt equation input
        ax_dx = plt.axes([eq_left, yo + 0.09, eq_width, 0.03], facecolor=self.axcolor)
        self.text_dx = TextBox(
            ax_dx, 'dx/dt = ',
            initial='win * (mu * x - y - x * (x**2 + y**2))',
            label_pad=0.01
        )

        # dy/dt equation input
        ax_dy = plt.axes([eq_left, yo + 0.05, eq_width, 0.03], facecolor=self.axcolor)
        self.text_dy = TextBox(
            ax_dy, 'dy/dt = ',
            initial='(1 - win) * (x + mu * y - y * (x**2 + y**2))',
            label_pad=0.01
        )

        # Apply button
        ax_apply = plt.axes([eq_left, yo + 0.13, 0.1, 0.03])
        self.apply_button = Button(ax_apply, 'Apply', color=self.axcolor, hovercolor='0.975')

    def connect_callbacks(
        self,
        update_callback: Callable,
        reset_callback: Callable,
        system_callback: Callable,
        apply_custom_callback: Optional[Callable] = None
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
        apply_custom_callback : callable, optional
            Callback for applying custom equations
        """
        self.sfreq.on_changed(update_callback)
        self.samp.on_changed(update_callback)
        self.sw.on_changed(update_callback)
        self.button.on_clicked(reset_callback)
        self.radio.on_clicked(system_callback)

        if apply_custom_callback:
            self.apply_button.on_clicked(apply_custom_callback)

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

    def get_custom_equations(self) -> Tuple[str, str]:
        """
        Get the current custom equation strings.

        Returns
        -------
        tuple
            (dx_dt_equation, dy_dt_equation)
        """
        return (self.text_dx.text, self.text_dy.text)

    def set_custom_equations(self, dx_eq: str, dy_eq: str):
        """
        Set the custom equation text boxes.

        Parameters
        ----------
        dx_eq : str
            Equation for dx/dt
        dy_eq : str
            Equation for dy/dt
        """
        self.text_dx.set_val(dx_eq)
        self.text_dy.set_val(dy_eq)


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

        # Custom system state
        self.custom_system = None
        self.custom_dx_eq = "mu * x - y"
        self.custom_dy_eq = "x + mu * y"

    def get_current_system(self) -> DynamicalSystem:
        """
        Get the current dynamical system.

        Returns
        -------
        DynamicalSystem
            Current system instance
        """
        if self.system_type == 'Custom' and self.custom_system is not None:
            # Update parameters on custom system
            self.custom_system.mu = self.mu
            self.custom_system.fc = self.cfreq
            self.custom_system.win = self.w
            return self.custom_system
        elif self.system_type == 'Custom':
            # Create default custom system if not set
            return self.create_custom_system(self.custom_dx_eq, self.custom_dy_eq)
        else:
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

    def create_custom_system(self, dx_eq: str, dy_eq: str) -> DynamicalSystem:
        """
        Create a custom system from equations.

        Parameters
        ----------
        dx_eq : str
            Equation for dx/dt
        dy_eq : str
            Equation for dy/dt

        Returns
        -------
        DynamicalSystem
            Custom system instance
        """
        parser = EquationParser()
        self.custom_dx_eq = dx_eq
        self.custom_dy_eq = dy_eq

        try:
            self.custom_system = parser.create_system_from_equations(
                dx_eq,
                dy_eq,
                mu=self.mu,
                fc=self.cfreq,
                win=self.w
            )
            return self.custom_system
        except Exception as e:
            print(f"Error creating custom system: {e}")
            # Fall back to Hopf system
            self.system_type = 'Hopf'
            return get_system('Hopf', mu=self.mu, fc=self.cfreq, win=self.w)

    def get_custom_equations(self) -> Tuple[str, str]:
        """
        Get the current custom equations.

        Returns
        -------
        tuple
            (dx_dt_equation, dy_dt_equation)
        """
        return (self.custom_dx_eq, self.custom_dy_eq)

    def get_system_equations(self, system_type: str) -> Tuple[str, str]:  # noqa: E501
        """
        Get the equation strings for a given system type.

        Parameters
        ----------
        system_type : str
            The system type name

        Returns
        -------
        tuple
            (dx_dt_equation, dy_dt_equation) as strings
        """
        # System equations including the fc scaling factor
        equations = {
            'Hopf': (
                'fc * win * (mu * x - y - x * (x**2 + y**2))',
                'fc * (1 - win) * (x + mu * y - y * (x**2 + y**2))'
            ),
            'VDPol': (
                'fc * (win * mu * x - y**2 * x - y)',
                'fc * x'
            ),
            'SN': (
                'fc * (mu - x**2)',
                'fc * (-y)'
            ),
            'global': (
                'fc * y',
                'fc * (mu * y + x - x**2 + x * y)'
            ),
            'Custom': (
                self.custom_dx_eq,
                self.custom_dy_eq
            )
        }
        return equations.get(system_type, (self.custom_dx_eq, self.custom_dy_eq))


# ---------------------------------------------------------------------------
# Chat panel
# ---------------------------------------------------------------------------

class ChatManager:
    """
    Manages the chat panel at the bottom of the figure.

    Layout (figure-normalised coordinates, y in [0, CHAT_TOP]):
        - History display area: a borderless axes with a Text object
        - Input TextBox: single-line text entry
        - Send Button: submits the query to the agent
        - Clear Button: wipes the conversation display

    The caller owns the agent and provides a ``context_fn`` callable that
    returns a current-state string on each query.
    """

    # Height of the entire chat strip (figure fraction)
    CHAT_HEIGHT = 0.18
    # Top of chat strip (figure fraction) – must match _setup_figure adjustment
    CHAT_TOP = 0.18

    # Max lines to keep visible in the history display
    MAX_DISPLAY_LINES = 20
    # Wrap width in characters for display
    WRAP_WIDTH = 100

    def __init__(
        self,
        fig,
        agent,
        context_fn: Callable[[], str],
        axcolor: str = 'lightgoldenrodyellow',
    ):
        """
        Parameters
        ----------
        fig : matplotlib.figure.Figure
        agent : DynSysAgent  (or None for offline mode)
        context_fn : callable() -> str
            Returns a string describing current system state.
        axcolor : str
            Background colour for widget axes.
        """
        self.fig = fig
        self.agent = agent
        self.context_fn = context_fn
        self.axcolor = axcolor

        self._display_lines: list[str] = []

        self._create_widgets()
        self._append_line("[System] Chat ready. Ask anything about the dynamics.")

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _create_widgets(self):
        """Create all chat-panel axes and widgets."""
        # --- history display axes (left 75 % of strip, top 70 % of strip) ---
        hist_left   = 0.01
        hist_bottom = 0.06
        hist_width  = 0.74
        hist_height = 0.10
        self._ax_hist = self.fig.add_axes(
            [hist_left, hist_bottom, hist_width, hist_height],
            facecolor='#f5f5f5',
        )
        self._ax_hist.set_xticks([])
        self._ax_hist.set_yticks([])
        self._ax_hist.set_navigate(False)
        for spine in self._ax_hist.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor('#cccccc')

        self._hist_text = self._ax_hist.text(
            0.005, 0.97, "",
            transform=self._ax_hist.transAxes,
            fontsize=7,
            verticalalignment='top',
            horizontalalignment='left',
            family='monospace',
            wrap=False,
            clip_on=True,
        )

        # --- input TextBox ---
        ax_input = self.fig.add_axes(
            [hist_left, 0.02, hist_width - 0.01, 0.03],
            facecolor=self.axcolor,
        )
        self.text_input = TextBox(ax_input, 'Ask: ', initial='', label_pad=0.01)

        # --- Send button ---
        ax_send = self.fig.add_axes([0.76, 0.02, 0.08, 0.03])
        self.btn_send = Button(
            ax_send, 'Send',
            color=self.axcolor, hovercolor='0.85'
        )

        # --- Clear button ---
        ax_clear = self.fig.add_axes([0.85, 0.02, 0.08, 0.03])
        self.btn_clear = Button(
            ax_clear, 'Clear',
            color=self.axcolor, hovercolor='0.85'
        )

        # Wire callbacks
        self.btn_send.on_clicked(self._on_send)
        self.btn_clear.on_clicked(self._on_clear)
        # Also allow Enter key in the TextBox to send
        self.text_input.on_submit(self._on_submit_text)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_send(self, event):
        """Send button clicked."""
        self._submit(self.text_input.text)

    def _on_submit_text(self, text: str):
        """TextBox submitted via Enter key."""
        self._submit(text)

    def _on_clear(self, event):
        """Clear button clicked."""
        self._display_lines.clear()
        if self.agent is not None:
            self.agent.reset_history()
        self._refresh_display()
        self._append_line("[System] Conversation cleared.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _submit(self, message: str):
        """Process a user message: display it, query agent, display reply."""
        message = message.strip()
        if not message:
            return

        # Echo user message
        self._append_line(f"You: {message}")

        # Clear input box
        self.text_input.set_val("")

        # Query agent (or show placeholder if unavailable)
        if self.agent is not None and self.agent.available:
            context = self.context_fn()
            self._append_line("[Agent thinking...]")
            self._refresh_display()
            self.fig.canvas.draw_idle()

            response = self.agent.chat(message, context)

            # Replace the "thinking" line with the real response
            if self._display_lines and self._display_lines[-1] == "[Agent thinking...]":
                self._display_lines.pop()
            for line in self._wrap_text(f"Agent: {response}"):
                self._display_lines.append(line)
        else:
            self._append_line("Agent: [unavailable – set GEMINI_KEY in .env]")

        self._trim_history()
        self._refresh_display()
        self.fig.canvas.draw_idle()

    def _append_line(self, line: str):
        """Append wrapped lines to the display buffer."""
        for wrapped in self._wrap_text(line):
            self._display_lines.append(wrapped)
        self._trim_history()
        self._refresh_display()

    def _wrap_text(self, text: str) -> list[str]:
        """Wrap a single message into display-width lines."""
        return textwrap.wrap(text, width=self.WRAP_WIDTH) or [text]

    def _trim_history(self):
        """Keep only the most recent MAX_DISPLAY_LINES lines."""
        if len(self._display_lines) > self.MAX_DISPLAY_LINES:
            self._display_lines = self._display_lines[-self.MAX_DISPLAY_LINES:]

    def _refresh_display(self):
        """Update the history text object."""
        self._hist_text.set_text("\n".join(self._display_lines))
