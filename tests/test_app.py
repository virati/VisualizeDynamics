"""Integration tests for the interactive flow field application."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add app and src to path
app_path = Path(__file__).parent.parent / "app"
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(app_path))
sys.path.insert(0, str(src_path))


class TestInteractiveFlowFieldClass:
    """Test the InteractiveFlowField class."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        # Import here to avoid importing at module level
        from interactive_flow_field import InteractiveFlowField

        # Create app without showing
        app = InteractiveFlowField()
        yield app

        # Cleanup
        plt.close(app.fig)

    def test_initialization(self, app):
        """Test app initializes correctly."""
        assert app.mesh_lim == 3
        assert app.mesh_res == 50
        assert app.system_type == 'Hopf'
        assert app.mu == 0
        assert app.cfreq == 3
        assert app.w == 0.5
        assert app.cx == 4
        assert app.cy == -2

    def test_figure_created(self, app):
        """Test that figure is created."""
        assert app.fig is not None
        assert isinstance(app.fig, plt.Figure)

    def test_axes_created(self, app):
        """Test that all axes are created."""
        assert app.tser_ax is not None
        assert app.phslice_ax is not None
        assert app.main_ax is not None

    def test_controls_created(self, app):
        """Test that GUI controls are created."""
        assert app.sfreq is not None
        assert app.samp is not None
        assert app.sw is not None
        assert app.button is not None
        assert app.radio is not None

    def test_initial_plots_created(self, app):
        """Test that initial plots are created."""
        assert app.vector_field is not None
        assert app.traj_scatter is not None
        assert app.start_marker is not None

    def test_trajectory_tracking(self, app):
        """Test trajectory tracking initialization."""
        assert app.tidx == 0
        assert app.trajectory_points == []


class TestAppMethods:
    """Test the app's methods."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_update_method_exists(self, app):
        """Test update method exists."""
        assert hasattr(app, 'update')
        assert callable(app.update)

    def test_update_increments_counter(self, app):
        """Test update increments the counter."""
        initial_tidx = app.tidx
        app.update(None)
        assert app.tidx == initial_tidx + 1

    def test_reset_method_exists(self, app):
        """Test reset method exists."""
        assert hasattr(app, 'reset')
        assert callable(app.reset)

    def test_set_system_type_method_exists(self, app):
        """Test set_system_type method exists."""
        assert hasattr(app, 'set_system_type')
        assert callable(app.set_system_type)

    def test_set_system_type_changes_system(self, app):
        """Test changing system type."""
        initial_type = app.system_type
        app.set_system_type('VDPol')
        assert app.system_type == 'VDPol'
        assert app.system_type != initial_type

    def test_on_click_method_exists(self, app):
        """Test on_click method exists."""
        assert hasattr(app, 'on_click')
        assert callable(app.on_click)


class TestAppSystemSwitching:
    """Test switching between different dynamical systems."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_switch_to_hopf(self, app):
        """Test switching to Hopf system."""
        app.set_system_type('Hopf')
        assert app.system_type == 'Hopf'

    def test_switch_to_vdpol(self, app):
        """Test switching to Van der Pol system."""
        app.set_system_type('VDPol')
        assert app.system_type == 'VDPol'

    def test_switch_to_saddle_node(self, app):
        """Test switching to Saddle-Node system."""
        app.set_system_type('SN')
        assert app.system_type == 'SN'

    def test_switch_to_global(self, app):
        """Test switching to Global system."""
        app.set_system_type('global')
        assert app.system_type == 'global'


class TestAppParameterControl:
    """Test parameter control through sliders."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_slider_initial_values(self, app):
        """Test sliders have correct initial values."""
        assert app.samp.val == 0  # mu
        assert app.sfreq.val == 3  # cfreq
        assert app.sw.val == 0.5  # w

    def test_slider_ranges(self, app):
        """Test sliders have correct ranges."""
        assert app.samp.valmin == -10
        assert app.samp.valmax == 8
        assert app.sfreq.valmin == 0
        assert app.sfreq.valmax == 15.0
        assert app.sw.valmin == 0
        assert app.sw.valmax == 1.0


class TestAppUpdateBehavior:
    """Test app update behavior."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_update_with_different_mu(self, app):
        """Test update with different mu values."""
        app.samp.set_val(1.0)
        # After update, mu should change
        # Note: set_val triggers the update callback
        assert app.mu == 1.0

    def test_update_with_different_cfreq(self, app):
        """Test update with different cfreq values."""
        app.sfreq.set_val(5.0)
        assert app.cfreq == 5.0

    def test_update_with_different_w(self, app):
        """Test update with different w values."""
        app.sw.set_val(0.7)
        assert app.w == 0.7


class TestAppStateManagement:
    """Test app state management."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_initial_state_position(self, app):
        """Test initial state position."""
        assert app.cx == 4
        assert app.cy == -2

    def test_trajectory_points_initially_empty(self, app):
        """Test trajectory points list is initially empty."""
        assert len(app.trajectory_points) == 0
        assert isinstance(app.trajectory_points, list)

    def test_update_counter_starts_at_zero(self, app):
        """Test update counter starts at zero."""
        assert app.tidx == 0


class TestAppIntegration:
    """Integration tests for complete app functionality."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_app_with_all_systems(self, app):
        """Test app works with all system types."""
        systems = ['Hopf', 'VDPol', 'SN', 'global']

        for system_name in systems:
            app.set_system_type(system_name)
            assert app.system_type == system_name

            # Trigger update
            app.update(None)

    def test_parameter_sweep(self, app):
        """Test app with parameter sweep."""
        # Sweep through mu values
        for mu in [-2, -1, 0, 1, 2]:
            app.samp.set_val(mu)
            assert app.mu == mu

        # Sweep through cfreq values
        for cfreq in [1, 5, 10]:
            app.sfreq.set_val(cfreq)
            assert app.cfreq == cfreq

    def test_multiple_updates(self, app):
        """Test multiple successive updates."""
        initial_tidx = app.tidx

        for i in range(5):
            app.update(None)
            assert app.tidx == initial_tidx + i + 1


class TestMainFunction:
    """Test the main entry point."""

    def test_main_function_exists(self):
        """Test main function exists."""
        from interactive_flow_field import main
        assert callable(main)


class TestAppModularity:
    """Test that app uses the backend modules correctly."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_app_uses_get_system(self, app):
        """Test app uses get_system from backend."""
        from vizdyn.systems import get_system

        # App should use this function internally
        system = get_system(app.system_type, mu=app.mu, fc=app.cfreq, win=app.w)
        assert system is not None

    def test_app_uses_analysis_functions(self, app):
        """Test app uses analysis functions from backend."""
        from vizdyn.analysis import compute_trajectory, compute_flow_field

        # These should work with app's current system
        from vizdyn.systems import get_system
        system = get_system(app.system_type, mu=app.mu, fc=app.cfreq, win=app.w)

        t, traj = compute_trajectory(system, (app.cx, app.cy))
        assert traj.shape[0] > 0

        X, Y, Z = compute_flow_field(system)
        assert Z.shape[0] == 2

    def test_app_uses_visualization_functions(self):
        """Test app uses visualization functions from backend."""
        from vizdyn.visualization import (
            plot_vector_field,
            plot_trajectory,
            plot_start_marker
        )

        # These functions should be available and callable
        assert callable(plot_vector_field)
        assert callable(plot_trajectory)
        assert callable(plot_start_marker)


class TestAppRobustness:
    """Test app robustness and error handling."""

    @pytest.fixture
    def app(self):
        """Create an app instance for testing."""
        from interactive_flow_field import InteractiveFlowField
        app = InteractiveFlowField()
        yield app
        plt.close(app.fig)

    def test_app_handles_extreme_parameters(self, app):
        """Test app handles extreme parameter values."""
        # Test with extreme mu
        app.samp.set_val(-10)
        app.update(None)

        app.samp.set_val(8)
        app.update(None)

        # Test with extreme cfreq
        app.sfreq.set_val(0)
        app.update(None)

        app.sfreq.set_val(15)
        app.update(None)

    def test_app_handles_boundary_w_values(self, app):
        """Test app handles boundary w values."""
        app.sw.set_val(0)
        app.update(None)

        app.sw.set_val(1.0)
        app.update(None)

    def test_multiple_system_switches(self, app):
        """Test multiple rapid system switches."""
        systems = ['Hopf', 'VDPol', 'SN', 'global']

        for _ in range(3):
            for system_name in systems:
                app.set_system_type(system_name)
                assert app.system_type == system_name
