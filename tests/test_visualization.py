"""Tests for the visualization module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

from vizdyn.visualization import (
    add_arrow,
    plot_vector_field,
    plot_trajectory,
    plot_start_marker,
    plot_phase_surface,
    plot_contour_projections,
)


@pytest.fixture
def fig_and_ax():
    """Create a figure and axes for testing."""
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def fig_and_ax_3d():
    """Create a figure and 3D axes for testing."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    yield fig, ax
    plt.close(fig)


class TestAddArrow:
    """Test arrow addition to lines."""

    def test_add_arrow_to_line(self, fig_and_ax):
        """Test adding arrow to a line."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        # Should not raise error
        add_arrow(line)

    def test_add_arrow_with_position(self, fig_and_ax):
        """Test adding arrow at specific position."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        add_arrow(line, position=5.0)

    def test_add_arrow_direction_left(self, fig_and_ax):
        """Test adding arrow pointing left."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        add_arrow(line, direction='left')

    def test_add_arrow_direction_right(self, fig_and_ax):
        """Test adding arrow pointing right."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        add_arrow(line, direction='right')

    def test_add_arrow_custom_color(self, fig_and_ax):
        """Test adding arrow with custom color."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        add_arrow(line, color='red')

    def test_add_arrow_custom_size(self, fig_and_ax):
        """Test adding arrow with custom size."""
        fig, ax = fig_and_ax
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        line, = ax.plot(x, y)

        add_arrow(line, size=100)


class TestPlotVectorField:
    """Test vector field plotting."""

    def test_plot_vector_field_basic(self, fig_and_ax):
        """Test basic vector field plotting."""
        fig, ax = fig_and_ax

        x = np.linspace(-3, 3, 10)
        y = np.linspace(-3, 3, 10)
        X, Y = np.meshgrid(x, y)

        # Create simple vector field
        U = -Y
        V = X
        Z = np.array([U.ravel(), V.ravel()])

        quiver = plot_vector_field(ax, X, Y, Z)
        assert quiver is not None

    def test_plot_vector_field_custom_kwargs(self, fig_and_ax):
        """Test vector field with custom kwargs."""
        fig, ax = fig_and_ax

        x = np.linspace(-3, 3, 5)
        y = np.linspace(-3, 3, 5)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X
        Z = np.array([U.ravel(), V.ravel()])

        quiver = plot_vector_field(ax, X, Y, Z, color='red', alpha=0.8)
        assert quiver is not None

    def test_plot_vector_field_overrides_defaults(self, fig_and_ax):
        """Test that custom kwargs override defaults."""
        fig, ax = fig_and_ax

        x = np.linspace(-3, 3, 5)
        y = np.linspace(-3, 3, 5)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X
        Z = np.array([U.ravel(), V.ravel()])

        quiver = plot_vector_field(ax, X, Y, Z, width=0.05, alpha=1.0)
        assert quiver is not None


class TestPlotTrajectory:
    """Test trajectory plotting."""

    def test_plot_trajectory_basic(self, fig_and_ax, sample_trajectory):
        """Test basic trajectory plotting."""
        fig, ax = fig_and_ax
        scatter = plot_trajectory(ax, sample_trajectory)

        assert isinstance(scatter, PathCollection)

    def test_plot_trajectory_custom_colormap(self, fig_and_ax, sample_trajectory):
        """Test trajectory with custom colormap."""
        fig, ax = fig_and_ax
        scatter = plot_trajectory(ax, sample_trajectory, colormap='viridis')

        assert isinstance(scatter, PathCollection)

    def test_plot_trajectory_custom_kwargs(self, fig_and_ax, sample_trajectory):
        """Test trajectory with custom kwargs."""
        fig, ax = fig_and_ax
        scatter = plot_trajectory(ax, sample_trajectory, s=50, alpha=0.5)

        assert isinstance(scatter, PathCollection)

    def test_plot_trajectory_different_lengths(self, fig_and_ax):
        """Test plotting trajectories of different lengths."""
        fig, ax = fig_and_ax

        traj1 = np.random.randn(50, 2)
        traj2 = np.random.randn(200, 2)

        scatter1 = plot_trajectory(ax, traj1)
        ax.clear()
        scatter2 = plot_trajectory(ax, traj2)

        assert isinstance(scatter1, PathCollection)
        assert isinstance(scatter2, PathCollection)


class TestPlotStartMarker:
    """Test start location marker plotting."""

    def test_plot_start_marker_basic(self, fig_and_ax):
        """Test basic start marker plotting."""
        fig, ax = fig_and_ax
        scatter = plot_start_marker(ax, 1.0, 0.5)

        assert isinstance(scatter, PathCollection)

    def test_plot_start_marker_custom_color(self, fig_and_ax):
        """Test start marker with custom color."""
        fig, ax = fig_and_ax
        scatter = plot_start_marker(ax, 1.0, 0.5, color='blue')

        assert isinstance(scatter, PathCollection)

    def test_plot_start_marker_custom_marker(self, fig_and_ax):
        """Test start marker with custom marker style."""
        fig, ax = fig_and_ax
        scatter = plot_start_marker(ax, 1.0, 0.5, marker='o')

        assert isinstance(scatter, PathCollection)

    def test_plot_start_marker_custom_size(self, fig_and_ax):
        """Test start marker with custom size."""
        fig, ax = fig_and_ax
        scatter = plot_start_marker(ax, 1.0, 0.5, s=500)

        assert isinstance(scatter, PathCollection)

    def test_plot_start_marker_different_positions(self, fig_and_ax):
        """Test plotting markers at different positions."""
        fig, ax = fig_and_ax

        positions = [(0, 0), (1, 1), (-1, -1), (2, -2)]
        for x, y in positions:
            scatter = plot_start_marker(ax, x, y)
            assert isinstance(scatter, PathCollection)


class TestPlotPhaseSurface:
    """Test 3D phase surface plotting."""

    def test_plot_phase_surface_basic(self, fig_and_ax_3d):
        """Test basic phase surface plotting."""
        fig, ax = fig_and_ax_3d

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        surface = plot_phase_surface(ax, X, Y, Z)
        assert surface is not None

    def test_plot_phase_surface_custom_alpha(self, fig_and_ax_3d):
        """Test phase surface with custom alpha."""
        fig, ax = fig_and_ax_3d

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        surface = plot_phase_surface(ax, X, Y, Z, alpha=0.5)
        assert surface is not None

    def test_plot_phase_surface_different_shapes(self, fig_and_ax_3d):
        """Test phase surface with different grid sizes."""
        fig, ax = fig_and_ax_3d

        for n in [10, 20, 30]:
            ax.clear()
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2

            surface = plot_phase_surface(ax, X, Y, Z)
            assert surface is not None


class TestPlotContourProjections:
    """Test contour projection plotting."""

    def test_plot_contour_projections_basic(self, fig_and_ax_3d):
        """Test basic contour projections."""
        fig, ax = fig_and_ax_3d

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        contours = plot_contour_projections(ax, X, Y, Z)
        assert isinstance(contours, list)
        assert len(contours) == 3  # Three projections

    def test_plot_contour_projections_custom_offset(self, fig_and_ax_3d):
        """Test contour projections with custom offset."""
        fig, ax = fig_and_ax_3d

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        contours = plot_contour_projections(ax, X, Y, Z, offset=5)
        assert isinstance(contours, list)
        assert len(contours) == 3

    def test_plot_contour_projections_different_functions(self, fig_and_ax_3d):
        """Test contour projections with different functions."""
        fig, ax = fig_and_ax_3d

        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)

        # Different test functions
        functions = [
            X**2 + Y**2,
            np.sin(X) * np.cos(Y),
            np.exp(-(X**2 + Y**2))
        ]

        for Z in functions:
            ax.clear()
            contours = plot_contour_projections(ax, X, Y, Z)
            assert len(contours) == 3


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_complete_2d_plot(self, fig_and_ax):
        """Test creating a complete 2D phase plot."""
        fig, ax = fig_and_ax

        # Create vector field
        x = np.linspace(-3, 3, 15)
        y = np.linspace(-3, 3, 15)
        X, Y = np.meshgrid(x, y)
        U = -Y
        V = X
        Z = np.array([U.ravel(), V.ravel()])

        # Plot vector field
        quiver = plot_vector_field(ax, X, Y, Z)

        # Add trajectory
        t = np.linspace(0, 2*np.pi, 100)
        trajectory = np.column_stack([np.cos(t), np.sin(t)])
        scatter = plot_trajectory(ax, trajectory)

        # Add start marker
        marker = plot_start_marker(ax, 1.0, 0.0)

        assert quiver is not None
        assert isinstance(scatter, PathCollection)
        assert isinstance(marker, PathCollection)

    def test_complete_3d_plot(self, fig_and_ax_3d):
        """Test creating a complete 3D phase plot."""
        fig, ax = fig_and_ax_3d

        # Create surface data
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2

        # Plot surface
        surface = plot_phase_surface(ax, X, Y, Z)

        # Add contour projections
        contours = plot_contour_projections(ax, X, Y, Z)

        # Set limits
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 20)

        assert surface is not None
        assert len(contours) == 3

    def test_multiple_trajectories(self, fig_and_ax):
        """Test plotting multiple trajectories on same axes."""
        fig, ax = fig_and_ax

        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            t = np.linspace(0, 2*np.pi, 100)
            r = 1 + i * 0.5
            traj = np.column_stack([r * np.cos(t), r * np.sin(t)])
            trajectories.append(traj)

        # Plot all trajectories
        scatters = []
        for traj in trajectories:
            scatter = plot_trajectory(ax, traj)
            scatters.append(scatter)

        assert len(scatters) == 3
        assert all(isinstance(s, PathCollection) for s in scatters)


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""

    def test_plot_trajectory_single_point(self, fig_and_ax):
        """Test plotting trajectory with single point."""
        fig, ax = fig_and_ax
        trajectory = np.array([[1.0, 0.5]])
        scatter = plot_trajectory(ax, trajectory)

        assert isinstance(scatter, PathCollection)

    def test_plot_trajectory_two_points(self, fig_and_ax):
        """Test plotting trajectory with two points."""
        fig, ax = fig_and_ax
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
        scatter = plot_trajectory(ax, trajectory)

        assert isinstance(scatter, PathCollection)

    def test_empty_vector_field(self, fig_and_ax):
        """Test plotting vector field with zeros."""
        fig, ax = fig_and_ax

        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((2, 25))

        quiver = plot_vector_field(ax, X, Y, Z)
        assert quiver is not None

    def test_start_marker_at_origin(self, fig_and_ax):
        """Test plotting start marker at origin."""
        fig, ax = fig_and_ax
        scatter = plot_start_marker(ax, 0.0, 0.0)

        assert isinstance(scatter, PathCollection)
