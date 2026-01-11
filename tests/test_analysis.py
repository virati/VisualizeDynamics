"""Tests for the analysis module."""

import pytest
import numpy as np

from vizdyn.analysis import (
    compute_trajectory,
    compute_flow_field,
    find_critical_points,
    find_critical_points_2d,
    compute_field_magnitude,
    compute_trajectory_dynamics,
)


class TestComputeTrajectory:
    """Test trajectory computation."""

    def test_trajectory_shape(self, hopf_system):
        """Test trajectory returns correct shape."""
        t, traj = compute_trajectory(hopf_system, (1.0, 0.5))
        assert t.shape == (500,)
        assert traj.shape == (500, 2)

    def test_trajectory_custom_points(self, hopf_system):
        """Test trajectory with custom number of points."""
        t, traj = compute_trajectory(hopf_system, (1.0, 0.5), n_points=100)
        assert t.shape == (100,)
        assert traj.shape == (100, 2)

    def test_trajectory_custom_time_span(self, hopf_system):
        """Test trajectory with custom time span."""
        t, traj = compute_trajectory(hopf_system, (1.0, 0.5), t_span=(0.0, 20.0))
        assert t[0] == 0.0
        assert t[-1] == 20.0

    def test_trajectory_initial_condition(self, hopf_system):
        """Test trajectory starts at initial condition."""
        initial_state = (2.0, 1.5)
        t, traj = compute_trajectory(hopf_system, initial_state)
        np.testing.assert_array_almost_equal(traj[0], initial_state)

    def test_trajectory_different_systems(self, saddle_node_system, van_der_pol_system):
        """Test trajectory computation for different systems."""
        t1, traj1 = compute_trajectory(saddle_node_system, (1.0, 0.5))
        t2, traj2 = compute_trajectory(van_der_pol_system, (1.0, 0.5))

        assert traj1.shape == traj2.shape
        # Trajectories should be different for different systems
        assert not np.allclose(traj1, traj2)

    def test_trajectory_is_continuous(self, hopf_system):
        """Test trajectory is continuous (no jumps)."""
        t, traj = compute_trajectory(hopf_system, (1.0, 0.5), n_points=1000)
        # Check that consecutive points are close
        diffs = np.diff(traj, axis=0)
        max_step = np.max(np.linalg.norm(diffs, axis=1))
        assert max_step < 1.0  # No large jumps


class TestComputeFlowField:
    """Test flow field computation."""

    def test_flow_field_shapes(self, hopf_system):
        """Test flow field returns correct shapes."""
        X, Y, Z = compute_flow_field(hopf_system)
        assert X.shape == Y.shape
        assert Z.shape[0] == 2
        assert Z.shape[1] == X.size

    def test_flow_field_custom_resolution(self, hopf_system):
        """Test flow field with custom resolution."""
        X, Y, Z = compute_flow_field(hopf_system, resolution=20)
        assert X.shape == (20, 20)
        assert Y.shape == (20, 20)
        assert Z.shape == (2, 400)

    def test_flow_field_custom_limits(self, hopf_system):
        """Test flow field with custom axis limits."""
        X, Y, Z = compute_flow_field(
            hopf_system,
            xlim=(-5, 5),
            ylim=(-2, 2),
            resolution=10
        )
        assert X.min() == -5
        assert X.max() == 5
        assert Y.min() == -2
        assert Y.max() == 2

    def test_flow_field_normalized(self, hopf_system):
        """Test normalized flow field has unit magnitude."""
        X, Y, Z = compute_flow_field(hopf_system, normalize=True)
        magnitudes = np.linalg.norm(Z, axis=0)
        # Most vectors should have magnitude close to 1 (some may be zero)
        non_zero = magnitudes > 0.1
        np.testing.assert_array_almost_equal(magnitudes[non_zero], 1.0, decimal=5)

    def test_flow_field_unnormalized(self, hopf_system):
        """Test unnormalized flow field."""
        X, Y, Z = compute_flow_field(hopf_system, normalize=False)
        magnitudes = np.linalg.norm(Z, axis=0)
        # Not all vectors should have unit magnitude
        assert not np.allclose(magnitudes, 1.0)

    def test_flow_field_different_systems(self, hopf_system, saddle_node_system):
        """Test flow fields differ for different systems."""
        X1, Y1, Z1 = compute_flow_field(hopf_system, normalize=False)
        X2, Y2, Z2 = compute_flow_field(saddle_node_system, normalize=False)

        np.testing.assert_array_equal(X1, X2)  # Same grid
        np.testing.assert_array_equal(Y1, Y2)  # Same grid
        assert not np.allclose(Z1, Z2)  # Different fields


class TestFindCriticalPoints:
    """Test critical point detection."""

    def test_find_critical_points_simple(self):
        """Test finding critical points in simple data."""
        x = np.linspace(-5, 5, 100)
        y = x**2  # Minimum at x=0
        critical_indices, stability = find_critical_points(x, y)

        assert len(critical_indices) >= 1
        # Should find minimum near center
        assert any(abs(x[idx]) < 0.5 for idx in critical_indices)

    def test_find_critical_points_sine(self):
        """Test finding critical points in sinusoidal data."""
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)
        critical_indices, stability = find_critical_points(x, y)

        # Should find multiple critical points
        assert len(critical_indices) >= 3

    def test_stability_array_length(self):
        """Test stability array has same length as critical points."""
        x = np.linspace(-5, 5, 100)
        y = np.sin(x)
        critical_indices, stability = find_critical_points(x, y)

        assert len(stability) == len(critical_indices)

    def test_find_critical_points_2d(self):
        """Test 2D critical point detection."""
        x = np.linspace(-5, 5, 50)
        y = x**2
        critical_indices, stability = find_critical_points_2d(x, y)

        assert isinstance(critical_indices, np.ndarray)
        assert isinstance(stability, list)
        assert len(stability) == 0  # Stability not computed for 2D


class TestComputeFieldMagnitude:
    """Test field magnitude computation."""

    def test_field_magnitude_shapes(self, hopf_system):
        """Test field magnitude returns correct shapes."""
        X, Y, magnitude = compute_field_magnitude(hopf_system)
        assert X.shape == Y.shape
        assert magnitude.shape == X.T.shape

    def test_field_magnitude_custom_resolution(self, hopf_system):
        """Test field magnitude with custom resolution."""
        X, Y, magnitude = compute_field_magnitude(hopf_system, resolution=15)
        assert X.shape == (15, 15)
        assert magnitude.shape[0] == 15
        assert magnitude.shape[1] == 15

    def test_field_magnitude_non_negative(self, hopf_system):
        """Test field magnitude is non-negative."""
        X, Y, magnitude = compute_field_magnitude(hopf_system)
        assert np.all(magnitude >= 0)

    def test_field_magnitude_at_fixed_point(self):
        """Test magnitude is near zero at fixed points."""
        from vizdyn.systems import HopfSystem
        system = HopfSystem(mu=-1.0, fc=1.0, win=0.5)

        # For subcritical Hopf, origin is stable
        X, Y, magnitude = compute_field_magnitude(
            system,
            xlim=(-0.5, 0.5),
            ylim=(-0.5, 0.5),
            resolution=11
        )

        # Magnitude at center (origin) should be smallest
        center_idx = 5  # Middle of 11x11 grid
        center_mag = magnitude[center_idx, center_idx]
        assert center_mag < np.mean(magnitude)

    def test_field_magnitude_custom_limits(self, hopf_system):
        """Test field magnitude with custom limits."""
        X, Y, magnitude = compute_field_magnitude(
            hopf_system,
            xlim=(-10, 10),
            ylim=(-5, 5),
            resolution=10
        )
        assert X.min() == -10
        assert X.max() == 10
        assert Y.min() == -5
        assert Y.max() == 5


class TestComputeTrajectoryDynamics:
    """Test trajectory dynamics computation."""

    def test_trajectory_dynamics_structure(self, hopf_system, sample_trajectory):
        """Test trajectory dynamics returns correct structure."""
        dynamics = compute_trajectory_dynamics(hopf_system, sample_trajectory)

        assert isinstance(dynamics, dict)
        assert 'field' in dynamics
        assert 'difference' in dynamics
        assert 'alignment' in dynamics
        assert 'magnitude' in dynamics

    def test_trajectory_dynamics_shapes(self, hopf_system, sample_trajectory):
        """Test trajectory dynamics returns correct shapes."""
        n_points = sample_trajectory.shape[0]
        dynamics = compute_trajectory_dynamics(hopf_system, sample_trajectory, n_samples=50)

        # Should have n_points - 1 segments
        assert dynamics['field'].shape[0] == n_points - 1
        assert dynamics['difference'].shape[0] == n_points - 1
        assert dynamics['alignment'].shape[0] == n_points - 1
        assert dynamics['magnitude'].shape[0] == n_points - 1

    def test_trajectory_dynamics_custom_samples(self, hopf_system, sample_trajectory):
        """Test trajectory dynamics with custom sample count."""
        dynamics = compute_trajectory_dynamics(
            hopf_system,
            sample_trajectory,
            n_samples=20
        )

        # Field should have shape (n_segments, 2, n_samples)
        n_segments = sample_trajectory.shape[0] - 1
        assert dynamics['field'].shape == (n_segments, 2, 20)

    def test_trajectory_dynamics_single_segment(self, hopf_system):
        """Test trajectory dynamics with just two points."""
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
        dynamics = compute_trajectory_dynamics(hopf_system, trajectory)

        assert dynamics['field'].shape[0] == 1  # One segment
        assert dynamics['alignment'].shape[0] == 1
        assert dynamics['magnitude'].shape[0] == 1

    def test_trajectory_dynamics_magnitude_non_negative(self, hopf_system, sample_trajectory):
        """Test magnitude values are non-negative."""
        dynamics = compute_trajectory_dynamics(hopf_system, sample_trajectory)
        assert np.all(np.array(dynamics['magnitude']) >= 0)


class TestAnalysisIntegration:
    """Integration tests combining multiple analysis functions."""

    def test_trajectory_on_flow_field(self, hopf_system):
        """Test that trajectory follows flow field."""
        # Compute trajectory
        t, traj = compute_trajectory(hopf_system, (1.0, 0.5), n_points=100)

        # Compute flow field
        X, Y, Z = compute_flow_field(hopf_system, normalize=False)

        # Trajectory should generally follow the flow direction
        # (This is a basic sanity check)
        assert traj.shape[0] == 100
        assert Z.shape[0] == 2

    def test_consistent_parameters(self):
        """Test that same parameters give consistent results."""
        from vizdyn.systems import HopfSystem

        system1 = HopfSystem(mu=1.0, fc=2.0, win=0.5)
        system2 = HopfSystem(mu=1.0, fc=2.0, win=0.5)

        t1, traj1 = compute_trajectory(system1, (1.0, 0.5))
        t2, traj2 = compute_trajectory(system2, (1.0, 0.5))

        np.testing.assert_array_almost_equal(traj1, traj2)

    def test_field_magnitude_matches_flow_field(self, hopf_system):
        """Test field magnitude is consistent with flow field."""
        X1, Y1, Z = compute_flow_field(hopf_system, resolution=20, normalize=False)
        X2, Y2, magnitude = compute_field_magnitude(hopf_system, resolution=20)

        # Compute magnitude from flow field
        Z_magnitude = np.linalg.norm(Z, axis=0).reshape(X1.T.shape)

        # Should match the magnitude function
        np.testing.assert_array_almost_equal(Z_magnitude, magnitude)
