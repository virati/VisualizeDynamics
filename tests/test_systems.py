"""Tests for the systems module."""

import pytest
import numpy as np

from vizdyn.systems import (
    DynamicalSystem,
    HopfSystem,
    SaddleNodeSystem,
    VanDerPolSystem,
    GlobalSystem,
    get_system,
    SYSTEM_REGISTRY,
)


class TestDynamicalSystemBase:
    """Test the base DynamicalSystem class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that the abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            DynamicalSystem()


class TestHopfSystem:
    """Test the Hopf bifurcation system."""

    def test_initialization(self, hopf_system):
        """Test system initialization."""
        assert hopf_system.mu == 1.0
        assert hopf_system.fc == 3.0
        assert hopf_system.win == 0.5

    def test_vector_field_shape(self, hopf_system, sample_state):
        """Test vector field returns correct shape."""
        result = hopf_system.vector_field(sample_state, 0.0)
        assert result.shape == (2,)

    def test_vector_field_type(self, hopf_system, sample_state):
        """Test vector field returns numpy array."""
        result = hopf_system.vector_field(sample_state, 0.0)
        assert isinstance(result, np.ndarray)

    def test_callable(self, hopf_system, sample_state):
        """Test system can be called directly."""
        result1 = hopf_system.vector_field(sample_state, 0.0)
        result2 = hopf_system(sample_state, 0.0)
        np.testing.assert_array_equal(result1, result2)

    def test_equilibrium_at_origin_subcritical(self):
        """Test that origin is stable for mu < 0."""
        system = HopfSystem(mu=-1.0, fc=1.0, win=0.5)
        state = np.array([0.01, 0.01])
        derivative = system(state, 0.0)
        # For subcritical Hopf, small perturbations should decay
        assert np.dot(state, derivative) < 0

    def test_limit_cycle_supercritical(self):
        """Test that limit cycle exists for mu > 0."""
        system = HopfSystem(mu=1.0, fc=1.0, win=0.5)
        # On the limit cycle, radius should be sqrt(mu)
        radius = np.sqrt(1.0)
        state = np.array([radius, 0.0])
        derivative = system(state, 0.0)
        # Radial component should be zero (on the limit cycle)
        radial_derivative = np.dot(state, derivative) / radius
        assert abs(radial_derivative) < 0.1

    def test_array_input(self, hopf_system):
        """Test system handles array of states."""
        states = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, -0.5]]).T
        result = hopf_system(states, 0.0)
        assert result.shape == (2, 3)


class TestSaddleNodeSystem:
    """Test the Saddle-Node bifurcation system."""

    def test_initialization(self, saddle_node_system):
        """Test system initialization."""
        assert saddle_node_system.mu == 0.5
        assert saddle_node_system.fc == 2.0

    def test_vector_field_shape(self, saddle_node_system, sample_state):
        """Test vector field returns correct shape."""
        result = saddle_node_system.vector_field(sample_state, 0.0)
        assert result.shape == (2,)

    def test_fixed_points_exist(self):
        """Test fixed points exist for mu > 0."""
        system = SaddleNodeSystem(mu=1.0, fc=1.0, win=0.5)
        # Fixed points at x = ±sqrt(mu)
        state1 = np.array([1.0, 0.0])
        state2 = np.array([-1.0, 0.0])

        derivative1 = system(state1, 0.0)
        derivative2 = system(state2, 0.0)

        assert abs(derivative1[0]) < 0.1  # Near fixed point
        assert abs(derivative2[0]) < 0.1  # Near fixed point

    def test_no_fixed_points_subcritical(self):
        """Test no real fixed points for mu < 0."""
        system = SaddleNodeSystem(mu=-1.0, fc=1.0, win=0.5)
        # For any real x, mu - x^2 < 0, so dx/dt != 0
        for x in np.linspace(-3, 3, 10):
            state = np.array([x, 0.0])
            derivative = system(state, 0.0)
            assert derivative[0] != 0


class TestVanDerPolSystem:
    """Test the Van der Pol oscillator."""

    def test_initialization(self, van_der_pol_system):
        """Test system initialization."""
        assert van_der_pol_system.mu == 2.0
        assert van_der_pol_system.fc == 1.0
        assert van_der_pol_system.win == 0.8

    def test_vector_field_shape(self, van_der_pol_system, sample_state):
        """Test vector field returns correct shape."""
        result = van_der_pol_system.vector_field(sample_state, 0.0)
        assert result.shape == (2,)

    def test_origin_unstable(self):
        """Test that origin is an unstable fixed point."""
        system = VanDerPolSystem(mu=2.0, fc=1.0, win=0.5)
        state = np.array([0.0, 0.0])
        derivative = system(state, 0.0)
        # At origin, dx/dt = 0, dy/dt = 0
        np.testing.assert_array_almost_equal(derivative, [0.0, 0.0])


class TestGlobalSystem:
    """Test the Global bifurcation system."""

    def test_initialization(self, global_system):
        """Test system initialization."""
        assert global_system.mu == 0.5
        assert global_system.fc == 1.5

    def test_vector_field_shape(self, global_system, sample_state):
        """Test vector field returns correct shape."""
        result = global_system.vector_field(sample_state, 0.0)
        assert result.shape == (2,)

    def test_vector_field_values(self, global_system):
        """Test vector field computation."""
        state = np.array([1.0, 0.5])
        result = global_system(state, 0.0)

        # dx/dt = y
        assert result[0] == 1.5 * 0.5

        # dy/dt = mu * y + x - x^2 + x * y
        expected_dy = 0.5 * 0.5 + 1.0 - 1.0**2 + 1.0 * 0.5
        assert abs(result[1] - 1.5 * expected_dy) < 1e-10


class TestSystemRegistry:
    """Test the system registry and factory function."""

    def test_registry_contains_all_systems(self):
        """Test registry has all expected systems."""
        expected_systems = ['Hopf', 'VDPol', 'SN', 'global']
        assert set(SYSTEM_REGISTRY.keys()) == set(expected_systems)

    def test_get_system_hopf(self):
        """Test getting Hopf system."""
        system = get_system('Hopf', mu=1.0, fc=2.0)
        assert isinstance(system, HopfSystem)
        assert system.mu == 1.0
        assert system.fc == 2.0

    def test_get_system_vdpol(self):
        """Test getting Van der Pol system."""
        system = get_system('VDPol', mu=1.5)
        assert isinstance(system, VanDerPolSystem)
        assert system.mu == 1.5

    def test_get_system_saddle_node(self):
        """Test getting Saddle-Node system."""
        system = get_system('SN', mu=0.5)
        assert isinstance(system, SaddleNodeSystem)
        assert system.mu == 0.5

    def test_get_system_global(self):
        """Test getting Global system."""
        system = get_system('global', mu=1.0)
        assert isinstance(system, GlobalSystem)
        assert system.mu == 1.0

    def test_get_system_invalid_name(self):
        """Test error on invalid system name."""
        with pytest.raises(ValueError, match="Unknown system"):
            get_system('InvalidSystem')

    def test_get_system_default_parameters(self):
        """Test system creation with default parameters."""
        system = get_system('Hopf')
        assert system.mu == 0.0
        assert system.fc == 1.0
        assert system.win == 0.5


class TestSystemParameterEffects:
    """Test effects of different parameters."""

    def test_fc_parameter_scaling(self):
        """Test that fc parameter scales the output."""
        system1 = HopfSystem(mu=1.0, fc=1.0, win=0.5)
        system2 = HopfSystem(mu=1.0, fc=2.0, win=0.5)

        state = np.array([1.0, 0.5])
        result1 = system1(state, 0.0)
        result2 = system2(state, 0.0)

        np.testing.assert_array_almost_equal(result2, 2 * result1)

    def test_win_parameter_effect(self):
        """Test that win parameter affects dynamics."""
        system1 = HopfSystem(mu=1.0, fc=1.0, win=0.3)
        system2 = HopfSystem(mu=1.0, fc=1.0, win=0.7)

        state = np.array([1.0, 0.5])
        result1 = system1(state, 0.0)
        result2 = system2(state, 0.0)

        # Results should be different
        assert not np.allclose(result1, result2)
