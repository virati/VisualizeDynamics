"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def hopf_system():
    """Create a Hopf system for testing."""
    from vizdyn.systems import HopfSystem
    return HopfSystem(mu=1.0, fc=3.0, win=0.5)


@pytest.fixture
def saddle_node_system():
    """Create a Saddle-Node system for testing."""
    from vizdyn.systems import SaddleNodeSystem
    return SaddleNodeSystem(mu=0.5, fc=2.0, win=0.5)


@pytest.fixture
def van_der_pol_system():
    """Create a Van der Pol system for testing."""
    from vizdyn.systems import VanDerPolSystem
    return VanDerPolSystem(mu=2.0, fc=1.0, win=0.8)


@pytest.fixture
def global_system():
    """Create a Global bifurcation system for testing."""
    from vizdyn.systems import GlobalSystem
    return GlobalSystem(mu=0.5, fc=1.5, win=0.5)


@pytest.fixture
def sample_state():
    """Create a sample 2D state."""
    return np.array([1.0, 0.5])


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    y = np.sin(t)
    return np.column_stack([x, y])


@pytest.fixture
def meshgrid():
    """Create a sample meshgrid for testing."""
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)
    return X, Y


@pytest.fixture(autouse=True)
def matplotlib_backend():
    """Use non-interactive backend for matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
