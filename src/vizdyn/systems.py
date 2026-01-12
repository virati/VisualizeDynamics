"""Dynamical system definitions and normal forms."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class DynamicalSystem(ABC):
    """Abstract base class for 2D dynamical systems."""

    def __init__(self, mu: float = 0.0, fc: float = 1.0, win: float = 0.5):
        """
        Initialize the dynamical system.

        Parameters
        ----------
        mu : float
            Bifurcation parameter
        fc : float
            Frequency/scaling parameter
        win : float
            Additional parameter for system shaping
        """
        self.mu = mu
        self.fc = fc
        self.win = win

    @abstractmethod
    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the vector field at a given state.

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y]
        t : float
            Time (for compatibility with odeint)

        Returns
        -------
        np.ndarray
            Derivative [dx/dt, dy/dt]
        """
        pass

    def __call__(self, state: np.ndarray, t: float) -> np.ndarray:
        """Allow the system to be called directly."""
        return self.vector_field(state, t)


class HopfSystem(DynamicalSystem):
    """Hopf bifurcation normal form."""

    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Hopf normal form vector field.

        dx/dt = w * (mu * x - y - x * (x^2 + y^2))
        dy/dt = q * (x + mu * y - y * (x^2 + y^2))

        where w = win and q = 1 - w
        """
        x = state[0]
        y = state[1]

        w = self.win
        q = 1 - w

        xd = w * (self.mu * x - y - x * (x**2 + y**2))
        yd = q * (x + self.mu * y - y * (x**2 + y**2))

        return self.fc * np.array([xd, yd])


class SaddleNodeSystem(DynamicalSystem):
    """Saddle-node bifurcation normal form."""

    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Saddle-node normal form vector field.

        dx/dt = mu - x^2
        dy/dt = -y
        """
        x = state[0]
        y = state[1]

        xd = self.mu - x**2
        yd = -y

        return self.fc * np.array([xd, yd])


class GlobalSystem(DynamicalSystem):
    """Global bifurcation normal form."""

    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Global bifurcation normal form vector field.

        dx/dt = y
        dy/dt = mu * y + x - x^2 + x * y
        """
        x = state[0]
        y = state[1]

        xd = y
        yd = self.mu * y + x - x**2 + x * y

        return self.fc * np.array([xd, yd])


class VanDerPolSystem(DynamicalSystem):
    """Van der Pol oscillator normal form."""

    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Van der Pol normal form vector field.

        dx/dt = win * mu * x - y^2 * x - y
        dy/dt = x
        """
        x = state[0]
        y = state[1]

        xd = self.win * self.mu * x - y**2 * x - y
        yd = x

        return self.fc * np.array([xd, yd])


class CustomSystem(DynamicalSystem):
    """User-defined custom dynamical system."""

    def __init__(
        self,
        dx_dt_func,
        dy_dt_func,
        mu: float = 0.0,
        fc: float = 1.0,
        win: float = 0.5
    ):
        """
        Initialize custom system with user-defined functions.

        Parameters
        ----------
        dx_dt_func : callable
            Function for dx/dt that takes (x, y, mu, fc, win) and returns float
        dy_dt_func : callable
            Function for dy/dt that takes (x, y, mu, fc, win) and returns float
        mu : float
            Bifurcation parameter
        fc : float
            Frequency/scaling parameter
        win : float
            Win parameter
        """
        super().__init__(mu, fc, win)
        self.dx_dt_func = dx_dt_func
        self.dy_dt_func = dy_dt_func

    def vector_field(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute vector field using user-defined functions.

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y]
        t : float
            Time (for compatibility with odeint)

        Returns
        -------
        np.ndarray
            Derivative [dx/dt, dy/dt]
        """
        # Handle both single states and arrays of states
        if state.ndim == 1:
            x = state[0]
            y = state[1]
            xd = self.dx_dt_func(x, y, self.mu, self.fc, self.win)
            yd = self.dy_dt_func(x, y, self.mu, self.fc, self.win)
            return np.array([xd, yd])
        else:
            # Array of states
            x = state[0, :]
            y = state[1, :]
            xd = self.dx_dt_func(x, y, self.mu, self.fc, self.win)
            yd = self.dy_dt_func(x, y, self.mu, self.fc, self.win)
            return np.array([xd, yd])


# System registry for easy lookup
SYSTEM_REGISTRY = {
    'Hopf': HopfSystem,
    'VDPol': VanDerPolSystem,
    'SN': SaddleNodeSystem,
    'global': GlobalSystem,
}


def get_system(name: str, **kwargs) -> DynamicalSystem:
    """
    Get a dynamical system by name.

    Parameters
    ----------
    name : str
        Name of the system ('Hopf', 'VDPol', 'SN', 'global')
    **kwargs
        Parameters to pass to the system constructor

    Returns
    -------
    DynamicalSystem
        Instantiated dynamical system
    """
    if name not in SYSTEM_REGISTRY:
        raise ValueError(f"Unknown system: {name}. Available: {list(SYSTEM_REGISTRY.keys())}")

    return SYSTEM_REGISTRY[name](**kwargs)
