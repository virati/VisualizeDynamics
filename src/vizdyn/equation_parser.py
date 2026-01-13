"""Parser for user-defined dynamical system equations."""

import numpy as np
import re
from typing import Callable, Tuple, Optional


class EquationParser:
    """Parse and convert string equations to callable functions."""

    # Safe namespace for eval
    SAFE_NAMESPACE = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "np": np,  # Allow numpy operations
    }

    @staticmethod
    def validate_equation(equation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an equation string for safety and syntax.

        Parameters
        ----------
        equation : str
            Equation string to validate

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        if not equation or not equation.strip():
            return False, "Equation cannot be empty"

        # Check for dangerous operations
        dangerous_patterns = [
            "__",  # Dunder methods
            "import",
            "exec",
            "eval",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
        ]

        equation_lower = equation.lower()
        for pattern in dangerous_patterns:
            if pattern in equation_lower:
                return False, f"Dangerous operation detected: {pattern}"

        # Check for valid characters (allow math operations and variables)
        valid_pattern = re.compile(
            r"^[x y mu fc win \d\.\+\-\*/\(\)\^\*\s sincostaexplgrtahbfmipqu]+$"
        )
        if not valid_pattern.match(equation.lower().replace("np.", "")):
            return False, "Equation contains invalid characters"

        # Try to parse as valid Python expression
        try:
            # Replace ^ with ** for exponentiation
            test_eq = equation.replace("^", "**")
            # Test compile (doesn't execute)
            compile(test_eq, "<string>", "eval")
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        return True, None

    @staticmethod
    def parse_equation(equation: str) -> Callable:
        """
        Parse an equation string and return a callable function.

        The equation can use variables: x, y, mu, fc, win
        and mathematical functions: sin, cos, exp, etc.

        Parameters
        ----------
        equation : str
            Equation string, e.g., "mu * x - y - x * (x**2 + y**2)"

        Returns
        -------
        callable
            Function that takes (x, y, mu, fc, win) and returns result

        Examples
        --------
        >>> parser = EquationParser()
        >>> func = parser.parse_equation("x * y + mu")
        >>> func(1.0, 2.0, 0.5, 1.0, 0.5)
        2.5
        """
        # Validate first
        is_valid, error_msg = EquationParser.validate_equation(equation)
        if not is_valid:
            raise ValueError(f"Invalid equation: {error_msg}")

        # Replace ^ with ** for exponentiation
        equation = equation.replace("^", "**")

        # Create the function
        def equation_func(x, y, mu, fc, win):
            """Evaluate the equation with given parameters."""
            # Create local namespace with parameters
            local_ns = EquationParser.SAFE_NAMESPACE.copy()
            local_ns.update({"x": x, "y": y, "mu": mu, "fc": fc, "win": win})

            try:
                result = eval(equation, {"__builtins__": {}}, local_ns)
                return result
            except Exception as e:
                raise RuntimeError(f"Error evaluating equation: {str(e)}")

        return equation_func

    @staticmethod
    def create_system_from_equations(
        dx_dt_eq: str, dy_dt_eq: str, mu: float = 0.0, fc: float = 1.0, win: float = 0.5
    ):
        """
        Create a CustomSystem from equation strings.

        Parameters
        ----------
        dx_dt_eq : str
            Equation for dx/dt
        dy_dt_eq : str
            Equation for dy/dt
        mu : float
            Bifurcation parameter
        fc : float
            Frequency parameter
        win : float
            Win parameter

        Returns
        -------
        CustomSystem
            System with user-defined equations

        Examples
        --------
        >>> parser = EquationParser()
        >>> system = parser.create_system_from_equations(
        ...     "mu * x - y",
        ...     "x + mu * y"
        ... )
        """
        from .systems import CustomSystem

        # Parse equations
        dx_dt_func = EquationParser.parse_equation(dx_dt_eq)
        dy_dt_func = EquationParser.parse_equation(dy_dt_eq)

        # Create and return system
        return CustomSystem(
            dx_dt_func=dx_dt_func, dy_dt_func=dy_dt_func, mu=mu, fc=fc, win=win
        )


def test_equation_parser():
    """Test the equation parser with various inputs."""
    parser = EquationParser()

    # Test valid equations
    test_cases = [
        ("x + y", True),
        ("mu * x - y * y", True),
        ("sin(x) + cos(y)", True),
        ("exp(-x**2 - y**2)", True),
        ("x^2 + y^2", True),  # Should convert ^ to **
        ("import os", False),  # Should reject
        ("__import__", False),  # Should reject
        ("", False),  # Empty should reject
    ]

    print("Testing equation parser...")
    for eq, should_pass in test_cases:
        is_valid, error = parser.validate_equation(eq)
        status = "✓" if is_valid == should_pass else "✗"
        print(f"{status} {eq}: valid={is_valid}")

    # Test parsing and evaluation
    print("\nTesting parsing and evaluation...")
    func = parser.parse_equation("x * y + mu")
    result = func(2.0, 3.0, 1.0, 1.0, 0.5)
    print(f"x * y + mu with x=2, y=3, mu=1: {result} (expected 7.0)")

    func2 = parser.parse_equation("sin(x) + cos(y)")
    result2 = func2(0.0, 0.0, 0.0, 1.0, 0.5)
    print(f"sin(0) + cos(0): {result2} (expected 1.0)")


if __name__ == "__main__":
    test_equation_parser()
