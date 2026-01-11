# VisualizeDynamics Test Suite

Comprehensive pytest test suite for the VisualizeDynamics project.

## Test Coverage

**Overall Coverage: 99%** (155 statements, 1 missed)

- `src/vizdyn/__init__.py`: 100%
- `src/vizdyn/analysis.py`: 100%
- `src/vizdyn/systems.py`: 98%
- `src/vizdyn/visualization.py`: 100%

## Test Organization

### `test_systems.py` - Dynamical Systems (30 tests)

Tests for all dynamical system implementations:

- **TestDynamicalSystemBase**: Abstract base class validation
- **TestHopfSystem**: Hopf bifurcation system (7 tests)
  - Initialization, vector field computation, equilibrium points, limit cycles
- **TestSaddleNodeSystem**: Saddle-node bifurcation (4 tests)
  - Fixed points, bifurcation behavior
- **TestVanDerPolSystem**: Van der Pol oscillator (3 tests)
  - System dynamics, unstable origin
- **TestGlobalSystem**: Global bifurcation (3 tests)
  - Vector field computation and values
- **TestSystemRegistry**: System factory and registry (6 tests)
  - System lookup, parameter handling, error cases
- **TestSystemParameterEffects**: Parameter sensitivity (2 tests)
  - Frequency scaling, win parameter effects

### `test_analysis.py` - Analysis Functions (28 tests)

Tests for computational analysis tools:

- **TestComputeTrajectory**: ODE integration (6 tests)
  - Shape validation, custom parameters, initial conditions, continuity
- **TestComputeFlowField**: Vector field computation (6 tests)
  - Grid generation, normalization, custom limits
- **TestFindCriticalPoints**: Critical point detection (4 tests)
  - 1D and 2D detection, stability analysis
- **TestComputeFieldMagnitude**: Magnitude computation (5 tests)
  - Shape validation, fixed points, boundary conditions
- **TestComputeTrajectoryDynamics**: Trajectory analysis (5 tests)
  - Field dynamics, alignment, magnitude along paths
- **TestAnalysisIntegration**: Integration tests (3 tests)
  - Cross-function consistency, parameter validation

### `test_visualization.py` - Visualization Utilities (33 tests)

Tests for plotting functions:

- **TestAddArrow**: Arrow annotations (6 tests)
  - Position, direction, color, size customization
- **TestPlotVectorField**: Quiver plots (3 tests)
  - Basic plotting, custom kwargs, default overrides
- **TestPlotTrajectory**: Trajectory plotting (4 tests)
  - Colormaps, custom parameters, variable lengths
- **TestPlotStartMarker**: Start markers (5 tests)
  - Position, color, marker style, size
- **TestPlotPhaseSurface**: 3D surface plots (3 tests)
  - Basic rendering, alpha blending, grid sizes
- **TestPlotContourProjections**: Contour projections (3 tests)
  - Wall projections, offsets, different functions
- **TestVisualizationIntegration**: Complete plots (3 tests)
  - 2D phase plots, 3D plots, multiple trajectories
- **TestVisualizationEdgeCases**: Edge cases (4 tests)
  - Single points, empty fields, boundary conditions

### `test_app.py` - Application Integration (30 tests)

Tests for the interactive application:

- **TestInteractiveFlowFieldClass**: App initialization (6 tests)
  - Figure creation, axes setup, controls, plots
- **TestAppMethods**: Method validation (6 tests)
  - Update, reset, system switching, event handling
- **TestAppSystemSwitching**: System selection (4 tests)
  - Switching between Hopf, VDPol, SN, Global systems
- **TestAppParameterControl**: GUI controls (2 tests)
  - Slider values and ranges
- **TestAppUpdateBehavior**: Update mechanics (3 tests)
  - Parameter changes via sliders
- **TestAppStateManagement**: State tracking (3 tests)
  - Initial conditions, trajectory points, counters
- **TestAppIntegration**: Full workflow (3 tests)
  - Multiple systems, parameter sweeps, successive updates
- **TestMainFunction**: Entry point (1 test)
- **TestAppModularity**: Backend integration (3 tests)
  - Usage of systems, analysis, visualization modules
- **TestAppRobustness**: Error handling (3 tests)
  - Extreme parameters, boundary values, rapid switching

## Running Tests

### Run all tests:
```bash
uv run pytest tests/
```

### Run with verbose output:
```bash
uv run pytest tests/ -v
```

### Run specific test file:
```bash
uv run pytest tests/test_systems.py
```

### Run specific test class:
```bash
uv run pytest tests/test_systems.py::TestHopfSystem
```

### Run specific test:
```bash
uv run pytest tests/test_systems.py::TestHopfSystem::test_limit_cycle_supercritical
```

### Run with coverage report:
```bash
uv run pytest tests/ --cov=src/vizdyn --cov-report=term-missing
```

### Run with coverage HTML report:
```bash
uv run pytest tests/ --cov=src/vizdyn --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Fixtures

Defined in `conftest.py`:

- `hopf_system`: HopfSystem instance with default parameters
- `saddle_node_system`: SaddleNodeSystem instance
- `van_der_pol_system`: VanDerPolSystem instance
- `global_system`: GlobalSystem instance
- `sample_state`: 2D state vector [1.0, 0.5]
- `sample_trajectory`: Circular trajectory (100 points)
- `meshgrid`: 10x10 grid for testing
- `matplotlib_backend`: Auto-configured non-interactive backend

## Test Markers

Available markers (defined in `pyproject.toml`):

- `slow`: Mark slow tests (deselect with `-m "not slow"`)
- `integration`: Mark integration tests

## Dependencies

Test dependencies (installed with `uv sync --extra dev`):

- `pytest>=8.0.0`: Testing framework
- `pytest-cov>=4.1.0`: Coverage reporting

## Key Testing Principles

1. **Isolation**: Each test is independent and uses fixtures
2. **Coverage**: 99% overall code coverage
3. **Documentation**: Clear test names and docstrings
4. **Edge Cases**: Boundary conditions and error handling
5. **Integration**: Tests verify module interactions
6. **Determinism**: All tests are deterministic and reproducible
7. **Performance**: Tests complete in ~37 seconds

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv sync --extra dev
    uv run pytest tests/ --cov=src/vizdyn
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >95% coverage
4. Add integration tests for cross-module features
5. Update this README if adding new test categories
