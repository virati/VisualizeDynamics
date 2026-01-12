# Custom Equation Feature

## Overview

The interactive flow field application now supports user-defined custom dynamical systems. Users can write equations directly in the GUI and immediately see the flow field update.

## Features

### 1. Text Input Widgets

Two text boxes are provided for entering custom equations:
- **dx/dt**: Equation for the x-component of the vector field
- **dy/dt**: Equation for the y-component of the vector field

### 2. Available Variables and Functions

#### Variables
- `x`: Current x coordinate
- `y`: Current y coordinate
- `mu`: Bifurcation parameter (controlled by Mu slider)
- `fc`: Frequency parameter (controlled by CFreq slider)
- `win`: Win parameter (controlled by W factor slider)

#### Mathematical Functions
- Trigonometric: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Exponential/Logarithmic: `exp`, `log`, `sqrt`
- Other: `abs`
- Constants: `pi`, `e`
- Numpy operations: Use `np.` prefix (e.g., `np.power(x, 2)`)

### 3. Operator Support

- Basic arithmetic: `+`, `-`, `*`, `/`
- Exponentiation: `**` or `^` (automatically converted)
- Parentheses for grouping: `(`, `)`

## Usage

### Basic Examples

#### 1. Linear System
```
dx/dt = mu * x - y
dy/dt = x + mu * y
```
This creates a linear oscillator with bifurcation parameter mu.

#### 2. Pendulum
```
dx/dt = y
dy/dt = -sin(x) - mu * y
```
Simple pendulum with damping (mu controls damping).

#### 3. Van der Pol Oscillator
```
dx/dt = y
dy/dt = mu * (1 - x**2) * y - x
```

#### 4. Lotka-Volterra (Predator-Prey)
```
dx/dt = x * (mu - y)
dy/dt = y * (x - fc)
```

#### 5. FitzHugh-Nagumo
```
dx/dt = x - x**3 / 3 - y + mu
dy/dt = fc * (x + 0.7 - 0.8 * y)
```

### Advanced Examples

#### Nonlinear Oscillator
```
dx/dt = -y - mu * x * (x**2 + y**2 - 1)
dy/dt = x - mu * y * (x**2 + y**2 - 1)
```

#### Duffing Oscillator
```
dx/dt = y
dy/dt = mu * x - x**3 - fc * y
```

#### Glycolysis Model
```
dx/dt = -x + mu * y + x**2 * y
dy/dt = fc - mu * y - x**2 * y
```

## GUI Workflow

1. **Select "Custom" from the system type radio buttons**
   - This switches to custom equation mode

2. **Enter equations in the text boxes**
   - Type your equations using the supported variables and functions
   - Default equations are provided as examples

3. **Click "Apply" button**
   - Validates the equations
   - Creates a new dynamical system
   - Updates all plots immediately

4. **Adjust parameters using sliders**
   - Mu, CFreq, and W sliders work with custom equations
   - Flow field updates in real-time

5. **Interact with the plot**
   - Left-click: Set new initial condition
   - Right-click: Draw manual trajectory
   - All features work with custom systems

## Safety Features

The equation parser includes several safety measures:

### Input Validation
- Checks for dangerous operations (`import`, `exec`, `eval`, etc.)
- Blocks access to file operations
- Validates syntax before evaluation

### Error Handling
- Invalid equations display error messages
- Falls back to Hopf system if custom system fails
- Prevents application crashes from bad input

### Secure Evaluation
- Restricted namespace (no `__builtins__`)
- Only safe mathematical operations allowed
- No access to file system or network

## Technical Implementation

### Components

#### 1. `CustomSystem` Class (`src/vizdyn/systems.py`)
```python
system = CustomSystem(
    dx_dt_func=lambda x, y, mu, fc, win: mu * x - y,
    dy_dt_func=lambda x, y, mu, fc, win: x + mu * y,
    mu=1.0,
    fc=1.0,
    win=0.5
)
```

#### 2. `EquationParser` (`src/vizdyn/equation_parser.py`)
- `validate_equation(equation)`: Check syntax and safety
- `parse_equation(equation)`: Convert string to callable
- `create_system_from_equations(dx_eq, dy_eq)`: Create `CustomSystem`

#### 3. GUI Integration (`app/gui_components.py`)
- `GUIManager`: Text boxes and Apply button
- `StateManager`: Custom system state management
- Event handling: Apply button callback

#### 4. Application Integration (`app/interactive_flow_field.py`)
- `on_apply_custom_equations()`: Handle Apply button
- `on_system_change()`: Auto-update when selecting Custom
- Full integration with existing features

## Example Session

```python
# 1. Start the application
python app/interactive_flow_field.py

# 2. Select "Custom" from radio buttons

# 3. Enter equations:
#    dx/dt: y
#    dy/dt: -sin(x) - 0.1 * y

# 4. Click "Apply"

# 5. Adjust Mu slider to change system behavior

# 6. Click on phase plane to explore trajectories
```

## Tips

### Creating Interesting Systems

1. **Start Simple**: Begin with linear terms, then add nonlinearities
2. **Use Parameters**: Leverage `mu`, `fc`, `win` for interactive exploration
3. **Check Stability**: Look for fixed points and limit cycles
4. **Test Boundaries**: Try extreme parameter values

### Debugging Equations

1. **Check Syntax**: Use parentheses to clarify order of operations
2. **Test Components**: Try each equation separately first
3. **Print Output**: Terminal shows equation text when applied
4. **Watch for NaNs**: Avoid division by zero or undefined operations

### Performance

- Keep equations relatively simple for real-time interaction
- Complex expressions may slow down flow field calculation
- Use numpy operations (`np.`) for better performance

## API Reference

### EquationParser Methods

```python
parser = EquationParser()

# Validate equation
is_valid, error_msg = parser.validate_equation("x + y")

# Parse to function
func = parser.parse_equation("mu * x - y")
result = func(x=1.0, y=0.5, mu=1.0, fc=1.0, win=0.5)

# Create system
system = parser.create_system_from_equations(
    dx_dt_eq="y",
    dy_dt_eq="-sin(x)",
    mu=0.0,
    fc=1.0,
    win=0.5
)
```

### StateManager Methods

```python
# Create custom system
state.create_custom_system(
    dx_eq="mu * x - y",
    dy_eq="x + mu * y"
)

# Get current system
system = state.get_current_system()

# Get custom equations
dx_eq, dy_eq = state.get_custom_equations()
```

## Future Enhancements

Potential improvements:
- Equation library/presets
- Save/load custom equations
- Multi-equation support (3D systems)
- Parameter fitting to data
- Automatic bifurcation analysis
- LaTeX rendering of equations

## Troubleshooting

### Common Issues

**Issue**: Equations don't update when clicking Apply
- **Solution**: Make sure "Custom" is selected in radio buttons

**Issue**: "Error applying custom equations" message
- **Solution**: Check equation syntax, ensure all variables are defined

**Issue**: Flow field looks wrong
- **Solution**: Verify equation logic, check parameter values

**Issue**: Application becomes slow
- **Solution**: Simplify equations or reduce mesh resolution

**Issue**: NaN or Inf values in plots
- **Solution**: Check for division by zero, avoid extreme parameter values

## Examples by Category

### Oscillators
- Harmonic: `dx/dt = y, dy/dt = -x`
- Damped: `dx/dt = y, dy/dt = -x - mu*y`
- Driven: `dx/dt = y, dy/dt = -x + mu*cos(fc*x)`

### Bifurcations
- Hopf: `dx/dt = mu*x - y - x*(x**2 + y**2), dy/dt = x + mu*y - y*(x**2 + y**2)`
- Saddle-node: `dx/dt = mu - x**2, dy/dt = -y`
- Pitchfork: `dx/dt = mu*x - x**3, dy/dt = -y`

### Biology
- Lotka-Volterra: `dx/dt = x*(mu - y), dy/dt = y*(x - fc)`
- SIR Model: `dx/dt = -mu*x*y, dy/dt = mu*x*y - fc*y`
- Brusselator: `dx/dt = mu - (fc + 1)*x + x**2*y, dy/dt = fc*x - x**2*y`

### Physics
- Pendulum: `dx/dt = y, dy/dt = -sin(x) - mu*y`
- Duffing: `dx/dt = y, dy/dt = mu*x - x**3 - fc*y`
- Lorenz (2D): `dx/dt = mu*(y - x), dy/dt = x*(fc - x) - y`
