# Integer Genetic Algorithm - Python Implementation

This package contains Python implementations of optimization algorithms originally written in MATLAB.

## Converted Algorithms

1. **BGA (Binary Genetic Algorithm)** - `run_bga`
2. **IDE (Improved Differential Evolution)** - `run_ide`
3. **PSO (Particle Swarm Optimization)** - `run_pso`
4. **RGA (Real-Coded Genetic Algorithm)** - `run_rga`

## Structure

```
integerGA_python/
├── algos/
│   ├── run_bga.py      # Binary Genetic Algorithm
│   ├── run_ide.py      # Improved Differential Evolution
│   ├── run_pso.py      # Particle Swarm Optimization
│   └── run_rga.py      # Real-Coded Genetic Algorithm
└── utils/
    ├── cauchy.py       # Cauchy distribution sampling
    ├── bga_support/    # Binary GA operators
    │   ├── crossover.py
    │   ├── mutation.py
    │   └── selection.py
    └── operators/      # General operators
        ├── de_operators.py          # DE mutation operators
        ├── polynomial_mutation.py
        ├── sb_crossover.py          # Simulated Binary Crossover
        └── selection.py
```

## Usage

### Basic Example

```python
import numpy as np
from integerGA_python.algos import run_bga, run_ide, run_pso, run_rga

# Define your objective function (to be maximized)
def my_objective_function(x):
    # Example: simple sphere function (negated for maximization)
    return -np.sum(x**2)

# Problem parameters
dim = 10          # Problem dimension
max_fes = 10000   # Maximum function evaluations
pop_size = 50     # Population size

# Run BGA
best_sol, convergence, best_obj = run_bga(
    dim=dim,
    obj_func=my_objective_function,
    max_fes=max_fes,
    pop_size=pop_size,
    verbose=True
)

# Run IDE
best_sol, convergence, best_obj = run_ide(
    dim=dim,
    obj_func=my_objective_function,
    max_fes=max_fes,
    pop_size=pop_size,
    verbose=True
)

# Run PSO
best_sol, convergence, best_obj = run_pso(
    dim=dim,
    obj_func=my_objective_function,
    max_fes=max_fes,
    pop_size=pop_size,
    w=0.7,    # Inertia weight
    c1=1.5,   # Cognitive coefficient
    c2=1.5,   # Social coefficient
    verbose=True
)

# Run RGA
best_sol, convergence, best_obj = run_rga(
    dim=dim,
    obj_func=my_objective_function,
    max_fes=max_fes,
    pop_size=pop_size,
    verbose=True
)
```

### Algorithm Parameters

#### BGA (Binary Genetic Algorithm)
- `dim`: Problem dimension
- `obj_func`: Objective function to maximize
- `max_fes`: Maximum function evaluations
- `pop_size`: Population size
- `p_single_point`: Probability of single point crossover (default: 0.1)
- `p_double_point`: Probability of double point crossover (default: 0.2)
- `verbose`: Print progress (default: True)

#### IDE (Improved Differential Evolution)
- `dim`: Problem dimension
- `obj_func`: Objective function to maximize
- `max_fes`: Maximum function evaluations
- `pop_size`: Population size
- `pbest_rate`: Rate for selecting from top individuals (default: 0.1)
- `mem_size`: Memory size for parameter adaptation (default: 5)
- `arc_rate`: Archive rate (default: 2.6)
- `verbose`: Print progress (default: True)

#### PSO (Particle Swarm Optimization)
- `dim`: Problem dimension
- `obj_func`: Objective function to maximize
- `max_fes`: Maximum function evaluations
- `pop_size`: Swarm size
- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive coefficient (default: 1.5)
- `c2`: Social coefficient (default: 1.5)
- `verbose`: Print progress (default: True)

#### RGA (Real-Coded Genetic Algorithm)
- `dim`: Problem dimension
- `obj_func`: Objective function to maximize
- `max_fes`: Maximum function evaluations
- `pop_size`: Population size
- `verbose`: Print progress (default: True)

## Notes

- All algorithms are designed for **maximization**. If you need to minimize, negate your objective function.
- Solution space is typically bounded in [0, 1] for continuous variables.
- For BGA, solutions are binary (0 or 1).
- Convergence array tracks the best objective value over generations.

## Differences from MATLAB

1. Arrays are 0-indexed in Python (vs 1-indexed in MATLAB)
2. NumPy is used for array operations
3. Dictionary structures replace MATLAB structs
4. More Pythonic naming conventions (snake_case)

## Dependencies

- NumPy

## License

Based on MATLAB implementations from various sources including Yarpiz (www.yarpiz.com).
