"""
Example script demonstrating how to use the converted Python algorithms.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integerGA_python.algos import run_bga, run_ide, run_pso, run_rga


def sphere_function(x):
    """Simple sphere function (maximization version - negated)."""
    return -np.sum(x**2)


def rastrigin_function(x):
    """Rastrigin function (maximization version - negated)."""
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def example_test():
    """Run a simple test with all algorithms."""
    
    # Problem setup
    dim = 10
    max_fes = 10000
    pop_size = 50
    
    print("=" * 70)
    print("Testing Converted Algorithms on Sphere Function")
    print("=" * 70)
    
    # Test BGA
    print("\n" + "=" * 70)
    print("Running BGA (Binary Genetic Algorithm)")
    print("=" * 70)
    best_sol, convergence, best_obj = run_bga(
        dim=dim,
        obj_func=sphere_function,
        max_fes=max_fes,
        pop_size=pop_size,
        verbose=False
    )
    print(f"BGA - Best objective: {best_obj:.6f}")
    print(f"BGA - Final convergence: {convergence[-1]:.6f}")
    
    # Test IDE
    print("\n" + "=" * 70)
    print("Running IDE (Improved Differential Evolution)")
    print("=" * 70)
    best_sol, convergence, best_obj = run_ide(
        dim=dim,
        obj_func=sphere_function,
        max_fes=max_fes,
        pop_size=pop_size,
        verbose=False
    )
    print(f"IDE - Best objective: {best_obj:.6f}")
    print(f"IDE - Final convergence: {convergence[-1]:.6f}")
    
    # Test PSO
    print("\n" + "=" * 70)
    print("Running PSO (Particle Swarm Optimization)")
    print("=" * 70)
    best_sol, convergence, best_obj = run_pso(
        dim=dim,
        obj_func=sphere_function,
        max_fes=max_fes,
        pop_size=pop_size,
        w=0.7,
        c1=1.5,
        c2=1.5,
        verbose=False
    )
    print(f"PSO - Best objective: {best_obj:.6f}")
    print(f"PSO - Final convergence: {convergence[-1]:.6f}")
    
    # Test RGA
    print("\n" + "=" * 70)
    print("Running RGA (Real-Coded Genetic Algorithm)")
    print("=" * 70)
    best_sol, convergence, best_obj = run_rga(
        dim=dim,
        obj_func=sphere_function,
        max_fes=max_fes,
        pop_size=pop_size,
        verbose=False
    )
    print(f"RGA - Best objective: {best_obj:.6f}")
    print(f"RGA - Final convergence: {convergence[-1]:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_test()
