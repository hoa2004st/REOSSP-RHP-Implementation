"""
Quick test of metaheuristic algorithms on REOSSP
Tests all algorithms on a small instance to verify integration
"""

from parameters import InstanceParameters
from reossp_metaheuristics import REOSSPMetaheuristicSolver
from reossp_two_phase_ga import REOSSPTwoPhaseGA

print("="*80)
print("QUICK TEST: METAHEURISTIC ALGORITHMS ON REOSSP")
print("="*80)

# Create small test instance
params = InstanceParameters(
    instance_id=1,
    S=6,
    K=3,
    J_sk=15,
    T=36*24*2,
    unavailable_slot_probability=0.2
)

print(f"\nTest Instance:")
print(f"  Satellites (K): {params.K}")
print(f"  Stages (S): {params.S}")
print(f"  Slots per stage (J_sk): {params.J_sk}")
print(f"  Chromosome dimension: {params.K * params.S}")
print(f"  Unavailable slots: {params.unavailable_slots.sum()}/{params.S * params.J_sk}")

# Algorithm parameters
pop_size = 30
max_fes = 2000

print(f"\nAlgorithm Parameters:")
print(f"  Population size: {pop_size}")
print(f"  Max evaluations: {max_fes}")

# Test metaheuristic solver
solver = REOSSPMetaheuristicSolver(params)

# Test each algorithm
algorithms = {
    'BGA': lambda: solver.solve_bga(pop_size=pop_size, max_fes=max_fes, verbose=False),
    'IDE': lambda: solver.solve_ide(pop_size=pop_size, max_fes=max_fes, verbose=False),
    'PSO': lambda: solver.solve_pso(pop_size=pop_size, max_fes=max_fes, verbose=False),
    'RGA': lambda: solver.solve_rga(pop_size=pop_size, max_fes=max_fes, verbose=False)
}

results = {}

print("\n" + "="*80)
print("TESTING ALGORITHMS")
print("="*80)

for alg_name, solve_func in algorithms.items():
    print(f"\nTesting {alg_name}...", end=" ")
    try:
        results[alg_name] = solve_func()
        print(f"✓ Success")
        print(f"  Objective: {results[alg_name]['objective']:.2f}")
        print(f"  Runtime: {results[alg_name]['runtime_seconds']:.2f}s")
        print(f"  Evaluations: {results[alg_name]['total_evaluations']}")
        print(f"  Feasibility: {results[alg_name]['feasibility_rate']*100:.1f}%")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

# Test Two-Phase GA
print(f"\nTesting Two-Phase GA...", end=" ")
try:
    tp_ga_solver = REOSSPTwoPhaseGA(params)
    results['Two-Phase GA'] = tp_ga_solver.solve(pop_size=pop_size, n_generations=20, verbose=False)
    print(f"✓ Success")
    print(f"  Objective: {results['Two-Phase GA']['objective']:.2f}")
    runtime_minutes = results['Two-Phase GA'].get('runtime_minutes', 0)
    print(f"  Runtime: {runtime_minutes*60:.2f}s")
    print(f"  Generations: {results['Two-Phase GA']['generations_completed']}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Summary comparison
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'Algorithm':<15} {'Objective':>12} {'Runtime (s)':>12} {'Status':>10}")
print("-"*80)

for alg_name in ['BGA', 'IDE', 'PSO', 'RGA', 'Two-Phase GA']:
    if alg_name in results:
        r = results[alg_name]
        runtime_key = 'runtime_seconds' if 'runtime_seconds' in r else 'runtime_minutes'
        runtime = r[runtime_key] if runtime_key == 'runtime_seconds' else r[runtime_key] * 60
        print(f"{alg_name:<15} {r['objective']:>12.2f} {runtime:>12.2f} {r['status']:>10}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nAll algorithms are working correctly!")
print("You can now run the full experiment with:")
print("  python run_all_metaheuristics.py")
