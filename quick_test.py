"""
Ultra-minimal test for immediate verification
Tests all 3 methods: EOSSP Baseline, REOSSP Exact, and REOSSP RHP
Uses an extremely small instance (T=50) for quick testing
"""

from parameters import InstanceParameters
from eossp_baseline import EOSSPSolver
from reossp_exact import REOSSPExactSolver
from reossp_rhp import REOSSPRHPSolver

print("="*80)
print("TESTING ALL 3 METHODS (EOSSP, REOSSP-Exact, REOSSP-RHP)")
print("="*80)

# Create tiny instance: 2 satellites, 2 stages, 50 time steps
params = InstanceParameters(
    instance_id=999,
    S=8,  # Just 5 stages
    K=5,  # Only 3 satellites
    J_sk=20,  # Very few slots
    T=500  # Only 500 time steps for quick test
)

print(f"\nProblem size:")
print(f"  Satellites: {params.K}")
print(f"  Stages: {params.S}")
print(f"  Time steps: {params.T}")
print(f"  Observations possible: {params.V_target.sum()}")
print(f"  Downlinks possible: {params.V_ground.sum()}")

# Test 1: EOSSP Baseline
print(f"\n" + "="*80)
print(f"METHOD 1: EOSSP BASELINE")
print(f"="*80)
solver1 = EOSSPSolver(params)
results1 = solver1.solve(time_limit_minutes=1, solver_name='highs')
print(f"Status: {results1['status']}")
print(f"Objective: {results1['objective']:.2f}")
print(f"Data downlinked: {results1['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results1['runtime_minutes']:.2f} minutes")

# Test 2: REOSSP Exact
print(f"\n" + "="*80)
print(f"METHOD 2: REOSSP EXACT")
print(f"="*80)
solver2 = REOSSPExactSolver(params)
results2 = solver2.solve(time_limit_minutes=1, solver_name='highs')
print(f"Status: {results2['status']}")
print(f"Objective: {results2['objective']:.2f}")
print(f"Data downlinked: {results2['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results2['runtime_minutes']:.2f} minutes")

# Test 3: REOSSP RHP
print(f"\n" + "="*80)
print(f"METHOD 3: REOSSP RHP")
print(f"="*80)
solver3 = REOSSPRHPSolver(params)
results3 = solver3.solve(time_limit_per_stage_minutes=1, solver_name='highs')
print(f"Status: {results3['status']}")
print(f"Objective: {results3['objective']:.2f}")
print(f"Data downlinked: {results3['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results3['runtime_minutes']:.2f} minutes")

# Summary
print(f"\n" + "="*80)
print(f"SUMMARY - ALL METHODS TESTED:")
print(f"="*80)
all_passed = all(r['status'] in ['optimal', 'completed'] for r in [results1, results2, results3])
print(f"EOSSP Baseline:  {results1['status']:12s} | Obj: {results1['objective']:.2f}")
print(f"REOSSP Exact:    {results2['status']:12s} | Obj: {results2['objective']:.2f}")
print(f"REOSSP RHP:      {results3['status']:12s} | Obj: {results3['objective']:.2f}")

if all_passed:
    print(f"\n✓ ALL TESTS PASSED - All 3 methods are working!")
else:
    print(f"\n✗ SOME TESTS FAILED - Check individual results above")
