"""
Ultra-minimal test for immediate verification
Tests all 3 methods: EOSSP Baseline, REOSSP Exact, and REOSSP RHP
Uses an extremely small instance (T=50) for quick testing
"""

import csv
import os
from datetime import datetime
from parameters import InstanceParameters
from eossp_baseline import EOSSPSolver
from reossp_exact import REOSSPExactSolver
from reossp_rhp import REOSSPRHPSolver

print("="*80)
print("TESTING ALL 3 METHODS (EOSSP, REOSSP-Exact, REOSSP-RHP)")
print("="*80)

# IMPORTANT: J_sk must be >= K (need enough slots for all satellites)
S, K, J_sk, T = 8, 5, 20, 36*24*14
time_limit_minutes = 60

params = InstanceParameters(instance_id=999, S=S, K=K, J_sk=J_sk, T=T)

print(f"\nProblem size:")
print(f"  Satellites: {params.K}")
print(f"  Stages: {params.S}")
print(f"  Time steps: {params.T}")
print(f"  Time steps per stage: {params.T // params.S}")
print(f"  Observations possible: {params.V_target.sum()}")
print(f"  Downlinks possible: {params.V_ground.sum()}")

# Test 1: EOSSP Baseline
print(f"\n" + "="*80)
print(f"METHOD 1: EOSSP BASELINE")
print(f"="*80)
solver1 = EOSSPSolver(params)
results1 = solver1.solve(time_limit_minutes=time_limit_minutes, solver_name='gurobi')
print(f"Status: {results1['status']}")
print(f"Objective: {results1['objective']:.2f}")
print(f"Data downlinked: {results1['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results1['runtime_minutes']:.2f} minutes")
print(f"\nModel Statistics:")
print(f"  Variables: {results1.get('num_variables', 'N/A')}")
print(f"  Constraints: {results1.get('num_constraints', 'N/A')}")
print(f"  Non-zeros: {results1.get('num_nonzeros', 'N/A')}")

# Test 2: REOSSP Exact
print(f"\n" + "="*80)
print(f"METHOD 2: REOSSP EXACT")
print(f"="*80)
solver2 = REOSSPExactSolver(params)
results2 = solver2.solve(time_limit_minutes=time_limit_minutes, solver_name='gurobi')
print(f"Status: {results2['status']}")
print(f"Objective: {results2['objective']:.2f}")
print(f"Data downlinked: {results2['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results2['runtime_minutes']:.2f} minutes")
print(f"\nModel Statistics:")
print(f"  Variables: {results2.get('num_variables', 'N/A')}")
print(f"  Constraints: {results2.get('num_constraints', 'N/A')}")
print(f"  Non-zeros: {results2.get('num_nonzeros', 'N/A')}")

# Test 3: REOSSP RHP
print(f"\n" + "="*80)
print(f"METHOD 3: REOSSP RHP")
print(f"="*80)
solver3 = REOSSPRHPSolver(params)
results3 = solver3.solve(time_limit_per_stage_minutes=time_limit_minutes, solver_name='gurobi')
print(f"Status: {results3['status']}")
print(f"Objective: {results3['objective']:.2f}")
print(f"Data downlinked: {results3['data_downlinked_gb']:.4f} GB")
print(f"Runtime: {results3['runtime_minutes']:.2f} minutes")
print(f"\nModel Statistics:")
print(f"  Total variables (all stages): {results3.get('num_variables', 'N/A')}")
print(f"  Total constraints (all stages): {results3.get('num_constraints', 'N/A')}")
print(f"  Number of stages solved: {results3.get('num_stages_solved', params.S)}")

# Summary
print(f"\n" + "="*80)
print(f"SUMMARY - ALL METHODS TESTED:")
print(f"="*80)
all_passed = all(r['status'] in ['optimal', 'completed'] for r in [results1, results2, results3])
print(f"EOSSP Baseline:  {results1['status']:12s} | Obj: {results1['objective']:10.2f} | Time: {results1['runtime_minutes']:6.2f}m")
print(f"REOSSP Exact:    {results2['status']:12s} | Obj: {results2['objective']:10.2f} | Time: {results2['runtime_minutes']:6.2f}m")
print(f"REOSSP RHP:      {results3['status']:12s} | Obj: {results3['objective']:10.2f} | Time: {results3['runtime_minutes']:6.2f}m")

print(f"\n" + "="*80)
print(f"CONSTRAINT COMPARISON:")
print(f"="*80)
print(f"EOSSP Baseline:  {results1.get('num_constraints', 'N/A'):>10} constraints | {results1.get('num_variables', 'N/A'):>10} variables")
print(f"REOSSP Exact:    {results2.get('num_constraints', 'N/A'):>10} constraints | {results2.get('num_variables', 'N/A'):>10} variables")
print(f"REOSSP RHP:      {results3.get('num_constraints', 'N/A'):>10} constraints | {results3.get('num_variables', 'N/A'):>10} variables (cumulative)")

print(f"\n" + "="*80)
print(f"ANALYSIS:")
print(f"="*80)

# Objective comparison
if results1['objective'] > 0:
    obj_diff_exact = results2['objective'] - results1['objective']
    obj_diff_rhp = results3['objective'] - results1['objective']
    print(f"REOSSP Exact vs Baseline: {obj_diff_exact:+.2f} ({obj_diff_exact/results1['objective']*100:+.2f}%)")
    print(f"REOSSP RHP vs Baseline:   {obj_diff_rhp:+.2f} ({obj_diff_rhp/results1['objective']*100:+.2f}%)")

# Runtime comparison
if results1['runtime_minutes'] > 0:
    print(f"\nRuntime ratio (Exact/Baseline): {results2['runtime_minutes']/results1['runtime_minutes']:.2f}x")
    print(f"Runtime ratio (RHP/Exact):      {results3['runtime_minutes']/results2['runtime_minutes']:.2f}x")
    print(f"Runtime ratio (RHP/Baseline):   {results3['runtime_minutes']/results1['runtime_minutes']:.2f}x")

# Possible reasons for unexpected behavior
print(f"\n" + "="*80)
print(f"POTENTIAL ISSUES:")
print(f"="*80)

if results1['objective'] > 0 and abs(results2['objective'] - results1['objective']) < 0.01:
    print(f"⚠ REOSSP Exact ≈ Baseline: Storage constraints may be non-binding")
    print(f"  → Check if storage capacity M_k is large enough")
    print(f"  → Verify data accumulation between stages is allowed")
    print(f"  → Instance may have abundant downlink opportunities")
    print(f"  → Propellant budget may be too restrictive")

if results3['runtime_minutes'] > results2['runtime_minutes']:
    print(f"⚠ RHP slower than Exact: Decomposition overhead > problem size benefit")
    print(f"  → {params.S} stages × {params.T//params.S} timesteps per stage")
    print(f"  → Multiple model builds and warm-start overhead")
    print(f"  → Consider reducing stages or increasing T for RHP to be beneficial")
    print(f"  → RHP typically faster for T > 5000-10000 timesteps")

if all_passed:
    print(f"\n✓ ALL TESTS PASSED - All 3 methods are working!")
else:
    print(f"\n✗ SOME TESTS FAILED - Check individual results above")

# Save results to CSV
print(f"\n" + "="*80)
print(f"SAVING RESULTS TO CSV")
print(f"="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = "results.csv"

# Prepare data for CSV
csv_data = {
    'timestamp': timestamp,
    'S': S,
    'K': K,
    'J_sk': J_sk,
    'T': T,
    'observations_possible': params.V_target.sum(),
    'downlinks_possible': params.V_ground.sum(),
    'time_limit_minutes': time_limit_minutes,
    
    # EOSSP Baseline results
    'eossp_status': results1['status'],
    'eossp_objective': results1['objective'],
    'eossp_data_downlinked_gb': results1['data_downlinked_gb'],
    'eossp_runtime_minutes': results1['runtime_minutes'],
    'eossp_num_variables': results1.get('num_variables', 'N/A'),
    'eossp_num_constraints': results1.get('num_constraints', 'N/A'),
    'eossp_num_nonzeros': results1.get('num_nonzeros', 'N/A'),
    
    # REOSSP Exact results
    'reossp_exact_status': results2['status'],
    'reossp_exact_objective': results2['objective'],
    'reossp_exact_data_downlinked_gb': results2['data_downlinked_gb'],
    'reossp_exact_runtime_minutes': results2['runtime_minutes'],
    'reossp_exact_num_variables': results2.get('num_variables', 'N/A'),
    'reossp_exact_num_constraints': results2.get('num_constraints', 'N/A'),
    'reossp_exact_num_nonzeros': results2.get('num_nonzeros', 'N/A'),
    
    # REOSSP RHP results
    'reossp_rhp_status': results3['status'],
    'reossp_rhp_objective': results3['objective'],
    'reossp_rhp_data_downlinked_gb': results3['data_downlinked_gb'],
    'reossp_rhp_runtime_minutes': results3['runtime_minutes'],
    'reossp_rhp_num_variables': results3.get('num_variables', 'N/A'),
    'reossp_rhp_num_constraints': results3.get('num_constraints', 'N/A'),
    'reossp_rhp_num_stages_solved': results3.get('num_stages_solved', S),
}

# Calculate figure of merit (objective / runtime) for each method
if results1['runtime_minutes'] > 0:
    csv_data['eossp_figure_of_merit'] = results1['objective'] / results1['runtime_minutes']
else:
    csv_data['eossp_figure_of_merit'] = 'N/A'

if results2['runtime_minutes'] > 0:
    csv_data['reossp_exact_figure_of_merit'] = results2['objective'] / results2['runtime_minutes']
else:
    csv_data['reossp_exact_figure_of_merit'] = 'N/A'

if results3['runtime_minutes'] > 0:
    csv_data['reossp_rhp_figure_of_merit'] = results3['objective'] / results3['runtime_minutes']
else:
    csv_data['reossp_rhp_figure_of_merit'] = 'N/A'

# Calculate comparison metrics
if results1['objective'] > 0:
    csv_data['exact_vs_baseline_obj_diff'] = results2['objective'] - results1['objective']
    csv_data['exact_vs_baseline_obj_pct'] = (results2['objective'] - results1['objective']) / results1['objective'] * 100
    csv_data['rhp_vs_baseline_obj_diff'] = results3['objective'] - results1['objective']
    csv_data['rhp_vs_baseline_obj_pct'] = (results3['objective'] - results1['objective']) / results1['objective'] * 100
else:
    csv_data['exact_vs_baseline_obj_diff'] = 'N/A'
    csv_data['exact_vs_baseline_obj_pct'] = 'N/A'
    csv_data['rhp_vs_baseline_obj_diff'] = 'N/A'
    csv_data['rhp_vs_baseline_obj_pct'] = 'N/A'

if results1['runtime_minutes'] > 0:
    csv_data['exact_vs_baseline_runtime_ratio'] = results2['runtime_minutes'] / results1['runtime_minutes']
    csv_data['rhp_vs_baseline_runtime_ratio'] = results3['runtime_minutes'] / results1['runtime_minutes']
else:
    csv_data['exact_vs_baseline_runtime_ratio'] = 'N/A'
    csv_data['rhp_vs_baseline_runtime_ratio'] = 'N/A'

if results2['runtime_minutes'] > 0:
    csv_data['rhp_vs_exact_runtime_ratio'] = results3['runtime_minutes'] / results2['runtime_minutes']
else:
    csv_data['rhp_vs_exact_runtime_ratio'] = 'N/A'

# Write to CSV (append mode)
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
    
    # Write header only if file doesn't exist
    if not file_exists:
        writer.writeheader()
    
    writer.writerow(csv_data)

if file_exists:
    print(f"✓ Results appended to: {csv_filename}")
else:
    print(f"✓ Results saved to new file: {csv_filename}")
print(f"  Total columns: {len(csv_data)}")
