"""
Main Execution Script
Run all 24 instances and generate results matching Table 2 from the paper
"""

import sys
import time
from datetime import datetime
import pickle

from parameters import generate_instance_set
from eossp_baseline import EOSSPSolver
from reossp_exact import REOSSPExactSolver
from reossp_rhp import REOSSPRHPSolver
from results_analysis import (
    create_results_table, 
    print_results_summary,
    save_results_to_csv,
    save_summary_to_csv,
    compare_with_paper_results,
    generate_comparison_report
)


def run_single_instance(params, run_exact=True, run_rhp=True, solver_name='highs'):
    """
    Run all three methods on a single instance
    
    Args:
        params: InstanceParameters object
        run_exact: Whether to run REOSSP-Exact (can be slow)
        run_rhp: Whether to run REOSSP-RHP
        solver_name: Which solver to use ('cbc', 'highs', or 'glpk')
        
    Returns:
        Dict with results from all methods
    """
    print(f"\n{'='*80}")
    print(f"Instance {params.instance_id}: S={params.S}, K={params.K}, J_sk={params.J_sk}")
    print(f"Using solver: {solver_name.upper()}")
    print(f"{'='*80}")
    
    results = {
        'instance_id': params.instance_id,
        'S': params.S,
        'K': params.K,
        'J_sk': params.J_sk,
    }
    
    # 1. EOSSP Baseline
    print("\n[1/3] Running EOSSP Baseline...")
    try:
        eossp_solver = EOSSPSolver(params)
        eossp_results = eossp_solver.solve(time_limit_minutes=60, solver_name=solver_name)
        results['eossp'] = eossp_results
        print(f"  ✓ Objective: {eossp_results['objective']:.2f}, "
              f"Runtime: {eossp_results['runtime_minutes']:.2f} min, "
              f"Data: {eossp_results['data_downlinked_gb']:.2f} GB")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        results['eossp'] = {
            'status': 'error',
            'objective': 0,
            'runtime_minutes': 0,
            'data_downlinked_gb': 0,
            'total_observations': 0,
            'total_downlinks': 0,
            'propellant_used': 0
        }
    
    # 2. REOSSP-Exact (optional, can be slow)
    if run_exact:
        print("\n[2/3] Running REOSSP-Exact...")
        try:
            reossp_solver = REOSSPExactSolver(params)
            reossp_results = reossp_solver.solve(time_limit_minutes=60, solver_name=solver_name)
            results['reossp_exact'] = reossp_results
            print(f"  ✓ Objective: {reossp_results['objective']:.2f}, "
                  f"Runtime: {reossp_results['runtime_minutes']:.2f} min, "
                  f"Data: {reossp_results['data_downlinked_gb']:.2f} GB, "
                  f"Propellant: {reossp_results['propellant_used']:.2f} m/s")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results['reossp_exact'] = {
                'status': 'error',
                'objective': 0,
                'runtime_minutes': 0,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0
            }
    else:
        print("\n[2/3] Skipping REOSSP-Exact (disabled)")
        results['reossp_exact'] = results['eossp'].copy()  # Use baseline as placeholder
    
    # 3. REOSSP-RHP
    if run_rhp:
        print("\n[3/3] Running REOSSP-RHP...")
        try:
            rhp_solver = REOSSPRHPSolver(params, lookahead=1)
            rhp_results = rhp_solver.solve(time_limit_per_stage_minutes=5, solver_name=solver_name)
            results['reossp_rhp'] = rhp_results
            print(f"  ✓ Objective: {rhp_results['objective']:.2f}, "
                  f"Runtime: {rhp_results['runtime_minutes']:.2f} min, "
                  f"Data: {rhp_results['data_downlinked_gb']:.2f} GB, "
                  f"Propellant: {rhp_results['propellant_used']:.2f} m/s")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results['reossp_rhp'] = {
                'status': 'error',
                'objective': 0,
                'runtime_minutes': 0,
                'data_downlinked_gb': 0,
                'total_observations': 0,
                'total_downlinks': 0,
                'propellant_used': 0
            }
    else:
        print("\n[3/3] Skipping REOSSP-RHP (disabled)")
        results['reossp_rhp'] = results['eossp'].copy()  # Use baseline as placeholder
    
    return results


def run_all_experiments(run_exact=True, run_rhp=True, subset=3, solver_name='highs'):
    """
    Run all 24 instances from the paper
    
    Args:
        run_exact: Whether to run REOSSP-Exact (time-consuming)
        run_rhp: Whether to run REOSSP-RHP
        subset: If specified, only run first N instances (for testing)
        solver_name: Which solver to use ('cbc', 'highs', or 'glpk')
        
    Returns:
        List of results dicts
    """
    print("\n" + "="*80)
    print("SATELLITE SCHEDULING EXPERIMENTS")
    print("Recreating Table 2 from Paper")
    print(f"Using solver: {solver_name.upper()}")
    print("="*80)
    
    # Generate all instances
    instances = generate_instance_set()
    
    if subset is not None:
        instances = instances[:subset]
        print(f"\nRunning SUBSET: First {subset} instances only")
    
    print(f"\nTotal instances to run: {len(instances)}")
    print(f"Methods: EOSSP-Baseline (always), "
          f"REOSSP-Exact ({'enabled' if run_exact else 'disabled'}), "
          f"REOSSP-RHP ({'enabled' if run_rhp else 'disabled'})")
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for i, params in enumerate(instances):
        print(f"\n\nProgress: {i+1}/{len(instances)}")
        
        result = run_single_instance(params, run_exact=run_exact, run_rhp=run_rhp, solver_name=solver_name)
        all_results.append(result)
        
        # Save intermediate results
        if (i + 1) % 5 == 0:
            print("\n[Checkpoint] Saving intermediate results...")
            with open('intermediate_results.pkl', 'wb') as f:
                pickle.dump(all_results, f)
    
    total_runtime = time.time() - start_time
    
    print(f"\n\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total runtime: {total_runtime/3600:.2f} hours")
    print(f"{'='*80}")
    
    return all_results


def main():
    """Main execution function"""
    
    # Configuration
    RUN_EXACT = True  # Set to False to skip REOSSP-Exact (faster testing)
    RUN_RHP = True
    SUBSET = None  # Set to integer to run only first N instances for testing
    SOLVER = 'highs'  # Using HiGHS solver
    
    # For quick testing, you can use:
    # SUBSET = 3  # Run only first 3 instances
    # RUN_EXACT = False  # Skip exact method
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Run EOSSP Baseline: True (always)")
    print(f"Run REOSSP-Exact: {RUN_EXACT}")
    print(f"Run REOSSP-RHP: {RUN_RHP}")
    print(f"Instance subset: {SUBSET if SUBSET else 'All (24 instances)'}")
    print(f"Solver: {SOLVER.upper()}")
    print("="*80)
    
    # Run experiments
    all_results = run_all_experiments(
        run_exact=RUN_EXACT,
        run_rhp=RUN_RHP,
        subset=SUBSET,
        solver_name=SOLVER
    )
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'all_results_{timestamp}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nRaw results saved to: {results_file}")
    
    # Create results table
    df = create_results_table(all_results)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS TABLE (Table 2 Format)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Print summary
    print_results_summary(df)
    
    # Compare with paper
    compare_with_paper_results(df)
    
    # Generate detailed comparison report
    report_file = f'comparison_report_{timestamp}.txt'
    generate_comparison_report(df, report_file)
    
    # Save to CSV
    csv_file = f'results_table2_{timestamp}.csv'
    save_results_to_csv(df, csv_file)
    
    summary_file = f'results_summary_{timestamp}.csv'
    save_summary_to_csv(df, summary_file)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {csv_file}")
    print(f"  - {summary_file}")
    print(f"  - {report_file}")
    print(f"  - {results_file}")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
