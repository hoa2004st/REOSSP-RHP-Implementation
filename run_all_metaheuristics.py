"""
Run All Metaheuristic Algorithms for REOSSP
Tests BGA, IDE, PSO, RGA, and Two-Phase GA on all problem instances
Saves comprehensive results to CSV
"""

import csv
import os
from datetime import datetime
from parameters import InstanceParameters
from reossp_two_phase_ga import REOSSPTwoPhaseGA
from reossp_metaheuristics import REOSSPMetaheuristicSolver

print("="*80)
print("COMPREHENSIVE METAHEURISTIC COMPARISON FOR REOSSP")
print("="*80)

# Problem parameters
SK_pairs = [(8, 5), (8, 6), (9, 5), (9, 6), (12, 5), (12, 6)]
unavailable_probs = [0.0, 0.2, 0.5, 0.8, 1.0]
J_sk, T = 20, 36*24*2

# Algorithm parameters
pop_size = 50
max_fes = 5000  # For BGA, IDE, PSO, RGA
n_generations = 30  # For Two-Phase GA (different evaluation model)

# Initialize CSV file
csv_filename = "results/results_all_metaheuristics.csv"
file_exists = os.path.isfile(csv_filename)

# Counter for instance ID
instance_counter = 0

# Iterate through all unavailable probabilities
for prob in unavailable_probs:
    print(f"\n" + "="*80)
    print(f"UNAVAILABLE PROBABILITY: {prob}")
    print(f"="*80)
    
    # Iterate through all SK_pairs
    for pair_idx, (S, K) in enumerate(SK_pairs, 1):
        instance_counter += 1
        
        print(f"\n" + "="*80)
        print(f"INSTANCE {instance_counter}: S={S}, K={K}, unavailable_prob={prob}")
        print(f"="*80)
        
        # Create instance parameters
        params = InstanceParameters(
            instance_id=instance_counter, 
            S=S, 
            K=K, 
            J_sk=J_sk, 
            T=T, 
            unavailable_slot_probability=prob
        )

        print(f"\nProblem size:")
        print(f"  Satellites (K): {params.K}")
        print(f"  Stages (S): {params.S}")
        print(f"  Slots per stage (J_sk): {params.J_sk}")
        print(f"  Chromosome dimension: {params.K * params.S}")
        print(f"  Observations possible: {params.V_target.sum()}")
        print(f"  Downlinks possible: {params.V_ground.sum()}")
        print(f"  Unavailable slots: {params.unavailable_slots.sum()}/{params.S * params.J_sk}")

        # Initialize CSV data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_data = {
            'timestamp': timestamp,
            'instance_id': instance_counter,
            'S': S,
            'K': K,
            'J_sk': J_sk,
            'T': T,
            'unavailable_prob': prob,
            'observations_possible': params.V_target.sum(),
            'downlinks_possible': params.V_ground.sum(),
            'unavailable_slots': params.unavailable_slots.sum(),
            'pop_size': pop_size,
            'max_fes': max_fes,
        }

        # Run metaheuristic algorithms (BGA, IDE, PSO, RGA)
        solver = REOSSPMetaheuristicSolver(params)
        
        algorithms = {
            'bga': ('BGA', solver.solve_bga),
            'ide': ('IDE', solver.solve_ide),
            'pso': ('PSO', solver.solve_pso),
            'rga': ('RGA', solver.solve_rga)
        }
        
        for alg_key, (alg_name, solve_func) in algorithms.items():
            print(f"\n{'='*80}")
            print(f"RUNNING {alg_name}")
            print(f"{'='*80}")
            
            try:
                results = solve_func(pop_size=pop_size, max_fes=max_fes, verbose=False)
                
                # Add results to CSV data
                csv_data[f'{alg_key}_status'] = results['status']
                csv_data[f'{alg_key}_objective'] = results['objective']
                csv_data[f'{alg_key}_total_observations'] = results['total_observations']
                csv_data[f'{alg_key}_total_downlinks'] = results['total_downlinks']
                csv_data[f'{alg_key}_data_downlinked_gb'] = results['data_downlinked_gb']
                csv_data[f'{alg_key}_runtime_minutes'] = results['runtime_minutes']
                csv_data[f'{alg_key}_propellant_used'] = results['propellant_used']
                csv_data[f'{alg_key}_evaluations'] = results['total_evaluations']
                csv_data[f'{alg_key}_feasible_solutions'] = results['feasible_solutions']
                csv_data[f'{alg_key}_feasibility_rate'] = results['feasibility_rate']
                
                print(f"  Status: {results['status']}")
                print(f"  Objective: {results['objective']:.2f}")
                print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
                print(f"  Feasibility rate: {results['feasibility_rate']*100:.1f}%")
                
            except Exception as e:
                print(f"  ✗ {alg_name} failed: {e}")
                # Add failure data
                csv_data[f'{alg_key}_status'] = 'Failed'
                csv_data[f'{alg_key}_objective'] = 0
                csv_data[f'{alg_key}_total_observations'] = 0
                csv_data[f'{alg_key}_total_downlinks'] = 0
                csv_data[f'{alg_key}_data_downlinked_gb'] = 0
                csv_data[f'{alg_key}_runtime_minutes'] = 0
                csv_data[f'{alg_key}_propellant_used'] = 0
                csv_data[f'{alg_key}_evaluations'] = 0
                csv_data[f'{alg_key}_feasible_solutions'] = 0
                csv_data[f'{alg_key}_feasibility_rate'] = 0

        # Run Two-Phase GA
        print(f"\n{'='*80}")
        print(f"RUNNING TWO-PHASE GA")
        print(f"{'='*80}")
        
        try:
            tp_ga_solver = REOSSPTwoPhaseGA(params)
            tp_ga_results = tp_ga_solver.solve(pop_size=pop_size, n_generations=n_generations, verbose=False)
            
            csv_data['tp_ga_status'] = tp_ga_results['status']
            csv_data['tp_ga_objective'] = tp_ga_results['objective']
            csv_data['tp_ga_total_observations'] = tp_ga_results['total_observations']
            csv_data['tp_ga_total_downlinks'] = tp_ga_results['total_downlinks']
            csv_data['tp_ga_data_downlinked_gb'] = tp_ga_results['data_downlinked_gb']
            csv_data['tp_ga_runtime_minutes'] = tp_ga_results['runtime_minutes']
            csv_data['tp_ga_propellant_used'] = tp_ga_results['propellant_used']
            csv_data['tp_ga_generations'] = tp_ga_results['generations_completed']
            
            print(f"  Status: {tp_ga_results['status']}")
            print(f"  Objective: {tp_ga_results['objective']:.2f}")
            print(f"  Runtime: {tp_ga_results['runtime_minutes']:.2f} minutes")
            
        except Exception as e:
            print(f"  ✗ Two-Phase GA failed: {e}")
            csv_data['tp_ga_status'] = 'Failed'
            csv_data['tp_ga_objective'] = 0
            csv_data['tp_ga_total_observations'] = 0
            csv_data['tp_ga_total_downlinks'] = 0
            csv_data['tp_ga_data_downlinked_gb'] = 0
            csv_data['tp_ga_runtime_minutes'] = 0
            csv_data['tp_ga_propellant_used'] = 0
            csv_data['tp_ga_generations'] = 0

        # Write to CSV
        print(f"\n{'='*80}")
        print(f"SAVING RESULTS TO CSV")
        print(f"{'='*80}")

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
                file_exists = True
            
            writer.writerow(csv_data)

        print(f"✓ Results saved to: {csv_filename}")
        
        # Print instance summary
        print(f"\n{'='*80}")
        print(f"INSTANCE {instance_counter} SUMMARY")
        print(f"{'='*80}")
        print(f"{'Algorithm':<15} {'Objective':>12} {'Runtime (min)':>15} {'Status':>10}")
        print(f"{'-'*80}")
        
        for alg_key, (alg_name, _) in algorithms.items():
            obj = csv_data.get(f'{alg_key}_objective', 0)
            runtime = csv_data.get(f'{alg_key}_runtime_minutes', 0)
            status = csv_data.get(f'{alg_key}_status', 'N/A')
            print(f"{alg_name:<15} {obj:>12.2f} {runtime:>15.2f} {status:>10}")
        
        # Add Two-Phase GA to summary
        tp_obj = csv_data.get('tp_ga_objective', 0)
        tp_runtime = csv_data.get('tp_ga_runtime_minutes', 0)
        tp_status = csv_data.get('tp_ga_status', 'N/A')
        print(f"{'Two-Phase GA':<15} {tp_obj:>12.2f} {tp_runtime:>15.2f} {tp_status:>10}")

# Final summary
print(f"\n{'='*80}")
print(f"ALL INSTANCES COMPLETED")
print(f"{'='*80}")
print(f"Total instances tested: {instance_counter}")
print(f"Results saved to: {csv_filename}")
print(f"\nBreakdown:")
print(f"  Unavailable probabilities: {len(unavailable_probs)}")
print(f"  S-K combinations: {len(SK_pairs)}")
print(f"  Total: {len(unavailable_probs)} × {len(SK_pairs)} = {instance_counter}")
print(f"\nAlgorithms tested per instance:")
print(f"  - BGA (Binary Genetic Algorithm)")
print(f"  - IDE (Improved Differential Evolution)")
print(f"  - PSO (Particle Swarm Optimization)")
print(f"  - RGA (Real-Coded Genetic Algorithm)")
print(f"  - Two-Phase GA (Custom)")
