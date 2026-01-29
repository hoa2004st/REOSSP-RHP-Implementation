"""
Run Two-Phase Genetic Algorithm for REOSSP
Tests all combinations of S, K, and unavailable_slot_probability
Saves results to results_tp_ga.csv
"""

import csv
import os
from datetime import datetime
from parameters import InstanceParameters
from reossp_two_phase_ga import REOSSPTwoPhaseGA

print("="*80)
print("TESTING TWO-PHASE GENETIC ALGORITHM")
print("="*80)

# Problem parameters
SK_pairs = [(8, 5), (8, 6), (9, 5), (9, 6), (12, 5), (12, 6)]
# unavailable_probs = [0.0, 0.1, 0.2, 0.5, 1.0]
SK_pairs = [(8, 5)]
unavailable_probs = [0.1]
J_sk, T = 20, 36*24*2

# GA parameters
pop_size = 50
n_generations = 30

# Initialize CSV file
csv_filename = "results/results_tp_ga_v2.csv"
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
        print(f"  Total timesteps (T): {params.T}")
        print(f"  Timesteps per stage: {params.T // params.S}")
        print(f"  Chromosome size: {params.K} × {params.S} = {params.K * params.S} genes")
        print(f"  Observations possible: {params.V_target.sum()}")
        print(f"  Downlinks possible: {params.V_ground.sum()}")
        print(f"  Unavailable slots: {params.unavailable_slots.sum()}/{params.S * params.J_sk}")

        # Run Two-Phase GA
        print(f"\n" + "="*80)
        print(f"RUNNING TWO-PHASE GA")
        print(f"="*80)
        
        solver = REOSSPTwoPhaseGA(params)
        results = solver.solve(pop_size=pop_size, n_generations=n_generations, verbose=True)
        
        print(f"\nResults:")
        print(f"  Status: {results['status']}")
        print(f"  Objective: {results['objective']:.2f}")
        print(f"  Runtime: {results['runtime_minutes']:.2f} minutes")
        print(f"  Total observations: {results['total_observations']}")
        print(f"  Total downlinks: {results['total_downlinks']}")
        print(f"  Data downlinked: {results['data_downlinked_gb']:.4f} GB")
        print(f"  Propellant used: {results['propellant_used']:.2f} m/s")
        print(f"  Generations: {results['generations_completed']}")

        # Save results to CSV
        print(f"\n" + "="*80)
        print(f"SAVING RESULTS TO CSV")
        print(f"="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for CSV
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
            'n_generations': n_generations,
            
            # Two-Phase GA results
            'tp_ga_status': results['status'],
            'tp_ga_objective': results['objective'],
            'tp_ga_total_observations': results['total_observations'],
            'tp_ga_total_downlinks': results['total_downlinks'],
            'tp_ga_data_downlinked_gb': results['data_downlinked_gb'],
            'tp_ga_runtime_minutes': results['runtime_minutes'],
            'tp_ga_propellant_used': results['propellant_used'],
            'tp_ga_generations_completed': results['generations_completed'],
            'tp_ga_final_population_size': results['final_population_size'],
        }

        # Calculate figure of merit (objective / runtime)
        if results['runtime_minutes'] > 0:
            csv_data['tp_ga_figure_of_merit'] = results['objective'] / results['runtime_minutes']
        else:
            csv_data['tp_ga_figure_of_merit'] = 'N/A'

        # Write to CSV (append mode)
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
                file_exists = True  # Set to True after writing header
            
            writer.writerow(csv_data)

        print(f"✓ Results saved to: {csv_filename}")
        print(f"  Total columns: {len(csv_data)}")

# Final summary
print(f"\n" + "="*80)
print(f"ALL INSTANCES COMPLETED")
print(f"="*80)
print(f"Total instances tested: {instance_counter}")
print(f"Results saved to: {csv_filename}")
print(f"\nBreakdown:")
print(f"  Unavailable probabilities: {len(unavailable_probs)}")
print(f"  S-K combinations: {len(SK_pairs)}")
print(f"  Total: {len(unavailable_probs)} × {len(SK_pairs)} = {instance_counter}")
