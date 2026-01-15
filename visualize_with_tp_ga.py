"""
Visualize comparison of all methods including Two-Phase GA
Creates line charts comparing:
- EOSSP Baseline
- REOSSP Exact
- REOSSP RHP
- Two-Phase GA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

print("="*80)
print("LOADING RESULTS")
print("="*80)

# Read the result files
df_0_0 = pd.read_csv('results/results_0.0.csv')
df_0_1 = pd.read_csv('results/results_0.1.csv')
df_0_2 = pd.read_csv('results/results_0.2.csv')
df_0_5 = pd.read_csv('results/results_0.5.csv')
df_1_0 = pd.read_csv('results/results_1.0.csv')

# Add unavailable_probability column to each dataframe
df_0_0['unavailable_prob'] = 0.0
df_0_1['unavailable_prob'] = 0.1
df_0_2['unavailable_prob'] = 0.2
df_0_5['unavailable_prob'] = 0.5
df_1_0['unavailable_prob'] = 1.0

# Combine all dataframes
df_all = pd.concat([df_0_0, df_0_1, df_0_2, df_0_5, df_1_0], ignore_index=True)

# Read Two-Phase GA results
try:
    df_tp_ga = pd.read_csv('results/results_tp_ga.csv')
    has_tp_ga = True
    print("✓ Two-Phase GA results loaded")
    print(f"  Total instances: {len(df_tp_ga)}")
except FileNotFoundError:
    print("⚠ Two-Phase GA results not found. Run run_tp_ga.py first.")
    has_tp_ga = False

print(f"✓ Other method results loaded")
print(f"  Total instances: {len(df_all)}")

# Prepare data for plotting
unavailable_probs = [0.0, 0.1, 0.2, 0.5, 1.0]

if has_tp_ga:
    methods = ['EOSSP Baseline', 'REOSSP Exact', 'REOSSP RHP', 'Two-Phase GA']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    metrics = {
        'Average Objective': {
            'EOSSP Baseline': ('eossp_objective', df_all),
            'REOSSP Exact': ('reossp_exact_objective', df_all),
            'REOSSP RHP': ('reossp_rhp_objective', df_all),
            'Two-Phase GA': ('tp_ga_objective', df_tp_ga)
        },
        'Runtime (minutes)': {
            'EOSSP Baseline': ('eossp_runtime_minutes', df_all),
            'REOSSP Exact': ('reossp_exact_runtime_minutes', df_all),
            'REOSSP RHP': ('reossp_rhp_runtime_minutes', df_all),
            'Two-Phase GA': ('tp_ga_runtime_minutes', df_tp_ga)
        },
        'Figure of Merit': {
            'EOSSP Baseline': ('eossp_figure_of_merit', df_all),
            'REOSSP Exact': ('reossp_exact_figure_of_merit', df_all),
            'REOSSP RHP': ('reossp_rhp_figure_of_merit', df_all),
            'Two-Phase GA': ('tp_ga_figure_of_merit', df_tp_ga)
        }
    }
else:
    methods = ['EOSSP Baseline', 'REOSSP Exact', 'REOSSP RHP']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    metrics = {
        'Average Objective': {
            'EOSSP Baseline': ('eossp_objective', df_all),
            'REOSSP Exact': ('reossp_exact_objective', df_all),
            'REOSSP RHP': ('reossp_rhp_objective', df_all)
        },
        'Runtime (minutes)': {
            'EOSSP Baseline': ('eossp_runtime_minutes', df_all),
            'REOSSP Exact': ('reossp_exact_runtime_minutes', df_all),
            'REOSSP RHP': ('reossp_rhp_runtime_minutes', df_all)
        },
        'Figure of Merit': {
            'EOSSP Baseline': ('eossp_figure_of_merit', df_all),
            'REOSSP Exact': ('reossp_exact_figure_of_merit', df_all),
            'REOSSP RHP': ('reossp_rhp_figure_of_merit', df_all)
        }
    }

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparison of Methods Across Different Unavailable Probability Settings', fontsize=16, fontweight='bold')

print("\n" + "="*80)
print("CREATING LINE CHARTS")
print("="*80)

for idx, (metric_name, method_data) in enumerate(metrics.items()):
    ax = axes[idx]
    
    print(f"\nProcessing: {metric_name}")
    
    # Calculate averages for each method and probability
    for i, method in enumerate(methods):
        col_name, df_source = method_data[method]
        
        data_points = []
        for prob in unavailable_probs:
            df_prob = df_source[df_source['unavailable_prob'] == prob]
            avg_value = df_prob[col_name].mean()
            data_points.append(avg_value)
        
        # Create line plot
        ax.plot(unavailable_probs, data_points, marker='o', linewidth=2.5, 
                markersize=8, label=method, color=colors[i], alpha=0.8)
        
        # Add value labels on each point
        for j, prob in enumerate(unavailable_probs):
            if not np.isnan(data_points[j]):
                ax.text(prob, data_points[j], f'{data_points[j]:.1f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        print(f"  {method}: {data_points}")
    
    # Customize subplot
    ax.set_xlabel('Unavailable Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xticks(unavailable_probs)
    ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)  # Y-axis starts at 0

# Adjust layout and save
plt.tight_layout()
output_filename = 'visualizations/method_comparison_with_tp_ga.png' if has_tp_ga else 'visualizations/method_comparison_line_charts.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Line charts saved to {output_filename}")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for metric_name, method_data in metrics.items():
    print(f"\n{metric_name}:")
    print("-" * 80)
    for prob in unavailable_probs:
        print(f"\n  Unavailable Probability: {prob:.1f}")
        for method in methods:
            col_name, df_source = method_data[method]
            df_prob = df_source[df_source['unavailable_prob'] == prob]
            avg_value = df_prob[col_name].mean()
            std_value = df_prob[col_name].std()
            print(f"    {method:20s}: Mean = {avg_value:8.2f}, Std = {std_value:8.2f}")

# Additional analysis for Two-Phase GA if available
if has_tp_ga:
    print("\n" + "="*80)
    print("TWO-PHASE GA SPECIFIC ANALYSIS")
    print("="*80)
    
    for prob in unavailable_probs:
        df_prob = df_tp_ga[df_tp_ga['unavailable_prob'] == prob]
        
        print(f"\n  Unavailable Probability: {prob:.1f}")
        print(f"    Avg Generations: {df_prob['tp_ga_generations_completed'].mean():.1f}")
        print(f"    Avg Population Size: {df_prob['tp_ga_final_population_size'].mean():.1f}")
        print(f"    Avg Propellant Used: {df_prob['tp_ga_propellant_used'].mean():.2f} m/s")
        print(f"    Avg Observations: {df_prob['tp_ga_total_observations'].mean():.1f}")
        print(f"    Avg Downlinks: {df_prob['tp_ga_total_downlinks'].mean():.1f}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
