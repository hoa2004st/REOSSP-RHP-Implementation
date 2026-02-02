"""
Visualize comparison of all metaheuristic algorithms
Creates comprehensive charts comparing:
- BGA (Binary Genetic Algorithm)
- IDE (Improved Differential Evolution)
- PSO (Particle Swarm Optimization)
- RGA (Real-Coded Genetic Algorithm)
- Two-Phase GA (Custom)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

print("="*80)
print("LOADING METAHEURISTIC COMPARISON RESULTS")
print("="*80)

# Read the result file
try:
    df = pd.read_csv('results/results_all_metaheuristics.csv')
    print(f"✓ Results loaded successfully")
    print(f"  Total instances: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
except FileNotFoundError:
    print("✗ Results file not found. Run run_all_metaheuristics.py first.")
    exit(1)

# Define algorithms and colors
algorithms = ['BGA', 'IDE', 'PSO', 'RGA', 'Two-Phase GA']
alg_keys = ['bga', 'ide', 'pso', 'rga', 'tp_ga']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Unavailable probabilities to plot
unavailable_probs = sorted(df['unavailable_prob'].unique())

print(f"\nUnique unavailable probabilities: {unavailable_probs}")
print(f"Algorithms: {algorithms}")

# Define metrics to compare
metrics = {
    'Average Objective': 'objective',
    'Runtime (minutes)': 'runtime_minutes',
    'Feasibility Rate (%)': 'feasibility_rate'
}

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Metaheuristic Algorithm Comparison Across Unavailable Probability Settings', 
             fontsize=16, fontweight='bold')

print("\n" + "="*80)
print("CREATING COMPARISON CHARTS")
print("="*80)

axes_flat = axes.flatten()

for idx, (metric_name, metric_suffix) in enumerate(metrics.items()):
    ax = axes_flat[idx]
    
    print(f"\nProcessing: {metric_name}")
    
    # Calculate averages for each algorithm and probability
    for i, (alg_name, alg_key) in enumerate(zip(algorithms, alg_keys)):
        col_name = f'{alg_key}_{metric_suffix}'
        
        # Check if column exists
        if col_name not in df.columns:
            if alg_key == 'tp_ga' and metric_suffix == 'feasibility_rate':
                # Two-Phase GA doesn't have feasibility rate, skip
                print(f"  {alg_name}: Column {col_name} not found, skipping")
                continue
            else:
                print(f"  {alg_name}: Column {col_name} not found, skipping")
                continue
        
        data_points = []
        for prob in unavailable_probs:
            df_prob = df[df['unavailable_prob'] == prob]
            
            # Handle feasibility rate (convert to percentage)
            if metric_suffix == 'feasibility_rate':
                avg_value = df_prob[col_name].mean() * 100
            else:
                avg_value = df_prob[col_name].mean()
            
            data_points.append(avg_value)
        
        # Create line plot
        ax.plot(unavailable_probs, data_points, marker='o', linewidth=2.5, 
                markersize=8, label=alg_name, color=colors[i], alpha=0.8)
        
        # Add value labels on each point
        for j, prob in enumerate(unavailable_probs):
            if not np.isnan(data_points[j]):
                ax.text(prob, data_points[j], f'{data_points[j]:.1f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        print(f"  {alg_name}: {[f'{x:.2f}' for x in data_points]}")
    
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
output_filename = 'visualizations/all_metaheuristics_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison charts saved to {output_filename}")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for metric_name, metric_suffix in metrics.items():
    print(f"\n{metric_name}:")
    print("-" * 100)
    
    for prob in unavailable_probs:
        print(f"\n  Unavailable Probability: {prob:.1f}")
        
        for alg_name, alg_key in zip(algorithms, alg_keys):
            col_name = f'{alg_key}_{metric_suffix}'
            
            if col_name not in df.columns:
                if not (alg_key == 'tp_ga' and metric_suffix == 'feasibility_rate'):
                    print(f"    {alg_name:20s}: N/A")
                continue
            
            df_prob = df[df['unavailable_prob'] == prob]
            
            if metric_suffix == 'feasibility_rate':
                avg_value = df_prob[col_name].mean() * 100
                std_value = df_prob[col_name].std() * 100
                print(f"    {alg_name:20s}: Mean = {avg_value:8.2f}%, Std = {std_value:8.2f}%")
            else:
                avg_value = df_prob[col_name].mean()
                std_value = df_prob[col_name].std()
                print(f"    {alg_name:20s}: Mean = {avg_value:8.2f}, Std = {std_value:8.2f}")

# Create a ranking table
print("\n" + "="*80)
print("ALGORITHM RANKING BY OBJECTIVE (Overall Average)")
print("="*80)

rankings = []
for alg_name, alg_key in zip(algorithms, alg_keys):
    col_name = f'{alg_key}_objective'
    if col_name in df.columns:
        avg_obj = df[col_name].mean()
        avg_runtime = df[f'{alg_key}_runtime_minutes'].mean()
        rankings.append({
            'Algorithm': alg_name,
            'Avg Objective': avg_obj,
            'Avg Runtime (min)': avg_runtime
        })

rankings_df = pd.DataFrame(rankings).sort_values('Avg Objective', ascending=False)
print("\n" + rankings_df.to_string(index=False))

# Create bar chart for overall comparison
print("\n" + "="*80)
print("CREATING OVERALL COMPARISON BAR CHART")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(algorithms))
width = 0.25

# Get overall averages
objectives = [rankings_df[rankings_df['Algorithm'] == alg]['Avg Objective'].values[0] 
              if alg in rankings_df['Algorithm'].values else 0 for alg in algorithms]

runtimes = [rankings_df[rankings_df['Algorithm'] == alg]['Avg Runtime (min)'].values[0] 
            if alg in rankings_df['Algorithm'].values else 0 for alg in algorithms]

# Normalize runtime for visualization (scale to similar range as objectives)
max_obj = max(objectives)
max_runtime = max(runtimes)
runtimes_normalized = [r * (max_obj / max_runtime) for r in runtimes]

# Create bars
bars1 = ax.bar(x - width/2, objectives, width, label='Avg Objective', color='#2ca02c', alpha=0.8)
bars2 = ax.bar(x + width/2, runtimes_normalized, width, label='Avg Runtime (normalized)', color='#ff7f0e', alpha=0.8)

# Customize
ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Overall Algorithm Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, runtime in zip(bars2, runtimes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{runtime:.1f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
output_filename = 'visualizations/overall_comparison_bars.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✓ Bar chart saved to {output_filename}")

plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
