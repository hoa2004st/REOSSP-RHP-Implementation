"""
Temporary visualization comparing 4 methods at unavailable_prob = 0.1
Compares: EOSSP Baseline, REOSSP Exact, REOSSP RHP, REOSSP Two-Phase GA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

print("="*80)
print("COMPARING 4 METHODS AT UNAVAILABLE_PROB = 0.1")
print("="*80)

# Read the result files
df_baseline = pd.read_csv('results/results_0.1.csv')  # Has EOSSP, REOSSP Exact, REOSSP RHP
df_tp_ga = pd.read_csv('results/results_tp_ga.csv')    # Has Two-Phase GA

print(f"\nBaseline methods data: {len(df_baseline)} instances")
print(f"Two-Phase GA data: {len(df_tp_ga)} instances")

# Filter Two-Phase GA for unavailable_prob = 0.1 (baseline is already filtered by filename)
df_tp_ga = df_tp_ga[df_tp_ga['unavailable_prob'] == 0.1].copy()

print(f"\nBaseline methods (already filtered): {len(df_baseline)} instances")
print(f"Filtered Two-Phase GA: {len(df_tp_ga)} instances")

# Get unique S-K pairs
SK_pairs = sorted(df_baseline[['S', 'K']].drop_duplicates().values.tolist())
print(f"\nS-K pairs: {SK_pairs}")

# Prepare data for plotting
methods = ['EOSSP Baseline', 'REOSSP Exact', 'REOSSP RHP', 'REOSSP Two-Phase GA']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Create x-axis labels from S-K pairs
x_labels = [f"S={s}, K={k}" for s, k in SK_pairs]
x_pos = np.arange(len(x_labels))

# Metrics to compare
metrics = {
    'Objective': {
        'ylabel': 'Objective Value',
        'columns': ['eossp_objective', 'reossp_exact_objective', 'reossp_rhp_objective', 'tp_ga_objective']
    },
    'Runtime': {
        'ylabel': 'Runtime (minutes)',
        'columns': ['eossp_runtime_minutes', 'reossp_exact_runtime_minutes', 'reossp_rhp_runtime_minutes', 'tp_ga_runtime_minutes']
    },
    'Figure of Merit': {
        'ylabel': 'Figure of Merit (Obj/Runtime)',
        'columns': ['eossp_figure_of_merit', 'reossp_exact_figure_of_merit', 'reossp_rhp_figure_of_merit', 'tp_ga_figure_of_merit']
    }
}

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle('Method Comparison at Unavailable Probability = 0.1', fontsize=18, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

for idx, (metric_name, metric_info) in enumerate(metrics.items()):
    ax = axes[idx]
    
    # Collect data for each method
    data_by_method = []
    
    for i, (s, k) in enumerate(SK_pairs):
        row_baseline = df_baseline[(df_baseline['S'] == s) & (df_baseline['K'] == k)]
        row_tp_ga = df_tp_ga[(df_tp_ga['S'] == s) & (df_tp_ga['K'] == k)]
        
        if len(row_baseline) > 0 and len(row_tp_ga) > 0:
            # Get values for each method
            values = []
            for col_idx, col_name in enumerate(metric_info['columns']):
                if col_name is None:
                    values.append(0)  # No data for this method
                elif col_idx < 3:  # Baseline methods
                    values.append(row_baseline[col_name].values[0])
                else:  # Two-Phase GA
                    values.append(row_tp_ga[col_name].values[0])
            data_by_method.append(values)
        else:
            data_by_method.append([0, 0, 0, 0])
    
    # Convert to numpy array for easier manipulation
    data_by_method = np.array(data_by_method)  # Shape: (n_pairs, n_methods)
    
    # Plot each method
    for method_idx, method_name in enumerate(methods):
        ax.plot(x_pos, data_by_method[:, method_idx], 
                marker=markers[method_idx], linewidth=2.5, markersize=10,
                label=method_name, color=colors[method_idx], alpha=0.8)
        
        # Add value labels on each point
        for i in range(len(x_pos)):
            value = data_by_method[i, method_idx]
            ax.text(x_pos[i], value, f'{value:.1f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold',
                   color=colors[method_idx])
    
    # Customize subplot
    ax.set_xlabel('Problem Instance (S, K)', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_info['ylabel'], fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

# Adjust layout and save
plt.tight_layout()
plt.savefig('visualizations/temp_4methods_comparison_0.1.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: visualizations/temp_4methods_comparison_0.1.png")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
