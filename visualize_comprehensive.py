"""
Comprehensive Visualization: All Methods Comparison
Compares traditional methods with metaheuristic algorithms:
- EOSSP Baseline
- REOSSP Exact
- REOSSP RHP
- BGA (Binary Genetic Algorithm)
- IDE (Improved Differential Evolution)
- PSO (Particle Swarm Optimization)
- RGA (Real-Coded Genetic Algorithm)
- Two-Phase GA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

print("="*80)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*80)

# Read traditional method results
print("\nLoading traditional method results...")
try:
    df_0_0 = pd.read_csv('results/results_0.0.csv')
    df_0_2 = pd.read_csv('results/results_0.2.csv')
    df_0_5 = pd.read_csv('results/results_0.5.csv')
    df_0_8 = pd.read_csv('results/results_0.8.csv')
    df_1_0 = pd.read_csv('results/results_1.0.csv')
    
    # Add unavailable_probability column
    df_0_0['unavailable_prob'] = 0.0
    df_0_2['unavailable_prob'] = 0.2
    df_0_5['unavailable_prob'] = 0.5
    df_0_8['unavailable_prob'] = 0.8
    df_1_0['unavailable_prob'] = 1.0
    
    df_traditional = pd.concat([df_0_0, df_0_2, df_0_5, df_0_8, df_1_0], ignore_index=True)
    print(f"✓ Traditional methods loaded: {len(df_traditional)} instances")
except Exception as e:
    print(f"✗ Error loading traditional results: {e}")
    df_traditional = None

# Read metaheuristic results
print("\nLoading metaheuristic results...")
try:
    df_meta = pd.read_csv('results/results_all_metaheuristics.csv')
    print(f"✓ Metaheuristic methods loaded: {len(df_meta)} instances")
except Exception as e:
    print(f"✗ Error loading metaheuristic results: {e}")
    df_meta = None

if df_traditional is None or df_meta is None:
    print("\n✗ Cannot proceed without both result files.")
    print("Run the following first:")
    print("  1. run.py (for traditional methods)")
    print("  2. run_all_metaheuristics.py (for metaheuristic methods)")
    exit(1)

# Define all methods
traditional_methods = {
    'EOSSP Baseline': ('eossp_objective', 'eossp_runtime_minutes'),
    'REOSSP Exact': ('reossp_exact_objective', 'reossp_exact_runtime_minutes'),
    'REOSSP RHP': ('reossp_rhp_objective', 'reossp_rhp_runtime_minutes')
}

metaheuristic_methods = {
    'BGA': ('bga_objective', 'bga_runtime_minutes'),
    'IDE': ('ide_objective', 'ide_runtime_minutes'),
    'PSO': ('pso_objective', 'pso_runtime_minutes'),
    'RGA': ('rga_objective', 'rga_runtime_minutes'),
    'Two-Phase GA': ('tp_ga_objective', 'tp_ga_runtime_minutes')
}

all_methods = {**traditional_methods, **metaheuristic_methods}

# Define colors (colorblind-friendly palette)
colors = {
    'EOSSP Baseline': '#1f77b4',    # Blue
    'REOSSP Exact': '#ff7f0e',      # Orange
    'REOSSP RHP': '#2ca02c',        # Green
    'BGA': '#d62728',                # Red
    'IDE': '#9467bd',                # Purple
    'PSO': '#8c564b',                # Brown
    'RGA': '#e377c2',                # Pink
    'Two-Phase GA': '#17becf'        # Cyan
}

# Markers for differentiation
markers = {
    'EOSSP Baseline': 'o',
    'REOSSP Exact': 's',
    'REOSSP RHP': '^',
    'BGA': 'D',
    'IDE': 'v',
    'PSO': '<',
    'RGA': '>',
    'Two-Phase GA': '*'
}

unavailable_probs = sorted(df_meta['unavailable_prob'].unique())
print(f"\nUnique unavailable probabilities: {unavailable_probs}")

# ============================================================================
# FIGURE 1: Objective Comparison
# ============================================================================
print("\n" + "="*80)
print("CREATING OBJECTIVE COMPARISON CHART")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

for method_name, (obj_col, runtime_col) in all_methods.items():
    data_points = []
    
    # Determine which dataframe to use
    if method_name in traditional_methods:
        df_source = df_traditional
    else:
        df_source = df_meta
    
    for prob in unavailable_probs:
        df_prob = df_source[df_source['unavailable_prob'] == prob]
        avg_value = df_prob[obj_col].mean()
        data_points.append(avg_value)
    
    # Plot line
    ax.plot(unavailable_probs, data_points, 
            marker=markers[method_name], 
            linewidth=2.5, 
            markersize=10, 
            label=method_name, 
            color=colors[method_name], 
            alpha=0.85)
    
    print(f"  {method_name:20s}: {[f'{x:.1f}' for x in data_points]}")

ax.set_xlabel('Unavailable Slot Probability', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Objective Value', fontsize=13, fontweight='bold')
ax.set_title('Objective Comparison: All Methods Across Unavailable Probabilities', 
             fontsize=15, fontweight='bold')
ax.set_xticks(unavailable_probs)
ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('visualizations/all_methods_objective_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/all_methods_objective_comparison.png")

# ============================================================================
# FIGURE 2: Runtime Comparison
# ============================================================================
print("\n" + "="*80)
print("CREATING RUNTIME COMPARISON CHART")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

for method_name, (obj_col, runtime_col) in all_methods.items():
    data_points = []
    
    if method_name in traditional_methods:
        df_source = df_traditional
    else:
        df_source = df_meta
    
    for prob in unavailable_probs:
        df_prob = df_source[df_source['unavailable_prob'] == prob]
        avg_value = df_prob[runtime_col].mean()
        data_points.append(avg_value)
    
    ax.plot(unavailable_probs, data_points, 
            marker=markers[method_name], 
            linewidth=2.5, 
            markersize=10, 
            label=method_name, 
            color=colors[method_name], 
            alpha=0.85)
    
    print(f"  {method_name:20s}: {[f'{x:.2f}' for x in data_points]}")

ax.set_xlabel('Unavailable Slot Probability', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Runtime (minutes)', fontsize=13, fontweight='bold')
ax.set_title('Runtime Comparison: All Methods Across Unavailable Probabilities', 
             fontsize=15, fontweight='bold')
ax.set_xticks(unavailable_probs)
ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('visualizations/all_methods_runtime_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/all_methods_runtime_comparison.png")

# ============================================================================
# FIGURE 3: Combined 2-Metric Dashboard
# ============================================================================
print("\n" + "="*80)
print("CREATING COMBINED DASHBOARD")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Comprehensive Method Comparison Dashboard', fontsize=16, fontweight='bold')

metrics = [
    ('Objective', 0),
    ('Runtime (minutes)', 1)
]

for metric_name, metric_idx in metrics:
    if metric_idx == 0:
        ax = axes[0, 0]
    else:
        ax = axes[0, 1]
    
    for method_name, cols in all_methods.items():
        data_points = []
        
        if method_name in traditional_methods:
            df_source = df_traditional
        else:
            df_source = df_meta
        
        col_name = cols[metric_idx]
        
        for prob in unavailable_probs:
            df_prob = df_source[df_source['unavailable_prob'] == prob]
            avg_value = df_prob[col_name].mean()
            data_points.append(avg_value)
        
        ax.plot(unavailable_probs, data_points, 
                marker=markers[method_name], 
                linewidth=2, 
                markersize=8, 
                label=method_name, 
                color=colors[method_name], 
                alpha=0.85)
    
    ax.set_xlabel('Unavailable Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xticks(unavailable_probs)
    ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

# Third subplot: Overall ranking bar chart
ax = axes[1, 0]
method_names = list(all_methods.keys())
avg_objectives = []
avg_runtimes = []

for method_name, (obj_col, runtime_col) in all_methods.items():
    if method_name in traditional_methods:
        df_source = df_traditional
    else:
        df_source = df_meta
    
    avg_objectives.append(df_source[obj_col].mean())
    avg_runtimes.append(df_source[runtime_col].mean())

# Normalize runtime for visualization
max_obj = max(avg_objectives)
max_runtime = max(avg_runtimes)
runtimes_normalized = [r * (max_obj / max_runtime) for r in avg_runtimes]

x = np.arange(len(method_names))
width = 0.35

bars1 = ax.bar(x - width/2, avg_objectives, width, label='Avg Objective', alpha=0.8)
bars2 = ax.bar(x + width/2, runtimes_normalized, width, label='Avg Runtime (normalized)', alpha=0.8)

# Color bars
for i, bar in enumerate(bars1):
    bar.set_color(colors[method_names[i]])
for i, bar in enumerate(bars2):
    bar.set_color(colors[method_names[i]])
    bar.set_alpha(0.5)

ax.set_xlabel('Method', fontsize=11, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Overall Average Performance', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/all_methods_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/all_methods_dashboard.png")

# ============================================================================
# FIGURE 5: Method Categorization (Traditional vs Metaheuristic)
# ============================================================================
print("\n" + "="*80)
print("CREATING CATEGORY COMPARISON CHART")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Traditional vs Metaheuristic Methods', fontsize=16, fontweight='bold')

# Left plot: Traditional methods
ax = axes[0]
for method_name, (obj_col, runtime_col) in traditional_methods.items():
    data_points = []
    for prob in unavailable_probs:
        df_prob = df_traditional[df_traditional['unavailable_prob'] == prob]
        avg_value = df_prob[obj_col].mean()
        data_points.append(avg_value)
    
    ax.plot(unavailable_probs, data_points, 
            marker=markers[method_name], 
            linewidth=3, 
            markersize=12, 
            label=method_name, 
            color=colors[method_name], 
            alpha=0.9)

ax.set_xlabel('Unavailable Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Objective', fontsize=12, fontweight='bold')
ax.set_title('Traditional Methods', fontsize=13, fontweight='bold')
ax.set_xticks(unavailable_probs)
ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

# Right plot: Metaheuristic methods
ax = axes[1]
for method_name, (obj_col, runtime_col) in metaheuristic_methods.items():
    data_points = []
    for prob in unavailable_probs:
        df_prob = df_meta[df_meta['unavailable_prob'] == prob]
        avg_value = df_prob[obj_col].mean()
        data_points.append(avg_value)
    
    ax.plot(unavailable_probs, data_points, 
            marker=markers[method_name], 
            linewidth=3, 
            markersize=12, 
            label=method_name, 
            color=colors[method_name], 
            alpha=0.9)

ax.set_xlabel('Unavailable Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Objective', fontsize=12, fontweight='bold')
ax.set_title('Metaheuristic Methods', fontsize=13, fontweight='bold')
ax.set_xticks(unavailable_probs)
ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('visualizations/traditional_vs_metaheuristic.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: visualizations/traditional_vs_metaheuristic.png")

# ============================================================================
# Summary Statistics Table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary_data = []
for method_name, (obj_col, runtime_col) in all_methods.items():
    if method_name in traditional_methods:
        df_source = df_traditional
    else:
        df_source = df_meta
    
    summary_data.append({
        'Method': method_name,
        'Avg Objective': df_source[obj_col].mean(),
        'Std Objective': df_source[obj_col].std(),
        'Avg Runtime (min)': df_source[runtime_col].mean(),
        'Std Runtime (min)': df_source[runtime_col].std(),
        'Category': 'Traditional' if method_name in traditional_methods else 'Metaheuristic'
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Avg Objective', ascending=False)

print("\n" + summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('visualizations/method_comparison_summary.csv', index=False)
print("\n✓ Saved: visualizations/method_comparison_summary.csv")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. all_methods_objective_comparison.png")
print("  2. all_methods_runtime_comparison.png")
print("  3. all_methods_dashboard.png")
print("  4. traditional_vs_metaheuristic.png")
print("  5. method_comparison_summary.csv")
