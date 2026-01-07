import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Read the three result files
df_0_0 = pd.read_csv('results/results_0.0.csv')
df_0_1 = pd.read_csv('results/results_0.1.csv')
df_0_2 = pd.read_csv('results/results_0.2.csv')

# Add unavailable_probability column to each dataframe
df_0_0['unavailable_prob'] = 0.0
df_0_1['unavailable_prob'] = 0.1
df_0_2['unavailable_prob'] = 0.2

# Combine all dataframes
df_all = pd.concat([df_0_0, df_0_1, df_0_2], ignore_index=True)

# Calculate averages for each method and unavailable probability
metrics = {
    'Average Objective': {
        'EOSSP Baseline': 'eossp_objective',
        'REOSSP Exact': 'reossp_exact_objective',
        'REOSSP RHP': 'reossp_rhp_objective'
    },
    'Figure of Merit': {
        'EOSSP Baseline': 'eossp_figure_of_merit',
        'REOSSP Exact': 'reossp_exact_figure_of_merit',
        'REOSSP RHP': 'reossp_rhp_figure_of_merit'
    },
    'Runtime (minutes)': {
        'EOSSP Baseline': 'eossp_runtime_minutes',
        'REOSSP Exact': 'reossp_exact_runtime_minutes',
        'REOSSP RHP': 'reossp_rhp_runtime_minutes'
    }
}

# Prepare data for plotting
unavailable_probs = [0.0, 0.1, 0.2]
methods = ['EOSSP Baseline', 'REOSSP Exact', 'REOSSP RHP']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparison of Methods Across Different Unavailable Probability Settings', fontsize=16, fontweight='bold')

for idx, (metric_name, method_cols) in enumerate(metrics.items()):
    ax = axes[idx]
    
    # Calculate averages for each method and probability
    data = []
    for prob in unavailable_probs:
        df_prob = df_all[df_all['unavailable_prob'] == prob]
        row = []
        for method in methods:
            col = method_cols[method]
            avg_value = df_prob[col].mean()
            row.append(avg_value)
        data.append(row)
    
    # Convert to numpy array for easier manipulation
    data = np.array(data)
    
    # Set up bar positions
    x = np.arange(len(unavailable_probs))
    width = 0.25
    
    # Create bars for each method
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[:, i], width, label=method, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Customize subplot
    ax.set_xlabel('Unavailable Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.1f}' for p in unavailable_probs])
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout and save
plt.tight_layout()
plt.savefig('visualizations/method_comparison_bar_charts.png', dpi=300, bbox_inches='tight')
print("Bar charts saved to visualizations/method_comparison_bar_charts.png")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for metric_name, method_cols in metrics.items():
    print(f"\n{metric_name}:")
    print("-" * 60)
    for prob in unavailable_probs:
        df_prob = df_all[df_all['unavailable_prob'] == prob]
        print(f"\n  Unavailable Probability: {prob:.1f}")
        for method in methods:
            col = method_cols[method]
            avg_value = df_prob[col].mean()
            std_value = df_prob[col].std()
            print(f"    {method:20s}: Mean = {avg_value:8.2f}, Std = {std_value:8.2f}")
