import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Set a professional style for Matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the maximum step for the x-axis
MAX_STEP = 1500

# Define the ablation scenarios (all 8 agents) and corresponding labels/colors
ABLATION_SCENARIOS = {
    '8agents': {'label': 'Full Model (MAPPO)', 'color': '#2ca02c'},          # Green
    '8agents_0highspeed': {'label': 'w/o High-Speed Reward', 'color': '#d62728'}, # Red
    '8agents_0incentive': {'label': 'w/o Incentive Reward', 'color': '#9467bd'} # Purple
}
# ---------------------

def plot_ablation_subplot(ax, df, metric_type, scenarios, ylabel):
    """
    Plots the mean performance with min/max error bands onto a given subplot axis (ax).
    This function expects 'Step' to be the x-axis name.
    """
    if df is None or df.empty:
        # Create a placeholder if data is missing, so the figure structure is maintained
        ax.text(0.5, 0.5, 'Data Missing', transform=ax.transAxes, 
                ha='center', va='center', color='red', fontsize=16)
        return

    # Filter data up to the maximum step
    df_filtered = df[df['Step'] <= MAX_STEP]

    for scenario_key, config in scenarios.items():
        # Construct the column names based on the metric type and scenario key
        # metric_type is 'Reward' or 'Episode_Length'
        mean_col = f'{scenario_key} - Training_{metric_type}_Mean'
        min_col = f'{scenario_key} - Training_{metric_type}_Mean__MIN'
        max_col = f'{scenario_key} - Training_{metric_type}_Mean__MAX'
        
        # Check for mean column existence
        if mean_col not in df_filtered.columns:
            print(f"Warning: Column {mean_col} not found. Skipping scenario {scenario_key}.")
            continue

        # 1. Plot the Mean Line
        ax.plot(
            df_filtered['Step'], 
            df_filtered[mean_col], 
            label=config['label'], 
            color=config['color'], 
            linewidth=2
        )

        # 2. Plot the Min/Max Error Band
        # Robustly handle the MIN/MAX column check (defaulting to mean if min/max not found)
        min_data = df_filtered[min_col] if min_col in df_filtered.columns else df_filtered[mean_col]
        max_data = df_filtered[max_col] if max_col in df_filtered.columns else df_filtered[mean_col]

        ax.fill_between(
            df_filtered['Step'],
            min_data,
            max_data,
            color=config['color'],
            alpha=0.2, 
            linewidth=0
        )

    # --- Plot Customization ---
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(0, MAX_STEP)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='lower right', fontsize=12)
    

def plot_combined_ablation(reward_df, length_df, scenarios, output_filename):
    """
    Creates a single figure with two subplots: Reward (top) and Episode Length (bottom).
    Sets individual titles and X-labels for each subplot.
    """
    # Create 2 rows, 1 column, sharing the X-axis
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # --- Top Subplot: Reward ---
    ax_reward = axes[0]
    plot_ablation_subplot(
        ax=ax_reward,
        df=reward_df,
        metric_type='Reward',
        scenarios=scenarios,
        ylabel='Average Training Reward'
    )
    # Set Title and X-Label for Reward Plot (as requested)
    ax_reward.set_title('Ablation Study: Average Training Reward', fontsize=14)
    # ax_reward.set_xlabel('Step', fontsize=14) 
    
    # --- Bottom Subplot: Episode Length ---
    ax_length = axes[1]
    plot_ablation_subplot(
        ax=ax_length,
        df=length_df,
        metric_type='Episode_Length',
        scenarios=scenarios,
        ylabel='Average Episode Length'
    )
    # Set Title and X-Label for Length Plot (as requested)
    ax_length.set_title('Ablation Study: Average Episode Length', fontsize=14)
    ax_length.set_xlabel('Step', fontsize=14)
    
    # 3. Global Legend and Saving
    
    # Use fig.suptitle to set an empty string 
    # fig.suptitle('asdf', fontsize=16) 
    
    # Create a single legend using the handles from the bottom plot
    # handles, labels = ax_length.get_legend_handles_labels()
    # fig.legend(handles, labels, 
    #            loc='lower center', 
    #            ncol=3, 
    #            fontsize=12,
    #            bbox_to_anchor=(0.5, 0.0),
    #            title='Scenario')
    
    # Adjust layout to prevent overlap and make space for the global legend
    fig.subplots_adjust(bottom=0.15, hspace=0.15)
    
    # Save the figure
    fig.savefig(output_filename, dpi=300)
    print(f"Figure saved to {output_filename}")


# --- Main Execution ---

# 1. Load Data
try:
    ablation_reward_df = pd.read_csv('ablation_reward.csv')
except FileNotFoundError:
    print("Error: 'ablation_reward.csv' not found.")
    ablation_reward_df = None

try:
    ablation_length_df = pd.read_csv('ablation_length.csv')
except FileNotFoundError:
    print("Error: 'ablation_length.csv' not found.")
    ablation_length_df = None

# 2. Generate Combined Plot
plot_combined_ablation(
    reward_df=ablation_reward_df,
    length_df=ablation_length_df,
    scenarios=ABLATION_SCENARIOS,
    output_filename='ablation_combined_vs_step.png'
)