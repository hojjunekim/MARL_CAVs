import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
MAX_STEP = 1500

ALL_SCENARIOS = {
    # MAPPO Cases
    'MAPPO_1': {'col_prefix': '1agent', 'label': 'MAPPO, 1 Agent', 'color': '#1f77b4', 'algo': 'mappo'},  
    'MAPPO_4': {'col_prefix': '4agents', 'label': 'MAPPO, 4 Agents', 'color': '#ff7f0e', 'algo': 'mappo'}, 
    'MAPPO_8': {'col_prefix': '8agents', 'label': 'MAPPO, 8 Agents', 'color': '#2ca02c', 'algo': 'mappo'}, 
    # PPO Cases
    'PPO_4': {'col_prefix': '4 agents 4 humans', 'label': 'CPPO, 4 Agents', 'color': '#a65628', 'algo': 'ppo'}, 
    'PPO_8': {'col_prefix': '8 agents', 'label': 'CPPO, 8 Agents', 'color': '#e377c2', 'algo': 'ppo'}       
}
# ---------------------

def load_and_standardize(file_path, algo_type, metric_name):
    """
    Loads CSV, standardizes column names, and applies PPO step scaling/shifting.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None

    df.columns = [col.strip() for col in df.columns]

    standard_df = pd.DataFrame()
    
    # 1. Standardize Step Column and Apply Scaling/Shifting
    step_col_raw = 'Step' if algo_type == 'mappo' else 'global_step'
    if step_col_raw not in df.columns:
        return None
    
    steps = df[step_col_raw].copy()
    
    if algo_type == 'ppo':
        # Apply scaling: divide by 10
        steps /= 10.0
        
        # Apply shifting: subtract the minimum step value to start near 0
        min_step = steps.min()
        if not np.isnan(min_step):
            steps -= min_step
        
    standard_df['Step'] = steps
    
    # 2. Iterate through all relevant scenarios and standardize metric columns
    for key, config in ALL_SCENARIOS.items():
        if config['algo'] == algo_type:
            prefix = config['col_prefix'].strip()

            if algo_type == 'mappo':
                metric_base = f'Training_{metric_name}_Mean'
            
            elif algo_type == 'ppo':
                metric_suffix = 'rew' if metric_name == 'Reward' else 'len'
                metric_base = f'rollout/ep_{metric_suffix}_mean'
            
            mean_key = f'{prefix} - {metric_base}'
            min_key = f'{prefix} - {metric_base}__MIN'
            max_key = f'{prefix} - {metric_base}__MAX'
            
            if mean_key in df.columns:
                standard_df[f'{key}_Mean'] = df[mean_key]
                standard_df[f'{key}_Min'] = df.get(min_key, df[mean_key])
                standard_df[f'{key}_Max'] = df.get(max_key, df[mean_key])
            else:
                continue 

    return standard_df.dropna(subset=['Step']).reset_index(drop=True)

# --- Plotting Functions (Unchanged Logic) ---

def plot_metric(ax, df, ylabel, metric_type):
    """Generic function to plot mean and error bands on a given axis."""
    
    for key, config in ALL_SCENARIOS.items():
        mean_col = f'{key}_Mean'
        min_col = f'{key}_Min'
        max_col = f'{key}_Max'

        if mean_col in df.columns:
            
            if config['algo'] == 'ppo':
                # Filter to only rows where PPO data is logged (non-NaN after merge)
                df_scenario = df.dropna(subset=[mean_col])
                plot_fmt = '.' # dot line
                line_width = 2
                marker_size = 0
            else:
                # MAPPO data plots as a dense line
                df_scenario = df
                plot_fmt = '-'
                line_width = 2
                marker_size = 0

            # 1. Plot the Mean Line
            ax.plot(
                df_scenario['Step'], 
                df_scenario[mean_col], 
                label=config['label'], 
                color=config['color'], 
                linewidth=line_width,
                marker=plot_fmt[1] if len(plot_fmt) > 1 else None,
                markersize=marker_size
            )

            # 2. Plot the Min/Max Error Band (using the full DF)
            ax.fill_between(
                df['Step'],
                df[min_col],
                df[max_col],
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


def plot_combined_performance():
    """Loads all data, standardizes it, and plots Reward (top) and Length (bottom) 
       in a single figure."""
    
    # 1. Load and Standardize Data
    reward_mappo = load_and_standardize('mappo_reward.csv', 'mappo', 'Reward')
    reward_ppo = load_and_standardize('ppo_reward.csv', 'ppo', 'Reward')
    
    length_mappo = load_and_standardize('mappo_length.csv', 'mappo', 'Episode_Length')
    length_ppo = load_and_standardize('ppo_length.csv', 'ppo', 'Length')

    # divide length_ppo by 2
    if length_ppo is not None:
        length_ppo['Step'] = length_ppo['Step']  # Steps already scaled in loading
        for col in length_ppo.columns:
            if col != 'Step':
                length_ppo[col] = length_ppo[col] / 2.0
    # Consolidate DataFrames by merging on the 'Step' column
    reward_df = pd.merge(reward_mappo, reward_ppo, on='Step', how='outer')
    length_df = pd.merge(length_mappo, length_ppo, on='Step', how='outer')
    
    # Filter data up to the maximum step and sort by step
    reward_df = reward_df[reward_df['Step'] <= MAX_STEP].sort_values(by='Step').reset_index(drop=True)
    length_df = length_df[length_df['Step'] <= MAX_STEP].sort_values(by='Step').reset_index(drop=True)

    if reward_df.empty or length_df.empty:
        print("Could not consolidate enough data to plot.")
        return

    # 2. Create Subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # --- Top Plot: Reward ---
    ax_reward = axes[0]
    plot_metric(ax_reward, reward_df, 'Average Training Reward', 'Reward')
    ax_reward.set_title('Average Training Reward Comparison (PPO vs. MAPPO)', fontsize=14)
    # ax_reward.set_xlabel('Step', fontsize=14) 
    
    # --- Bottom Plot: Length ---
    ax_length = axes[1]
    plot_metric(ax_length, length_df, 'Average Episode Length', 'Length')
    ax_length.set_title('Average Episode Length Comparison (PPO vs. MAPPO)', fontsize=14)
    ax_length.set_xlabel('Step', fontsize=14)
    ax_length.set_ylim(bottom=0)
    
    # Global Legend 
    handles, labels = ax_length.get_legend_handles_labels()
    scenario_keys_ordered = list(ALL_SCENARIOS.keys())
    legend_map = {label: handle for handle, label in zip(handles, labels)}
    
    ordered_labels = [ALL_SCENARIOS[key]['label'] for key in scenario_keys_ordered if ALL_SCENARIOS[key]['label'] in labels]
    ordered_handles = [legend_map[label] for label in ordered_labels]

    # fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=3, title='Scenario', bbox_to_anchor=(0.5, 0.0))
    
    fig.subplots_adjust(bottom=0.15, hspace=0.15) 
    
    # Save the figure
    fig.savefig('ppo_vs_mappo_comparison.png', dpi=300)
    print("Figure saved to ppo_vs_mappo_comparison.png")


# --- Execute the Combined Plotting ---
plot_combined_performance()