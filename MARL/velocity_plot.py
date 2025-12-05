import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 

# Data extracted from the user's provided results
agents = np.array([1, 4, 8])
# Create new x-coordinates for equal spacing
x_indices = np.arange(len(agents)) + 1 # Use [1, 2, 3] for plotting x-coordinates

avg_speed = np.array([2.02, 2.06, 2.15])
min_speed = np.array([2.00, 1.99, 1.87])
max_speed = np.array([2.07, 2.25, 3.62])

alpha = 10

avg_speed = avg_speed * alpha
min_speed = min_speed * alpha
max_speed = max_speed * alpha

# Calculate the asymmetric error bars
lower_error = avg_speed - min_speed
upper_error = max_speed - avg_speed
error_bars = np.array([lower_error, upper_error])

# Set a professional style (e.g., 'default' is usually clean)
plt.style.use('default') 

# Create the figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# Plot the average speed points with min/max error range, using x_indices
ax.errorbar(x_indices, avg_speed, yerr=error_bars, 
            fmt='o', # Marker only - removes connecting line
            color='black', 
            ecolor='gray', # Error bar line color
            capsize=5,
            markerfacecolor='black', 
            markeredgecolor='black',
            zorder=3) 

# --- Add Min/Max Text Annotations ---
for i in range(len(agents)):
    # Annotate Max Speed above the upper error cap, using x_indices
    ax.text(x_indices[i], max_speed[i] + 0.05, f'{max_speed[i]:.2f}', 
            ha='center', va='bottom', fontsize=9)
    # Annotate Min Speed below the lower error cap, using x_indices
    ax.text(x_indices[i], min_speed[i] - 0.05, f'{min_speed[i]:.2f}', 
            ha='center', va='top', fontsize=9)

# Configuration
ax.set_xticks(x_indices) # Set ticks at the new x-coordinates
ax.set_xticklabels([f'{a}' for a in agents]) # Keep original agent labels
ax.set_xlabel('Number of Controlled Agents', fontsize=12)
ax.set_ylabel('Average Velocity [km/h]', fontsize=12) 
# ax.set_title('Average Velocity vs. Number of Controlled Agents', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax.set_ylim(bottom=15, top=40)
# Set x-limits tightly around the new index points (1, 2, 3)
ax.set_xlim(0.5, 3.5)

# Create a clean legend for the plotted point and error range
legend_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
           markeredgecolor='black', markersize=7, label='Average Velocity'),
    Line2D([0], [0], color='gray', lw=1.5, marker='|', 
           markeredgecolor='gray', markersize=10, 
           label='Min/Max Range')
]
ax.legend(handles=legend_lines, loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('speed_range_final_academic_equal_spacing.png', dpi=300)