import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Function to add Gaussian noise to a line
def add_gaussian_noise(x_vals, y_vals, std=2.0, sample_interval=25):
    """
    Add Gaussian noise to a line by sampling at intervals and interpolating.
    
    Args:
        x_vals: x-axis values
        y_vals: y-axis values (base trajectory)
        std: standard deviation of noise
        sample_interval: interval at which to sample noise points
    
    Returns:
        y_vals with noise added
    """
    # Create noise sampling points
    noise_points = np.arange(x_vals[0], x_vals[-1] + sample_interval, sample_interval)
    noise_points = noise_points[noise_points <= x_vals[-1]]
    
    # Get base values at noise points
    base_values = np.interp(noise_points, x_vals, y_vals)
    
    # Add Gaussian noise
    noise_values = base_values + np.random.normal(0, std, len(noise_points))
    
    # Interpolate back to original x values
    noisy_y = np.interp(x_vals, noise_points, noise_values)
    
    return noisy_y

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Generate x-axis data (RL locking / elicitation steps)
x = np.linspace(0, 2000, 1000)

# Baseline performance (constant at 60%)
baseline = np.full_like(x, 60)

# SFT/RL locking line with exponential decay (only up to x=1000)
decay_rate = 5  # Controls how fast it decays (higher = faster)
locking_x = x[x <= 1000]
locking_y = 60 * np.exp(-decay_rate * locking_x / 1000)

# Full elicitation lines (light green) - all recover to 60%
# Starting at x=0
full_elicit_0 = np.full_like(x, 60)
full_elicit_0 = add_gaussian_noise(x, full_elicit_0, std=3)
full_elicit_0 = np.clip(full_elicit_0, 45, 75)

# Starting at x=100
full_elicit_100_x = x[x >= 100]
decay_at_100 = 60 * np.exp(-decay_rate * 100 / 1000)
full_elicit_100_y = np.zeros(len(full_elicit_100_x))
for i, xi in enumerate(full_elicit_100_x):
    if xi <= 100 + 400:
        progress = (xi - 100) / 400
        full_elicit_100_y[i] = decay_at_100 + (60 - decay_at_100) * progress
    else:
        full_elicit_100_y[i] = 60
full_elicit_100_y = add_gaussian_noise(full_elicit_100_x, full_elicit_100_y, std=2)
full_elicit_100_y = np.clip(full_elicit_100_y, 0, 100)

# Over elicitation lines (dark green) - all go to 80%
# Starting at x=0
over_elicit_0 = np.zeros_like(x)
for i, xi in enumerate(x):
    if xi <= 800:
        over_elicit_0[i] = 60 + (20 * xi / 800)
    else:
        over_elicit_0[i] = 80
over_elicit_0 = add_gaussian_noise(x, over_elicit_0, std=2)
over_elicit_0 = np.clip(over_elicit_0, 55, 85)

# Starting at x=150
over_elicit_150_x = x[x >= 150]
decay_at_150 = 60 * np.exp(-decay_rate * 150 / 1000)
over_elicit_150_y = np.zeros(len(over_elicit_150_x))
for i, xi in enumerate(over_elicit_150_x):
    if xi <= 150 + 500:
        progress = (xi - 150) / 500
        over_elicit_150_y[i] = decay_at_150 + (80 - decay_at_150) * progress
    else:
        over_elicit_150_y[i] = 80
over_elicit_150_y = add_gaussian_noise(over_elicit_150_x, over_elicit_150_y, std=2)
over_elicit_150_y = np.clip(over_elicit_150_y, 0, 100)

# Partial elicitation lines (yellow) - recover to different levels
# Starting at x=250, recovers to 40%
partial_elicit_250_x = x[x >= 250]
decay_at_250 = 60 * np.exp(-decay_rate * 250 / 1000)
partial_elicit_250_y = np.zeros(len(partial_elicit_250_x))
for i, xi in enumerate(partial_elicit_250_x):
    if xi <= 250 + 800:
        progress = (xi - 250) / 800
        partial_elicit_250_y[i] = decay_at_250 + (40 - decay_at_250) * progress
    else:
        partial_elicit_250_y[i] = 40
partial_elicit_250_y = add_gaussian_noise(partial_elicit_250_x, partial_elicit_250_y, std=1.5)
partial_elicit_250_y = np.clip(partial_elicit_250_y, 0, 100)

# Starting at x=400, recovers to 38%
partial_elicit_400_x = x[x >= 400]
decay_at_400 = 60 * np.exp(-decay_rate * 400 / 1000)
partial_elicit_400_y = np.zeros(len(partial_elicit_400_x))
for i, xi in enumerate(partial_elicit_400_x):
    if xi <= 400 + 500:
        progress = (xi - 400) / 500
        partial_elicit_400_y[i] = decay_at_400 + (38 - decay_at_400) * progress
    else:
        partial_elicit_400_y[i] = 38
partial_elicit_400_y = add_gaussian_noise(partial_elicit_400_x, partial_elicit_400_y, std=1.5)
partial_elicit_400_y = np.clip(partial_elicit_400_y, 0, 100)

# No elicitation lines (red) - stay flat at decay level
# Starting at x=200
no_elicit_200_x = x[x >= 200]
decay_at_200 = 60 * np.exp(-decay_rate * 200 / 1000)
no_elicit_200_y = np.full(len(no_elicit_200_x), decay_at_200)
no_elicit_200_y = add_gaussian_noise(no_elicit_200_x, no_elicit_200_y, std=0.5)

# Starting at x=500
no_elicit_500_x = x[x >= 500]
decay_at_500 = 60 * np.exp(-decay_rate * 500 / 1000)
no_elicit_500_y = np.full(len(no_elicit_500_x), decay_at_500)
no_elicit_500_y = add_gaussian_noise(no_elicit_500_x, no_elicit_500_y, std=0.5)

# Starting at x=750
no_elicit_750_x = x[x >= 750]
decay_at_750 = 60 * np.exp(-decay_rate * 750 / 1000)
no_elicit_750_y = np.full(len(no_elicit_750_x), decay_at_750)
no_elicit_750_y = add_gaussian_noise(no_elicit_750_x, no_elicit_750_y, std=0.5)

# Plot the lines with 6 distinct legend entries
# Baseline and locking (always show in legend)
ax.plot(x, baseline, 'gray', linestyle='--', linewidth=2, label='Baseline performance')
ax.plot(locking_x, locking_y, 'black', linewidth=2, label='SFT/RL locking')

# Full elicitation lines (light green) - only first gets legend
light_green = 'lightgreen'
ax.plot(x, full_elicit_0, light_green, linewidth=2, alpha=0.8, label='Full elicitation')
ax.plot(full_elicit_100_x, full_elicit_100_y, light_green, linewidth=2, alpha=0.8)

# Over elicitation lines (dark green) - only first gets legend  
dark_green = 'darkgreen'
ax.plot(x, over_elicit_0, dark_green, linewidth=2, label='Over elicitation')
ax.plot(over_elicit_150_x, over_elicit_150_y, dark_green, linewidth=2)

# Partial elicitation lines (yellow) - only first gets legend
ax.plot(partial_elicit_250_x, partial_elicit_250_y, 'gold', linewidth=2, label='Partial elicitation')
ax.plot(partial_elicit_400_x, partial_elicit_400_y, 'gold', linewidth=2)

# No elicitation lines (red) - all same color, only first gets legend
red_color = 'crimson'
ax.plot(no_elicit_200_x, no_elicit_200_y, red_color, linewidth=2, label='No elicitation')
ax.plot(no_elicit_500_x, no_elicit_500_y, red_color, linewidth=2)
ax.plot(no_elicit_750_x, no_elicit_750_y, red_color, linewidth=2)

# Set axis labels and title
ax.set_xlabel('RL locking / elicitation steps', fontsize=12)
ax.set_ylabel('Accuracy reward (%)', fontsize=12)
ax.set_title('RL Elicitation on Model Organism: Possible Elicitation Outcomes', fontsize=14)

# Set axis limits
ax.set_xlim(0, 2000)
ax.set_ylim(0, 100)

# Add grid for better readability
ax.grid(True, alpha=0.3)

# Add legend with 6 entries in 3 columns
ax.legend(loc='upper right', fontsize=10, ncol=3, columnspacing=1)

# Adjust layout and display
plt.tight_layout()
plt.savefig('/workspace/exploration-hacking/notebooks/elicitation_run.png', dpi=150)
plt.show()

# Print some key values for verification
print(f"Initial locking value: {locking_y[0]:.1f}%")
print(f"Locking value at step 1000: {locking_y[-1]:.1f}%")
print(f"Decay at x=100: {decay_at_100:.1f}%")
print(f"Decay at x=150: {decay_at_150:.1f}%")
print(f"Decay at x=200: {decay_at_200:.1f}%") 
print(f"Decay at x=250: {decay_at_250:.1f}%")
print(f"Decay at x=400: {decay_at_400:.1f}%")
print(f"Decay at x=500: {decay_at_500:.1f}%")
print(f"Decay at x=750: {decay_at_750:.1f}%")
print("\nLegend has 6 entries: Baseline, Locking, Full elicitation (light green), Over elicitation (dark green), Partial (yellow), No elicitation (red)")