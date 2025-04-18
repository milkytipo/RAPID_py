import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import os

# Set the base path for the model
model_path = "./model_saved_3hour_flood3"
file_name = "discharge_only_flood"
# file_name = "drone1_flood_est"
file_name = "drone1_discharge_est"
file_name = "prob_target_map"
file_name = "prob_x_flood_map"
# file_name = "coverage_area_map"
file_name_pos = "drone1_pos"
# Load the shapefile
shp_path = "./rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp"
shp_data = gpd.read_file(shp_path)
days = 20

# Load the estimation of discharge data without treating the first row as the header
discharge_data_path = f"{model_path}/{file_name}.csv"
drone_data_path = f"{model_path}/{file_name_pos}.csv"
discharge_data = pd.read_csv(discharge_data_path, header=None)
drone_data = pd.read_csv(drone_data_path, header=None)

print(type(drone_data))
# Load the reach ID data without treating the first row as the header
reach_id_data_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
reach_id_data = pd.read_csv(reach_id_data_path, header=None)

# Ensure the discharge data columns match the reach IDs
reach_ids = reach_id_data.iloc[:, 0].values
discharge_data.columns = reach_ids

# Combine the discharge data with the shapefile data
for t in range(discharge_data.shape[0]):
    shp_data[f"Q{t+1}"] = discharge_data.iloc[t].reindex(shp_data.COMID).values

# Simulate the strmOrder column
np.random.seed(42)  # For reproducibility
shp_data['strmOrder'] = np.random.randint(1, 5, size=len(shp_data))

# Filter out polylines with stream order > 0
shp_sub = shp_data[shp_data['strmOrder'] > 0]

# Calculate global maximum discharge for color scaling
Qcols = [col for col in shp_sub.columns if col.startswith('Q')]

# To set color bar range
globQmax = max(shp_sub[Qcols[0:days]].max().max(), 0)

# Set up color and line width gradients
colfun = plt.cm.viridis
colfun = plt.cm.inferno
lwdRamp = np.linspace(0.15, 12, 50)

# Create a figure and axes
fig, ax = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [6, 1]})

# Function to update the plot
def update(i):
    ax[0].clear()
    ax[1].clear()
    
    # Map plotting
    Qcol = Qcols[i]
    Q = shp_sub[Qcol]
    norm = plt.Normalize(vmin=0, vmax=globQmax)
    colors = colfun(norm(Q))
    lwds = np.interp(Q, (0, globQmax), (0.15, 12))
    
    # Plot the river network
    shp_sub.plot(ax=ax[0], color=colors, linewidth=lwds)
    ax[0].set_title(f"Discharge at Day {i+1}")
    ax[0].set_xlim(shp_sub.total_bounds[0], shp_sub.total_bounds[2])
    ax[0].set_ylim(shp_sub.total_bounds[1], shp_sub.total_bounds[3])
    
    # Add legend
    ax[0].legend(loc='upper right')
    
    # Color bar plotting
    sm = plt.cm.ScalarMappable(cmap=colfun, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax[1])
    cbar.set_label("Q (cms)")
    
    # Plot the drone location and sensing range
    drones_pos = drone_data.iloc[i].to_numpy().reshape(-1, 2)
    
    for idx in range((drones_pos.shape[0])):
        lat, log = drones_pos[idx]
        sensing_range = 0.18 * 2.5
        ax[0].plot(log, lat, 'ro', markersize=10, label=f'{idx}th Drone Location ')  # Red dot for the drone
        circle = Circle((log, lat), sensing_range, color='r', alpha=0.05, label='Sensing Range')
        ax[0].add_patch(circle)

# Create animation
ani = FuncAnimation(fig, update, frames=days, repeat=False)

plt.show()

# Create gif directory if it doesn't exist
gif_path = f"{model_path}/gif"
if not os.path.exists(gif_path):
    os.makedirs(gif_path, exist_ok=True)

# Save the animation
ani.save(f"{gif_path}/{file_name}-100.gif", writer='pillow', fps=3)
