import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# --- Configuration ---

# Set the base path for the model and drone data
model_path = "./model_saved_3hour_flood2"

# List of all discharge/estimation files to be plotted
# file_names = [
#     "discharge_only_flood",
#     "drones_flood_est",
#     "drones_discharge_est",
#     "flood_est_ground_truth",
#     "prob_target_map",
#     "prob_x_flood_map",
#     "coverage_area_map",
#     "grountruth_discharge",
#     "discharge_est"
# ]

# drone_pos_file = "drones_pos_default_map"
# file_names = [
#     "drones_discharge_est_default_map",
#     "prob_target_map_default_map",
#     "coverage_area_map_default_map",
#     "gt_discharge",
#     "discharge_ckf_est_input_est_obs_3"
# ]


drone_pos_file = "drones_pos_no_default_map"
file_names = [
    "drones_discharge_est_no_default_map",
    "prob_target_map_no_default_map",
    "coverage_area_map_no_default_map",
    "gt_discharge",
    "discharge_ckf_est_input_est_obs_3"
]

# Path to the shapefile and reach ID data
shp_path = "./rapid_data/NHDFlowline_San_Guad/NHDFlowline_San_Guad.shp"
reach_id_data_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"

# Animation settings
days = 80 # Number of frames (days) in the animation

# --- 1. Load Base Data (Shapefile, Reach IDs, Drone Positions) ---

print("Loading base data...")
# Load the river network shapefile
shp_data = gpd.read_file(shp_path)

# Load the reach ID data
reach_id_data = pd.read_csv(reach_id_data_path, header=None)
reach_ids = reach_id_data.iloc[:, 0].values

# Load the drone position data
drone_data_path = f"{model_path}/{drone_pos_file}.csv"
drone_data = pd.read_csv(drone_data_path, header=None)

# --- 2. Process Data for All Files ---

print("Processing data for each input file...")
all_shp_data = []
# Simulate the strmOrder column for filtering (for reproducibility)
np.random.seed(42)
shp_data['strmOrder'] = np.random.randint(1, 5, size=len(shp_data))

# Loop through each file, load its data, and merge it with the shapefile
for file_name in file_names:
    print(f"  - Loading data for {file_name}")
    discharge_data_path = f"{model_path}/{file_name}.csv"
    
    # Load the discharge data for the current file
    discharge_data = pd.read_csv(discharge_data_path, header=None)
    discharge_data.columns = reach_ids
    
    # Create a fresh copy of the shapefile data to avoid overwriting
    shp_data_copy = shp_data.copy()
    
    # Combine the discharge data with the shapefile data for each time step
    for t in range(discharge_data.shape[0]):
        # Ensure we don't exceed the number of days for the animation
        if t < days:
            shp_data_copy[f"Q{t+1}"] = discharge_data.iloc[t].reindex(shp_data_copy.COMID).values
        
    all_shp_data.append(shp_data_copy)

# Set up color function
colfun = plt.cm.inferno

# Create gif directory if it doesn't exist
gif_path = f"{model_path}/gif"
os.makedirs(gif_path, exist_ok=True)


# --- 3. Generate and Save a Separate Animation for Each File ---

for idx, file_name in enumerate(file_names):
    print(f"\n--- Generating animation for: {file_name} ---")
    
    # Get the specific shapefile data for the current file
    current_shp_data = all_shp_data[idx]
    shp_sub = current_shp_data[current_shp_data['strmOrder'] > 0]

    # Calculate the max Q for THIS file only to set its unique color scale
    Qcols = [col for col in shp_sub.columns if col.startswith('Q')]
    local_qmax = shp_sub[Qcols[0:days]].max().max()
    print(f"  - Max discharge for this file (local_qmax): {local_qmax:.2f}")

    # Create a new figure and axes for each animation
    fig, ax = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [6, 1]})

    # Function to update the plot for the current file
    def update(i):
        # Clear previous frame's content
        ax[0].clear()
        ax[1].clear()
        
        # --- Map plotting ---
        Qcol = f'Q{i+1}'
        if Qcol not in shp_sub.columns:
            print(f"Warning: Column {Qcol} not found for {file_name}. Skipping frame.")
            return
            
        Q = shp_sub[Qcol]
        
        # Normalize colors and line widths based on THIS FILE'S local_qmax
        norm = plt.Normalize(vmin=0, vmax=local_qmax)
        colors = colfun(norm(Q.fillna(0)))
        lwds = np.interp(Q.fillna(0), (0, local_qmax), (0.2, 8.0))
        
        # Plot the river network
        shp_sub.plot(ax=ax[0], color=colors, linewidth=lwds)
        ax[0].set_title(f"{file_name}\nDischarge at Day {i+1}", fontsize=12)
        ax[0].set_xlim(shp_data.total_bounds[0], shp_data.total_bounds[2])
        ax[0].set_ylim(shp_data.total_bounds[1], shp_data.total_bounds[3])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        
        # Plot the drone location and sensing range
        drones_pos = drone_data.iloc[i].to_numpy().reshape(-1, 2)
        for drone_idx in range(drones_pos.shape[0]):
            lat, log = drones_pos[drone_idx]
            sensing_range = 0.18
            ax[0].plot(log, lat, 'ro', markersize=5, alpha=0.7) # Red dot for the drone
            circle = Circle((log, lat), sensing_range, color='blue', alpha=0.1)
            ax[0].add_patch(circle)

        # --- Color bar plotting ---
        sm = plt.cm.ScalarMappable(cmap=colfun, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=ax[1], label="Discharge Q (cms)")
        
        print(f"  - Frame {i+1}/{days} for {file_name}")

    # Create and save the animation for the current file
    ani = FuncAnimation(fig, update, frames=days, repeat=False)
    
    output_filename = f"{gif_path}/{file_name}.gif"
    ani.save(output_filename, writer='pillow', fps=5)
    
    print(f"Animation successfully saved to: {output_filename}")
    
    # Close the figure to free up memory before the next loop iteration
    plt.close(fig)

print("\nAll animations have been generated successfully.")