import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Define file paths
# ----------------------------------------------------
reach_info_path = "./rapid_data/NHDFlowline_San_Guad/reach_info.csv"
sorted_id_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
obs_id_path = './rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'

# ----------------------------------------------------
# 2. Load and process the data
# ----------------------------------------------------
try:
    # Load the geometry data for the reaches
    geo_data = pd.read_csv(reach_info_path, skiprows=7, header=None)
    geo_data.columns = ['ID', 'Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 'Length_km']

    # Load the observation IDs
    obs_ids = pd.read_csv(obs_id_path, header=None)
    obs_ids.columns = ['ID']

except FileNotFoundError as e:
    print(f"Error: Could not find a required file: {e.filename}")
    print("Please ensure all CSV files are in the correct directory.")
    exit()

print(f"obs id list : {obs_ids['ID'].tolist()}")

# Merge observation IDs with the geometry data to get their coordinates
obs_reach_data = pd.merge(geo_data, obs_ids, on='ID', how='inner')

# Check if the merge was successful
if obs_reach_data.empty:
    print("Error: No common IDs were found between the observation and geometry files. Cannot create the plot.")
    exit()

# ----------------------------------------------------
# 3. Initialize and create the plot
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# First, plot ALL reaches from the original geometry file for context
for _, row in geo_data.iterrows():
    ax.plot([row['Start_Longitude'], row['End_Longitude']],
            [row['Start_Latitude'], row['End_Latitude']],
            'b-', linewidth=0.5, alpha=0.5) # Faint blue lines for context

# Then, iterate through the OBSERVED reaches to highlight and label them
for _, row in obs_reach_data.iterrows():
    start_lon = row['Start_Longitude']
    start_lat = row['Start_Latitude']
    end_lon = row['End_Longitude']
    end_lat = row['End_Latitude']

    # Re-plot the observed reach with a thicker, green line to make it stand out
    ax.plot([start_lon, end_lon], [start_lat, end_lat], 'g-', linewidth=1.5)

    # Calculate the midpoint to place the label
    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2

    # Add the 'ID' as a text label in red
    ax.text(mid_lon, mid_lat, str(row['ID']), fontsize=7, ha='center', color='red', weight='bold')

# ----------------------------------------------------
# 4. Set plot properties and save the figure
# ----------------------------------------------------
ax.set_title("River Network with Observed Reaches (obs_id) Highlighted")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# Save the figure to a file
plt.savefig('river_network_with_obs_id.png', dpi=300, bbox_inches='tight')

plt.close(fig) # Close the figure to free up memory

print("Plotting complete. Image with observed reaches has been saved as 'river_network_with_obs_id.png'")