import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Define file paths
# ----------------------------------------------------
reach_info_path = "./rapid_data/NHDFlowline_San_Guad/reach_info.csv"
sorted_id_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"

# ----------------------------------------------------
# 2. Load and process the data
# ----------------------------------------------------
try:
    # Load the geometry data for the reaches
    geo_data = pd.read_csv(reach_info_path, skiprows=7, header=None)
    geo_data.columns = ['ID', 'Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 'Length_km']

    # Load the hydro-sequenced (sorted) reach IDs
    sorted_ids = pd.read_csv(sorted_id_path, header=None)
    sorted_ids.columns = ['ID']
    # The correct label is the index from this file, which represents the upstream/downstream order
    sorted_ids['sorted_index'] = sorted_ids.index

except FileNotFoundError as e:
    print(f"Error: Could not find a required file: {e.filename}")
    print("Please ensure both CSV files are in the correct directory.")
    exit()

# Merge the two dataframes to link geometry with the sorted index based on the common 'ID'
merged_data = pd.merge(geo_data, sorted_ids, on='ID', how='inner')

# Check if the merge was successful
if merged_data.empty:
    print("Error: No common IDs were found between the two files. Cannot create the plot.")
    exit()

# ----------------------------------------------------
# 3. Create a 10% random sample for labeling
# ----------------------------------------------------
# Use random_state for reproducible results
label_sample = merged_data.sample(frac=0.1, random_state=1)

# ----------------------------------------------------
# 4. Initialize and create the plot
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# First, plot ALL reaches from the original geometry file for context
for _, row in geo_data.iterrows():
    ax.plot([row['Start_Longitude'], row['End_Longitude']], 
            [row['Start_Latitude'], row['End_Latitude']], 
            'b-', linewidth=0.5, alpha=0.5) # Faint blue lines for context

# Then, iterate through the SAMPLE to plot and add labels
for _, row in label_sample.iterrows():
    start_lon = row['Start_Longitude']
    start_lat = row['Start_Latitude']
    end_lon = row['End_Longitude']
    end_lat = row['End_Latitude']
    
    # Re-plot the sampled reach with a slightly thicker line to make it stand out
    ax.plot([start_lon, end_lon], [start_lat, end_lat], 'b-', linewidth=1.2)
    
    # Calculate the midpoint to place the label
    mid_lon = (start_lon + end_lon) / 2
    mid_lat = (start_lat + end_lat) / 2
    
    # Add the 'sorted_index' as a text label in red
    ax.text(mid_lon, mid_lat, str(row['sorted_index']), fontsize=6, ha='center', color='red')

# ----------------------------------------------------
# 5. Set plot properties and save the figure
# ----------------------------------------------------
ax.set_title("River Network with 10% of Reaches Labeled by Hydro-Sequence Index")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# Save the figure to a file
plt.savefig('river_network_sample_with_sorted_index.png', dpi=300, bbox_inches='tight')

plt.close(fig) # Close the figure to free up memory

print("Plotting complete. Image with a 10% sample of reaches labeled by sorted index has been saved as 'river_network_sample_with_sorted_index.png'")