import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Final Gauge ID Lists (Corrected)
#
#    The lists now correctly contain 23 assimilation and
#    13 validation IDs as per your requirements.
# ----------------------------------------------------

# List of IDs for "Assimilation gauges" (23 total)
assimilation_ids = [
    # San Antonio Basin (South)
    3589508, 3585724, 3585678, 3585554,  # Far upstream section
    10835030, 10833740, 10835974,         # Upstream tributaries
    7851041, 10840810, 7851771, 7852265, # Mid-stream section
    1639209,                            # "Goliad" area and downstream

    # Guadalupe Basin (North)
    1630223, 1631023, 1631099,         # Far upstream section
    3589120, 3586192,                  # North-side tributaries
    1631087, 1631387, 1631587,         # "Cuero" area
    1622713,                           # Tributary near "Victoria"
    1639225,                           # Far-east tributary (Corrected: Added this ID)

    # River Mouth
    1638907,
]

# List of IDs for "Validation gauges" (13 total)
validation_ids = [
    # San Antonio Basin (South)
    10836388, 10836420, 10840488,       # "Fall City" and nearby tributaries
    10840572,
    3838221, 3838999,                   # Mid-stream tributary

    # Guadalupe Basin (North)
    1619595, 1619649,                   # Upstream tributary
    1622763,                            # "Victoria" main stem
    1620031, 1637447, 1623207,         # Tributaries around "Victoria"

    # Post-Confluence
    3840125,
]


# ----------------------------------------------------
# 2. Define file path and load geometry data
# ----------------------------------------------------
# This assumes the file structure from the original prompt
reach_info_path = "./rapid_data/NHDFlowline_San_Guad/reach_info.csv"
sorted_id_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
try:
    # Load the geometry data for all reaches
    geo_data = pd.read_csv(reach_info_path, skiprows=7, header=None)
    geo_data.columns = ['ID', 'Start_Latitude', 'Start_Longitude', 'End_Latitude', 'End_Longitude', 'Length_km']

except FileNotFoundError as e:
    print(f"Error: Could not find the geometry file: {e.filename}")
    exit()

# ----------------------------------------------------
# 3. Filter geometry data for the specified gauges
# ----------------------------------------------------
# Get the coordinates for assimilation gauges
assimilation_reaches = geo_data[geo_data['ID'].isin(assimilation_ids)].copy()
# Get the coordinates for validation gauges
validation_reaches = geo_data[geo_data['ID'].isin(validation_ids)].copy()

# Calculate midpoints for plotting markers
def get_midpoint(row):
    return pd.Series({
        'mid_lon': (row['Start_Longitude'] + row['End_Longitude']) / 2,
        'mid_lat': (row['Start_Latitude'] + row['End_Latitude']) / 2
    })

assimilation_reaches[['mid_lon', 'mid_lat']] = assimilation_reaches.apply(get_midpoint, axis=1)
validation_reaches[['mid_lon', 'mid_lat']] = validation_reaches.apply(get_midpoint, axis=1)


# ----------------------------------------------------
# 4. Initialize and create the plot
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# First, plot the entire river network for context
for _, row in geo_data.iterrows():
    ax.plot([row['Start_Longitude'], row['End_Longitude']],
            [row['Start_Latitude'], row['End_Latitude']],
            'b-', linewidth=0.5, alpha=0.3) # Faint blue lines for context

# Plot Assimilation gauges as orange circles
ax.scatter(assimilation_reaches['mid_lon'], assimilation_reaches['mid_lat'],
           marker='o', s=100, c='orange', ec='black', zorder=5, label=f'Assimilation Gauges ({len(assimilation_reaches)})')

# Plot Validation gauges as purple squares
ax.scatter(validation_reaches['mid_lon'], validation_reaches['mid_lat'],
           marker='s', s=80, c='purple', ec='black', zorder=5, label=f'Validation Gauges ({len(validation_reaches)})')


# ----------------------------------------------------
# 5. Set plot properties and save the figure
# ----------------------------------------------------
ax.set_title("Assimilation and Validation Gauges")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.legend() # Add a legend to explain the markers

# Save the figure to a file
plt.savefig('river_network_final_gauges.png', dpi=300, bbox_inches='tight')

plt.close(fig) # Close the figure to free up memory

print("Plotting complete! Image has been saved as 'river_network_final_gauges.png'")
print(f"Successfully plotted {len(assimilation_reaches)} assimilation gauges and {len(validation_reaches)} validation gauges.")

# Final verification check
if len(assimilation_reaches) != 23 or len(validation_reaches) != 13:
    print(f"\nWarning: The final plot does not have 23 and 13 points respectively.")
    print("This may happen if some IDs from your list are not present in the 'reach_info.csv' file.")