import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the observation data and river reach IDs
obs_data_path = './rapid_data/Qobs_San_Guad_2010_2013_full.csv'
obs_ids_path = './rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'

obs_data = pd.read_csv(obs_data_path, header=None)
obs_reach_ids = pd.read_csv(obs_ids_path, header=None)

# Load the open-loop simulated discharge data
open_loop_data_path = './model_saved_official/discharge_est_offical.csv'
open_loop_data_path = './model_saved_3hour/open_loop_est.csv'
open_loop_data_path = './model_saved_3hour_w_input/open_loop_est.csv'
# open_loop_data_path = './model_saved_official/discharge_est_ave.csv'
open_loop_data = pd.read_csv(open_loop_data_path, header=None)
print(f"open_loop_data {open_loop_data} with shape {open_loop_data.shape}")
# Load the IDs for the open-loop simulation
open_loop_ids_path = './rapid_data/riv_bas_id_San_Guad_hydroseq.csv'
open_loop_ids = pd.read_csv(open_loop_ids_path, header=None)
days = 366+365+365+365 #2010 year 366 days
# Extract the first 366 days from the observation data
obs_data_366_days = obs_data.iloc[:days]

# Extract the list of open-loop IDs
open_loop_ids_list = open_loop_ids[0].tolist()

# Extract the list of relevant reach IDs from the observation data
obs_ids_list = obs_reach_ids[0].tolist()
# Find the indices of the relevant reaches in the open-loop data
relevant_indices = [open_loop_ids_list.index(reach_id) for reach_id in obs_ids_list if reach_id in open_loop_ids_list]

# Extract the relevant simulated data using the matched indices
open_loop_data_relevant = open_loop_data.iloc[:days, relevant_indices]

# Define a function to calculate NSE
def calculate_nse(observed, simulated):
    observed_mean = np.mean(observed)
    nse = 1 - (np.sum((observed - simulated) ** 2) / np.sum((observed - observed_mean) ** 2))
    return nse

# Initialize a list to store the NSE values
nse_values = []

# Calculate NSE for each gauge
for i in range(len(relevant_indices)):
    observed = obs_data_366_days.iloc[:, i]
    simulated = open_loop_data_relevant.iloc[:, i]
    nse = calculate_nse(observed, simulated)
    nse_values.append(nse)

# Create a DataFrame to store NSE values along with gauge IDs
nse_df = pd.DataFrame({
    'Gauge ID': [obs_ids_list[i] for i in range(len(relevant_indices))],
    'NSE': nse_values
})

# Save the NSE values DataFrame to a CSV file
nse_csv_path = './figure/nse_values.csv'
nse_df.to_csv(nse_csv_path, index=False)

# Choose a few representative gauges for visualization
selected_gauges = [1619595, 1620031, 1630223, 1631087, 1631099, 1639225, 1639225,3840125]

# Create and save hydrographs comparing observed and simulated discharge for the selected gauges
for gauge_id in selected_gauges:
    plt.figure(figsize=(8, 5))
    print(f"gauge_id {gauge_id} | {open_loop_ids_list.index(gauge_id)}")
    observed = obs_data_366_days.iloc[:,obs_ids_list.index(gauge_id)]    
    simulated = open_loop_data.iloc[:, open_loop_ids_list.index(gauge_id)]
    plt.plot(observed, label='Observed', color='green')
    plt.plot(simulated, label='Simulated (Open-loop)', color='blue')
    plt.title(f'Gauge ID: {gauge_id}')
    plt.xlabel('Day')
    plt.ylabel('Discharge (m³/s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/hydrograph_gauge_{gauge_id}.png')
    plt.close()