# This file is to calculate the RMSE of the flood estimation
# there are three types of estimation:
# 1. discharge estimation
# 2. lateral-inflow estimation
# 3. external-water estimation
# The data folder is ./model_saved_3hour_flood3
# For 1. discharge flood estimation, ground truth is discharge_est.csv, compared with drone1_discharge_est.csv
# For 2. lateral-inflow flood estimation, ground truth is river_lateral_est_ground_truth_flood.csv, compared with drone1_river_lateral_est.csv
# For 3. external-water flood estimation, ground truth is flood_est_ground_truth.csv, compared with drone1_flood_est.csv

# please plot the RMSE of each estimation type. x is the timestep, y is the RMSE
# The RMSE is calculated as the square root of the mean of the squared differences between the estimated and ground truth values.
# The RMSE is calculated for each timestep, and the plot should show the RMSE over time.        
# Set the base path for the model
import pandas as pd
import matplotlib.pyplot as plt 
model_path = "./model_saved_3hour_flood3"
file_name_drone = "drone1_discharge_est"
# file_name_drone = "drone1_river_lateral_est"
# file_name_drone = "drone1_flood_est"

file_name_gt = "discharge_est"
# file_name_gt = "river_lateral_est_ground_truth_flood"
# file_name_gt = "flood_est_ground_truth"

# Load the estimation of flood data without treating the first row as the header
flood_data_path = f"{model_path}/{file_name_drone}.csv"
flood_data = pd.read_csv(flood_data_path, header=None)
# Load the ground truth flood data without treating the first row as the header
ground_truth_flood_data_path = f"{model_path}/{file_name_gt}.csv"
ground_truth_flood_data = pd.read_csv(ground_truth_flood_data_path, header=None)
# Ensure the flood data columns match the ground truth data
flood_data.columns = ground_truth_flood_data.columns
# Calculate RMSE for each timestep
rmse_values = ((flood_data - ground_truth_flood_data) ** 2).mean(axis=1).apply(lambda x: x ** 0.5)
# Plot the RMSE values
plt.figure(figsize=(10, 6))
plt.plot(rmse_values, marker='o', linestyle='-', color='b')
plt.title('RMSE of Flood Estimation Over Time')
plt.xlabel('Timestep')
plt.ylabel('RMSE')
plt.grid()
plt.xticks(range(len(rmse_values)), rotation=45)
plt.tight_layout()
plt.savefig(f"{model_path}/fig_eval/rmse_flood_estimation.png")