# This file is to calculate the RMSE of the flood estimation
# there are three types of estimation:
# 1. discharge estimation
# 2. lateral-inflow estimation
# 3. external-water estimation
# The data folder is ./model_saved_3hour_flood3
# For 1. discharge flood estimation, ground truth is obs_synthetic_3.csv, compared with drones_discharge_est.csv and discharge_est.csv
# For 2. lateral-inflow flood estimation, ground truth is river_lateral_est_ground_truth_flood.csv, compared with drones_river_lateral_est.csv
# For 3. external-water flood estimation, ground truth is flood_est_ground_truth.csv, compared with drones_flood_est.csv

# please plot the RMSE of each estimation type. x is the timestep, y is the RMSE
# The RMSE is calculated as the square root of the mean of the squared differences between the estimated and ground truth values.
# The RMSE is calculated for each timestep, and the plot should show the RMSE over time.        

    
# Set the base path for the model
import pandas as pd
import os
import matplotlib.pyplot as plt 
model_path = "./model_saved_3hour_flood3_2"

gif_path = f"{model_path}/fig_eval"
os.makedirs(gif_path, exist_ok=True)

file_name_drone = "drones_discharge_est_input_est_no_x_map_default_map"
# file_name_drone = "drones_discharge_est_fixed_gauge_input_est"
file_name_drone = "discharge_ckf_est_no_input_est_obs_3"
# file_name_drone = "drones_discharge_est_no_default_map"

file_name_gt = "gt_discharge"
file_name_ckf = "drones_discharge_est_no_input_est_no_x_map_default_map"
file_name_ckf = "drones_discharge_est_fixed_gauge_input_est"


# file_name_ckf = "discharge_ckf_est_input_est_obs_3"
# file_name_ckf = "discharge_ckf_no_est_input_est_obs_1"
# file_name_drone = "discharge_ckf_no_est_input_est_obs_3"


flood_data_path = f"{model_path}/{file_name_drone}.csv"
flood_data = pd.read_csv(flood_data_path, header=None)
# Load the ground truth flood data without treating the first row as the header
ground_truth_flood_data_path = f"{model_path}/{file_name_gt}.csv"
ground_truth_flood_data = pd.read_csv(ground_truth_flood_data_path, header=None)

# Load the CKF flood estimation data without treating the first row as the header
ckf_flood_data_path = f"{model_path}/{file_name_ckf}.csv"
ckf_flood_data = pd.read_csv(ckf_flood_data_path, header=None)


# 将 ground truth 数据的每一行复制四次，以匹配其他数据
# ground_truth_flood_data = ground_truth_flood_data.loc[ground_truth_flood_data.index.repeat(4)].reset_index(drop=True)
# ===============================================
# 从 drone 数据中每4行取1行（即每天只取一个点）
# flood_data = flood_data.iloc[::4].reset_index(drop=True)
# ckf_flood_data = ckf_flood_data.iloc[::4].reset_index(drop=True)
# ===============================================

# Ensure the CKF flood data columns match the ground truth data
ckf_flood_data.columns = ground_truth_flood_data.columns
# Calculate RMSE for each timestep for CKF
rmse_ckf_values = ((ckf_flood_data - ground_truth_flood_data) ** 2).mean(axis=1).apply(lambda x: x ** 0.5)
# Ensure the flood data columns match the ground truth data
flood_data.columns = ground_truth_flood_data.columns
# Calculate RMSE for each timestep
rmse_values = ((flood_data - ground_truth_flood_data) ** 2).mean(axis=1).apply(lambda x: x ** 0.5)
# Plot the RMSE values for CKF and the flood estimation
plt.figure(figsize=(10, 6))
plt.plot(rmse_values, marker='o', linestyle='-', color='b', label='Flood Estimation')
plt.plot(rmse_ckf_values, marker='x', linestyle='--', color='r', label='CKF Estimation')
plt.title('RMSE of Flood Estimation Over Time')
plt.xlabel('Timestep')
plt.ylabel('RMSE')
plt.grid()
plt.xticks(range(len(rmse_values)), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{model_path}/fig_eval/rmse_flood_estimation.png")

# also calculate the MAE (Mean Absolute Error) for each estimation type
mae_values = (flood_data - ground_truth_flood_data).abs().mean(axis=1)
mae_ckf_values = (ckf_flood_data - ground_truth_flood_data).abs().mean(axis=1)
# Plot the MAE values for CKF and the flood estimation
plt.figure(figsize=(10, 6))
plt.plot(mae_values, marker='o', linestyle='-', color='b', label='Flood Estimation MAE')
plt.plot(mae_ckf_values, marker='x', linestyle='--', color='r', label='CKF Estimation MAE')
plt.title('MAE of Flood Estimation Over Time')
plt.xlabel('Timestep')
plt.ylabel('MAE')
plt.grid()
plt.xticks(range(len(mae_values)), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{model_path}/fig_eval/mae_flood_estimation.png")


# plt.figure(figsize=(10, 6))
# plt.plot(rmse_values, marker='o', linestyle='-', color='b')
# plt.title('RMSE of Flood Estimation Over Time')
# plt.xlabel('Timestep')
# plt.ylabel('RMSE')
# plt.grid()
# plt.xticks(range(len(rmse_values)), rotation=45)
# plt.tight_layout()
# plt.savefig(f"{model_path}/fig_eval/rmse_flood_estimation.png")