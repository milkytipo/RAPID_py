import os
import pickle
import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import multivariate_normal as scipy_gaussian
from utility import PreProcessor
import utility
from typing import Optional, Dict, Any
import netCDF4
from datetime import datetime, timezone
from tqdm import tqdm
from utility import find_rank_and_rightmost_columns


def inject_flood_gaussian(cur_day, peak_day, flood_vector, rainfall_dict, reach_id_to_index):
    # Apply Gaussian rainfall directly to reaches in rainfall_dict
    sigma = 2
    for reach_id, rain_value in rainfall_dict.items():
        if rain_value > 0:  # Only apply rainfall if it's nonzero
            reach_index = reach_id_to_index.get(reach_id)  # Convert Reach ID to sorted index
            if reach_index is not None:  # Ensure the reach exists
                rain = rain_value * np.exp(-((cur_day - peak_day) ** 2) / (2 * sigma ** 2)) 
                flood_vector[reach_index] = rain  

    return flood_vector


def generate_gaussian_rainfall(peak_reach_id = 0, max_rainfall = 5):
    """
    Simulate the flood injection, index is the reach ID to inject, water is the discharge

    Returns:
        np.ndarray: vector of added flood discharge of each each
    """
    peak_reach_id = 1000 
    peak_day = 5
    max_rainfall = 1

    # Load the ordered reach coordinates with Euclidean distances
    ordered_reach_coords = utility.river_geo_info()

    # Select a reach to inject discharge (center of Gaussian distribution)
    selected_reach = ordered_reach_coords.iloc[peak_reach_id]  # Example: First reach

    # Extract coordinates for the selected reach
    selected_x = selected_reach["Start Longitude"]
    selected_y = selected_reach["Start Latitude"]

    # Compute Euclidean distances from the selected reach to all other reaches
    ordered_reach_coords["Distance to Center"] = np.sqrt(
        (ordered_reach_coords["Start Longitude"] - selected_x) ** 2 +
        (ordered_reach_coords["Start Latitude"] - selected_y) ** 2
    )

    # Find the 100 closest reaches
    num_closest = 100  # Adjustable parameter
    closest_reaches = ordered_reach_coords.nsmallest(num_closest, "Distance to Center")

    # Apply Gaussian function to inject water
    sigma = closest_reaches["Distance to Center"].max() / 0.01  # Spread parameter based on max distance

    # Compute Gaussian-distributed rainfall for the selected reaches
    closest_reaches["Gaussian Rainfall"] = max_rainfall * np.exp(- (closest_reaches["Distance to Center"] ** 2) / (2 * sigma ** 2))

    # Convert reach IDs to a dictionary for quick lookup
    rainfall_dict = closest_reaches.set_index("Reach ID")["Gaussian Rainfall"].to_dict()

    # # Save the rainfall dictionary to a file
    # rainfall_dict_path = "./rapid_data/rainfall_dict_output.txt"
    # with open(rainfall_dict_path, "w") as file:
    #     for reach_id, rainfall in rainfall_dict.items():
    #         file.write(f"{reach_id}: {rainfall}\n")

    # Load Reach ID to index mapping
    sorted_ids_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
    sorted_ids = pd.read_csv(sorted_ids_path, header=None, names=['Reach ID']) 

    reach_id_to_index = {reach_id: idx for idx, reach_id in enumerate(sorted_ids['Reach ID'])}
    
    return rainfall_dict, reach_id_to_index, peak_day
    # num_reaches = self.u[0].shape[0] 
    # r_peak = num_reaches // 4  
    # # Gaussian-distributed rainfall 
    # flood_vector = 5 * np.exp(-((np.arange(num_reaches) - r_peak) ** 2) / (2 * sigma ** 2))
    # return flood_vector  


if __name__ == '__main__':
    from flood_data_generator_base import RAPIDKF
    rapid_kf = RAPIDKF(load_mode=1, sub_dir_path="model_saved_3hour_flood3",)
    rapid_kf.simulate_flood(flood_type = "gaussian")
