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


def inject_flood_fixed(flood_vector):
    """    
    Simulate the flood injection, index is the reach ID to inject, water is the discharge
    
    Returns:
        np.ndarray: vector of added flood discharge of each reach
    """
    flood_vector[0] = 20

    return flood_vector


if __name__ == '__main__':
    from flood_data_generator_base import RAPIDKF
    rapid_kf = RAPIDKF(load_mode=1, sub_dir_path="model_saved_3hour_flood2",)
    rapid_kf.simulate_flood(flood_type = "fixed")
