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

class RAPIDKF:
    """
    Class for the Kalman Filter (KF) model used in river network modeling.
    
    Attributes:
        epsilon (float): Threshold for muskingum parameter.
        radius (int): Radius for KF estimation.
        i_factor (float): Scaling factor for covariance P.
        days (int): Total number of days from 2010 to 2013.
        month (int): Total months calculated from days.
        timestep (int): Current timestep in the simulation.
    """

    def __init__(self, load_mode: int = 0) -> None:
        """
        Initializes the RAPIDKF class.

        Args:
            load_mode (int): Mode for loading data (0 = file, 1 = pickle, 2 = both).
        """
        np.random.seed(42)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.sub_dir_path = "model_saved_3hour_flood3"
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.join(dir_path, self.sub_dir_path)):
            os.makedirs(os.path.join(dir_path, self.sub_dir_path), exist_ok=True)
        self.epsilon: float = 0  # Muskingum parameter threshold
        self.radius: int = 20
        self.i_factor: float = 2.58  # Enforced on covariance P
        self.days: int = 366 + 365 + 365 + 365  # 2010 to 2013
        self.days: int = 20  # 2010 to 2013
        self.month: int = self.days // 365 * 12
        self.timestep: int = 0
        
        if load_mode in [0, 2]:
            self.load_file(dir_path)
        if load_mode in [1, 2]:
            self.load_pkl(dir_path)

    def load_pkl(self, dir_path: str) -> None:
        """
        Loads the saved model data from a pickle file.

        Args:
            dir_path (str): Directory path of the pickle file.
        """
        dis_name = os.path.join(self.sub_dir_path,'load_coef.pkl')
        with open(os.path.join(dir_path, dis_name), 'rb') as f:
            saved_dict: Dict[str, Any] = pickle.load(f)
        
        self.Ae = saved_dict['Ae']
        self.A0 = saved_dict['A0']
        self.Ae_day = saved_dict['Ae_day']
        self.A0_day = saved_dict['A0_day']
        self.A4 = saved_dict['A4']
        self.A5 = saved_dict['A5']
        self.H1 = saved_dict['H1']
        self.H2 = saved_dict['H2']
        self.S = saved_dict['S']
        self.P = saved_dict['P']
        self.R = saved_dict['R']
        self.u = saved_dict['u']
        self.obs_data = saved_dict['obs_data']
        self.id2sortedid = saved_dict['id2sortedid']
        
        Qou_ncf = os.path.join(dir_path, self.sub_dir_path, 'Qout_San_Guad_20100101_20131231_VIC0125_3H_utc_py_2.nc')
        m3r_ncf = os.path.join(dir_path, 'rapid_data/m3_riv_San_Guad_20100101_20131231_VIC0125_3H_utc_err_R286_D_scl.nc')
        self.Qou_ncf = Qou_ncf
        self.Qout_nc(m3r_ncf, Qou_ncf, self.id2sortedid)
        
    def simulate_flood(self, sim_mode: int = 0) -> None:
        """
        Simulates the Kalman Filter model.

        Args:
            sim_mode (int): Mode for simulation (0 = open loop, 1 = Kalman Filter estimation).
        """
        if sim_mode == 0:
            print(f"Load_csv file")
        elif sim_mode == 1: 
            print(f"Compute all csv from scratch")
            
        ### Update P scale:
        # self.P = self.P * 24 * 3600    
        # Create a netCDF file for the river discharge comparison
        g = netCDF4.Dataset(self.Qou_ncf, 'a')
        Qout = g.variables['Qout']
        
        state_estimation = []
        discharge_estimation = []
        open_loop_x = []
        flood_est = []
        obs_synthetic = []
        
        self.Q0 = np.zeros_like(self.u[0])
        evolution_steps = 8  # Number of steps for each day
        added_flood = np.zeros((self.days*evolution_steps,self.u[0].shape[0]))
        origin_inflow = np.zeros_like(added_flood)
        inject_flood_inflow = np.zeros_like(added_flood)
        discharge_only_flood = np.zeros((self.days,self.u[0].shape[0]))  
            
        # Define file paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, self.sub_dir_path)

        file_names = [
            "injected_flood.csv",
            "inject_w_inflow.csv",
            "original_inflow.csv",
            "discharge_from_obs1.csv",
            "open_loop_est.csv",
            "discharge_only_flood.csv",
            "percentile_90.csv",
            "obs_synthetic.csv"
        ]
        
        file_paths = [os.path.join(dir_path, file) for file in file_names]

        # Check if all files exist
        if all(os.path.exists(file) for file in file_paths) and sim_mode == 0:
            print("All CSV files exist. Loading data instead of recalculating...")

            added_flood = np.loadtxt(file_paths[0], delimiter=",")
            inject_flood_inflow = np.loadtxt(file_paths[1], delimiter=",")
            origin_inflow = np.loadtxt(file_paths[2], delimiter=",")
            discharge_obs_kf1 = np.loadtxt(file_paths[3], delimiter=",")
            open_loop_x = np.loadtxt(file_paths[4], delimiter=",")
            discharge_only_flood = np.loadtxt(file_paths[5], delimiter=",")
            percentile_90 = np.loadtxt(file_paths[6], delimiter=",")
            obs_synthetic = np.loadtxt(file_paths[7], delimiter=",")
        else:
            print('Data is needed')
            
        '''
        Simulation under synthetic data
        '''
        # self.S = np.eye(self.P.shape[0])
        log = -97
        lat = 29
        sensing_range = 0.5
        self.drone_pos_initial(log, lat, sensing_range)
        # self.H = np.dot(self.S, self.Ae_day)
        self.B = self.S.T
        self.timestep = 0
        self.Q0 = np.zeros_like(self.u[0])
        drone_pos = []
        prob_flood_map = []
        prob_flood_est = np.zeros_like(self.u[0])

        for timestep in tqdm(range(self.days)):
            discharge_avg = np.zeros_like(self.u[0])
            self.x = np.zeros_like(self.u[0])
            drone_pos.append((log,lat,sensing_range))
            
            # Kalman Filter estimation (updates every 3 hours)
            for i in range(evolution_steps):
                self.x += self.u[timestep * evolution_steps + i]  / evolution_steps
                # self.x += added_flood[timestep * evolution_steps + i] / evolution_steps
                
            gt_obs = obs_synthetic[timestep]
            gt_obs = self.S @ gt_obs
            self.update(gt_obs, timestep, True)

            for i in range(evolution_steps):
                discharge_avg += self.update_discharge()

            discharge_avg /= evolution_steps
            Qout[timestep, :] = discharge_avg[:]
            state_estimation.append(copy.deepcopy(self.get_state()))
            discharge_estimation.append(discharge_avg)
            flood_est.append(self.S.T @ self.u_flood)
            
            prob_flood_obs = self.sigmoid_prob(self.u_flood, percentile_90)
            prob_flood_est = self.flood_prob_update(prob_flood_obs, self.S)
            
            prob_flood_map.append(prob_flood_est)
            
            # Dynamics of drone
            self.timestep += 1
            log, lat, sensing_range = self.drone_pos_update(log,lat,sensing_range)

        # Save results to the created directory
        Qout_df = pd.DataFrame(Qout[:])
        Qout_df.to_csv(os.path.join(dir_path, "drone1_Qout.csv"), index=False)
        np.savetxt(os.path.join(dir_path, "drone1_discharge_est.csv"), discharge_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "drone1_river_lateral_est.csv"), state_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "drone1_flood_est.csv"), flood_est, delimiter=",")
        np.savetxt(os.path.join(dir_path, "drone1_pos.csv"), drone_pos, delimiter=",")
        np.savetxt(os.path.join(dir_path, "prob_flood_map.csv"), prob_flood_map, delimiter=",")
        g.close()
    
    def drone_pos_initial(self, log, lat, sensing_range):
        # Load the ordered reach coordinates with Euclidean distances
        ordered_reach_coords = utility.river_geo_info()
        ordered_reach_coords["Distance to Center"] = np.sqrt(
            (ordered_reach_coords["Start Longitude"] - log) ** 2 +
            (ordered_reach_coords["Start Latitude"] - lat) ** 2
        )
        # Find the 1000 closest reaches
        closest_reaches = ordered_reach_coords[ordered_reach_coords["Distance to Center"] < sensing_range]
        sensing_ids = closest_reaches["Reach ID"].values
        sorted_ids_path = "./rapid_data/riv_bas_id_San_Guad_hydroseq.csv"
        sorted_ids = pd.read_csv(sorted_ids_path, header=None, names=['Reach ID'])  # Assuming no header
            
        S_mat = np.zeros((len(sensing_ids), len(sorted_ids)), dtype=int)
        for i, obs in enumerate(sensing_ids):
            index = np.where(sorted_ids == obs)[0]
            S_mat[i, index] = 1
        
        H_mat = np.dot(S_mat, self.Ae_day)
        self.S = S_mat
        self.H = H_mat
        self.B = self.S.T
        
        # Find the reach at the boundary of sensing range
        connectivity_path = "./rapid_data/rapid_connect_San_Guad.csv"
        connect_data = pd.read_csv(connectivity_path, header=None)
        column_names = ['id', 'downId', 'numUp', 'upId1', 'upId2', 'upId3', 'upId4']
        connect_data.columns = column_names
        
        boundary_reach_id = []
        far_reaches = ordered_reach_coords[ordered_reach_coords["Distance to Center"] > 0.95*sensing_range]
        far_ids = far_reaches["Reach ID"].values
        for sensing_id in sensing_ids:
            down_row_index =  connect_data[connect_data['id'] == sensing_id]
            
            for i in range(1,5):
                up_id = down_row_index[f'upId{i}'].values[0]
                if up_id in sensing_ids:
                    break
                if i == 4 and sensing_id in far_ids:
                    boundary_reach_id.append(sensing_id)
        
        boundary_id_transform = np.zeros(len(sorted_ids))   
        for b_id in boundary_reach_id:
            index = np.where(sorted_ids == b_id)[0]
            boundary_id_transform[index] = 1
        
        
        self.boundary_id_transform = boundary_id_transform
    
    def drone_pos_update(self, log, lat, range):
        range *= 1.1
        self.drone_pos_initial(log, lat, range)
        return log,lat, range
    
    def sigmoid_prob(self, values, percentiles):
        percentiles = self.S @ percentiles
        # percentiles = self.S @ ( percentiles *0 + 10)
        
        # Calculate probabilities using the sigmoid function
        probabilities = 1 / (1 + np.exp(-(values - percentiles)))
        
        return probabilities
    
    def flood_prob_update(self, prob_flood_obs, S):
        S = S.T
        rows, cols = S.shape
        output = np.zeros_like(S)
        
        # Iterate through each column
        for j in range(cols):
            col = S[:, j]
            first_one_index = np.argmax(col)  # Find the index of the '1'
            output[:first_one_index + 1, j] = 1  # Fill ones until and including the '1'

        prob_flood_obs1 = (S.T @ self.boundary_id_transform) * prob_flood_obs
        prob_flood_obs2 = S @ prob_flood_obs
        flood_prob_map = output @ prob_flood_obs1 + prob_flood_obs2
        
        # Element-wise log-odds prob fusion
        # flood_prob_map = np.zeros(S.shape[0])
        # for j in range(cols):
        #     print(prob_flood_obs[j])
        #     # flood_prob_map *= (1 - output[:, j] * prob_flood_obs[j])
        #     # flood_prob_map += np.log(prob_flood_obs[j]/(1-prob_flood_obs[j])) * output[:, j]
        #     print(flood_prob_map[j])
        # log_odds = np.log(prob_flood_obs/(1-prob_flood_obs)) 
        # flood_prob_map = 1 / (1+np.exp(-log_odds))
        
        return flood_prob_map
            
    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """
        Predicts the next state of the system.

        Args:
            u (np.ndarray): Optional inflow data for the prediction.
        """
        self.x = u if u is not None else np.zeros_like(self.x)
        # self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        self.timestep += 1

    def update(self, z: np.ndarray, timestep: int, input_type: Optional[bool] = False) -> None:
        """
        Updates the Kalman Filter with new measurements.

        Args:
            z (np.ndarray): Observation data.
            timestep (int): Current timestep.
            input_type: if do input estimation
        """
        diag_R = 0.01 * z ** 2
        self.R = np.diag(diag_R)
        z = z - np.dot(self.S, np.dot(self.A0_day, self.Q0))
        
        if input_type:
            self.u_flood, self.u_flood_var = self.input_estimation(z)
            self.u_flood[self.u_flood < 0] = 0
            self.x = self.x + np.dot(self.B,self.u_flood)
            innovation=  z - np.dot(self.H, self.x)
        else: 
            innovation = z - np.dot(self.H, self.x)
        
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, innovation)
        # self.P = self.P - np.dot(np.dot(K,self.H),self.P) 
        
    def update_discharge(self) -> np.ndarray:
        """
        Updates the discharge using the current state.

        Returns:
            np.ndarray: Averaged discharge.
        """
        Q0 = copy.deepcopy(self.Q0)
        ### Method1
        Q0_ave = np.zeros_like(self.Q0)
        for _ in range(12):
            self.Q0 = self.A5 @ self.x + self.A4 @ self.Q0
            Q0_ave += self.Q0
            
        # ### Method2
        # self.Q0 = self.H1 @ self.x + self.H2 @ self.Q0
        # Q0_ave = self.Q0

        return Q0_ave / 12
    
    
    def input_estimation(self,z): 
        """
        Estimate the unkown input in the system

        Returns:
            np.ndarray: vector of unkown input and corresponding var
        """
        F = self.H @ self.B
        S = self.H @ self.P @ self.H.T + self.R
        M_1 = np.linalg.inv(F.T @ np.linalg.inv(S) @ F)
        M_2 = F.T @ np.linalg.inv(S)
        M = M_1 @ M_2
        innovation = z - self.H @ self.x
        u = M @ innovation
        u_var = M_1

        return u, u_var 
    
    def get_state(self) -> np.ndarray:
        """
        Returns the current state.

        Returns:
            np.ndarray: Current state.
        """
        return self.x

    def get_discharge(self) -> np.ndarray:
        """
        Returns the current discharge.

        Returns:
            np.ndarray: Current discharge.
        """
        return self.Q0
    
    def Qout_nc(self, m3r_ncf, Qou_ncf, IV_bas_tot):
        """
        Generates a netCDF file for the river discharge comparison.

        Args:
            m3r_ncf: m3_riv netCDF file.
            Qou_ncf: Output .nc file
            IV_bas_tot: maping from index of sorted_ID to id in unsored ID
        """
        # -------------------------------------------------------------------------
        # Get UTC date and time
        # -------------------------------------------------------------------------
        YS_dat = datetime.now(timezone.utc)
        YS_dat = YS_dat.replace(microsecond=0)
        YS_dat = YS_dat.isoformat()+'+00:00'

        # -------------------------------------------------------------------------
        # Open one file and create the other
        # -------------------------------------------------------------------------
        f = netCDF4.Dataset(m3r_ncf, 'r')
        g = netCDF4.Dataset(Qou_ncf, 'w', format='NETCDF4')

        # -------------------------------------------------------------------------
        # Copy dimensions
        # -------------------------------------------------------------------------
        YV_exc = ['nerr']
        for nam, dim in f.dimensions.items():
            if nam not in YV_exc:
                g.createDimension(nam, len(dim) if not dim.isunlimited() else None)

        g.createDimension('nerr', 3)

        # -------------------------------------------------------------------------
        # Create variables
        # -------------------------------------------------------------------------
        g.createVariable('Qout', 'float32', ('time', 'rivid'))
        g['Qout'].long_name = ('average river water discharge downstream of '
                            'each river reach')
        g['Qout'].units = 'm3 s-1'
        g['Qout'].coordinates = 'lon lat'
        g['Qout'].grid_mapping = 'crs'
        g['Qout'].cell_methods = 'time: mean'

        g.createVariable('Qout_err', 'float32', ('nerr', 'rivid'))
        g['Qout_err'].long_name = ('average river water discharge uncertainty '
                                'downstream of each river reach')
        g['Qout_err'].units = 'm3 s-1'
        g['Qout_err'].coordinates = 'lon lat'
        g['Qout_err'].grid_mapping = 'crs'
        g['Qout_err'].cell_methods = 'time: mean'

        # -------------------------------------------------------------------------
        # Copy variables variables
        # -------------------------------------------------------------------------
        YV_exc = ['m3_riv', 'm3_riv_err']
        YV_sub = ['rivid', 'lon', 'lat']
        for nam, var in f.variables.items():
            if nam not in YV_exc:
                if nam in YV_sub:
                    g.createVariable(nam, var.datatype, var.dimensions)
                    g[nam][:] = f[nam][IV_bas_tot]

                else:
                    g.createVariable(nam, var.datatype, var.dimensions)
                    g[nam][:] = f[nam][:]

                g[nam].setncatts(f[nam].__dict__)
                # copy variable attributes all at once via dictionary

        # -------------------------------------------------------------------------
        # Populate global attributes
        # -------------------------------------------------------------------------
        g.Conventions = f.Conventions
        g.title = f.title
        g.institution = f.institution
        g.source = 'RAPID, ' + 'runoff: ' + os.path.basename(m3r_ncf)
        g.history = 'date created: ' + YS_dat
        g.references = ('https://doi.org/10.1175/2011JHM1345.1, '
                        'https://github.com/c-h-david/rapid')
        g.comment = ''
        g.featureType = f.featureType

        # -------------------------------------------------------------------------
        # Close all files
        # -------------------------------------------------------------------------
        f.close()
        g.close()

if __name__ == '__main__':
    rapid_kf = RAPIDKF(load_mode=1)
    rapid_kf.simulate_flood()
