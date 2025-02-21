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
        self.sub_dir_path = "model_saved_3hour_flood2"
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
        
    def load_file(self, dir_path: str) -> None:
        """
        Loads model parameters from CSV files and initializes the model.

        Args:
            dir_path (str): Directory path where the CSV files are stored.
        """
        # Introduction to the model parameters:
        # These correspond to Equations (6) and (7) in the paper:
        # "Underlying fundamentals of Kalman filtering for river network modeling"

        # Matrix definitions:
        # A4 = [I - C1N]^-1(C3 + C2N)
        # A5 = [I - C1N]^-1(C1 + C2)
        # Ae is used for open-loop simulation, while Ae_day is for Kalman Filter (KF) estimation.
        # Parameter updates (open-loop and KF estimation):
        # Ae += (12 - p)/12 * A4^p @ A5  where p ranges from 0 to 11
        # A0 += 1/12 * A4^p  where p ranges from 1 to 12
        # Ae_day += (96 - p)/96 * A4^p @ A5  where p ranges from 0 to 95
        # A0_day += 1/96 * A4^p  where p ranges from 1 to 96
        # H1 and H2 updates:
        # H1 += A4^p @ A5  where p ranges from 0 to 95
        # H2 = A4^96

        params = {
            'id_path_unsorted': os.path.join(dir_path, 'rapid_data/rivid.csv'),
            'id_path_sorted': os.path.join(dir_path, 'rapid_data/riv_bas_id_San_Guad_hydroseq.csv'),
            'connectivity_path': os.path.join(dir_path, 'rapid_data/rapid_connect_San_Guad.csv'),
            'm3riv_path': os.path.join(dir_path, 'rapid_data/m3_riv.csv'),
            'x_path': os.path.join(dir_path, 'rapid_data/x_San_Guad_2004_1.csv'),
            'k_path': os.path.join(dir_path, 'rapid_data/k_San_Guad_2004_1.csv'),
            'obs_path': os.path.join(dir_path, 'rapid_data/Qobs_San_Guad_2010_2013_full.csv'),
            'obs_id_path': os.path.join(dir_path, 'rapid_data/obs_tot_id_San_Guad_2010_2013_full.csv'),
            'vic_model_path': os.path.join(dir_path, 'rapid_data/m3_riv_vic_month.csv'),
            'ens_model_path': os.path.join(dir_path, 'rapid_data/m3_riv_ens_month.csv'),
            'days': self.days,
            'month': self.month,
            'radius': self.radius,
            'epsilon': self.epsilon,
            'i_factor': self.i_factor,
        }

        data_processor = PreProcessor()
        self.Ae, self.A0, self.Ae_day, self.A0_day, self.S, self.P, self.R, self.u, self.obs_data, \
            self.A4, self.A5, self.H1, self.H2, self.id2sortedid = data_processor.pre_processing(**params)

        saved_dict = {
            'Ae': self.Ae,
            'A0': self.A0,
            'Ae_day': self.Ae_day,
            'A0_day': self.A0_day,
            'A4': self.A4,
            'A5': self.A5,
            'H1': self.H1,
            'H2': self.H2,
            'S': self.S,
            'P': self.P,
            'R': self.R,
            'u': self.u,
            'obs_data': self.obs_data,
            'id2sortedid': self.id2sortedid,
        }

        dis_name = os.path.join(self.sub_dir_path,'load_coef.pkl')
        with open(os.path.join(dir_path, dis_name), 'wb') as f:
            pickle.dump(saved_dict, f)

    def simulate_flood(self, sim_mode: int = 0) -> None:
        """
        Simulates the Kalman Filter model.

        Args:
            sim_mode (int): Mode for simulation (0 = open loop, 1 = Kalman Filter estimation).
        """
        if sim_mode == 0:
            print(f"Simulation started with mode: open loop")
        elif sim_mode == 1: 
            print(f"Simulation started with mode: Klaman Filter estimation")
            
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
        
        self.H = np.dot(self.S, self.Ae_day)
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
            "percentile_90.csv"
        ]
        
        file_paths = [os.path.join(dir_path, file) for file in file_names]

        # Check if all files exist
        if all(os.path.exists(file) for file in file_paths):
            print("All CSV files exist. Loading data instead of recalculating...")

            added_flood = np.loadtxt(file_paths[0], delimiter=",")
            inject_flood_inflow = np.loadtxt(file_paths[1], delimiter=",")
            origin_inflow = np.loadtxt(file_paths[2], delimiter=",")
            discharge_obs_kf1 = np.loadtxt(file_paths[3], delimiter=",")
            open_loop_x = np.loadtxt(file_paths[4], delimiter=",")
            discharge_only_flood = np.loadtxt(file_paths[5], delimiter=",")
            percentile_90 = np.loadtxt(file_paths[6], delimiter=",")
        
        else:
            print("Some files are missing. Proceeding with the full simulation...")     
            # Find the 90th percentile along each column (axis=0 for each reach)
            percentile_90 = np.percentile(self.u, 90, axis=0)  
            np.savetxt(os.path.join(dir_path, "percentile_90.csv"), percentile_90, delimiter=",")
            
            '''
            Open-loop simulation only added flood(predict inflow every 3 hours)
            '''
            for timestep in tqdm(range(self.days)):
                self.x = np.zeros_like(self.u[0])

                for i in range(evolution_steps):
                    # Unkown input
                    index = timestep * evolution_steps + i
                    inject_flood_inflow[index] = copy.deepcopy(self.u[index])
                    
                    if timestep <= 10:
                        added_flood[index][0] = 20
                        origin_inflow[index] = self.u[index]
                        inject_flood_inflow[index] += added_flood[index]
                        
                    self.predict(added_flood[index])
                    discharge_only_flood[timestep] += self.update_discharge()/evolution_steps

            np.savetxt(os.path.join(dir_path, "injected_flood.csv"), added_flood, delimiter=",")
            np.savetxt(os.path.join(dir_path, "inject_w_inflow.csv"), inject_flood_inflow, delimiter=",")
            np.savetxt(os.path.join(dir_path, "original_inflow.csv"), origin_inflow, delimiter=",")
            np.savetxt(os.path.join(dir_path, "discharge_only_flood.csv"), discharge_only_flood, delimiter=",")
            
            '''
            discharge estimation from original observation (updates every 3 hours)
            '''
            self.timestep = 0
            self.Q0 = np.zeros_like(self.u[0])
            discharge_obs_kf1 = np.zeros_like(discharge_only_flood)
            
            for timestep in tqdm(range(self.days)):
                self.x = np.zeros_like(self.u[0])
                for i in range(evolution_steps):
                    self.x += self.u[timestep * evolution_steps + i] / evolution_steps
                
                self.timestep += 1
                
                self.update(self.obs_data[timestep], timestep)
                
                for i in range(evolution_steps):
                    discharge_obs_kf1[timestep] += self.update_discharge()/evolution_steps
                
            np.savetxt(os.path.join(dir_path, "discharge_from_obs1.csv"), discharge_obs_kf1, delimiter=",")
            
            '''
            Open-loop simulation with added flood 
            '''
            self.timestep = 0
            self.Q0 = np.zeros_like(self.u[0])
            
            for timestep in tqdm(range(self.days)):
                discharge_avg = np.zeros_like(self.u[0])
                self.x = np.zeros_like(self.u[0])
                    
                for i in range(evolution_steps):
                    self.predict(inject_flood_inflow[timestep * evolution_steps + i])
                    discharge_avg += self.update_discharge()

                discharge_avg /= evolution_steps
                open_loop_x.append(discharge_avg)
                
            np.savetxt(os.path.join(dir_path, "open_loop_est.csv"), open_loop_x, delimiter=",")
        
        '''
        Simulation under synthetic data
        '''
        # self.S = np.eye(self.P.shape[0])
        self.H = np.dot(self.S, self.Ae_day)
        self.B = self.S.T
        self.timestep = 0
        self.Q0 = np.zeros_like(self.u[0])
        
        for timestep in tqdm(range(self.days)):
            obs_synthetic.append(discharge_obs_kf1[timestep] + discharge_only_flood[timestep])
            
        np.savetxt(os.path.join(dir_path, "obs_synthetic.csv"), obs_synthetic, delimiter=",")

        for timestep in tqdm(range(self.days)):
            discharge_avg = np.zeros_like(self.u[0])
            self.x = np.zeros_like(self.u[0])

            # Kalman Filter estimation (updates every 3 hours)
            for i in range(evolution_steps):
                self.x += self.u[timestep * evolution_steps + i]  / evolution_steps
                # self.x += added_flood[timestep * evolution_steps + i] / evolution_steps
                
            self.timestep += 1

            gt_obs = discharge_obs_kf1[timestep] + discharge_only_flood[timestep]
            gt_obs = self.S @ gt_obs
            self.update(gt_obs, timestep, True)

            for i in range(evolution_steps):
                discharge_avg += self.update_discharge()

            discharge_avg /= evolution_steps
            Qout[timestep, :] = discharge_avg[:]

            state_estimation.append(copy.deepcopy(self.get_state()))
            discharge_estimation.append(discharge_avg)
            
            flood_est.append(self.S.T @ self.u_flood)

        # Save results to the created directory
        Qout_df = pd.DataFrame(Qout[:])
        Qout_df.to_csv(os.path.join(dir_path, "Qout.csv"), index=False)
        np.savetxt(os.path.join(dir_path, "discharge_est.csv"), discharge_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "river_lateral_est.csv"), state_estimation, delimiter=",")
        np.savetxt(os.path.join(dir_path, "flood_est.csv"), flood_est, delimiter=",")
        g.close()
        
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
