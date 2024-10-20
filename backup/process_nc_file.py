import netCDF4 as nc
import pandas as pd

# Open the NetCDF file
dataset = nc.Dataset('../rapid_data/m3_riv_San_Guad_20100101_20131231_VIC0125_3H_utc_err_R286_D_scl.nc')

# Create an empty DataFrame
df = pd.DataFrame()
print(dataset.variables.keys())
print(f"***** M3_RIV: {dataset.variables['m3_riv']}")
# print(f"***** M3_RIV Error: {dataset.variables['m3_riv_err']}")
print(f"***** rivid: {dataset.variables['rivid']}")
print(f"***** time: {dataset.variables['time']}")
print(f"***** time_bnds: {dataset.variables['time_bnds']}")
# print(dataset.variables['m3_riv_err'])


# # Extract 'm3_riv' data and save to CSV
# m3_riv_data = dataset.variables['m3_riv'][:].astype('float64') 
# m3_riv_df = pd.DataFrame(m3_riv_data)
# m3_riv_df.to_csv('./rapid_data/m3_riv.csv', float_format='%.15f', index=False)

# # Extract 'rivid' data and save to CSV
# comid_data = dataset.variables['rivid'][:]
# comid_df = pd.DataFrame(comid_data)
# comid_df.to_csv('./rapid_data/rivid.csv', index=False)

# # # Extract 'm3_riv_err' data and save to CSV
# comid_data = dataset.variables['m3_riv_err'][:].astype('float64') 
# comid_df = pd.DataFrame(comid_data)
# comid_df.to_csv('./rapid_data/m3_riv_err.csv', float_format='%.15f', index=False)

# # # Extract 'time' data and save to CSV
# comid_data = dataset.variables['time'][:]
# comid_df = pd.DataFrame(comid_data)
# comid_df.to_csv('./rapid_data/time.csv', index=False)

# # # Extract 'time_bnds' data and save to CSV
# comid_data = dataset.variables['time_bnds'][:]
# comid_df = pd.DataFrame(comid_data)
# comid_df.to_csv('./rapid_data/time_bnds.csv', index=False)

# # Close the dataset
dataset.close()

