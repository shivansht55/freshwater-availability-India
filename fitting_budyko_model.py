# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:23:14 2025

@author: shiva
"""

import xarray as xr
import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Define the budyko function
def budyko_m1(aridity, omega):
    evapratio = 1 + aridity - (1 + aridity**omega)**(1/omega)
    return evapratio

# Define the function to fit omega
def budyko_single_fit(omega, evapratio, aridity):
    return abs(evapratio - budyko_m1(aridity, omega))

# Load data from .nc files
pet_ds = xr.open_dataset('pet_1980_2020.nc')  
p_ds = xr.open_dataset('pr_1980_2020.nc')  
et_ds = xr.open_dataset('GLEAM_1980_2020..nc')  

pet_data = pet_ds['pet'].values  # Assuming 'pet' is the variable name in your .nc file
p_data = p_ds['rf'].values  # Assuming 'precip' is the variable name in your .nc file
et_data = et_ds['E'].values  # Assuming 'et' is the variable name in your .nc file
latitudes = pet_ds['lat'].values
longitudes = pet_ds['lon'].values
time_steps = pet_ds['time'].values

# Ensure the datasets have the same grids (latitudes and longitudes)
if not (np.array_equal(latitudes, p_ds['lat'].values) and np.array_equal(latitudes, et_ds['lat'].values) and
        np.array_equal(longitudes, p_ds['lon'].values) and np.array_equal(longitudes, et_ds['lon'].values)):
    raise ValueError("The datasets do not have the same grids.")

# Create a folder to store the results
import os
if not os.path.exists('calibrated_omega_results'):
    os.makedirs('calibrated_omega_results')

# Iterate through each time step and location
for time_index, time_step in enumerate(time_steps):
    results = pd.DataFrame(columns=['Latitude', 'Longitude', 'Calibrated_Omega'])

    for lat_index, lat in enumerate(latitudes):
        for lon_index, lon in enumerate(longitudes):
            pet = pet_data[time_index, lat_index, lon_index]
            p = p_data[time_index, lat_index, lon_index]
            et = et_data[time_index, lat_index, lon_index]

            if not np.isnan(pet) and not np.isnan(p) and not np.isnan(et) and pet > 0 and p > 0 and et > 0:
                aridity = pet / p
                evapratio = et / p

                # Calibrate omega
                initial_guess = 4  # Initial guess for omega
                fit = minimize(budyko_single_fit, initial_guess, args=(evapratio, aridity), bounds=[(0, None)])
                calibrated_omega = fit.x[0]

                results = results.append({'Latitude': lat, 'Longitude': lon, 'Calibrated_Omega': calibrated_omega}, ignore_index=True)

    # Export the results to a CSV file for each time step
    results.to_csv(f'calibrated_omega_results/calibrated_omega_values_{time_index}.csv', index=False)