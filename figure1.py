# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:26:58 2025

@author: shiva
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read and Prepare the Data
data = pd.read_csv("budyko_plot.txt", delimiter=',')  #budyko_plot.txt has statewise ae/p and pet/p values for each grid 
data = data.dropna()  

# Step 2: Plot the Data on the Budyko Curve
states = data['State_Name'].unique()

# Define colors and markers
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(states)))
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', '*', 'x', '+', '|', '_', '.', ',', 
           '1', '2', '3', '4', '8', 's', 'p', '*', 'h', '+', 'x', 'D', 'o', '^']

# Budyko curve data
budyko_curve_x = np.arange(0, 16, 0.05)
budyko_curve_y = np.power((budyko_curve_x * np.tanh(1 / budyko_curve_x) * (1 - np.exp(-budyko_curve_x))), 0.5)

# Energy limit
energy_limit_x = np.arange(0, 1.01, 0.05)
energy_limit_y = energy_limit_x

# Water limit starting from energy limit endpoint
water_limit_x = np.arange(1, 16, 0.05)
water_limit_y = np.ones_like(water_limit_x)

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting scatter points
for i, state in enumerate(states):
    state_data = data[data['State_Name'] == state]
    x = state_data['pet/p']
    y = state_data['ae/p']
    ax.scatter(x, y, label=state, color=colors[i], marker=markers[i], s=50)

# Extended axis limits for better spacing
plt.xlim(0, 17)

# Budyko curve as solid line
plt.plot(budyko_curve_x, budyko_curve_y, linestyle='-', linewidth=2, color='k')

# Energy limit
plt.plot(energy_limit_x, energy_limit_y, linestyle='--', linewidth=2, color='k')
plt.text(0.3, 0.8, 'Energy Limit', fontsize=10, rotation=90, verticalalignment='center')

# Water limit
plt.plot(water_limit_x, water_limit_y, linestyle='--', linewidth=2, color='k')
plt.text(8, 1.02, 'Water Limit', fontsize=10, horizontalalignment='center')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax.set_xlabel(r'Aridity Index (E$_{P}$/P)', fontsize=14)
ax.set_ylabel(r'Evaporation Ratio (E/P)', fontsize=14)


ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5)
plt.savefig("Figure_1.png", dpi=500, bbox_inches='tight')
plt.show()
