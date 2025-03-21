# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:44:37 2025

@author: shiva
"""

import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the names from the text file
names_file = 'state_list1.txt'  # list of states
with open(names_file, 'r') as file:
    names = [line.strip() for line in file.readlines()]

# Step 2: Read data from the first text file
data_file = 'correlation_values_pic.txt'  #correlation with picontrol for each state
data = np.loadtxt(data_file, delimiter=' ')  

# Step 3: Calculate percentile values for each column
percentiles = [10, 20, 50, 80, 90]
percentile_values = np.percentile(data, percentiles, axis=0)

# Step 4: Read the 29 values from the second text file
values_file = 'correlation_values_statewise.txt'  #correlation with historical simulation statewise
values = np.loadtxt(values_file)  # Assuming it's a single column of 29 values

# Step 5: Create a dictionary to read population density
population_density_data = {
    'Andhra Pradesh': 305,
    'Arunachal Pradesh': 17,
    'Assam': 398,
    'Bihar': 1106,
    'Chhattisgarh': 189,
    'Gujarat': 308,
    'Haryana': 573,
    'Himachal Pradesh': 123,
    'Jharkhand': 414,
    'Karnataka': 319,
    'Kerala': 860,
    'Madhya Pradesh': 236,
    'Maharashtra': 365,
    'Manipur': 128,
    'Meghalaya': 132,
    'Mizoram': 52,
    'Nagaland': 119,
    'Odisha': 270,
    'Punjab': 551,
    'Rajasthan': 200,
    'Sikkim': 86,
    'Tamil Nadu': 555,
    'Telangana': 312,
    'Tripura': 350,
    'Uttar Pradesh': 829,
    'Uttarakhand': 189,
    'West Bengal': 1028,
}

# Step 6: Extract population densities corresponding to the states
population_densities = np.array([population_density_data[name] for name in names if name in population_density_data])

# Normalize population density for color mapping
norm = plt.Normalize(population_densities.min(), population_densities.max())
colors = plt.cm.Reds(norm(population_densities))  
# Step 7: Create a scatter plot
plt.figure(figsize=(12, 6))  

# Create boxplot for percentile values
plt.boxplot(percentile_values, vert=True, showcaps=True, showfliers=True)

# Scatter plot of historical values
scatter = plt.scatter(range(1, len(values) + 1), values, marker="o", s=100, c=colors, label='Historical')

# Manually set the color for "India"
for i, name in enumerate(names):
    if name == 'India':
        plt.scatter(i + 1, values[i], marker="o", s=100, c='red', label='India')

# Set the x-axis labels using the names from the file
plt.xticks(range(1, len(names) + 1), names, rotation=90, fontsize=10)
plt.yticks(fontsize=12)  # Rotate the labels for better readability

# Add labels and a title
plt.xlabel('States', fontsize=10)
plt.ylabel('Correlation', fontsize=10)

# Add a color bar to show population density
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])  # Only needed for older Matplotlib versions
plt.colorbar(sm, label='Population Density (persons/km\u00B2)')


# Save the figure as a .png file
plt.savefig('Figure4.png', dpi=600, bbox_inches='tight')

plt.show() 