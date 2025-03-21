# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:57:16 2025

@author: shiva
"""

import matplotlib.pyplot as plt
import numpy as np

# List of text files
data1 = np.genfromtxt('anomaly_wtrav_observed_values.txt', dtype=float) #text_file with anomalies of water avaialbility using  observed data
data2 = np.genfromtxt('anomaly_wtrav_Amon_IITM-ESM_hist_ssp585.txt', dtype=float) #text file with anomalies of water availability calculated using IITM-ESM data

plt.plot(data1, color='red', linewidth=1.5, label='Observed')
plt.plot(data2, color='black', linewidth=1.5, label='IITM-ESM')

data_list = [data1, data2]
merged_data = np.vstack(data_list)

plt.xticks((0, 7, 13, 19, 25, 31, 36), labels=('1982', '1989', '1995', '2001', '2007', '2013', "2018"))
plt.xlabel('Years')
plt.ylabel('Anomaly (mm/year)', fontsize=10)
plt.legend(loc='upper right', ncol=2)
plt.title('India', fontsize=10)

# Adding annotation (a) to the first plot
plt.annotate('(a)', xy=(0, 1), xycoords='axes fraction', fontsize=10, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Save the plot as a TIFF file
plt.savefig('plot_fig3(a).png', dpi=600, format='png')
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'slopes.txt' #slope of anomalies of 1000 bootstrapped IITM-ESM picontrol chunks
df = pd.read_csv(file_path, header=None)

# Plot the histogram with KDE
sns.histplot(df, kde=True, label='piControl')

# Add vertical lines for observed and IITM-ESM trends
plt.axvline(-5.603, color='red', linestyle='-', linewidth=3, label='Observed') #slope of hist wtrav anomalies
plt.axvline(-1.8566795833333334, color='black', linestyle='-', linewidth=3, label='IITM-ESM') #slope of iitm-esm wtrav anomalies

# Add labels and title
plt.xlabel('Anomalies Trend in All-India Average Water Availability (mm/year)', fontsize=10)
plt.ylabel('Count', fontsize=10)

# Add legend
plt.legend()
plt.title('Trend Distribution', fontsize=10)
# Adding annotation (b) to the plot
plt.annotate('(b)', xy=(0, 1), xycoords='axes fraction', fontsize=10, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Save the plot as a TIFF file
plt.savefig('plot_fig3(b).png', dpi=500, format='png')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = 'correlations_alter.txt' #contains 1000 correlations between anomalies of wtrav obs and picontrol
df = pd.read_csv(file_path,header=None)

sns.histplot(df, kde=True,label='piControl')

plt.axvline(0.4276, color='black', linestyle='-', linewidth= 3, label='IITM-ESM') #correlations between anomalies of wtrav obs and hist
plt.xlabel('Correlation',fontsize=10)
plt.ylabel('Count', fontsize=10)

plt.legend()
plt.title('Detection Test', fontsize=10)
# Adding annotation (a) to the first plot
plt.annotate('(c)', xy=(0, 1), xycoords='axes fraction', fontsize=10, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Save the plot as a TIFF file
plt.savefig('plot_fig3(c).png', dpi=500, format='png')
plt.show()