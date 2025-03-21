# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:32:43 2025

@author: shiva
"""

import xarray as xr
import numpy as np
import pymannkendall as mk 
#import cartopy.crs as ccrs

#import cartopy.io.shapereader as shpreader
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import datetime

import cartopy.crs as ccrs
import geopandas as gpd
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
#import metpy.calc as mpcalc
#from metpy.units import units
import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from shapely.geometry import Polygon
from cartopy.feature import ShapelyFeature
import glob


observed1 = xr.open_dataset("pr_5_year_running_mean.nc",decode_times=False)
observed1 = observed1['rf']
obs1 = observed1.values

lats1 = observed1.lat.values
lons1 = observed1.lon.values

result1 = np.zeros([len(observed1.lat.values),
                                           len(observed1.lon.values)])
result1[:] = np.nan

sig1 = np.zeros([len(observed1.lat.values),
                                           len(observed1.lon.values)])
sig1[:] = np.nan
l1= []
ln1 =[]
c1=1
for i,lat in enumerate(lats1):
    #print(c)
    #c+=1
    for j,lon in enumerate(lons1):
        if (np.isnan(obs1[0,i,j])):
            result1[i,j] = np.nan
        else:
            y1 = obs1[:,i,j]
            test1=mk.original_test(y1,alpha=0.05)
            result1[i,j]=test1.slope
            if (test1.p < 0.05):
                l1.append(lats1[i])
                ln1.append(lons1[j])
                sig1[i,j] = 1
            
ds1 = xr.Dataset({"trend": (('lat','lon'), result1)},coords={'lat': lats1, 'lon': lons1,})
#ds.trend.plot()


ds1.to_netcdf("pr_5_year_running_mean_trend.nc") 


ds1 =ds1.sel(lat =slice(5, 38),lon = slice(67, 100))

lat = [5.5,   6.5,   7.5,   8.5,   9.5,
10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,  18.5,  19.5,
20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,  27.5,  28.5,  29.5,
30.5,  31.5,  32.5,  33.5,  34.5,  35.5,  36.5,  37.5,]
#######################################################################################################################################
observed2 = xr.open_dataset("pet_5_year_running_mean.nc",decode_times=False)
observed2 = observed2['pet']
obs2 = observed2.values

lats2 = observed2.lat.values
lons2 = observed2.lon.values

result2 = np.zeros([len(observed2.lat.values),
                                           len(observed2.lon.values)])
result2[:] = np.nan

sig2 = np.zeros([len(observed2.lat.values),
                                           len(observed2.lon.values)])
sig2[:] = np.nan
l2= []
ln2 =[]
c2=1
for i,lat in enumerate(lats2):
    #print(c)
    #c+=1
    for j,lon in enumerate(lons2):
        if (np.isnan(obs2[0,i,j])):
            result2[i,j] = np.nan
        else:
            y2 = obs2[:,i,j]
            test2=mk.original_test(y2,alpha=0.05)
            result2[i,j]=test2.slope
            if (test2.p < 0.05):
                l2.append(lats2[i])
                ln2.append(lons2[j])
                sig2[i,j] = 1
            
ds2 = xr.Dataset({"trend": (('lat','lon'), result2)},coords={'lat': lats2, 'lon': lons2,})
#ds.trend.plot()


ds2.to_netcdf("pet_5_year_running_mean_trend.nc") 


ds2 =ds2.sel(lat =slice(5, 38),lon = slice(67, 100))

lat = [5.5,   6.5,   7.5,   8.5,   9.5,
10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,  18.5,  19.5,
20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,  27.5,  28.5,  29.5,
30.5,  31.5,  32.5,  33.5,  34.5,  35.5,  36.5,  37.5,]
#######################################################################################################################################
observed3 = xr.open_dataset("calibrated_E.nc",decode_times=False)
observed3 = observed3['evapotranspiration']
obs3 = observed3.values

lats3 = observed3.lat.values
lons3 = observed3.lon.values

result3 = np.zeros([len(observed3.lat.values),
                                           len(observed3.lon.values)])
result3[:] = np.nan

sig3 = np.zeros([len(observed3.lat.values),
                                           len(observed3.lon.values)])
sig3[:] = np.nan
l3= []
ln3 =[]
c3=1
for i,lat in enumerate(lats3):
    #print(c)
    #c+=3
    for j,lon in enumerate(lons3):
        if (np.isnan(obs3[0,i,j])):
            result3[i,j] = np.nan
        else:
            y3 = obs3[:,i,j]
            test3=mk.original_test(y3,alpha=0.05)
            result3[i,j]=test3.slope
            if (test3.p < 0.05):
                l3.append(lats3[i])
                ln3.append(lons3[j])
                sig3[i,j] = 1
            
ds3 = xr.Dataset({"trend": (('lat','lon'), result3)},coords={'lat': lats3, 'lon': lons3,})
#ds.trend.plot()


ds3.to_netcdf("calibrated_E_trend.nc") 


ds3 =ds3.sel(lat =slice(5, 38),lon = slice(67, 100))

lat = [5.5,   6.5,   7.5,   8.5,   9.5,
10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,  18.5,  19.5,
20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,  27.5,  28.5,  29.5,
30.5,  31.5,  32.5,  33.5,  34.5,  35.5,  36.5,  37.5,]
#######################################################################################################################################
observed4 = xr.open_dataset("water_availability.nc",decode_times=False)
observed4 = observed4['wtrav']
obs4 = observed4.values

lats4 = observed4.lat.values
lons4 = observed4.lon.values

result4 = np.zeros([len(observed4.lat.values),
                                           len(observed4.lon.values)])
result4[:] = np.nan

sig4 = np.zeros([len(observed4.lat.values),
                                           len(observed4.lon.values)])
sig4[:] = np.nan
l4= []
ln4 =[]
c4=1
for i,lat in enumerate(lats4):
    #print(c)
    #c+=4
    for j,lon in enumerate(lons4):
        if (np.isnan(obs4[0,i,j])):
            result4[i,j] = np.nan
        else:
            y4 = obs4[:,i,j]
            test4=mk.original_test(y4,alpha=0.05)
            result4[i,j]=test4.slope
            if (test4.p < 0.05):
                l4.append(lats4[i])
                ln4.append(lons4[j])
                sig4[i,j] = 1
            
ds4 = xr.Dataset({"trend": (('lat','lon'), result4)},coords={'lat': lats4, 'lon': lons4,})
#ds.trend.plot()


ds4.to_netcdf("water_availability_trend.nc") 


ds4 =ds4.sel(lat =slice(5, 38),lon = slice(67, 100))

lat4 = [5.5,   6.5,   7.5,   8.5,   9.5,
10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,  18.5,  19.5,
20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,  27.5,  28.5,  29.5,
30.5,  31.5,  32.5,  33.5,  34.5,  35.5,  36.5,  37.5,]

fig = plt.figure(figsize=(10, 8))

# First subplot: Precipitation
ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
cf1 = ds1.trend.plot(x='lon', y='lat', add_colorbar=False, cmap="coolwarm", vmin=-10, vmax=10)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
plt.scatter(ln1, l1, s=5, c='k', alpha=1, marker="+")
plt.title('Precipitation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
shapefile1 = gpd.read_file('Z:/Shivansh/Shapefile.shp', name="INDIA")
shapefile1.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=0.5)
gl1 = ax1.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl1.xlabels_top = False
gl1.ylabels_right = False
ax1.annotate('(a)', xy=(0, 1), xycoords='axes fraction', fontsize=12, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')

# Second subplot: Potential Evapotranspiration
ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
cf2 = ds2.trend.plot(x='lon', y='lat', add_colorbar=False, cmap="coolwarm", vmin=-10, vmax=10)
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'))
plt.scatter(ln2, l2, s=5, c='k', alpha=1, marker="+")
plt.title('Potential Evapotranspiration')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
shapefile2 = gpd.read_file('Z:/Shivansh/Shapefile.shp', name="INDIA")
shapefile2.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=0.5)
gl2 = ax2.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.xlabels_top = False
gl2.ylabels_right = False
ax2.annotate('(b)', xy=(0, 1), xycoords='axes fraction', fontsize=12, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Third subplot: Evapotranspiration
ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
cf3 = ds3.trend.plot(x='lon', y='lat', add_colorbar=False, cmap="coolwarm", vmin=-10, vmax=10)
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'))
plt.scatter(ln3, l3, s=5, c='k', alpha=1, marker="+")
plt.title('Evapotranspiration')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
shapefile3 = gpd.read_file('Z:/Shivansh/Shapefile.shp', name="INDIA")
shapefile3.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=0.5)
gl3 = ax3.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl3.xlabels_top = False
gl3.ylabels_right = False
ax3.annotate('(c)', xy=(0, 1), xycoords='axes fraction', fontsize=12, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Fourth subplot: Water availability
ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
cf4 = ds4.trend.plot(x='lon', y='lat', add_colorbar=False, cmap="coolwarm", vmin=-10, vmax=10)
ax4.add_feature(cfeature.COASTLINE.with_scale('50m'))
plt.scatter(ln4, l4, s=5, c='k', alpha=1, marker="+")
plt.title('Water availability')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
shapefile4 = gpd.read_file('Z:/Shivansh/Shapefile.shp', name="INDIA")
shapefile4.plot(ax=ax4, facecolor='none', edgecolor='black', linewidth=0.5)
gl4 = ax4.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl4.xlabels_top = False
gl4.ylabels_right = False
ax4.annotate('(d)', xy=(0, 1), xycoords='axes fraction', fontsize=12, fontweight='bold',
             xytext=(-16, 10), textcoords='offset points', ha='center', va='center')
# Create a single colorbar
cbar_ax = fig.add_axes([0.35, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(cf4, cax=cbar_ax, orientation='horizontal')
cbar.set_label('mm/year', fontsize=9)

plt.subplots_adjust(hspace=0.3,wspace=0)  # Adjust the space between subplots


plt.savefig('Figure_2.png', dpi=600)
plt.show()

