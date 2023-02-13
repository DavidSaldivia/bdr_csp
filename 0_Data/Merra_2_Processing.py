# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 18:09:05 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
import xarray as xr
import scipy.optimize as spo
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import time
import os
from os.path import isfile
import pickle
import cantera as ct

from pvlib.location import Location
import pvlib.solarposition as plsp
from datetime import datetime,timedelta
import pvlib.irradiance as plir

import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cf                 # import features

from scipy.interpolate import interp2d, interpn
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator as rgi
from numba import jit, njit
pd.set_option('display.max_columns', None)

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)

###############################################

fldr_data = 'MERRA-2_Raw'
tz = 'Australia/Brisbane'
year = 2016

list_files = os.listdir(fldr_data)
list_files.sort()
list_slv =  [x for x in list_files if ((str(year) in x) and ('slv') in x)]
list_rad =  [x for x in list_files if ((str(year) in x) and ('rad') in x)]

data = xr.Dataset()
for i in range(len(list_slv)):
               
    file_rad = os.path.join(fldr_data,list_rad[i])
    data_rad = xr.open_dataset(file_rad)
    file_slv = os.path.join(fldr_data,list_slv[i])
    data_slv = xr.open_dataset(file_slv)
    data_slv = data_slv.where(data_slv.lon>=110., drop=True).assign(data_rad)
    data = data.merge(data_slv)
    if i%10==0:
        print('{:d} days alread processed'.format(i+1))
print(data)

print('Processing the data to generate DNI, azimuth and elevation')
data['WS'] = (data['V10M']**2+data['U10M']**2)**0.5
data  = data.drop(['V10M','U10M'])
GHI   = data.SWGDN
Pr    = data.SLP
Tdew  = data.T2MDEW
times = data.time
lats  = data.lat
lons  = data.lon

DNI_disc   = np.empty([len(times),len(lats),len(lons)])
DNI_dirint = np.empty([len(times),len(lats),len(lons)])
azimuth= np.empty([len(times),len(lats),len(lons)])
elev   = np.empty([len(times),len(lats),len(lons)])

for i in range (0,len(lats)):
    for j in range (0,len(lons)):
        lat=lats[i].values
        lon=lons[j].values
        times1 = pd.to_datetime(times)
        GHI1 = GHI[:,i,j].to_series()
        sol_pos = Location(lat, lon, tz=tz).get_solarposition(times1)
        zen_ij = sol_pos.zenith
        DNIij = plir.disc(GHI1,zen_ij,times1,pressure=Pr[:,i,j],min_cos_zenith=0.065, max_zenith=87, max_airmass=12)
        DNI_disc[:,i,j]     = DNIij["dni"]
        
        DNIij = plir.dirint(GHI1,zen_ij,times1,pressure=Pr[:,i,j], temp_dew=Tdew[:,i,j], min_cos_zenith=0.065, max_zenith=87, use_delta_kt_prime=True)
        DNI_dirint[:,i,j]     = DNIij
        
        azimuth[:,i,j] = sol_pos.azimuth
        elev[:,i,j]    = sol_pos.elevation
    print(lat,lon)

DNI_disc_xr   = xr.DataArray(DNI_disc,coords=[times,lats,lons],name='dni_disc').astype('float32')
DNI_dirint_xr = xr.DataArray(DNI_dirint, coords=[times,lats,lons], name='dni_dirint').astype('float32')
azixr = xr.DataArray(azimuth,coords=[times,lats,lons],name='azimuth').astype('float32')
elevxr = xr.DataArray(elev,coords=[times,lats,lons],name='elevation').astype('float32')
data = data.assign(DNI_disc=DNI_disc_xr)
data = data.assign(DNI_dirint=DNI_dirint_xr)
data = data.assign(azimuth=azixr)
data = data.assign(elevation=elevxr)

fldr_out = 'MERRA-2'
file_out = os.path.join(fldr_out,'MERRA2_Processed_{:d}.nc'.format(year))
data.to_netcdf(file_out)

#%% MAP 1: PLOTTING VARIABLES
#year = 2017
DNI_disc = data.sum(dim='time')['DNI_disc']/365/1000. #(in kWh/day)
DNI_dirint = data.sum(dim='time')['DNI_dirint']/365/1000. #(in kWh/day)
GHI = data.sum(dim='time')['SWGDN']/365/1000. #(in kWh/day)
Tavg = data.mean(dim='time')['T2M'] - 273.15 #(C)


AUS_PROJ = 3112
NSW_PROJ = 3308
AUS_BOUNDS = [110.,155.,-45.,-10.]
NSW_BOUNDS = [138.,154.,-38.,-28.]

cits = pd.DataFrame([['Sydney',151.21, -33.86],
          ['Melbourne',144.9631, -37.8136],
          ['Brisbane',153.0281, -27.4678],
          ['Perth',115.8589, -31.9522],
          ['Adelaide',138.6011, -34.9289],
          ['Darwin',130.8411, -12.4381],
          ['Canberra',149.1269, -35.2931],
          ['Hobart',147.3257, -42.8826]],
          columns=['city','lon','lat'])

# for var in [GHI,DNI,Tavg]:
for var in [DNI_disc,DNI_dirint]:    
    f_s = 16
    fig = plt.figure(figsize=(12, 9))
    # ax = fig.add_subplot(111,projection=ccrs.epsg(AUS_PROJ))
    ax = fig.add_subplot(111,projection=ccrs.PlateCarree())
    ax.set_extent(AUS_BOUNDS)
    ax.set_title("{:d}-{:}".format(year,var.name))
    
    vmin = var.min()
    vmax = var.max()
    vmin = 2.
    vmax = 8.
    levels=10
    surf = ax.contourf(var.lon, var.lat, var, 10, transform=ccrs.PlateCarree(),cmap=cm.viridis, vmin=vmin, vmax=vmax,levels=levels)
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    cb.ax.locator_params(nbins=7)
    
    res='50m'
    ocean = cf.NaturalEarthFeature('physical', 'ocean', res, edgecolor='face', facecolor= cf.COLORS['water'])
    lakes = cf.NaturalEarthFeature('physical', 'lakes', res, edgecolor='k', facecolor= cf.COLORS['water'], lw=0.5)
    borders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces', res, facecolor='none', edgecolor='k', lw=0.5)
    
    ax.add_feature(borders)
    ax.add_feature(ocean)
    ax.add_feature(lakes)
    ax.add_feature(cf.COASTLINE)
    

    # surf = ax.contourf(var.lon, var.lat, var, 10, transform=ccrs.PlateCarree(),cmap=cm.viridis, vmin=vmin, vmax=vmax,levels=levels)
    # cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    # cb.ax.tick_params(labelsize=f_s)
    # cb.ax.locator_params(nbins=4)
    
    for i,row in cits.iterrows():
        ax.scatter(row['lon'], row['lat'], transform=ccrs.PlateCarree(), c='r', s=20)
        ax.annotate(row['city'], xy=(row['lon']+0.25, row['lat']), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red', ha='left', va='center')
    fig.savefig(fldr_out+'{}_{}.png'.format(var.name,year), bbox_inches='tight')
    # ax.gridlines(linestyle=":")
    plt.show()
    
#%% MERRA ALL TOGETHER

fldr = 'MERRA-2'

data_all = xr.Dataset()
for year in range(2016,2021):
    print('Processing year {:0d}'.format(year))
    file_data = os.path.join(fldr,'MERRA2_Processed_{:d}.nc'.format(year))
    data_year = xr.open_dataset(file_data)
    data_year = data_year.drop(['SLP','T2MDEW', 'DNI_disc','azimuth','elevation'])
    data_all = data_all.merge(data_year)

file_out = os.path.join(fldr,'MERRA2_Processed_All.nc')
data_all.to_netcdf(file_out)
