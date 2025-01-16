# -*- coding: utf-8 -*-

"""
Created on Tue Mar 15 20:55:18 2022

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
import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cf                 # import features
import os
import cantera as ct
from cartopy import config
from pvlib.location import Location
import pvlib.solarposition as plsp

from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy import stats

from numba import jit, njit
pd.set_option('display.max_columns', None)


absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)
newPath = os.path.join(mainDir, '2_Optic_Analysis')
sys.path.append(newPath)
import BeamDownReceiver as BDR

newPath = os.path.join(mainDir, '5_SPR_Models')
sys.path.append(newPath)
import SolidParticleReceiver as SPR
import PerformancePowerCycle as PPC
stime = time.time()
# sys.exit()
##################################################
#%% MAP 1: PLOTTING VARIABLES
# AUS_PROJ = 3112
# NSW_PROJ = 3308
# AUS_BOUNDS = [110.,155.,-45.,-10.]
# NSW_BOUNDS = [138.,154.,-38.,-28.]

# cits = pd.DataFrame([['Sydney',151.21, -33.86],
#           ['Melbourne',144.9631, -37.8136],
#           ['Brisbane',153.0281, -27.4678],
#           ['Perth',115.8589, -31.9522],
#           ['Adelaide',138.6011, -34.9289],
#           ['Darwin',130.8411, -12.4381],
#           ['Canberra',149.1269, -35.2931],
#           ['Hobart',147.3257, -42.8826]],
#           columns=['city','lon','lat'])

# # for var in [GHI,DNI,Tavg]:
# # fig = plt.figure(figsize=(12, 9))
# # ax = fig.add_subplot(111,projection=ccrs.epsg(AUS_PROJ))
    
# f_s = 16
# fig = plt.figure(figsize=(12, 9))
# ax = fig.add_subplot(111,projection=ccrs.epsg(AUS_PROJ))
# ax.set_extent(AUS_BOUNDS)
# # ax.set_title("{:d}-{:02d} {:}".format(year,month,var.name))
# res='50m'
# ocean = cf.NaturalEarthFeature('physical', 'ocean', res, edgecolor='face', facecolor= cf.COLORS['water'])
# lakes = cf.NaturalEarthFeature('physical', 'lakes', res, edgecolor='k', facecolor= cf.COLORS['water'], lw=0.5)
# borders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces', res, facecolor='none', edgecolor='k', lw=1.0)

# ax.add_feature(borders)
# ax.add_feature(ocean)
# ax.add_feature(lakes)
# ax.add_feature(cf.COASTLINE)

# # vmin = var.min()
# # vmax = var.max()
# # vmin = 3
# # vmax = 10
# levels=10
# colors = {None:'k', 'WA':'gold', 'SA':'purple', 'NT':'darkorange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'skyblue', 'TAS':'green'}
# df_states['color'] = df_states['state'].map(colors)
# ax.scatter(df_states.lon, df_states.lat, c=df_states.color,s=10, transform=ccrs.PlateCarree())
# # surf = ax.contourf(var.lon, var.lat, var.transpose(), 10, transform=ccrs.PlateCarree(),cmap=cm.viridis, vmin=vmin, vmax=vmax,levels=levels)
# # cb = fig.colorbar(surf, shrink=0.25, aspect=4)
# # cb.ax.tick_params(labelsize=f_s)
# # cb.ax.locator_params(nbins=4)

# for i,row in cits.iterrows():
#     ax.scatter(row['lon'], row['lat'], transform=ccrs.PlateCarree(), c='r', s=20)
#     ax.annotate(row['city'], xy=(row['lon']+0.25, row['lat']), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red', ha='left', va='center')

# plt.show()
# sys.exit()

#####################################################
#%%% DEFINING THE CONDITIONS OF THE PLANT
#####################################################

#Fixed values
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.818161

CSTo,R2,SF,TOD = PPC.Getting_BaseCase(zf,Prcv,Qavg,fzv)

Tavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
T_amb_des = CSTo['T_amb'] - 273.15
T_pb_max  = CSTo['T_pb_max'] - 273.15         #Maximum temperature in Power cycle
DNI_min = 300.

Tavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
T_stg = 6
SM    = 1.25
Ntower = 4
Fr_pb  = 0.4
cost_pb = 'Sandia'

CSTo['T_stg'] = T_stg
CSTo['SM']    = SM

#%%% DEFINING LOCATION
location = 2

if location == 1:       #Closest location to Mount Isa (example)
    lat = -20.7
    lon = 139.5
    state = 'QLD'
    year = 2019
    
elif location == 2:     #High-Radiation place in SA
    lat = -27.0
    lon = 135.5
    state = 'SA'
    year = 2019

elif location == 3:     #High-Radiation place in VIC
    lat = -34.5
    lon = 141.5
    state = 'VIC'
    year = 2019
    
elif location == 4:
    lat = -31.0
    lon = 142.0
    state = 'NSW'
    year = 2019

#####################################################
#%%% GETTING THE FUNCTIONS FOR HOURLY EFFICIENCIES
Type = CSTo['Type']
file_ae = 'Preliminaries/1-Grid_AltAzi_vF.csv'
f_eta_opt = PPC.Eta_optical(file_ae, lbl='eta_SF')
eta_SF_des = f_eta_opt((lat,90.+lat,0)).item()
print(eta_SF_des)

file_PB = 'Preliminaries/sCO2_OD_lookuptable.csv'
f_eta_pb = PPC.Eta_PowerBlock(file_PB)
# eta_pb_des = f_eta_pb(T_amb_des, T_pb_max)[0]
eta_pb_des = f_eta_pb((T_pb_max,T_amb_des)).item()
print(eta_pb_des)

f_eta_TPR = PPC.Eta_TPR_0D(CSTo)
eta_rcv_des = f_eta_TPR([Tavg,T_amb_des+273.15,CSTo['Qavg']])[0]
print(eta_rcv_des)

#####################################################
#%% DEFINING THE REGION TO ANALYZE
#####################################################

# fldr_weather = mainDir+'\\0_Data\\MERRA-2\\'
# file_weather  = 'MERRA2_Processed_{:d}.nc'.format(year)
file_weather  = os.path.join(mainDir,'0_Data','MERRA-2','MERRA2_Processed_All.nc')
data_weather = xr.open_dataset(file_weather)
df_states = pd.read_csv('Preliminaries/States_lats.csv')

fldr_rslt = 'Geography/'

lon_lim = 128.
lats = np.array(data_weather.lat)
lons = np.array(data_weather.lon)
lons = lons[lons>lon_lim]
# lats = [lat,]
# lons = [lon,]

Revs,Gens,CF_PBs,CF_SFs, LCOHs, PbPs, NPVs, ROIs = [np.empty([len(lats),len(lons)]) for i in range(8)]

for i,j in [(i,j) for i in range(len(lats)) for j in range(len(lons))]:
    lat = lats[i]
    lon = lons[j]
    
    state0 = df_states[(df_states.lat==lat)&(df_states.lon==lon)]['state'].values[0]
    state = state0
    if state0=='WA':     state = 'SA'
    if state0=='NT':     state = 'SA'
    if state0 != state0:          #These cases are not evaluated
        Revs[i,j]    = 0.
        CF_PBs[i,j]  = 0.
        CF_SFs[i,j]  = 0.
        Gens[i,j]    = 0.
        LCOHs[i,j]   = 0.
        PbPs[i,j]    = 0.
        NPVs[i,j]    = 0.
        ROIs[i,j]    = 0.
        
        print(str(state0)+'\t'+'\t'.join('{:9.3f}'.format(x) for x in [lat, lon, 0., 0., 0., 0., 0., 0., 0.,0.]))
        continue
    
    CSTo,R2,SF,TOD = PPC.Getting_BaseCase(zf,Prcv,Qavg,fzv)
    CSTo['T_stg']  = T_stg
    CSTo['SM']     = SM
    CSTo['Ntower'] = Ntower
    CSTo['lat']    = lat
    CSTo['lng']    = lon
    
    CSTo['Costs_i']['Fr_pb'] = Fr_pb
    CSTo['Costs_i']['C_pb_est'] = cost_pb
    
    ####################################################
    #Design power block efficiency and capacity
    Prcv   = CSTo['P_rcv']
    P_pb   = Prcv / SM          #[MW] Design value for Power from receiver to power block
    P_el   = Ntower * eta_pb_des * P_pb  #[MW] Design value for Power Block generation
    Q_stg  = P_pb * T_stg       #[MWh] Capacity of storage per tower
    CSTo['P_pb']  = P_pb
    CSTo['P_el']  = P_el
    CSTo['Q_stg'] = Q_stg
    
    year_i = 2016
    year_f = 2020
    dT = 0.5   #Time in hours
    # df = PPC.Generating_DataFrame(lat,lon,state,year,dT)   
    df = PPC.Generating_DataFrame_v3(lat,lon,state,year_i,year_f,dT)
    
    args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    aux = PPC.Annual_Performance(df,CSTo,SF,args)
    
    args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    aux = PPC.Annual_Performance(df,CSTo,SF,args)
    CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_C,RoI,PbP,NPV,E_dispatch = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_C','RoI','PbP','NPV','E_dispatch']]
    
    LCOHs[i,j]   = LCOH
    Revs[i,j]    = Rev_tot
    CF_PBs[i,j]  = CF_pb
    CF_SFs[i,j]  = CF_sf
    Gens[i,j]    = E_dispatch
    PbPs[i,j]    = PbP
    NPVs[i,j]    = NPV
    ROIs[i,j]    = RoI
    
    print(state+'\t'+'\t'.join('{:9.3f}'.format(x) for x in [lat, lon, Rev_tot, CF_sf, CF_pb, E_dispatch, LCOH, PbP]))
        
    #%% SAVING SIMULATION
    
    Results = data_weather.sum(dim='time').sel(lat=lats,lon=lons)
    aux_xr = xr.DataArray(Revs,coords=[Results.lat,Results.lon],name='Rev').astype('float32')
    Results = Results.assign(Rev=aux_xr)
    
    aux_xr = xr.DataArray(CF_SFs,coords=[Results.lat,Results.lon],name='CF_SF').astype('float32')
    Results = Results.assign(CF_SF=aux_xr)
    
    aux_xr = xr.DataArray(CF_PBs,coords=[Results.lat,Results.lon],name='CF_PB').astype('float32')
    Results = Results.assign(CF_PB=aux_xr)
    
    aux_xr = xr.DataArray(Gens,coords=[Results.lat,Results.lon],name='Gen').astype('float32')
    Results = Results.assign(Gen=aux_xr)
    
    LCOHs = np.where(LCOHs==0.,LCOHs.max(),LCOHs)
    aux_xr = xr.DataArray(LCOHs,coords=[Results.lat,Results.lon],name='LCOH').astype('float32')
    Results = Results.assign(LCOH=aux_xr)
    
    aux_xr = xr.DataArray(PbPs,coords=[Results.lat,Results.lon],name='PbP').astype('float32')
    Results = Results.assign(PbP=aux_xr)
    
    aux_xr = xr.DataArray(NPVs,coords=[Results.lat,Results.lon],name='NPV').astype('float32')
    Results = Results.assign(NPV=aux_xr)
    
    aux_xr = xr.DataArray(ROIs,coords=[Results.lat,Results.lon],name='ROI').astype('float32')
    Results = Results.assign(ROI=aux_xr)
    
    file_Rslts = fldr_rslt+'Results_Dispatch_Fpb_{:.2f}_Ntower_{:d}.nc'.format(Fr_pb,Ntower)
    Results.to_netcdf(file_Rslts)

Results = Results.rename({'DNI_dirint':'DNI'})
#####################################################    
#%% MAP 1: PLOTTING VARIABLES

AUS_PROJ = 3112
NSW_PROJ = 3308
AUS_BOUNDS = [110.,155.,-45.,-10.]
AUS_BOUNDS_noWA = [128., 155.,-45.,-10.]
NSW_BOUNDS = [138.,154.,-38.,-28.]

cits = pd.DataFrame([['Sydney',151.21, -33.86],
          ['Melbourne',144.9631, -37.8136],
          ['Brisbane',153.0281, -27.4678],
          #['Perth',115.8589, -31.9522],
          ['Adelaide',138.6011, -34.9289],
          ['Darwin',130.8411, -12.4381],
          ['Canberra',149.1269, -35.2931],
          ['Hobart',147.3257, -42.8826]],
          columns=['city','lon','lat'])

P_SF   = Prcv/eta_rcv_des
P_pb   = Prcv / SM
P_el   = eta_pb_des * P_pb
Rev    = Results.Rev / P_el
j=0
for var in [Results.DNI, Results.Rev, Results.Gen, Results.PbP, Results.ROI, Results.CF_PB ]:
    f_s = 16
    fig = plt.figure(figsize=(12, 9))
    # ax = fig.add_subplot(111,projection=ccrs.epsg(AUS_PROJ))
    ax = fig.add_subplot(111,projection=ccrs.PlateCarree())
    ax.set_extent(AUS_BOUNDS_noWA)
    # ax.set_title(r'{:d} - {:} '.format(year,var.name) + units[j])
    res='50m'
    ocean = cf.NaturalEarthFeature('physical', 'ocean', res, edgecolor='face', facecolor= cf.COLORS['water'])
    lakes = cf.NaturalEarthFeature('physical', 'lakes', res, edgecolor='k', facecolor= cf.COLORS['water'], lw=0.5)
    borders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces', res, facecolor='none', edgecolor='k', lw=1.0)
    
    vmin = var.min()
    vmax = var.max()
    
    if var.name =='DNI':
        var = var/365/1000/(year_f-year_i+1)
        # vmin = 1.5e6
        # vmax = 3.0e6
        vmin = 5.0
        vmax = 8.0
        cmap = cm.viridis
        units = r'[$kWh/m^2\;d$]'
        loc_lbl = 0.81
        
    if var.name =='Rev':
        vmin = 0
        vmax = 12
        cmap = cm.viridis
        units = '[MM AUD/yr]'
        loc_lbl = 0.81
        
    if var.name =='Gen':
        var = var/1000.
        vmin = 40.
        vmax = 100.0
        cmap = cm.viridis
        units = '[GWh/yr]'
        loc_lbl = 0.81
        
    if var.name =='PbP':
        vmin = 10
        vmax = 40
        var = var.where(var>0.1).fillna(vmax)
        cmap = cm.viridis_r
        units = '[yrs]'
        loc_lbl = 0.80
    
    if var.name =='ROI':
        vmin = 0.00
        vmax = 0.70
        # var = var.where(var>0.1).fillna(vmax)
        cmap = cm.viridis
        units = '[-]' 
        loc_lbl = 0.80

    if var.name =='CF_PB':
        vmin = 0.00
        vmax = 0.50
        # var = var.where(var>0.1).fillna(vmax)
        cmap = cm.viridis
        units = '[-]' 
        loc_lbl = 0.80    
    levels = np.linspace(vmin, vmax, 11)
    surf = ax.contourf(var.lon, var.lat, var, 10, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,levels=levels, extend='both')
    
    cb = fig.colorbar(surf, shrink=0.4, aspect=6)
    cb.ax.tick_params(labelsize=f_s)
    
    fig.text(loc_lbl,0.71,r'{}'.format(var.name),fontsize=f_s,horizontalalignment='center')
    fig.text(loc_lbl,0.68,r'{}'.format(units),fontsize=f_s,horizontalalignment='center')
    
    ax.add_feature(borders)
    ax.add_feature(ocean)
    ax.add_feature(lakes)
    ax.add_feature(cf.COASTLINE)

    fig.savefig(fldr_rslt+'Fpb_{:.2f}_Ntower_{:d}_{}.png'.format(Fr_pb,Ntower,var.name), bbox_inches='tight')

    if var.name =='DNI':
        loc_lats = [-20.7,-27.0,-34.5,-31.0]
        loc_lons = [139.5,135.5,141.5,142.0]
        ax.scatter(loc_lons, loc_lats, transform=ccrs.PlateCarree(), c='r',marker='*', s=150)
        fig.savefig(fldr_rslt+'DNI_locations.png', bbox_inches='tight')
    
    
    j+=1

#%% MORE PLOTS
###############################################################
Fr_pb  = 0.4
Ntower = 4
year_i = 2016
year_f = 2020
fldr_rslt = 'Geography/'
file_Rslts = fldr_rslt+'Results_Dispatch_Fpb_{:.2f}_Ntower_{:d}.nc'.format(Fr_pb,Ntower)
Results = xr.open_dataset(file_Rslts)
Results = Results.rename({'DNI_dirint':'DNI'})

df_states = pd.read_csv('Preliminaries/States_lats.csv')

dfRes = Results.to_dataframe()[['DNI', 'Rev', 'CF_SF', 'CF_PB', 'Gen', 'LCOH', 'PbP', 'NPV', 'ROI']]
dfRes = df_states.merge(dfRes,how='left',on=('lat','lon'))

# colors = {'WA':'gold', 'SA':'green', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'purple'}
colors = {'WA':'red', 'SA':'gold', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}
f_s = 16
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
for st in ['SA', 'QLD', 'VIC', 'NSW']:
    dfaux = dfRes[dfRes.state==st]
    Xaux = dfaux.DNI/365/1000/(year_f-year_i+1)
    Yaux = dfaux.Rev
    ax.scatter(Xaux,Yaux,s=20,c=colors[st])
    ax.scatter([],[],s=100,c=colors[st],label=st) #Only for bigger legend
    
    m, n, r2, p_value, std_err = stats.linregress(Xaux,Yaux)
    X_reg = np.linspace(Xaux.min()*0.95,Xaux.max()*1.05,100)
    Y_reg = m*X_reg+n
    ax.plot(X_reg,Y_reg,ls='--',lw=1,c=colors[st])
    print(st,m,n,r2)

ax.set_xlim(4.4,8.2)
ax.legend(fontsize=f_s)
ax.grid()
ax.set_xlabel('DNI [kWh/m2/day]',fontsize=f_s)
ax.set_ylabel('Annual Revenue [MM AUD]',fontsize=f_s)
ax.tick_params(axis='both', which='major', labelsize=f_s-2)

ax2 = ax.twinx()
mn, mx = ax.get_ylim()
ax2.set_ylim(mn/1.4, mx/1.4)
ax2.set_ylabel('Annual Revenue [USD]',fontsize=f_s)
ax2.tick_params(axis='both', which='major', labelsize=f_s-2)

fig.savefig(fldr_rslt+'DNI-Fpb_{:.2f}_Nt_{:d}_Rev.png'.format(Fr_pb,Ntower), bbox_inches='tight')
plt.show()

f_s = 16
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
for st in ['SA', 'QLD', 'VIC', 'NSW']:
    dfaux = dfRes[dfRes.state==st]
    Xaux = dfaux.DNI/365/1000/(year_f-year_i+1)
    Yaux = dfaux.Gen/1000
    ax.scatter(Xaux,Yaux,s=10,c=colors[st])
    ax.scatter([],[],s=100,c=colors[st],label=st) #Only for bigger legend
    
    m, n, r2, p_value, std_err = stats.linregress(Xaux,Yaux)
    X_reg = np.linspace(Xaux.min(),Xaux.max(),100)
    Y_reg = m*X_reg+n
    # ax.plot(X_reg,Y_reg,ls='-',lw=1,c=colors[st])
    print(st,m,n,r2)

ax.set_xlim(4.4,8.2)
ax.legend(fontsize=f_s)
ax.grid()
ax.set_ylabel('Annual Generation [GWh]',fontsize=f_s)
ax.set_xlabel('DNI [kWh/m2/day]',fontsize=f_s)
ax.tick_params(axis='both', which='major', labelsize=f_s-2)
fig.savefig(fldr_rslt+'DNI-Fpb_{:.2f}_Nt_{:d}_Gen.png'.format(Fr_pb,Ntower), bbox_inches='tight')
plt.show()

f_s = 16
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
for st in ['SA', 'QLD', 'VIC', 'NSW']:
    dfaux = dfRes[(dfRes.state==st)]
    Xaux = dfaux.DNI/365/1000/(year_f-year_i+1)
    Yaux = dfaux.PbP
    ax.scatter(Xaux,Yaux,s=10,c=colors[st])
    ax.scatter([],[],s=100,c=colors[st],label=st) #Only for bigger legend
    
    # m, n, r2, p_value, std_err = stats.linregress(1/Xaux,Yaux)
    # X_reg = np.linspace(Xaux.min(),Xaux.max(),100)
    # Y_reg = m/X_reg+n
    # ax.plot(X_reg,Y_reg,ls='-',lw=1,c=colors[st])
    # print(st,m,n,r2)
ax.plot([0,8.5],[30,30],lw=3,c='gray',ls='--')

ax.legend(fontsize=f_s)
ax.set_ylim(0,50)
ax.set_xlim(4.4,8.2)
ax.set_ylabel('Payback Period [yrs]',fontsize=f_s)
ax.set_xlabel('DNI [kWh/m2/day]',fontsize=f_s)
ax.tick_params(axis='both', which='major', labelsize=f_s-2)
ax.grid()
fig.savefig(fldr_rslt+'DNI-Fpb_{:.2f}_Nt_{:d}_PbP.png'.format(Fr_pb,Ntower), bbox_inches='tight')
plt.show()

f_s = 16
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
for st in ['SA', 'QLD', 'VIC', 'NSW']:
    dfaux = dfRes[dfRes.state==st]
    Xaux = dfaux.DNI/365/1000/(year_f-year_i+1)
    Yaux = dfaux.ROI
    ax.scatter(Xaux,Yaux,s=10,c=colors[st])
    ax.scatter([],[],s=100,c=colors[st],label=st) #Only for bigger legend
    
    m, n, r2, p_value, std_err = stats.linregress(Xaux,Yaux)
    X_reg = np.linspace(Xaux.min()*0.95,Xaux.max()*1.05,100)
    Y_reg = m*X_reg+n
    ax.plot(X_reg,Y_reg,ls='--',lw=1,c=colors[st])
    print(st,m,n,r2)
    
ax.plot([0,8.5],[0,0],lw=3,c='gray',ls='--')

ax.legend(fontsize=f_s)
ax.set_xlim(4.4,8.2)
ax.set_ylim(-0.5,1)
ax.set_ylabel('Return on Investment [-]',fontsize=f_s)
ax.set_xlabel('DNI [kWh/m2/day]',fontsize=f_s)
ax.tick_params(axis='both', which='major', labelsize=f_s-2)
ax.grid()
fig.savefig(fldr_rslt+'DNI-Fpb_{:.2f}_Nt_{:d}_ROI.png'.format(Fr_pb,Ntower), bbox_inches='tight')
plt.show()


# ###############################################
# ### OBTAINING BEST AND AVERAGE VALUES
# for st in ['SA', 'NSW', 'QLD', 'VIC']:
#     dfaux = dfRes[dfRes.state==st].copy()
#     dfaux['DNI_day'] = dfaux.DNI/365/1000/(year_f-year_i+1)
#     best = dfaux[dfaux.PbP==dfaux.PbP.min()]
#     print(st)
#     print(best)
#     print()


# for st in ['SA', 'NSW', 'QLD', 'VIC']:
#     dfaux = dfRes[dfRes.state==st].copy()
#     dfaux['DNI_day'] = dfaux.DNI/365/1000/(year_f-year_i+1)
#     feasible = dfaux[dfaux.DNI_day>6.0].describe()
#     print(st)
#     print(feasible)
#     print()

#%%% Plotting per state
dfRes = Results.to_dataframe()[['DNI','Rev','CF_SF','CF_PB','Gen','LCOH']]
dfRes = df_states.merge(dfRes,how='left',on=('lat','lon'))

colors = {'WA':'red', 'SA':'gold', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}

f_s = 16
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
for st in ['WA', 'SA', 'NT', 'QLD', 'VIC', 'NSW', 'TAS']:
    dfaux = dfRes[dfRes.state==st]
    ax.scatter(dfaux.DNI/365/1000.,dfaux.Rev,s=20,c=colors[st])
    ax.scatter([],[],s=100,c=colors[st],label=st)
ax.legend(fontsize=f_s)
ax.grid()
fig.savefig(fldr_rslt+'{:0d}_Rev-DNI_{:.0f}.png'.format(year,T_stg), bbox_inches='tight')
plt.show()

f_s = 16
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
for st in ['WA', 'SA', 'NT', 'QLD', 'VIC', 'NSW', 'TAS']:
    dfaux = dfRes[dfRes.state==st]
    ax.scatter(dfaux.DNI/365/1000.,dfaux.Gen,s=10,c=colors[st],label=st)
ax.legend(fontsize=f_s)
ax.grid()
plt.show()

f_s = 16
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
for st in ['WA', 'SA', 'NT', 'QLD', 'VIC', 'NSW', 'TAS']:
    dfaux = dfRes[dfRes.state==st]
    ax.scatter(dfaux.DNI/365/1000.,dfaux.LCOH,s=10,c=colors[st],label=st)
ax.legend(fontsize=f_s)
ax.grid()
plt.show()

#%% DNI and Tamb plots
# DNI = data_weather.sum(dim='time')['DNI']/365/1000. 
DNI = Results.DNI / 365/1000. #(in kWh/day)
Tamb = Results.T10M / 8760 #(in kWh/day)
for var in [DNI, Tamb]:
    f_s = 16
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111,projection=ccrs.epsg(AUS_PROJ))
    ax.set_extent(AUS_BOUNDS)
    ax.set_title("{:d} - {:} [kWh/day]".format(year,var.name))
    res='50m'
    ocean = cf.NaturalEarthFeature('physical', 'ocean', res, edgecolor='face', facecolor= cf.COLORS['water'])
    lakes = cf.NaturalEarthFeature('physical', 'lakes', res, edgecolor='k', facecolor= cf.COLORS['water'], lw=0.5)
    borders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces', res, facecolor='none', edgecolor='k', lw=0.5)
    
    ax.add_feature(borders)
    ax.add_feature(ocean)
    ax.add_feature(lakes)
    ax.add_feature(cf.COASTLINE)
    vmin = var.min()
    vmax = var.max()
    # vmin = 3
    # vmax = 50
    levels=10
    surf = ax.contourf(var.lon, var.lat, var, 10, transform=ccrs.PlateCarree(), cmap=cm.viridis, vmin=vmin, vmax=vmax,levels=levels)
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    cb.ax.locator_params(nbins=4)
    
    for i,row in cits.iterrows():
        ax.scatter(row['lon'], row['lat'], transform=ccrs.PlateCarree(), c='r', s=20)
        ax.annotate(row['city'], xy=(row['lon']+0.25, row['lat']), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), color='red', ha='left', va='center')
    
# ax.gridlines(linestyle=":")
plt.show()

sys.exit()

#%%% Getting states curves

colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}

f_s=16
fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
df_SP = pd.read_csv(fldr_SP+file_SP)
df_SP['time'] = pd.to_datetime(df_SP['time'])
df_SP.set_index('time', inplace=True)
tz = 'Australia/Brisbane'

df_SP.index = df_SP.index.tz_convert(tz)
SP_hour = df_SP.groupby(df_SP.index.hour).mean()
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

for st in ['QLD', 'SA', 'NSW', 'VIC']:
    ax.plot(SP_hour.index,SP_hour['SP_'+st],lw=3,c=colors[st],ls='-', label=st)
ax.legend(fontsize=f_s)
ax.set_ylim(45,225)
ax.set_xlabel('Hour',fontsize=f_s)
ax.set_ylabel('Avg Spot Price [AUD/MWh]',fontsize=f_s)
ax.tick_params(axis='both', which='major', labelsize=f_s-2)
ax.grid()
fig.savefig('SP_daily_{:0d}.png'.format(year), bbox_inches='tight')
plt.show()