# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 02:51:52 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
import scipy.optimize as spo
from scipy.integrate import quad
import scipy.interpolate as spi
import scipy.linalg as sla
import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from os.path import isfile
import os
import pickle
import time
import sys

from pvlib.location import Location
import pvlib.solarposition as plsp
from datetime import datetime,timedelta
import pvlib.irradiance as plir

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)
newPath = os.path.join(mainDir, '2_Optic_Analysis')
sys.path.append(newPath)
import BeamDownReceiver as BDR

newPath = os.path.join(mainDir, '5_SPR_Models')
sys.path.append(newPath)
import SolidParticleReceiver as SPR

newPath = os.path.join(mainDir, '7_Overall_Optim_Dispatch')
sys.path.append(newPath)
import PerformancePowerCycle as PPC

#Fixed values
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.818161

CSTo,R2,SF,TOD = PPC.Getting_BaseCase(zf,Prcv,Qavg,fzv)
CSTo['file_weather'] = 'Weather/Alice_Springs_Real2000_Created20130430.csv'
CSTo['lat'] = -25.
CSTo['T_pb_max'] = 875.

Etas = SF[SF['hel_in']].mean()

C = SPR.BDR_Cost(SF,CSTo)

lat = CSTo['lat']
lon = CSTo['lng']
T_pavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
T_amb_des = CSTo['T_amb']-273.15
T_pb_max = CSTo['T_pb_max']
A_SF   = CSTo['S_SF']
A_rcv  = CSTo['Arcv']
eta_rfl = CSTo['eta_rfl']
DNI_min = 400.

file_ae = 'SkyGrid/1-GridAltAzi_vF_temp.csv'
f_eta_SF = PPC.Eta_optical(file_ae,lbl='eta_SF')
eta_SF_des = f_eta_SF((lat,90.+lat,0)).item()

f_eta_cos = PPC.Eta_optical(file_ae,lbl='eta_cos')
eta_cos_des = f_eta_cos((lat,90.+lat,0)).item()

f_eta_blk = PPC.Eta_optical(file_ae,lbl='eta_blk')
eta_blk_des = f_eta_blk((lat,90.+lat,0)).item()

f_eta_att = PPC.Eta_optical(file_ae,lbl='eta_att')
eta_att_des = f_eta_att((lat,90.+lat,0)).item()

f_eta_BDR = PPC.Eta_optical(file_ae,lbl='eta_BDR')
eta_BDR_des = f_eta_BDR((lat,90.+lat,0)).item()

f_eta_TPR = PPC.Eta_TPR_0D(CSTo)
eta_rcv_des = f_eta_TPR([T_pavg,T_amb_des+273.15,CSTo['Qavg']]).item()

print(eta_SF_des, eta_cos_des,eta_blk_des, eta_att_des, eta_BDR_des, eta_rcv_des)

#%%% WITH REAL DATA
file_weather = CSTo['file_weather']
df_w = pd.read_csv(file_weather,header=1)

cols_o = ['Date (MM/DD/YYYY)', 'Time (HH:MM)', 'DNI (W/m^2)','Dry-bulb (C)','Dew-point (C)','RHum (%)', 'Wdir (degrees)', 'Wspd (m/s)']
df_w = df_w[cols_o]
cols_n = ['date','time','DNI','Tamb','Tdp','RH','Wdir','Wspd']
df_w.rename(columns={cols_o[i]:cols_n[i] for i in range(len(cols_n))},inplace=True)

tz = 'Australia/Darwin'
df_w['datetime'] = pd.to_datetime(df_w[['date', 'time']].agg(' '.join, axis=1))
df_w['datetime'] = df_w.datetime.dt.tz_localize(tz)
df_w.set_index('datetime',inplace=True)

sol_pos = Location(lat, lon).get_solarposition(df_w.index)
df_w['azi'] = sol_pos.azimuth
df_w['ele'] = sol_pos.elevation
print(df_w)

Npoints = len(df_w)
lats = lat*np.ones(Npoints)
azis = df_w.azi.to_numpy()
azis = np.where(azis>180,360-azis,azis)
eles = df_w.ele.to_numpy()

T_pavgs  = T_pavg * np.ones(Npoints)
TambsC  = df_w.Tamb.to_numpy()                  #Ambient temperature in Celcius
TambsK  = df_w.Tamb.to_numpy() + 273.15         #Ambient temperature in Kelvin
T_maxC  = (T_pb_max-273.15) * np.ones(Npoints)

df_w['eta_cos'] = f_eta_cos(np.array([lats,eles,azis]).transpose())
df_w['eta_blk'] = f_eta_blk(np.array([lats,eles,azis]).transpose())
df_w['eta_att'] = f_eta_att(np.array([lats,eles,azis]).transpose())
df_w['eta_hel'] = df_w['eta_cos']*df_w['eta_blk']*df_w['eta_att']*eta_rfl
df_w['eta_BDR'] = f_eta_BDR(np.array([lats,eles,azis]).transpose())

df_w['eta_SF']  = f_eta_SF(np.array([lats,eles,azis]).transpose())
df_w['eta_SF2'] = df_w['eta_hel']*df_w['eta_BDR']

Qavgs = df_w.DNI * df_w.eta_SF * (A_SF / A_rcv) * 1e-6
df_w['eta_rcv'] = f_eta_TPR(np.array([T_pavgs,TambsK,Qavgs]).transpose())

df_w['P_in']  = df_w.DNI * A_SF * 1e-6
df_w['P_hel'] = df_w['P_in']  * df_w['eta_hel']
df_w['P_SF']  = df_w['P_hel'] * df_w['eta_BDR']
df_w['P_rcv'] = df_w['P_SF']  * df_w['eta_rcv']

print(df_w[df_w.index.dayofyear==1])

dff = df_w.groupby(df_w.index.month).sum()
fig, ax1 = plt.subplots(figsize=(9,6))
fs = 16
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax1.bar(dff.index, dff.P_in-dff.P_hel, bottom=dff.P_hel,label='P_in',alpha=0.7,tick_label=months)
ax1.bar(dff.index, dff.P_hel-dff.P_SF,bottom=dff.P_SF, label='P_BDR',alpha=0.7)
ax1.bar(dff.index, dff.P_SF-dff.P_rcv,bottom=dff.P_rcv, label='P_SF',alpha=0.7)
ax1.bar(dff.index, dff.P_rcv,label='P_rcv',alpha=0.7)
    # ax1.scatter(df2.T_hot,df2.eta_pb,s=10,label=r'$q_{{rcv}}={:.2f}[MW/m^2]$'.format(qi))
ax1.set_ylabel('Monthly Energy [MWh]',fontsize=fs)
ax1.set_xlabel('Month [-]',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.tick_params(axis='x', which='major', rotation=45,labelsize=fs-2)
ax1.legend(loc=1)
ax1.grid(axis='y')
plt.show()

print(dff.P_rcv.sum())

#########################################
#%%% WITH RMY
file_weather = 'Weather/AUS_NT.Alice.Springs.Airport.943260_RMY.epw'
fldr_rslt = 'AnnualPerformance'
df_w = pd.read_csv(file_weather,header=7,names=np.arange(35))
cols_o = [0,1,2,3,6,7,8,9,14]
cols_n = ['year','month','day','hour','Tamb','Tdp','RH','Pr','DNI']

df_w = df_w[cols_o]
df_w.rename(columns={cols_o[i]:cols_n[i] for i in range(len(cols_n))},inplace=True)
df_w['datetime'] = pd.to_datetime(df_w[['year', 'month', 'day', 'hour']])
df_w['datetime'] = df_w['datetime'] - pd.to_timedelta(0.5, unit='h')
df_w['datetime'] = df_w.datetime.dt.tz_localize(tz)
df_w.set_index('datetime',inplace=True)

sol_pos = Location(lat, lon).get_solarposition(df_w.index)
df_w['azi'] = sol_pos.azimuth
df_w['ele'] = sol_pos.elevation
print(df_w)

Npoints = len(df_w)
lats = lat*np.ones(Npoints)
azis = df_w.azi.to_numpy()
azis = np.where(azis>180,360-azis,azis)
eles = df_w.ele.to_numpy()

T_pavgs  = T_pavg * np.ones(Npoints)
TambsC  = df_w.Tamb.to_numpy()                  #Ambient temperature in Celcius
TambsK  = df_w.Tamb.to_numpy() + 273.15         #Ambient temperature in Kelvin
T_maxC  = (T_pb_max-273.15) * np.ones(Npoints)

df_w['eta_cos'] = f_eta_cos(np.array([lats,eles,azis]).transpose())
df_w['eta_blk'] = f_eta_blk(np.array([lats,eles,azis]).transpose())
df_w['eta_att'] = f_eta_att(np.array([lats,eles,azis]).transpose())
df_w['eta_hel'] = df_w['eta_cos']*df_w['eta_blk']*df_w['eta_att']*eta_rfl
df_w['eta_BDR'] = f_eta_BDR(np.array([lats,eles,azis]).transpose())

df_w['eta_SF']  = f_eta_SF(np.array([lats,eles,azis]).transpose())
df_w['eta_SF2'] = df_w['eta_hel']*df_w['eta_BDR']

df_w['DNI_u'] = np.where(df_w.DNI>DNI_min,df_w.DNI,0.)      #DNI useful (DNI>DNI_min)

Qavgs = df_w.DNI_u * df_w.eta_SF * (A_SF / A_rcv) * 1e-6
df_w['eta_rcv'] = f_eta_TPR(np.array([T_pavgs,TambsK,Qavgs]).transpose())

df_w['P_in']  = df_w.DNI_u * A_SF * 1e-6
df_w['P_hel'] = df_w['P_in']  * df_w['eta_hel']
df_w['P_SF']  = df_w['P_hel'] * df_w['eta_BDR']
df_w['P_rcv'] = df_w['P_SF']  * df_w['eta_rcv']

df_w['P_cos'] = df_w['P_in'] * df_w['eta_cos']  #!!!

print(df_w[df_w.index.dayofyear==1])

#Annual analysis
P_rcv  = CSTo['P_rcv']

AE_in  = df_w['P_in'].sum()      #[MWh]
AE_hel = df_w['P_hel'].sum()      #[MWh]
AE_SF  = df_w['P_SF'].sum()      #[MWh]
AE_rcv = df_w['P_rcv'].sum()      #[MWh]
CF_sf  = AE_rcv / (P_rcv * len(df_w))

eta_hel  = AE_hel/AE_in
eta_BDR  = AE_SF/AE_hel
eta_rcv  = AE_rcv/AE_SF
eta_StH  = AE_rcv/AE_in


AE_cos = df_w['P_cos'].sum()      # !!!
eta_cos  = AE_cos/AE_in           # !!!

eta_StH_avg = Etas['Eta_SF']*CSTo['eta_rcv']
AE_with_avg = (eta_StH_avg * df_w.DNI_u * A_SF * 1e-6).sum()
CF_sf_avg   = AE_with_avg / (P_rcv * len(df_w))

dff = df_w.groupby(df_w.index.month).sum()

days_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
dff['Prcv/per_day'] = dff.P_rcv/days_month

################################################
#% PLOTTING
# fig, ax1 = plt.subplots(figsize=(9,6))
fig, axes = plt.subplots(1, 2,figsize=(16,6))
ax1,ax2 = [axes[0],axes[1]]
fs = 16
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax1.bar(dff.index, dff.P_in-dff.P_hel, bottom=dff.P_hel,label=r'$Q_{loss,hel}$', alpha=0.7,tick_label=months)
ax1.bar(dff.index, dff.P_hel-dff.P_SF,bottom=dff.P_SF, label=r'$Q_{loss,BDR}$',alpha=0.7)
ax1.bar(dff.index, dff.P_SF-dff.P_rcv,bottom=dff.P_rcv, label=r'$Q_{loss,rcv}$',alpha=0.7)
ax1.bar(dff.index, dff.P_rcv,label=r'$P_{rcv}$',alpha=0.7)
    # ax1.scatter(df2.T_hot,df2.eta_pb,s=10,label=r'$q_{{rcv}}={:.2f}[MW/m^2]$'.format(qi))
ax1.set_ylabel('Monthly Energy [MWh]',fontsize=fs)
ax1.set_xlabel('Month [-]',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.tick_params(axis='x', which='major', rotation=45,labelsize=fs-2)
# ax1.legend(loc=1, fontsize=fs,bbox_to_anchor=(1.22, 1))
ax1.legend(bbox_to_anchor=(0.4,-0.37), loc="lower left", borderaxespad=0, ncol=4, fontsize=fs)
ax1.grid(axis='y')
# plt.show()


# fig.savefig(os.path.join(fldr_rslt,'AnnualPerformance_abs.png'), bbox_inches='tight')

#############################################
dff = df_w.groupby(df_w.index.month).sum()
# fig, ax1 = plt.subplots(figsize=(9,6))
fs = 16
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax2.bar(dff.index, (dff.P_in-dff.P_hel)/dff.P_in, bottom=dff.P_hel/dff.P_in,label=r'$1-\eta_{hel}$', alpha=0.7,tick_label=months)
ax2.bar(dff.index, (dff.P_hel-dff.P_SF)/dff.P_in,bottom=dff.P_SF/dff.P_in, label=r'$1-\eta_{BDR}$',alpha=0.7)
ax2.bar(dff.index, (dff.P_SF-dff.P_rcv)/dff.P_in,bottom=dff.P_rcv/dff.P_in, label=r'$1-\eta_{rcv}$',alpha=0.7)
ax2.bar(dff.index, dff.P_rcv/dff.P_in,label='$\eta_{StH}$',alpha=0.7)

ax2.plot([1,12],[0.4707,0.4707],ls='--',lw=3,c='grey')

ax2.set_ylabel('Efficiency [-]',fontsize=fs)
ax2.set_xlabel('Month [-]',fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.tick_params(axis='x', which='major', rotation=45,labelsize=fs-2)
# ax2.legend(loc=1, fontsize=fs,bbox_to_anchor=(1.22, 1))
ax2.grid(axis='y')
plt.show()
# fig.savefig(os.path.join(fldr_rslt,'AnnualPerformance_Rel.png'), bbox_inches='tight')
fig.savefig(os.path.join(fldr_rslt,'AnnualPerformance_both.png'), bbox_inches='tight')

print(dff.P_rcv.sum())
print((dff.P_rcv.mean()-dff.P_rcv.min())/dff.P_rcv.mean(),(dff.P_rcv.max()-dff.P_rcv.mean())/dff.P_rcv.mean())

print((dff.P_in.mean()-dff.P_in.min())/dff.P_in.mean(),(dff.P_in.max()-dff.P_in.mean())/dff.P_in.mean())