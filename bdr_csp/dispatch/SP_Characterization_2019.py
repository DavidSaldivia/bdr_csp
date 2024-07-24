# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:17:26 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
import xarray as xr
import scipy.optimize as spo
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import time
import os
import cantera as ct
from pvlib.location import Location
import pvlib.solarposition as plsp

from scipy.interpolate import interp2d, interpn
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
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
# sys.exit()

############################################################
#%% SELECTING DATA

# for st_SP in ['QLD','SA','NSW','VIC']:

st_SP = 'SA'
year   = 2019
SP_max = 500.
fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
df_SP = pd.read_csv(fldr_SP+file_SP)
df_SP['time'] = pd.to_datetime(df_SP['time'])
df_SP.set_index('time', inplace=True)
tz = 'Australia/Brisbane'
df_SP.index = df_SP.index.tz_convert(tz)

df = df_SP[['Demand_'+st_SP,'SP_'+st_SP]].copy()
df.rename(columns={ 'Demand_'+st_SP:'Demand', 'SP_'+st_SP:'SP' },inplace=True)

df['Weekday'] = df.index.dayofweek < 5
df['Month']   = df.index.month
df['Quarter'] = df.index.to_period('Q')

df = df[df.SP < SP_max]
SP_whole   = df.groupby(df.index.hour).mean()['SP']
SP_weekday = df[df.Weekday].groupby(df[df.Weekday].index.hour).mean()['SP']
SP_weekend = df[~df.Weekday].groupby(df[~df.Weekday].index.hour).mean()['SP']

#############################################################
#%% GENERATING FOURIER SERIES FITTING
def fourier(x, *params):
    params = np.array(params).reshape(2,-1)
    a = params[0, :]
    b = params[1, :]
    
    terms = len(a)
    P = 24.
    ret = 0
    for deg in range(terms):
        ret += a[deg] * np.cos(2*(deg+1)*np.pi/P * x - b[deg])
        
    return ret
# num fourier terms

X = SP_weekday.index
Y = SP_weekday
Y = SP_weekend
Y = SP_whole

A_ini = 20
phi_ini = 6

fs=16
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)

ax1.plot(X,Y,lw=3,ls=':', marker='o', c='k', label='data',zorder=10)
for terms in range(2,7):
    a0 = Y.mean()
    p0 = [A_ini]*terms + [phi_ini]*terms
    coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
    
    Yc = a0 + fourier(X,*coefs)
    r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
    Xc = np.linspace(0,24,100)
    Yc = a0 + fourier(Xc,*coefs)
    
    print(r2,a0,coefs[:min(terms,5)])
    ax1.plot(Xc,Yc, lw=2, label='{:d} terms'.format(terms))

ax1.set_xlabel('Time (hr)',fontsize=fs)
ax1.set_ylabel('Hourly Average Spot Price [AUD/MWh]',fontsize=fs)
ax1.set_xlim(-1,24)
ax1.set_ylim(40,140)
ax1.set_xticks(np.arange(0,25,4))
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.tick_params(axis='x', which='major',labelsize=fs-2)
ax1.legend(loc=2,fontsize=fs-1)
ax1.grid()

ax2 = ax1.twinx()
mn, mx = ax1.get_ylim()
ax2.set_ylim(mn/1.4, mx/1.4)
ax2.set_ylabel('Hourly Average Spot Price [USD/MWh]',fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)

fldr_rslt = 'SpotPrice/'
fig.savefig(fldr_rslt+'General_fitting.png', bbox_inches='tight')

plt.show()
sys.exit()
#####################################
#%% FITTING PER STATE

f_s = 14
fig = plt.figure(figsize=(12,8))
fig, axes = plt.subplots(2, 2,figsize=(14,8))
axs = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
i=0

for st_SP in ['QLD','SA','NSW','VIC']:
    year   = 2019
    SP_max = 500.
    fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
    file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
    df_SP = pd.read_csv(fldr_SP+file_SP)
    df_SP['time'] = pd.to_datetime(df_SP['time'])
    df_SP.set_index('time', inplace=True)
    tz = 'Australia/Brisbane'
    df_SP.index = df_SP.index.tz_convert(tz)
    
    df = df_SP[['Demand_'+st_SP,'SP_'+st_SP]].copy()
    df.rename(columns={ 'Demand_'+st_SP:'Demand', 'SP_'+st_SP:'SP' },inplace=True)
    
    # print(st_SP)
    # print(df)
    
    df['Weekday'] = df.index.dayofweek < 5
    df['Month']   = df.index.month
    df['Quarter'] = df.index.to_period('Q')
    
    df = df[df.SP < SP_max]
    SP_whole   = df.groupby(df.index.hour).mean()['SP']
    SP_weekday = df[df.Weekday].groupby(df[df.Weekday].index.hour).mean()['SP']
    SP_weekend = df[~df.Weekday].groupby(df[~df.Weekday].index.hour).mean()['SP']
    
    X = SP_weekday.index
    Y = SP_weekday
    Y = SP_weekend
    Y = SP_whole
    
    #Fitting with only two terms
    terms = 2
    a0 = Y.mean()
    p0 = [A_ini]*terms + [phi_ini]*terms
    coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
    Yc = a0 + fourier(X,*coefs)
    r2_2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
    #Fitting with 5 terms
    terms = 5
    a0 = Y.mean()
    p0 = [A_ini]*terms + [phi_ini]*terms
    coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
    Yc = a0 + fourier(X,*coefs)
    
    r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
    Amps = np.append([a0], coefs[:terms])
    phis = np.append([0.], coefs[terms:])
    
    phis = np.where(Amps<0,phis-np.pi,phis)
    Amps = np.where(Amps<0,abs(Amps),Amps)
    
    P=24.
    Xc = np.linspace(0,24,100)
    H = [Amps[i] * np.cos(2*i*np.pi/P * Xc - phis[i]) for i in range(terms+1)]
    
    A0,A1,A2 = Amps[0],Amps[1],Amps[2]
    
    P0 = Amps[0]
    P1 = Amps[0] + (Amps[1] + Amps[2])
    P2 = Amps[0] + abs(Amps[2] - Amps[1])
    q1 = P1/P0; q2 = P2/P0
    
    phi_x = [0] + [phis[i]*P/(2*np.pi*i) for i in range(1,terms+1)]     #Phase in hours
    
    # fig = plt.figure(figsize=(9,6))
    # ax1 = fig.add_subplot(111)
    ax = axs[i]
    ax.plot(X,Y,lw=3,ls=':', marker='s', c='k', label='data',zorder=10)
    # ax1.plot(X,Yc, label=terms)
    ax.plot(Xc,H[0], ls='-', lw=2, label='0th harmonic')
    ax.plot(Xc,H[1]+H[0], ls='-',lw=2, label='1st harmonic')
    ax.plot(Xc,H[2]+H[0], ls='-',lw=2, label='2nd harmonic')
    # ax1.plot(Xc,H[3], label='3rd harmonic')
    # ax.grid()
    
    states = ['a) Queensland (QLD)', 'b) South Australia (SA)', 'c) New South Wales (NSW)', 'd) Victoria (VIC)']
    # title = '{}: P0={:.1f}, P1={:.1f}, P2={:.1f} ($R^2_2$={:.2f})'.format(st_SP, P0, P1, P2, r2_2)
    ax.set_title(states[i], fontsize=f_s)
    
    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn/1.4, mx/1.4)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)
    
    if i in [2,3]:
        ax.set_xlabel('Time (hr)',fontsize=fs)
    
    if i in [0]:
        ax.set_ylabel('Hourly Average Spot Price [AUD/MWh]',fontsize=fs, y=-0.1)
    ax.set_xlim(-1,24)
    ax.set_ylim(40,150)
    ax.set_xticks(np.arange(0,25,4))
    
    if i==1:
        ax2.set_ylabel('Hourly Average Spot Price [USD/MWh]',fontsize=fs, y=-0.1)
    
    if i==2:
        ax.legend(bbox_to_anchor=(0.25,-0.37), loc="lower left", borderaxespad=0, ncol=4, fontsize=f_s)
    
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    i+=1
    ax.grid()
    
fldr_rslt = 'SpotPrice/'
fig.savefig(fldr_rslt+'Fitting_State.png', bbox_inches='tight')
plt.show()

print(Amps[0],Amps[1],Amps[2],P1,P2,r2,q1,q2)

sys.exit()

####################################
#%% MODIFYING THE CURVE

P0_new = P0*1.0
P1_new = P1*1.1
P2_new = P2*1.0

A0_new = P0_new
A1_new = (P1_new-P2_new)/2                  #Assuming A2 > A1
A2_new = (P1_new+P2_new)/2 - P0_new

Amps_new = np.append([A0_new,A1_new,A2_new],Amps[3:])
coefs2 = np.append(Amps_new[1:],phis[1:])
Yc2 = A0_new + fourier(X,*coefs2)

fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)
ax1.plot(X,Y,label='data')
ax1.plot(X,Yc, label=terms)
ax1.plot(X,Yc2, label='higher diff')

ax1.legend(loc=0)
ax1.grid()
plt.show()


#%% REVENUE AS FUNCTION OF P0, P1, P2
#####################################################
#%%% DEFINING THE CONDITIONS OF THE PLANT
#####################################################

#Fixed values
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.818161

CSTo,R2,SF,TOD = PPC.Getting_BaseCase(zf,Prcv,Qavg,fzv)

Type = CSTo['Type']
Tavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
T_amb_des = CSTo['T_amb'] - 273.15
T_pb_max  = CSTo['T_pb_max'] - 273.15         #Maximum temperature in Power cycle
DNI_min = 300.


T_stg = 6
SM    = 1.25

Ntower = 1
Fr_pb  = 0.47


CSTo['T_stg']  = T_stg
CSTo['SM']     = SM
CSTo['Ntower'] = Ntower


#%%% DEFINING LOCATION
location = 2

if location == 1:
    lat = -20.7
    lon = 139.5
    state = 'QLD'
    year = 2019
elif location == 2:
    lat = -27.0
    lon = 132.5
    state = 'SA'
    year = 2019

elif location == 3:
    lat = -33.4
    lon = 148.1
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

#%%% RUNNING THE SIMULATIONS

st_SPs = ['QLD','SA','NSW','VIC']
Ws = [True,False]
Qs = ['2019Q1','2019Q2','2019Q3','2019Q4']
data = []
cols = ['st_SP','W','Q','P0','P1','P2','q1','q2','r2_2','CF_sf','CF_pb','Rev_tot','Rev_day','LCOH','LCOE','PbP']
print('\t'.join('{:}'.format(x) for x in cols))
for (st_SP,W,Q) in [(st_SP,W,Q) for st_SP in st_SPs for W in Ws for Q in Qs]:
    
    #Obtaining the Fourier fitting
    dT = 0.5            #Time in hours
    df = PPC.Generating_DataFrame_1year(lat,lon,st_SP,year,dT)
    df = df[df.SP < SP_max]
    df['Weekday'] = df.index.dayofweek < 5
    df['Month']   = df.index.month
    df['Quarter'] = df.index.to_period('Q')
    
    df  = df[(df.Quarter==Q)&(df.Weekday==W)]
    SP_Q = df.groupby(df.index.hour).mean()['SP']
    
    X = SP_Q.index
    Y = SP_Q
    
    terms = 2
    a0 = Y.mean()
    p0 = [A_ini]*terms + [phi_ini]*terms
    coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
    Yc = a0 + fourier(X,*coefs)
    r2_2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
    terms = 5
    a0 = Y.mean()
    p0 = [A_ini]*terms + [phi_ini]*terms
    coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
    Yc = a0 + fourier(X,*coefs)
    r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
    Amps = np.append([a0], coefs[:terms])
    phis = np.append([0.], coefs[terms:])
    
    phis = np.where(Amps<0,phis-np.pi,phis)
    Amps = np.where(Amps<0,abs(Amps),Amps)

    A0,A1,A2 = Amps[0],Amps[1],Amps[2]

    P0 = Amps[0]
    P1 = Amps[0] + (Amps[1] + Amps[2])
    P2 = Amps[0] + abs(Amps[2] - Amps[1])
    q1 = P1/P0; q2 = P2/P0
 
    ################################################################
    #Design power block efficiency and capacity
    Prcv   = CSTo['P_rcv']
    P_pb   = Prcv / SM          #[MW] Design value for Power from receiver to power block
    P_el   = eta_pb_des * P_pb  #[MW] Design value for Power Block generation
    Q_stg  = P_pb * T_stg       #[MWh] Capacity of storage for full power block
    CSTo['P_pb']  = P_pb
    CSTo['P_el']  = P_el
    CSTo['Q_stg'] = Q_stg
    CSTo['Costs_i']['Fr_pb'] = Fr_pb
    ################################################################
    
    args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    aux = PPC.Annual_Performance(df,CSTo,SF,args)
    CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE, PbP = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','PbP']]
    
    data.append([st_SP,W,Q,P0,P1,P2,q1,q2,r2_2,CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,PbP])
    text_r = '{}\t{}\t{}\t'.format(st_SP,W,Q)
    text_r = text_r + '\t'.join('{:8.3f}'.format(x) for x in data[-1][3:])
    print(text_r)

res = pd.DataFrame(data,columns=cols)

# #%%% PLOTTING
# var_y = 'Rev_day'

# title = {'P0':'- Average value',
#          'P1':'- Evening Peak',
#          'P2':'- Morning Peak'}

# fig, axs = plt.subplots(1, 3,figsize=(14,6))
# f_s = 18
# j=0
# for var_x in ['P0','P1','P2']:
    
#     ax1 = axs[j]
#     var2_n = 'W'
#     for var2 in res[var2_n].unique():
#         res2 = res[res[var2_n]==var2]
        
#         X = res2[var_x]
#         Y = res2[var_y]
#         coef = np.polyfit(X,Y,1)
#         fit = np.poly1d(coef)
        
#         m, n, r2, p_value, std_err = stats.linregress(X,Y)
#         X_reg = np.linspace(X.min(),X.max(),100)
#         Y_reg = m*X_reg+n
#         lbl2 = 'Weekday' if var2 else 'Weekend'
#         ax1.scatter(X,Y,label=lbl2)
#         ax1.plot(X_reg,Y_reg,ls=':',lw=2,label=r'$R^2=${:.2f}'.format(r2))
        
#     ax1.set_xlabel(var_x+' [AUD/MWh]',fontsize=f_s)
#     if j==0:    ax1.set_ylabel('Daily Average Revenue [k AUD/day]',fontsize=f_s)
#     ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
#     ax1.legend(loc=0)
    
#     ax1.set_title('{} = f({}) {}'.format(var_y,var_x,title[var_x]), fontsize=f_s)
#     ax1.grid()
#     j+=1
    
# plt.show()


# from sklearn import linear_model
# for W in [True,False]:
#     x = res[res.W==W][['P0','P1','P2']]
#     y = res[res.W==W]['Rev_tot']
#     regr = linear_model.LinearRegression(fit_intercept = False)
#     regr.fit(x, y)
#     n = regr.intercept_
#     m = regr.coef_
#     r2 = regr.score(x, y)
#     print(m,n,r2)

#%%% PLOTTING
fldr_rslt = 'SpotPrice/'
var_y = 'Rev_day'

title = {'P0':'Average',
         'P1':'Evening Peak',
         'P2':'Morning Peak'}

fig, axs = plt.subplots(1, 3,figsize=(14,4))
colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}
fs = 16
j=0
for var_x in ['P0','P1','P2']:
    
    ax = axs[j]
    
    X = res[var_x]
    Y = res[var_y]
    coef = np.polyfit(X,Y,1)
    fit = np.poly1d(coef)
    
    m, n, r2, p_value, std_err = stats.linregress(X,Y)
    X_reg = np.linspace(X.min(),X.max(),100)
    Y_reg = m*X_reg+n
    
    for (st_SP,W) in [(st_SP,W) for st_SP in st_SPs for W in Ws]:
        Xaux = res[(res.st_SP==st_SP)&(res.W==W)][var_x]
        Yaux = res[(res.st_SP==st_SP)&(res.W==W)][var_y]
        c = colors[st_SP]
        m = '*' if W else 'd'
        ax.scatter(Xaux,Yaux,s=70,c=c,marker=m)
    
    ax.plot(X_reg,Y_reg,ls=':',lw=3,c='gray',label=r'$R^2=${:.2f}'.format(r2))
    
    ax.set_xlabel(var_x+' [AUD/MWh]',fontsize=fs)
    if j==0:    ax.set_ylabel('Daily Average Revenue [k AUD/day]',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.set_yticks(np.arange(5,12.1,1))    
    ax.legend(loc=0)
    
    ax.set_title('{} - {}'.format(var_x,title[var_x]), fontsize=fs)
    ax.grid()
    
    j+=1

ax2 = axs[0].twinx()
for st_SP in st_SPs:
    ax2.scatter([],[],s=50,c=colors[st_SP],marker='s',label=st_SP)
ax2.legend(loc=7,ncol=4,bbox_to_anchor=(1.75,-0.3),fontsize=fs-2)
ax2.set_yticks([])
ax3 = axs[1].twinx()
ax3.scatter([],[],s=50,c='k',marker='*',label='Weekdays')
ax3.scatter([],[],s=50,c='k',marker='d',label='Weekend')
ax3.legend(loc=7,ncol=4,bbox_to_anchor=(1.75,-0.3),fontsize=fs-2)
ax3.set_yticks([])
fig.savefig(fldr_rslt+'Revenue_P0P1P2.png', bbox_inches='tight')
plt.show()


from sklearn import linear_model
x = res[['P0','P1','P2']]
y = res['Rev_tot']
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)
n = regr.intercept_
m = regr.coef_
r2 = regr.score(x, y)
print(m,n,r2)

sys.exit()
#%% COMPARING FITITNG FOR DIFFERENT PROFILES

#%%% FITTINGS PER STATE

# year_i = 2011
# year_f = 2020

# fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
# file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
# # fldr_SP  = mainDir+'\\0_Data\\SpotPrice_NEM\\'
# # file_SP  = 'NEM_TRADINGPRICE_2010-2020.CSV'
# df_SP = pd.read_csv(fldr_SP+file_SP)
# df_SP['time'] = pd.to_datetime(df_SP['time'])
# df_SP.set_index('time', inplace=True)
# tz = 'Australia/Brisbane'
# df_SP.index = df_SP.index.tz_convert(tz)

# year  = 2019
# SP_max = 500.

# f_s = 14
# fig = plt.figure(figsize=(12,8))
# fig, axes = plt.subplots(2, 2,figsize=(14,8))
# axs = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
# i=0

# for st_SP in ['QLD','SA','NSW','VIC']:
    
#     df = df_SP[['Demand_'+st_SP,'SP_'+st_SP]].copy()
#     df.rename(columns={ 'Demand_'+st_SP:'Demand', 'SP_'+st_SP:'SP' },inplace=True)
    
#     df['Weekday'] = df.index.dayofweek < 5
#     df['Month']   = df.index.month
#     df['Quarter'] = df.index.to_period('Q')
    
#     df = df[df.SP < SP_max]
#     SP_whole   = df.groupby(df.index.hour).mean()['SP']
#     SP_weekday = df[df.Weekday].groupby(df[df.Weekday].index.hour).mean()['SP']
#     SP_weekend = df[~df.Weekday].groupby(df[~df.Weekday].index.hour).mean()['SP']
    
#     X = SP_weekday.index
#     Y = SP_weekday
#     # Y = SP_weekend
#     # Y = SP_whole
    
#     terms = 2
#     a0 = Y.mean()
#     p0 = [A_ini]*terms + [phi_ini]*terms
#     coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#     Yc = a0 + fourier(X,*coefs)
#     r2_2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
#     terms = 10
#     a0 = Y.mean()
#     p0 = [A_ini]*terms + [phi_ini]*terms
#     coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#     Yc = a0 + fourier(X,*coefs)
#     r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
    
#     Amps = np.append([a0], coefs[:terms])
#     phis = np.append([0.], coefs[terms:])
    
#     phis = np.where(Amps<0,phis-np.pi,phis)
#     Amps = np.where(Amps<0,abs(Amps),Amps)
    
#     P=24.
#     H = [Amps[i] * np.cos(2*i*np.pi/P * X - phis[i]) for i in range(terms+1)]
    
#     A0,A1,A2 = Amps[0],Amps[1],Amps[2]
    
#     P0 = Amps[0]
#     P1 = Amps[0] + (Amps[1] + Amps[2])
#     P2 = Amps[0] + abs(Amps[2] - Amps[1])
#     q1 = P1/P0; q2 = P2/P0
    
#     phi_x = [0] + [phis[i]*P/(2*np.pi*i) for i in range(1,terms+1)]     #Phase in hours
    
#     ax = axs[i]
#     ax.plot(X,Y,label='data')
#     ax.plot(X,Yc, label='fit w/ {:d} terms'.format(terms))
#     ax.plot(X,H[1], label='1st harmonic')
#     ax.plot(X,H[2], label='2nd harmonic')
#     ax.plot(X,H[3], label='3rd harmonic')
    
#     title = '{}: P0={:.1f}, P1={:.1f}, P2={:.1f} ($R^2_2$={:.2f})'.format(st_SP,P0,P1,P2,r2_2)
#     ax.set_title(title, fontsize=f_s)
    
#     if i in [2,3]:  ax.set_xlabel('Time (hr)')
#     ax.set_ylabel('Spot Price (AUD/MWh)')
#     ax.set_ylim(-25,150)
#     if i==1:
#         ax.legend(bbox_to_anchor=(1.04,-0.30), loc="lower left", borderaxespad=0)
#     ax.grid()
    
#     i+=1
# plt.show()

# #%%% FITTINGS FOR WEEKENDS/WEEKDAYS

# fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
# file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
# df_SP = pd.read_csv(fldr_SP+file_SP)
# df_SP['time'] = pd.to_datetime(df_SP['time'])
# df_SP.set_index('time', inplace=True)
# tz = 'Australia/Brisbane'
# df_SP.index = df_SP.index.tz_convert(tz)

# year  = 2019
# SP_max = 500.

# f_s = 14
# fig = plt.figure(figsize=(12,8))
# fig, axes = plt.subplots(2, 2,figsize=(14,8))
# axs = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
# i=0

# for st_SP in ['QLD','SA','NSW','VIC']:
    
#     df = df_SP[['Demand_'+st_SP,'SP_'+st_SP]].copy()
#     df.rename(columns={ 'Demand_'+st_SP:'Demand', 'SP_'+st_SP:'SP' },inplace=True)
    
#     df['Weekday'] = df.index.dayofweek < 5
#     df['Month']   = df.index.month
#     df['Quarter'] = df.index.to_period('Q')
    
#     df = df[df.SP < SP_max]
#     SP_whole   = df.groupby(df.index.hour).mean()['SP']
#     SP_weekday = df[df.Weekday].groupby(df[df.Weekday].index.hour).mean()['SP']
#     SP_weekend = df[~df.Weekday].groupby(df[~df.Weekday].index.hour).mean()['SP']
    
#     ax = axs[i]
    
#     j=0
#     for W in [True,False]:
#         df2  = df[df.Weekday==W]
#         SP_W = df2.groupby(df2.index.hour).mean()['SP']
        
#         X = SP_W.index
#         Y = SP_W
        
#         terms = 2
#         a0 = Y.mean()
#         p0 = [A_ini]*terms + [phi_ini]*terms
#         coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#         Yc = a0 + fourier(X,*coefs)
#         r2_2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
        
#         terms = 10
#         a0 = Y.mean()
#         p0 = [A_ini]*terms + [phi_ini]*terms
#         coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#         Yc = a0 + fourier(X,*coefs)
#         r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
        
#         Amps = np.append([a0], coefs[:terms])
#         phis = np.append([0.], coefs[terms:])
        
#         phis = np.where(Amps<0,phis-np.pi,phis)
#         Amps = np.where(Amps<0,abs(Amps),Amps)
    
#         A0,A1,A2 = Amps[0],Amps[1],Amps[2]
    
#         P0 = Amps[0]
#         P1 = Amps[0] + (Amps[1] + Amps[2])
#         P2 = Amps[0] + abs(Amps[2] - Amps[1])
#         q1 = P1/P0; q2 = P2/P0
        
#         lbl = 'Weekday' if W else 'Weekend'
        
#         ax.plot(X,Y, c='C'+str(j))
#         ax.plot(X,Yc, c='C'+str(j), label='{} ($R^2_2$={:.2f})'.format(lbl,r2_2))
#         j+=1
    
#     title = '{}'.format(st_SP)
#     ax.set_title(title, fontsize=f_s)
    
#     if i in [2,3]:  ax.set_xlabel('Time (hr)')
#     ax.set_ylabel('Spot Price (AUD/MWh)')
#     ax.set_ylim(0,170)
#     # ax.legend(bbox_to_anchor=(1.04,-0.30), loc="lower left", borderaxespad=0)
#     ax.legend(loc=0)
#     ax.grid()
    
#     i+=1
# plt.show()
# #%%% FITTINGS PER QUARTER

# # fldr_SP = mainDir+'\\0_Data\\SpotPrices_old\\'
# # file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)

# fldr_SP = mainDir+'\\0_Data\\SpotPrice_NEM\\'
# file_SP  = 'NEM_TRADINGPRICE_2010-2020.CSV'
# df_SP = pd.read_csv(fldr_SP+file_SP)

# df_SP['time'] = pd.to_datetime(df_SP['time'])
# df_SP.set_index('time', inplace=True)
# tz = 'Australia/Brisbane'
# # df_SP.index = df_SP.index.tz_convert(tz)

# year  = 2019
# SP_max = 500.

# A_ini = 20
# phi_ini = 6

# f_s = 14
# fig = plt.figure(figsize=(12,8))
# fig, axes = plt.subplots(2, 2,figsize=(14,8))
# axs = [axes[0,0],axes[0,1],axes[1,0],axes[1,1]]
# i=0

# for st_SP in ['QLD1','SA1','NSW1','VIC1']:
    
#     # df = df_SP[['Demand_'+st_SP,'SP_'+st_SP]].copy()
#     df = df_SP[['PERIODID',st_SP]].copy()
#     # df.rename(columns={ 'Demand_'+st_SP:'Demand', 'SP_'+st_SP:'SP' },inplace=True)
#     df.rename(columns={ st_SP:'SP' },inplace=True)
    
#     df['Weekday'] = df.index.dayofweek < 5
#     df['Month']   = df.index.month
#     df['Quarter'] = df.index.to_period('Q')
    
#     df = df[df.SP < SP_max]
#     SP_whole   = df.groupby(df.index.hour).mean()['SP']
#     SP_weekday = df[df.Weekday].groupby(df[df.Weekday].index.hour).mean()['SP']
#     SP_weekend = df[~df.Weekday].groupby(df[~df.Weekday].index.hour).mean()['SP']
    
#     ax = axs[i]
    
#     Quarters = ['{:d}Q{:d}'.format(year,j) for j in [1,2,3,4]]
#     # Quarters = df.Quarter.unique()
#     j=0
#     for Q in Quarters:
#         df2  = df[(df.Quarter==Q)&(df.Weekday)]
#         # df2  = df[(df.Quarter==Q)]
#         SP_Q = df2.groupby(df2.index.hour).mean()['SP']
#         # SP_Q = df2.groupby(df2.PERIODID).mean()['SP']
        
#         X = SP_Q.index
#         Y = SP_Q
        
#         terms = 2
#         a0 = Y.mean()
#         p0 = [A_ini]*terms + [phi_ini]*terms
#         coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#         Yc = a0 + fourier(X,*coefs)
#         r2_2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
        
#         terms = 10
#         a0 = Y.mean()
#         p0 = [A_ini]*terms + [phi_ini]*terms
#         coefs, covariance = spo.curve_fit( fourier, X, Y-a0, maxfev=10000, p0=p0)
#         Yc = a0 + fourier(X,*coefs)
#         r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-np.mean(Y))**2))
        
#         Amps = np.append([a0], coefs[:terms])
#         phis = np.append([0.], coefs[terms:])
        
#         phis = np.where(Amps<0,phis-np.pi,phis)
#         Amps = np.where(Amps<0,abs(Amps),Amps)
    
#         A0,A1,A2 = Amps[0],Amps[1],Amps[2]
    
#         P0 = Amps[0]
#         P1 = Amps[0] + (Amps[1] + Amps[2])
#         P2 = Amps[0] + abs(Amps[2] - Amps[1])
#         q1 = P1/P0; q2 = P2/P0
        
    
#         ax.plot(X,Y, c='C'+str(j), label='{} ($R^2_2$={:.2f})'.format(Q,r2_2))
#         ax.plot(X,Yc, c='C'+str(j),ls=':')
#         j+=1
    
#     # lbl = 'Weekdays'
#     # lbl = 'Weekend'
#     lbl = 'All days'
#     title = '{} ({})'.format(st_SP,lbl)
#     ax.set_title(title, fontsize=f_s)
    
#     if i in [2,3]:  ax.set_xlabel('Time (hr)')
#     ax.set_ylabel('Spot Price (AUD/MWh)')
#     ax.set_ylim(-10,175)
#     # ax.legend(bbox_to_anchor=(1.04,-0.30), loc="lower left", borderaxespad=0)
#     ax.legend(loc=0)
#     ax.grid()
    
#     i+=1
# plt.show()

#%% GENERATING NORMAL DISTRIBUTION

# X = SP_weekday.index
# Y = SP_weekday
# Y = SP_weekend
# # Y = SP_whole
# fig = plt.figure(figsize=(9,6))
# ax1 = fig.add_subplot(111)
# for Y in [SP_weekday,SP_weekend,SP_whole]:
#     Yd = Y.sort_values(ascending=False)
#     ax1.plot(X,Yd,label='data')
# # ax1.plot(X,Yc, label=terms)
# ax1.legend(loc=0)
# ax1.grid()
# plt.show()

# def gaussian(x, *params):
#     A1,A2,mu2,sigma1,sigma2 = params
#     ret = A1*np.exp(-x**2/(2**0.5*sigma1)**2) + A2*np.exp(-(x-mu2)**2/(2**0.5*sigma2)**2)
#     return ret

# def gaussian2(x, *params):
#     A1,mu1,sigma1 = params
#     ret = A1*np.exp(-(x-mu1)**2/(2**0.5*sigma1)**2)
#     return ret

# Yd = SP_whole.sort_values(ascending=False, ignore_index=True)
# # p0 = [120, 60, 5, 2, 2]
# p0 = [10, 0, 2]
# coefs, covariance = spo.curve_fit( gaussian2, X, Yd, maxfev=10000, p0=p0)
# Yc = gaussian(X,*coefs)
# r2 = 1 - (np.sum((Yd - Yc)**2) / np.sum((Yd-np.mean(Yd))**2))

