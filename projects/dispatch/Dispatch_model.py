
import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import time
from os.path import isfile
from scipy.interpolate import interp1d
pd.set_option('display.max_columns', None)

import bdr_csp.PerformancePowerCycle as PPC

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

def get_data_location(location: int) -> tuple[float,float,str]:
    if location == 1:       #Closest location to Mount Isa (example)
        return (-20.7, 139.5, "QLD")
        
    elif location == 2:     #High-Radiation place in SA
        return (-27.0, 135.5, "SA")
    
    elif location == 3:     #High-Radiation place in VIC
        return (-34.5, 141.5, "VIC")
        
    elif location == 4:
        return (-31.0, 142.0, "NSW")
    
    elif location == 5:
        return (-25.0, 119.0, "SA")


# Defining the conditions of the plant
# Fixed values
location = 2
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.818161

CSTo,R2,SF,TOD = PPC.getting_basecase(zf,Prcv,Qavg,fzv)
Type = CSTo['Type']
Tavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
T_amb_des = CSTo['T_amb'] - 273.15
T_pb_max  = CSTo['T_pb_max'] - 273.15         #Maximum temperature in Power cycle
DNI_min = 300.
year = 2019

lat,lon,state = get_data_location(location)

# Getting the functions for hourly efficiencies

file_ae = os.path.join(
    DIR_PROJECT, 'Preliminaries','1-Grid_AltAzi_vF.csv'
)
f_eta_opt = PPC.eta_optical(file_ae, lbl='eta_SF')
eta_SF_des = f_eta_opt((lat,90.+lat,0)).item()
print(eta_SF_des)

file_PB = os.path.join(
    DIR_PROJECT, 'Preliminaries','sCO2_OD_lookuptable.csv'
)
f_eta_pb = PPC.eta_power_block(file_PB)
eta_pb_des = f_eta_pb((T_pb_max,T_amb_des)).item()
print(eta_pb_des)

f_eta_TPR = PPC.eta_TPR_0D(CSTo)
eta_rcv_des = f_eta_TPR([Tavg,T_amb_des+273.15,CSTo['Qavg']])[0]
print(eta_rcv_des)

# df = []
# qis    = [0.25,0.50,0.75,1.0,2.0,3.0]
# T_hots = np.arange(750,1151,25)
# for (T_hot,qi) in [(T_hot,qi) for T_hot in T_hots for qi in qis]:
    
#     T_turb  = T_hot - 50.    #(C)
#     T_p     = T_hot + 273.15 #(K)
#     Tamb    = T_amb + 273.15 #(K)
#     eta_th = f_eta_HPR2([T_p,Tamb,qi])[0]
#     eta_pb = f_eta_pb(T_amb,T_turb)[0]
#     eta_overall = eta_SF_des * eta_th * eta_pb
#     df.append([T_hot,qi,eta_th,eta_pb,eta_overall])
#     print(df[-1])

# df = pd.DataFrame(df,columns=['T_hot','qi','eta_th','eta_pb','eta_overall'])

# fig, ax1 = plt.subplots(figsize=(9,6))
# fs = 18
# for qi in qis:
#     df2 = df[df.qi==qi]
#     ax1.plot(df2.T_hot,df2.eta_overall,lw=3,label=r'$q_{{rcv}}={:.2f}[MW/m^2]$'.format(qi))
#     # ax1.scatter(df2.T_hot,df2.eta_pb,s=10,label=r'$q_{{rcv}}={:.2f}[MW/m^2]$'.format(qi))
# ax1.set_ylabel('Overall efficiency [-]',fontsize=fs)
# ax1.set_xlabel('Particle temperature [C]',fontsize=fs)
# ax1.tick_params(axis='both', which='major', labelsize=fs-2)
# ax1.legend(loc=0)
# ax1.grid()
# plt.show()

# sys.exit()


# SIMULATIONS FOR DIFFERENT STORAGE CAPACITY, SOLAR MULTIPLE, NUMBER OF TOWERS AND LOCATIONS

Ntower = 1
location = 2

Ntowers = [1,2,3,4]
locations = [1,2,3,4]
T_stgs = np.arange(4,16.1,2)
SMs     = np.arange(1.0,4,0.125)

# Ntowers = [1]
# # locations = [2]
# T_stgs = [8]
# SMs     = [2]

year_i = 2016
year_f = 2020

loc_prev = 0

res_tot = []
cols = ['loc', 'Ntower', 'T_stg', 'SM',
        'Prcv', 'P_el', 'Q_stg', 'CF_sf', 
        'CF_pb', 'Rev_tot', 'LCOH', 'LCOE', 
        'C_pb', 'C_C', 'RoI', 'PbP', 
        'NPV','Stg_min','date_sim']

fldr_rslt = 'Results_SM_T_stg/'

Fr_pb = 0.0
# cost_pb = 'Neises'
cost_pb = 'Sandia'

file_results = fldr_rslt+'results_{}_Fr_{:.2f}.csv'.format(cost_pb,Fr_pb)
if isfile(file_results):
    res_tot = pd.read_csv(file_results,index_col=0)
else:
    res_tot = pd.DataFrame([],columns=cols)


for (location,Ntower) in [(location,Ntower) for location in locations for Ntower in Ntowers]:
    
    (lat,lon,state) = get_data_location(location)
    
    dT = 0.5            #Time in hours
    if location != loc_prev:
        print('Charging new dataset')
        df_initial = PPC.generating_df_aus(lat,lon,state,year_i,year_f,dT)
        loc_prev = location
    
    df = df_initial.copy()
    
    data = []
    print('\t'.join('{:}'.format(x) for x in cols))
    
    for (T_stg,SM) in [(T_stg,SM) for T_stg in T_stgs for SM in SMs]:
        
        CSTo,R2,SF,TOD = PPC.getting_basecase(zf,Prcv,Qavg,fzv)
        CSTo['eta_pb']  = eta_pb_des
        CSTo['eta_rcv'] = eta_rcv_des
        
        CSTo['T_stg']  = T_stg
        CSTo['SM']     = SM
        CSTo['Ntower'] = Ntower
        CSTo['lat']    = lat
        CSTo['lng']    = lon
        
        CSTo['costs_in']['Fr_pb'] = Fr_pb
        CSTo['costs_in']['C_pb_est'] = cost_pb

        #Design power block efficiency and capacity
        Prcv   = CSTo['P_rcv']
        P_pb   = Prcv / SM          #[MW] Design value for Power from receiver to power block
        P_el   = Ntower * eta_pb_des * P_pb  #[MW] Design value for Power Block generation
        Q_stg  = P_pb * T_stg       #[MWh] Capacity of storage per tower
        CSTo['P_pb']  = P_pb
        CSTo['P_el']  = P_el
        CSTo['Q_stg'] = Q_stg
        
        # Annual performance
        args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
        aux = PPC.annual_performance(df,CSTo,SF,args)
        CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV,Stg_min = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_pb','C_C','RoI','PbP','NPV','Stg_min']]
        
        date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        data.append([location,Ntower,T_stg,SM,Prcv,P_el,Q_stg,CF_sf,CF_pb,Rev_tot,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV,Stg_min,date_sim])
        print('\t'.join('{:8.3f}'.format(x) for x in data[-1][:-1]))
        
        # sys.exit()
        res = pd.DataFrame(data,columns=cols)
    
    print(location,df["DNI"].sum()/(365*5+1)/2,df["DNI"].mean()*48)
    
    res_tot = pd.concat([res_tot,res],ignore_index=True)
    # res_tot.to_csv(file_results)
        
    #%%% PLOTTING SM-STORAGE
    #################################################
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res["T_stg"]==T_stg]
        ax1.plot(res2["SM"],res2["LCOE"],lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_ylabel('LCOE [AUD/MWh]',fontsize=fs)
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    # ax1.set_ylim(100,300)
    ax1.grid()
    
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn/1.4, mx/1.4)
    ax2.set_ylabel('LCOE [USD/MWh]',fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)
    
    ax2.plot([1.0,4.0],[95.6,95.6],lw=3,ls='-.',c='gray')
    ax2.annotate('SAM sim.',(3.5,97.0),c='gray',rotation='0',fontsize=fs-4)
    
    SP_avg = {'QLD':71.50, 'SA':85.6, 'VIC': 78.24, 'NSW': 76.29}
    
    ax1.plot([1.0,4.0],[SP_avg[state],SP_avg[state]],lw=3,ls='--',c='gray')
    ax1.annotate('SP avg.',(3.5,SP_avg[state]+1.5),c='gray',rotation='0',fontsize=fs-4)
    
    
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_LCOE.png'.format(cost_pb,Fr_pb,state,Ntower), bbox_inches='tight')
    plt.show()
    
    ########################################
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.C_C,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.set_ylabel('Capital Cost [MM AUD]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_CC.png'.format(cost_pb,Fr_pb,state,Ntower), bbox_inches='tight')
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.Rev_tot,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.set_ylabel('Annual Total Revenue [MM AUD]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_Revenue.png'.format(cost_pb,Fr_pb,state, Ntower), bbox_inches='tight')
    plt.show()
    
    # fig, ax1 = plt.subplots(figsize=(9,6))
    # fs = 18
    # for T_stg in res.T_stg.unique():
    #     res2 = res[res.T_stg==T_stg]
    #     ax1.plot(res2.SM,res2.TimeOp_sf,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    # ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    # ax1.set_ylabel('Time of Operation (Solar Field) [hrs]',fontsize=fs)
    # ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    # ax1.legend(loc=0)
    # ax1.grid()
    # fig.savefig(fldr_rslt+'{}_{}_Ntower_{:0d}_SM_TimeOp_SF.png'.format(cost_pb,state,Ntower), bbox_inches='tight')
    # plt.show()
    
    # fig, ax1 = plt.subplots(figsize=(9,6))
    # fs = 18
    # for T_stg in res.T_stg.unique():
    #     res2 = res[res.T_stg==T_stg]
    #     ax1.plot(res2.SM,res2.TimeOp_pb,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    # ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    # ax1.set_ylabel('Time of Operation (Power Block) [hrs]',fontsize=fs)
    # ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    # ax1.legend(loc=0)
    # ax1.grid()
    # fig.savefig(fldr_rslt+'{}_{}_Ntower_{:0d}_SM_TimeOp_pb.png'.format(cost_pb,state,Ntower), bbox_inches='tight')
    # plt.show()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.P_el,res2.Rev_tot,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_xlabel('Power Cycle Size [MWe]',fontsize=fs)
    ax1.set_ylabel('Annual Total Revenue [MM AUD]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.P_el,res2.Q_stg,lw=3,marker='o',label=r'$Q_{{stg}}={:.0f}$MWh'.format(T_stg))
    ax1.set_xlabel('Power Cycle Size [MWe]',fontsize=fs)
    ax1.set_ylabel('Storage Capacity [MWh]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.C_pb,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.set_ylabel('PB Specific Cost [AUD/MW]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    
    ax1.grid()
    ########################################
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.C_C,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_ylabel('Total Capital Cost [MM USD]',fontsize=fs)
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.PbP,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_ylabel('Payback Period [yr]',fontsize=fs)
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.set_ylim(0,50)
    ax1.legend(loc=0)
    ax1.grid()
    
    ax1.plot([1.0,4.0],[30.,30.],lw=3,ls='--',c='gray')
    ax1.annotate('lifetime',(3.5,28.),c='gray',rotation='0',fontsize=fs-4)
    
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_PbP.png'.format(cost_pb,Fr_pb,state,Ntower), bbox_inches='tight')
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.CF_pb,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_ylabel('Capacity Factor (electrical) [-]',fontsize=fs)
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_CFel.png'.format(cost_pb,Fr_pb,state,Ntower), bbox_inches='tight')
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    for T_stg in res.T_stg.unique():
        res2 = res[res.T_stg==T_stg]
        ax1.plot(res2.SM,res2.RoI,lw=3,marker='o',label=r'$T_{{stg}}={:.0f}$hr'.format(T_stg))
    ax1.set_ylabel('Return on Investment [-]',fontsize=fs)
    ax1.set_xlabel('Solar Multiple [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0)
    ax1.grid()
    ax1.plot([1.0,4.0],[0.,0.],lw=3,ls='--',c='gray')
    
    fig.savefig(fldr_rslt+'{}_{:.2f}_{}_Ntower_{:0d}_SM_RoI.png'.format(cost_pb,Fr_pb,state,Ntower), bbox_inches='tight')
    plt.show()
    
sys.exit()


# SIMULATIONS FOR POWER BLOCK COSTS


Ntower = 1
location = 2

Ntowers = [1,2,3,4]
locations = [1,2,3,4]
res_tot = []

cols = ['loc', 'Ntower', 'T_stg', 'SM', 'Fr_pb', 'Prcv', 'P_el', 'Q_stg', 'CF_sf', 'CF_pb', 'Rev_tot', 'LCOH', 'LCOE', 'C_pb', 'C_C', 'RoI', 'PbP', 'NPV']

year_i = 2016
year_f = 2020
fldr_rslt = 'Results_SM_T_stg/'
cost_pb = 'Sandia'
file_results = fldr_rslt+'results_Fpb_{}.csv'.format(cost_pb)
if isfile(file_results):
    res_tot = pd.read_csv(file_results,index_col=0)
else:
    res_tot = pd.DataFrame([],columns=cols)

for location in locations:
    
    if location == 1:       #Closest location to Mount Isa (example)
        lat = -20.7
        lon = 139.5
        state = 'QLD'
    elif location == 2:     #High-Radiation place in SA
        lat = -27.0
        lon = 135.5
        state = 'SA'
    
    elif location == 3:     #High-Radiation place in VIC
        lat = -34.5
        lon = 141.5
        state = 'VIC'
        
    elif location == 4:
        lat = -31.0
        lon = 142.0
        state = 'NSW'
    
    dT = 0.5            #Time in hours
    df = PPC.Generating_DataFrame(lat,lon,state,year_i,year_f,dT)
    T_stg = 8
    SM    = 1.5
    Fr_pbs = np.arange(0.0,1.01,0.1)
    data = []
    print('\t'.join('{:}'.format(x) for x in cols))
    for (Ntower,Fr_pb) in [(Ntower,Fr_pb) for Ntower in Ntowers for Fr_pb in Fr_pbs]:
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
        
        ####################################################
        # Obtaining the efficiencies in hourly basis
        args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
        aux = PPC.Annual_Performance(df,CSTo,SF,args)
        CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_pb','C_C','RoI','PbP','NPV']]
        
        data.append([location,Ntower,T_stg,SM,Fr_pb,Prcv,P_el,Q_stg,CF_sf,CF_pb,Rev_tot,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV])
        print('\t'.join('{:8.3f}'.format(x) for x in data[-1]))
    
    res = pd.DataFrame(data,columns=cols)
    res_tot = pd.concat([res_tot,res],ignore_index=True)
res_tot.to_csv(file_results)
#%%% PLOTTING CHANGE IN POWER BLOCK

fldr_rslt = 'Results_SM_T_stg/'
cost_pb = 'Sandia'
file_results = fldr_rslt+'results_Fpb_{}.csv'.format(cost_pb)
# res_tot = pd.read_csv(file_results,index_col=0)

loc=2
Ntower=1
res_tot = pd.read_csv(file_results,index_col=0)
locations = [1,2,3,4]
states = {1:'QLD',2:'SA',3:'VIC',4:'NSW'}
colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}
markers=['o','v','s','d','P','*','H']

########################################
### PAYBACK PERIOD
fig, ax1 = plt.subplots(figsize=(9,6))
fs = 18

for loc in locations:
    state=states[loc]
    res= res_tot[(res_tot['loc']==loc)&(res_tot['Ntower']==Ntower)]
    res = res.sort_values(['Fr_pb'])
    ax1.plot(res.Fr_pb,res.PbP,lw=3,marker=markers[loc],c=colors[state],markersize=10,label=state)
    C_pb0 = res.iloc[0]['C_pb']
    
    print(state)
    print(res)

ax1.set_ylabel('Payback period [yrs]',fontsize=fs)
ax1.set_xlabel('Power block cost reduction [-]',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)

f = lambda x: (1-x)*C_pb0
g = lambda x: 1-x/C_pb0
ax2 = ax1.secondary_xaxis(-0.17, functions=(f,g))
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.set_xlabel('Power block cost [USD/kWh]',fontsize=fs)

ax1.set_xlim(-.05,1.05)
ax1.set_ylim(0,50)
ax1.legend(loc=1, fontsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslt+'0_C_pb_vs_PbP_clean.png', bbox_inches='tight')

ax1.plot([0,0],[0,50],lw=3,ls='--',c='gray')
ax1.plot([g(1000.),g(1000.)],[0,50],lw=3,ls='--',c='gray')
ax1.plot([g(600.),g(600.)],[0,50],lw=3,ls='--',c='gray')
ax1.plot([1.0,1.0],[0,50],lw=3,ls='--',c='gray')
ax1.plot([-0.05,1.05],[30,30],lw=3,ls='--',c='gray')


ax1.annotate('Current est.',(0.0,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Competitive',(0.41,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Sunshot',(0.65,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('No PB Cost',(1.0,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Plant lifetime',(-.05,28),c='gray',rotation='0',fontsize=fs-4)

fig.savefig(fldr_rslt+'0_C_pb_vs_PbP_notes.png', bbox_inches='tight')
plt.show()

#########################################
### LCOE
fig, ax1 = plt.subplots(figsize=(9,6))
fs = 18

SP_avg = {'QLD':71.50, 'SA':85.6, 'VIC': 78.24, 'NSW': 76.29}

SP_avgs = []
Fpbrs = []

for loc in locations:
    state=states[loc]
    res= res_tot[(res_tot['loc']==loc)&(res_tot['Ntower']==Ntower)]
    res = res.sort_values(['Fr_pb'])
    ax1.plot(res.Fr_pb,res.LCOE,lw=3,marker=markers[loc],c=colors[state],markersize=10,label=state)
    C_pb0 = res.iloc[0]['C_pb']
    
    Fpbrs.append(interp1d(res.LCOE,res.Fr_pb,bounds_error=False,fill_value='extrapolate')(SP_avg[state]))
    

ax1.scatter(Fpbrs,SP_avg.values(),c='C3',s=100,marker='o',zorder=10)

ax1.set_ylabel('LCOE [AUD/MWh]',fontsize=fs)
ax1.set_xlabel('Power block cost reduction [-]',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)

f = lambda x: (1-x)*C_pb0
g = lambda x: 1-x/C_pb0
ax3 = ax1.secondary_xaxis(-0.17, functions=(f,g))
ax3.tick_params(axis='both', which='major', labelsize=fs-2)
ax3.set_xlabel('Power block cost [USD/kWh]',fontsize=fs)


ymin=50; ymax=160
ax1.set_xlim(-.05,1.05)
ax1.set_ylim(ymin,ymax)
ax1.grid()

ax2 = ax1.twinx()
mn, mx = ax1.get_ylim()
ax2.set_ylim(mn/1.4, mx/1.4)
ax2.set_ylabel('LCOE [USD/MWh]',fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)

ax1.legend(loc=1, fontsize=fs-2, facecolor='white', framealpha=1)

fig.savefig(fldr_rslt+'0_C_pb_vs_LCOE_clean.png', bbox_inches='tight')

ax1.plot([0.0,0.0],[ymin,ymax],lw=3,ls='--',c='gray')
ax1.plot([g(1000.),g(1000.)],[ymin,ymax],lw=3,ls='--',c='gray')
ax1.plot([g(600.),g(600.)],[ymin,ymax],lw=3,ls='--',c='gray')
ax1.plot([1.0,1.0],[ymin,ymax],lw=3,ls='--',c='gray')
ax2.plot([-0.05,0.82],[95.6,95.6],lw=3,ls='--',c='gray')

ax1.annotate('Current est.',(0.0,50),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Competitive',(0.41,50),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Sunshot',(0.65,50),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('No PB Cost',(0.95,50),c='gray',rotation='30',fontsize=fs-4)
ax2.annotate('SAM sim.',(-0.05,92),c='gray',rotation='0',fontsize=fs-4)

fig.savefig(fldr_rslt+'0_C_pb_vs_LCOE_notes.png', bbox_inches='tight')
plt.show()


##########################################
## FUNCTION OF PAYBACK PERIOD
loc=2
res_tot = pd.read_csv(file_results)
locations = [1,2,3,4]
Ntowers = [1,2,3,4]
states = {1:'QLD',2:'SA',3:'VIC',4:'NSW'}
colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}
markers=['o','v','s','d','*','H','P']

fig, ax1 = plt.subplots(figsize=(9,6))
fs = 18
for Ntower in Ntowers:
    state=states[loc]
    res= res_tot[(res_tot['loc']==loc)&(res_tot['Ntower']==Ntower)]
    res = res.sort_values(['Fr_pb'])
    ax1.plot(res.Fr_pb,res.PbP,lw=3,marker=markers[loc],c='C'+str(Ntower),markersize=10,label=Ntower)
    # ax1.plot(res.Fr_pb,res.RoI,lw=3,marker=markers[loc],c=colors[state],markersize=10,label=state)
    if Ntower==1:
        C_pb0 = res.iloc[0]['C_pb']

ax1.set_ylabel('Payback period [yrs]',fontsize=fs)
ax1.set_xlabel('Power block cost reduction [-]',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)

f = lambda x: (1-x)*C_pb0
g = lambda x: 1-x/C_pb0
ax2 = ax1.secondary_xaxis(-0.17, functions=(f,g))
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.set_xlabel('Power block cost [USD/kWh]',fontsize=fs)

ax1.set_xlim(-.05,1.05)
ax1.set_ylim(0,50)
ax1.legend(loc=1, fontsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslt+'0_C_pb_vs_PbP_Ntowers_clean.png', bbox_inches='tight')

ax1.plot([0,0],[0,50],lw=3,ls='--',c='gray')
ax1.plot([g(1000.),g(1000.)],[0,50],lw=3,ls='--',c='gray')
ax1.plot([g(600.),g(600.)],[0,50],lw=3,ls='--',c='gray')
ax1.plot([1.0,1.0],[0,50],lw=3,ls='--',c='gray')
ax1.plot([-0.05,1.05],[30,30],lw=3,ls='--',c='gray')

ax1.annotate('Current est.',(0.0,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Competitive',(0.41,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('Sunshot',(0.65,0),c='gray',rotation='30',fontsize=fs-4)
ax1.annotate('No PB Cost',(1.0,0),c='gray',rotation='30',fontsize=fs-4)

fig.savefig(fldr_rslt+'0_C_pb_vs_PbP_Ntowers_notes.png', bbox_inches='tight')
plt.show()

#%%% PLOTTING CHANGE IN N TOWERS

loc=2
Fr_pb=0.4

for var_y in ['PbP', 'LCOE']:
    
    locations = [1,2,3,4]
    states = {1:'QLD',2:'SA',3:'VIC',4:'NSW'}
    colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}
    markers=['o','v','s','d','*','H','P']
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    width = 0.2
    widths = [0., -3*width/2, -width/2, width/2, 3*width/2]
    for loc in locations:
        state=states[loc]
        res= res_tot[(res_tot['loc']==loc)&(res_tot['Fr_pb']==Fr_pb)]
        res = res.sort_values(['Fr_pb'])
        # ax1.plot(res.Ntower, res.LCOE, lw=3, marker=markers[loc], c=colors[state], markersize=10, label=state)
        rects1 = ax1.bar(res.Ntower + widths[loc], res[var_y], width, label=state,color=colors[state], alpha=0.6)
        # ax1.bar_label(rects1,labels=['QLD','SA','VIC','NSW'], padding=3)
        ax1.bar_label(rects1,fmt='%.1f', padding=3)
    
    if var_y == 'LCOE':
        
        ylabel = 'LCOE [AUD/MWh]'
        ax1.set_ylim(50,140)

        ax2 = ax1.twinx()
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(mn/1.4, mx/1.4)
        ax2.set_ylabel('LCOE [USD/MWh]',fontsize=fs)
        ax2.tick_params(axis='both', which='major', labelsize=fs)

        ax2.plot([0.55,4.4],[95.6,95.6],lw=3,ls='--',c='gray')
        ax2.annotate('SAM sim.',(4.0,97.0),c='gray',rotation='0',fontsize=fs-4)
        
    elif var_y == 'PbP':
        
        ax1.plot([0.55,4.4],[30,30],lw=3,ls='--',c='gray')
        ax1.annotate('Plant lifetime',(3.5,28),c='gray',rotation='0',fontsize=fs-4)
        
        ylabel = 'Payback period [yrs]'
        ax1.set_ylim(0,40)
        ax1.legend(loc=1,bbox_to_anchor=(1.25, 1),fontsize=fs-1)
        
    ax1.set_ylabel(ylabel,fontsize=fs)
    ax1.set_xlabel('Number of towers [-]',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.set_xticks(locations)
    
    # ax1.set_xlim(-.05,1.05)
    
    ax1.grid()
    
    fig.savefig(fldr_rslt+'0_N_tower_vs_{}.png'.format(var_y), bbox_inches='tight')
    plt.show()

#%%% FINDING THE MINIMUM NUMERICALLY

def func_min_PbP(X,*args):
    
    (CSTo, df, f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min) = args
    T_stg, SM = X
    CSTo['T_stg']    = T_stg
    CSTo['SM']       = SM
    
    Ntower = CSTo['Ntower']
    Prcv   = CSTo['P_rcv']
    P_pb   = Prcv / SM          #[MW] Design value for Power from receiver to power block
    P_el   = Ntower * eta_pb_des * P_pb  #[MW] Design value for Power Block generation
    Q_stg  = P_pb * T_stg       #[MWh] Capacity of storage per tower
    CSTo['P_pb']  = P_pb
    CSTo['P_el']  = P_el
    CSTo['Q_stg'] = Q_stg
    args2 = f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min
    aux = PPC.Annual_Performance(df,CSTo,SF,args2)
    CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_pb','C_C','RoI','PbP','NPV']]
    # print(aux)
    print('\t'.join('{:8.3f}'.format(x) for x in X)+'\t'+'\t'.join('{:8.3f}'.format(aux[x]) for x in aux.keys()))
    return PbP

year_i = 2016
year_f = 2020
Ntowers = [1,2,3,4]
locations = [2,3,1,4]
res_tot = []
cols = ['loc', 'Ntower', 'T_stg', 'SM', 'Prcv', 'P_el', 'Q_stg', 'CF_sf', 'CF_pb', 'Rev_tot', 'LCOH', 'LCOE', 'C_pb', 'C_C', 'RoI', 'PbP', 'NPV']
data = []
print('\t'.join('{:}'.format(x) for x in cols))
for (location,Ntower) in [(location,Ntower) for location in locations for Ntower in Ntowers]:
    
    CSTo,R2,SF,TOD = PPC.Getting_BaseCase(zf,Prcv,Qavg,fzv)
    
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
        
    dT = 0.5            #Time in hours
    df = PPC.Generating_DataFrame(lat,lon,state,year_i,year_f,dT)
    
    CSTo['Ntower'] = Ntower
    CSTo['lat']    = lat
    CSTo['lng']    = lon
    
    hint = (12,1.5)          #T_stg, SM
    bnds1 = (6, 24 )
    bnds2 = (1.0,5.0)
    args = (CSTo, df, f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    sol = spo.minimize(func_min_PbP, x0=hint, args=args, method='Nelder-Mead', bounds=(bnds1,bnds2),tol=1e-2, options={'xatol':1e-2,'fatol':1e-2,'maxiter':100})
    T_stg = sol.x[0]
    SM = sol.x[1]
    
    CSTo['T_stg']    = T_stg
    CSTo['SM']       = SM
    CSTo['Ntower']   = Ntower
    ####################################################
    #Design power block efficiency and capacity
    Prcv   =CSTo['P_rcv']
    P_pb   = Prcv / SM          #[MW] Design value for Power from receiver to power block
    P_el   = Ntower * eta_pb_des * P_pb  #[MW] Design value for Power Block generation
    Q_stg  = P_pb * T_stg       #[MWh] Capacity of storage per tower
    CSTo['P_pb']  = P_pb
    CSTo['P_el']  = P_el
    CSTo['Q_stg'] = Q_stg
    
    ####################################################
    # Obtaining the efficiencies in hourly basis
    args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    aux = PPC.Annual_Performance(df,CSTo,SF,args)
    CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV = [aux[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_pb','C_C','RoI','PbP','NPV']]
    
    data.append([location,Ntower,T_stg,SM,Prcv,P_el,Q_stg,CF_sf,CF_pb,Rev_tot,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV])
    print('\t'.join('{:8.3f}'.format(x) for x in data[-1]))
    df = pd.DataFrame(data,columns=cols)
# fig, ax1 = plt.subplots(figsize=(9,6))
# fs = 18
# Ndays = 10
# df2 = df[df.index.dayofyear<=Ndays]
# ax1.scatter(df2.index,df2.E_stg_cum_tot,s=10)

# # ax1.set_ylim(20,35)
# ax1.set_xlim(df2.index.min(),df2.index.max())
# ax1.set_ylabel('Time',fontsize=fs)
# ax1.set_xlabel('Energy on Storage',fontsize=fs)
# ax1.tick_params(axis='both', which='major', labelsize=fs-2)
# # ax1.legend(loc=1,bbox_to_anchor=(1.25, 0.98),fontsize=fs-2)
# ax1.grid()
# plt.show()


sys.exit()

# E:        Energy (MWh) 
# E_SF:     Solar energy from Solar Field (receiver aperture)
# E_rcv:    Thermal energy delivered from receiver to storage
# E_pb:     Thermal energy delivered to Power block to generate electricity
# E_el:     Power generation from the power block (in Energy units)
#
# L_stg:    Level of energy in Storage unit
# OM_SF:    if True, Solar field is feeding receiver and storage
# OM_PB:    if True, Power block is generating electricity
