# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:28:27 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
# from scipy.optimize import fsolve, curve_fit, fmin
import scipy.optimize as spo
import scipy.interpolate as spi
import scipy.linalg as sla
from scipy.integrate import quad
import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from os.path import isfile
import os
import sys
import pickle

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)
newPath = os.path.join(mainDir, '2_Optic_Analysis')
sys.path.append(newPath)
newPath = os.path.join(mainDir, '5_SPR_Models')
sys.path.append(newPath)

import BeamDownReceiver as BDR
import SolidParticleReceiver as SPR
import time
import sys

pd.set_option('display.max_columns', None)

#############################################################
def BDTPR_UpdateCosts(CST_run):
    #This function is the same than BDR_costs, but the SF variable is removed
    
    zf,fzv,rmin,rmax,zmax,P_SF,eta_rc,Type,Array,rO,Cg,xrc,yrc,zrc,A_h1, S_HB = [CST_run[x] for x in ['zf', 'fzv', 'rmin', 'rmax', 'zmax', 'P_SF', 'eta_rcv', 'Type', 'Array', 'rO_TOD', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'A_h1', 'S_HB']]
    
    T_stg   = CST_run['T_stg']                   # hr storage
    SM      = CST_run['SM']                   # (-) Solar multiple (P_rcv/P_pb)
    T_pH    = CST_run['T_pH']
    T_pC    = CST_run['T_pC']
    HtD_stg = CST_run['HtD_stg']
    
    S_HB    = CST_run['S_HB']
    S_TOD   = CST_run['S_TOD']
    Arcv    = CST_run['Arcv']
    M_p     = CST_run['M_p']
    
    Ci      = CST_run['Costs_i']
    
    N_hel = int(CST_run['N_hel'])
    S_land = CST_run['S_land']
    
    #Solar field related costs
    R_ftl  = Ci['R_ftl']                   # Field to land ratio, SolarPilot
    C_land = Ci['C_land']                  # USD per m2 of land. SAM/SolarPilot (10000 USD/acre)
    C_site = Ci['C_site']                 # USD per m2 of heliostat. SAM/SolarPilot
    C_hel  = Ci['C_hel']                  # USD per m2 of heliostat. Projected by Pfal et al (2017)
    C_tpr  = Ci['C_tpr']                  # [USD/m2]
    C_tow = Ci['C_tow']             # USD fixed. SAM/SolarPilot, assumed tower 25% cheaper
    e_tow = Ci['e_tow']                 # Scaling factor for tower
    C_parts = Ci['C_parts']
    tow_type = Ci['tow_type']
    
    #Factors to apply on costs. Usually they are 1.0.
    F_HB   = Ci['F_HB']
    F_TOD  = Ci['F_TOD']
    F_rcv  = Ci['F_rcv']
    F_tow  = Ci['F_tow']
    F_stg  = Ci['F_stg']
    
    C_OM = Ci['C_OM']                    # % of capital costs.
    DR   = Ci['DR']                    # Discount rate
    Ny   = Ci['Ny']                      # Horizon project
    
    C_xtra = Ci['C_xtra']                   # Engineering and Contingency
    
    P_rcv = CST_run['P_rcv_sim'] if 'P_rcv_sim' in CST_run else CST_run['P_rcv'] #[MW] Nominal power stored by receiver
    P_pb  = P_rcv / SM              #[MWth]  Thermal energy delivered to power block
    Q_stg = P_pb * T_stg            #[MWh]  Energy storage required to meet T_stg
    
    #######################################
    Co = {}      #Everything in MM USD, unless explicitely indicated
    
    #######################################
    # CAPACITY FACTOR
    CF_sf = Ci['CF_sf']
    Co['CF_sf'] = CF_sf
    #######################################
    # LAND CALCULATIONS
    S_hel = N_hel * A_h1
    
    Co['land'] = ( C_land*S_land*R_ftl + C_site*S_hel )  / 1e6
    
    #######################################
    # MIRROR MASSES
    M_HB_fin  = CST_run['M_HB_fin']          # Fins components of HB weight
    M_HB_t    = CST_run['M_HB_tot']          # Total weight of HB
    
    M_TOD_mirr  = (S_TOD * 15.1 /1000)
    F_TOD_pipe  = 2.0                       # Extra factor due to fins (must be corrected)
    M_TOD_t     = M_TOD_mirr*(1 + F_TOD_pipe)
    
    #######################################
    # MIRROR COSTS
    F_struc_hel = 0.17
    zhel        = 4.0
    F_HB_fin    = M_HB_fin / (M_HB_t - M_HB_fin)
    F_mirr_HB   = 0.34               # Double than heliostat's
    F_struc_HB  = F_struc_hel * np.log((zmax-3.0)/0.003) / np.log((zhel-3.0)/0.003)
    F_cool_HB   = F_struc_HB * F_HB_fin
    F_oth_HB    = 0.20               # Double than heliostat's
    
    F_mirr_TOD  = 0.68              # Fourfold than heliostat's
    F_struc_TOD = F_struc_hel * np.log((zrc-3.0)/0.003) / np.log((zhel-3.0)/0.003)
    F_cool_TOD  = (8000 + 252.9*S_TOD**0.91) / (C_hel*S_TOD)  # Normalized Hall's correlation
    F_oth_TOD   = 0.20               # Double than heliostats
    
    C_HB  = C_hel * (F_mirr_HB + F_struc_HB + F_cool_HB + F_oth_HB)
    C_TOD = C_hel * (F_mirr_TOD + F_struc_TOD + F_cool_TOD + F_oth_TOD)
    
    Co['hel'] = C_hel * S_hel /1e6
    Co['HB']  = F_HB * C_HB * S_HB /1e6
    Co['TOD'] = F_TOD * C_TOD * S_TOD /1e6
    
    #######################################
    # TOWER COST
    if tow_type == 'Conv':
        Co['tow'] = C_tow * np.exp(e_tow*zmax) /1e6
    elif 'Antenna':
        M_HB_1   = M_HB_t / 4.          #Weight per column
        M_TOD_1  = M_TOD_t / 4.
        C_tow_HB = ( (123.21 + 362.6*M_HB_1) * np.exp(0.0224*zmax) / 1e6 ) * 4
        C_tow_TOD = ( (123.21 + 362.6*M_TOD_1) * np.exp(0.0224*zrc) / 1e6 ) * 4
        Co['tow'] = F_tow * (C_tow_HB + C_tow_TOD)
    
    #######################################
    # STORAGE COST
    dT_stg  = T_pH - T_pC                   # K
    Tp_avg  = (T_pH + T_pC)/2.              # K
    cp_stg  = 148*Tp_avg**0.3093
    rho_stg = 1810.
    
    M_stg = (Q_stg * 3600 * 1e6) / (cp_stg * dT_stg)            #[kg]
    V_stg = M_stg / rho_stg                                     #[m3]
    C_stg_p  = 1.1 * C_parts * M_stg  /1e6                      #[MM USD]
    
    C_stg_H = 1000. * (1 + 0.3*(1 + (T_pH - 273.15 - 600)/400.))
    C_stg_C = 1000. * (1 + 0.3*(1 + (T_pC - 273.15 - 600)/400.))
    D_stg   = ( 4. * V_stg / np.pi / HtD_stg) **(1./3.)
    H_stg   = HtD_stg*D_stg
    A_tank  = np.pi * ( D_stg**2/2. + D_stg * H_stg)
    C_stg_t = A_tank * (C_stg_H + C_stg_C) / 1e6            #[MM USD]
    
    Co['stg'] = F_stg*C_stg_p + C_stg_t      #[MM USD]
    Co['H_stg'] = H_stg
    Co['V_stg'] = V_stg
    #######################################
    # RECEIVER COST
    # Co['rcv'] = (C_rcv * Prcv) * (50/Prcv)**e_rcv /1e6
    C_tpr = 37400           #[USD/m2]
    C_rcv_tpr = C_tpr * Arcv
    hlift_CH = H_stg + 2.
    hlift_HC = H_stg + 5.
    C_lift = ( 58.37*hlift_CH + 58.37*hlift_HC/SM ) * M_p
    Co['rcv'] = F_rcv * ( C_rcv_tpr + C_lift ) /1e6
    
    #######################################
    # TOTAL HEAT COSTS
    Co['Heat'] = (Co['land'] + Co['hel'] + Co['HB'] + Co['TOD'] + Co['rcv'] + Co['tow'] + Co['stg']) * C_xtra
    Co['SCH'] = Co['Heat'] / P_rcv if P_rcv>0 else np.inf      #(USD/Wp) Specific cost of Heat (from receiver)
    
    #######################################
    # LEVELIZED COST OF HEAT
    #Levelised cost of heat (sun-to-storage)
    TPF  = (1./DR)*(1. - 1./(1.+DR)**Ny)
    P_yr = CF_sf * 8760 * P_rcv  /1e6            #[TWh_th/yr]
    Co['LCOH'] =  Co['Heat'] * (1. + C_OM*TPF) / (P_yr * TPF) if P_yr>0 else np.inf #USD/MWh delivered from receiver
    
    Co['land_prod'] = P_rcv/(S_land/1e4)            #MW/ha
    
    return Co

#%% CHARGING THE BASECASE
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.8181

CSTi = BDR.CST_BaseCase()
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs
fldr_data    = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))

CSTi['zf']    = zf
CSTi['fzv']   = fzv
CSTi['Qavg']  = Qavg
CSTi['P_rcv'] = Prcv
#Estimating eta_rcv to obtain the TOD characteristics
CSTi['type_rcvr'] = 'TPR_0D'
eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
CSTi['eta_rcv'] = eta_rcv_i
CSTi['Arcv'] = Arcv
CSTi['rO_TOD'] = rO_TOD
CSTi['type_rcvr'] = 'TPR_2D'

R2, SF, CSTo = SPR.Simulation_Coupled(CSTi)

hlst = SF[SF.hel_in].index
N_hel   = len(hlst)

Type = CSTo['Type']
rmin,rmax,S_HB,S_TOD,Arcv,lat,lng = [CSTo[x] for x in ['rmin', 'rmax', 'S_HB', 'S_TOD', 'Arcv', 'lat', 'lng']]
Type,Array,rO,Cg,zrc = [CSTo[x] for x in ['Type','Array','rO_TOD','Cg_TOD','zrc']]
P_SF, Q_max, Q_av, eta_rcv, Qstg, Mstg, t_res = [CSTo[x] for x in ['P_SF', 'Q_max', 'Q_av', 'eta_rcv', 'Q_stg', 'M_p', 't_res']]
T_p_av = CSTo['T_p'].mean()
T_p_mx = CSTo['T_p'].max()

Etas = SF[SF['hel_in']].mean()
hlst = SF[SF.hel_in].index
N_hel   = len(hlst)
eta_SF,eta_hbi,eta_BDR  = Etas['Eta_SF'], Etas['Eta_hbi'], Etas['Eta_BDR']
eta_TOD = Etas['Eta_tdi']*Etas['Eta_tdr']
eta_StH = eta_SF * eta_rcv

TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO, 'Cg':Cg},0.,0.,zrc)

fldr_rslt = 'Sensitivity_Analysis/'
file_CST = fldr_rslt+'CSTo_default.plk'
pickle.dump(CSTo,open(file_CST,'wb'))
# sys.exit()

#%% PARAMETRIC ANALYSIS

#%%% GENERATING SET TO RUN FOR PARAMETRIC

fldr_rslt = 'Sensitivity_Analysis/'
file_parametric = fldr_rslt+'Sensitivity_Analysis_Parametric.csv'

fldr_rslt = 'Sensitivity_Analysis/'

CSTo = pickle.load(open(file_CST,'rb'))
CF_sf_base = CSTo['Costs']['CF_sf']
CSTo['S_land'] = 53822.9*np.exp(0.024*zf) # Similar to Prcv
CSTi['Costs_i']['CF_sf'] = CF_sf_base
CSTo['Costs_i']['CF_sf'] = CF_sf_base
default = BDTPR_UpdateCosts(CSTo)
LCOH_base = default['LCOH']
print(default)

#Each entry has a triple (min,max,Nruns, econ? )
ranges = {
    'F_HB'  : ( 0., 2.0, 9, True ),
    'F_TOD' : ( 0., 2.0, 9, True ),
    'F_stg' : ( 0., 2.0, 9, True ),
    'F_rcv' : ( 0., 2.0, 9, True ),
    'F_tow' : ( 0., 2.0, 9, True ),
    'T_stg'  : ( 0, 16, 9, False ),
    'SM'     : ( 0.25, 4, 9, False ),

    'C_hel'  : ( 10., 190, 9, True ),
    'C_land' : ( 0, 5, 9, True ),
    'C_site' : ( 0, 32, 9, True ),
    
    'C_OM'   : ( 0, 0.05, 9, True ),
    'DR'     : ( 0.01, 0.1, 9, True ),
    'Ny'     : ( 20, 40, 9, True ),
    'C_xtra' : ( 1.0, 1.6, 9, True ),
    'CF_sf'  : ( 0.1, 0.4, 9, True ),
    }

list_vars = list(ranges.keys())
cols = list_vars + ['LCOH_base','LCOH_new','LCOH_pct','var']
runs = pd.DataFrame([],columns=cols)

#Defining a default raw
default_row = {}
for key in ranges.keys():
    
    if ranges[key][3]:
        value = CSTi['Costs_i'][key]
    else:
        value = CSTi[key]
    default_row[key] = value
print(default_row)

for key in ranges.keys():
    ini   = ranges[key][0]
    fin   = ranges[key][1]
    steps = ranges[key][2]
    new_set = np.linspace(ini,fin,steps)
    
    for new_value in new_set:
        new_row = default_row.copy()
        new_row[key] = new_value
        new_row['var'] = key
        runs = runs.append(new_row,ignore_index=True)


runs['LCOH_base'] = LCOH_base
print(runs)

#%%% RUNNING THE SYSTEM. FOR EACH PARAMETER THE SYSTEM IS RUN 9 TIMES

#######################################################
#######################################################
stime = time.time()
i=0

for index,row in runs.iterrows():

    CST_run = CSTo.copy()
    
    ####################################
    #UPDATING OPTIMISED DESIGN PARAMETERS (only needed for change in zf)
    # CST_run['zf']    = zf
    # CST_run['fzv']   = 0.78 + 0.24*np.exp(-0.040*zf)
    # CST_run['Qavg']  = 1.54 - 1.93*np.exp(-0.047*zf)
    # CST_run['P_rcv'] = 4.0*np.exp(0.029*zf)
    
    ####################################
    #UPDATING PARAMETERS DEPENDING ON OPTIMISATION
    # CST_run['S_HB']  = 132.50*np.exp(0.044*zf) # Similar to Prcv
    # CST_run['S_TOD'] = 10.40*np.exp(0.032*zf) # Similar to Prcv
    # CST_run['Arcv']  = 5.44*np.exp(0.021*zf) # Similar to Prcv
    # CST_run['N_hel'] = 986.51*np.exp(0.028*zf) # Similar to Prcv
    CST_run['S_land'] = 53822.9*np.exp(0.024*zf) # Similar to Prcv
    
    CST_run['type_weather'] = None
    
    ####################################
    #UPDATING INDEPENDENT PARAMETERS
    for lbl in list_vars:
        if ranges[lbl][3]:
            CST_run['Costs_i'][lbl] = row[lbl]
        else:
            CST_run[lbl] = row[lbl]

    #RUNNING THE NEW SIMULATION    
    CST_run['S_land'] = 53822.9*np.exp(0.024*zf)
    
    Co = BDTPR_UpdateCosts(CST_run)
    
    runs.loc[index,'LCOH_new'] = Co['LCOH']
    runs.loc[index,'LCOH_pct'] = Co['LCOH']/row['LCOH_base']

runs.to_csv(file_parametric)
print(runs)
# sys.exit()


#%%% PLOTING THE RESULTS
# file_parametric='Sensitivity_Parametric_Analysis.csv'
runs = pd.read_csv(file_parametric,index_col=0,header=0)
LCOH_range = 0.4
LCOH_base = runs['LCOH_base'].mean()
# LCOH_base = 22.9

#########################################
lbl_params1 = ['F_HB','F_TOD','F_stg','F_rcv','F_tow','SM','T_stg']
lbls = [r'$C_{HB}$',r'$C_{TOD}$',r'$C_{stg}$',r'$C_{rcv}$',r'$C_{tow}$','SM',r'$T_{stg}$']
ms=['o','v','s','d','*','H','P']

f_s = 18
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111,aspect='equal')
i=0
for label in lbl_params1:
    aux2 = runs[runs['var']==label].copy()
    aux2['var_pct'] = aux2[label] / default_row[label]
    ax1.plot(aux2.var_pct, aux2.LCOH_pct,label=lbls[i],marker=ms[i],lw=2,markersize=10)
    i+=1

ax1.set_xlabel(r'Independent parameters change $\left( \frac{P_{new}}{P_{base}} \right)$',fontsize=f_s-2)
ax1.set_ylabel(r'LCOH change $\left( \frac{LCOH_{new}}{LCOH_{base}} \right)$',fontsize=f_s-2)
ax1.set_xlim(0,2)
ax1.set_ylim(1-LCOH_range,1+LCOH_range)
ax1.legend(bbox_to_anchor=(1.2,0.99),loc=1,fontsize=f_s-4)
ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
ax1.grid()
fig.savefig(fldr_rslt+'Sensitivity_BDR.png', bbox_inches='tight')
plt.show()

###########################################
lbl_params2 = ['C_hel','C_land','C_OM','DR','Ny','C_xtra','CF_sf']
lbls = [r'$C_{hel}$',r'$C_{land}$',r'$C_{O&M}$','DR',r'$N_y$',r'$C_{EPC}$',r'$CF_{sf}$']
f_s = 18
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111,aspect='equal')
i=0
for label in lbl_params2:
    aux2 = runs[runs['var']==label].copy()
    aux2['var_pct'] = aux2[label] / default_row[label]
    ax1.plot(aux2.var_pct, aux2.LCOH_pct,label=lbls[i],marker=ms[i],lw=2,markersize=10)
    i+=1

ax1.set_xlabel(r'Independent parameters change $\left( \frac{P_{new}}{P_{base}} \right)$',fontsize=f_s-2)
ax1.set_ylabel(r'LCOH change $\left( \frac{LCOH_{new}}{LCOH_{base}} \right)$',fontsize=f_s-2)
ax1.set_xlim(0,2)
ax1.set_ylim(1-LCOH_range,1+LCOH_range)
ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
ax1.legend(bbox_to_anchor=(1.20,0.99),loc=1,fontsize=f_s-4)
ax1.grid()
fig.savefig(fldr_rslt+'Sensitivity_General.png', bbox_inches='tight')
plt.show()

sys.exit()

#%% RANDOM ACCUMULATED ANALYSIS

#%%% GENERATING SET TO RUN FOR ACCUMULATED

fldr_rslt = 'Sensitivity_Analysis/'
file_random = 'Sensitivity_Analysis_Random.csv'

CSTo = pickle.load(open(file_CST,'rb'))
CF_sf_base = CSTo['Costs']['CF_sf']
CSTo['S_land'] = 53822.9*np.exp(0.024*zf) # Similar to Prcv
CSTi['Costs_i']['CF_sf'] = CF_sf_base
CSTo['Costs_i']['CF_sf'] = CF_sf_base
default = BDTPR_UpdateCosts(CSTo)
LCOH_base = default['LCOH']
print(default)

#Each entry has a triple (min,max,Nruns, econ? )
ranges = {
    'F_HB'  : ( 0., 2.0, 9, True ),
    'F_TOD' : ( 0., 2.0, 9, True ),
    'F_stg' : ( 0., 2.0, 9, True ),
    'F_rcv' : ( 0., 2.0, 9, True ),
    'F_tow' : ( 0., 2.0, 9, True ),
    'T_stg'  : ( 0, 16, 9, False ),
    'SM'     : ( 1, 4, 9, False ),

    'C_hel'  : ( 10., 190, 9, True ),
    'C_land' : ( 0, 5, 9, True ),
    'C_site' : ( 0, 32, 9, True ),
    
    'C_OM'   : ( 0, 0.05, 9, True ),
    'DR'     : ( 0.01, 0.1, 9, True ),
    'Ny'     : ( 20, 40, 9, True ),
    'C_xtra' : ( 1.0, 1.6, 9, True ),
    'CF_sf'  : ( 0.1, 0.4, 9, True ),
    }

list_vars = list(ranges.keys())
cols = list_vars + ['LCOH_base','LCOH_new','LCOH_pct']
runs = pd.DataFrame([],columns=cols)

Nrows = 1000000
for key in ranges.keys():
    ini   = ranges[key][0]
    fin   = ranges[key][1]
    runs[key] = np.random.uniform(ini,fin,size=Nrows)

runs['LCOH_base'] = LCOH_base
print(runs)



#%%% RUNNING THE SYSTEM FOR EACH SET OF RANDOM PARAMETERS

#######################################################
#######################################################
stime = time.time()
LCOH_new = []
LCOH_pct = []

for index,row in runs.iterrows():

    CST_run = CSTo.copy()
    
    ####################################
    #UPDATING OPTIMISED DESIGN PARAMETERS (only needed for change in zf)
    # CST_run['zf']    = zf
    # CST_run['fzv']   = 0.78 + 0.24*np.exp(-0.040*zf)
    # CST_run['Qavg']  = 1.54 - 1.93*np.exp(-0.047*zf)
    # CST_run['P_rcv'] = 4.0*np.exp(0.029*zf)
    
    ####################################
    #UPDATING PARAMETERS DEPENDING ON OPTIMISATION
    # CST_run['S_HB']  = 132.50*np.exp(0.044*zf) # Similar to Prcv
    # CST_run['S_TOD'] = 10.40*np.exp(0.032*zf) # Similar to Prcv
    # CST_run['Arcv']  = 5.44*np.exp(0.021*zf) # Similar to Prcv
    # CST_run['N_hel'] = 986.51*np.exp(0.028*zf) # Similar to Prcv
    CST_run['S_land'] = 53822.9*np.exp(0.024*zf) # Similar to Prcv
    
    CST_run['type_weather'] = None
    
    ####################################
    #UPDATING INDEPENDENT PARAMETERS
    for lbl in list_vars:
        if ranges[lbl][3]:
            CST_run['Costs_i'][lbl] = row[lbl]
        else:
            CST_run[lbl] = row[lbl]

    #RUNNING THE NEW SIMULATION    
    Co = BDTPR_UpdateCosts(CST_run)
    LCOH_new.append(Co['LCOH'])
    LCOH_pct.append(Co['LCOH']/row['LCOH_base'])
    # runs.loc[index,'LCOH_new'] = Co['LCOH']
    # runs.loc[index,'LCOH_pct'] = Co['LCOH']/row['LCOH_base']

runs['LCOH_new'] = LCOH_new
runs['LCOH_pct'] = LCOH_pct

runs.to_csv(file_random)

print(runs)

#%%% PLOTTING

# runs = pd.read_csv(file_random,index_col=0,header=0)

cLCOH = 'mediumblue'
cP = 'orangered'
    
f_s = 18
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)

Y = np.arange(0,1.0,0.01)
X = []
for q in Y:
    X.append(runs.LCOH_new.quantile(q))
X = np.array(X)

ax1.plot(X, Y, lw=3, c=cLCOH,label='LCOH probability')

#Base case
idx_base = (np.abs(X - LCOH_base)).argmin()
q_base = Y[idx_base]
ax1.plot([LCOH_base,LCOH_base],[0,q_base],lw=3,ls='--',c='grey')
ax1.plot([0,LCOH_base],[q_base,q_base],lw=3,ls='--',c='grey', label='baseline')

#P90
q = 0.9
idx_q = (np.abs(Y - q)).argmin()
LCOH_90 = X[idx_q]
ax1.plot([LCOH_90,LCOH_90],[0,q],lw=3,ls='-.',c=cP)
ax1.plot([0,LCOH_90],[q,q],lw=3,ls='-.',c=cP, label='P90')

#P50
q = 0.5
idx_q = (np.abs(Y - q)).argmin()
LCOH_50 = X[idx_q]
ax1.plot([LCOH_50,LCOH_50],[0,q],lw=3,ls=':',c=cP)
ax1.plot([0,LCOH_50],[q,q],lw=3,ls=':',c=cP, label='P50')

ax1.set_xlabel(r'Levelised Cost of Heat $(USD/MWh)$',fontsize=f_s-2)
ax1.set_ylabel(r'Probability of non-exceedance (-)',fontsize=f_s-2)
ax1.set_xlim(0,80)
ax1.set_ylim(0,1)
ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
ax1.legend(bbox_to_anchor=(1.4,0.99),loc=1,fontsize=f_s-4)
ax1.grid()
fig.savefig(fldr_rslt+'Sensitivity_Random.png', bbox_inches='tight')
plt.show()