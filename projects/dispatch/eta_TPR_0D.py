# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:06:14 2022

@author: z5158936
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:13:56 2022

@author: z5158936
"""
import pandas as pd
import numpy as np
# from scipy.optimize import fsolve, curve_fit, fmin
import scipy.sparse as sps
import scipy.optimize as spo
from scipy.integrate import quad
import scipy.interpolate as spi
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy import constants as cte

from scipy.sparse.linalg import LinearOperator, spilu

import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import os
from os.path import isfile
import pickle
import time
import sys

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)
newPath = os.path.join(mainDir, '2_Optic_Analysis')
sys.path.append(newPath)
import bdr_csp.BeamDownReceiver as BDR

newPath = os.path.join(mainDir, '5_SPR_Models')
sys.path.append(newPath)
import bdr_csp.SolidParticleReceiver as SPR


CSTi = BDR.CST_BaseCase()
zf   = 50.
Prcv = 20.
Qavg = 1.00
fzv  = 0.822273
# fzv  = 0.815064
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs

case = 'zf{:.0f}_Q_avg{:.2f}_Prcv{:.0f}'.format(zf, Qavg, Prcv)
file_BaseCase = 'Cases/'+case+'.plk'

if isfile(file_BaseCase):
# if False:
    [CSTo,R2,SF,TOD] = pickle.load(open(file_BaseCase,'rb'))
else:
    fldr_data = os.path.join(mainDir, '0_Data','MCRT_Datasets_Final')
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
    
    zf,Type,Array,rO,Cg,xrc,yrc,zrc,Npan = [CSTo[x] for x in ['zf', 'Type', 'Array', 'rO_TOD', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'N_pan']]
    TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},xrc,yrc,zrc)
    
    pickle.dump([CSTo,R2,SF,TOD],open(file_BaseCase,'wb'))


Tavg = (CSTo['T_pC']+CSTo['T_pH'])/2.
dT = 200.
CST = CSTo.copy()
Tps   = np.arange(600,2001,100, dtype='int64')
Tambs = np.arange(-5,56,5, dtype='int64') + 273.15
Qavgs   = np.arange(0.25,2.1,0.25)
eta_ths = np.zeros((len(Tps),len(Tambs),len(Qavgs)))
data = []
for (i,j,k) in [(i,j,k) for i in range(len(Tps)) for j in range(len(Tambs)) for k in range(len(Qavgs))]:
    Tp    = Tps[i]
    T_amb = Tambs[j]
    Qavg    = Qavgs[k]
    CST['T_pC']  = Tp - dT/2
    CST['T_pH']  = Tp + dT/2
    CST['T_amb'] = T_amb
    CST['Qavg'] = Qavg
    eta_rcv = SPR.Initial_eta_rcv(CST)[0]
    eta_ths[i,j,k] = eta_rcv
    data.append([Tp,T_amb,Qavg,eta_rcv])

df = pd.DataFrame(data,columns=['Tp','T_amb','Qavg','eta_rcv'])

df.to_csv('Efficiencies/eta_TPR_0D.csv')
f_eta_TPR = rgi((Tps,Tambs,Qavgs),eta_ths)  #Function to interpolate

print(f_eta_TPR([Tavg, 25+273.15, 1.00]))

#%% Efficiency plot
fldr_rslt = 'Efficiencies/'
fig, ax1 = plt.subplots(figsize=(9,6))
fs = 18
markers=['o','v','s','d','*','H','P']
i=0
T_amb = 30+273.15
for Qavg in Qavgs[1:]:
    dfq = df[(df.Qavg==Qavg)&(df.T_amb==T_amb)]
    ax1.plot(dfq.Tp,dfq.eta_rcv,lw=2.0,marker=markers[i],markersize=10,label=r'{:.2f}$[MW/m^2]$'.format(Qavg))
    i+=1
ax1.set_xlim(min(Tps),max(Tps))
ax1.set_ylim(0.6,0.9)
ax1.set_xlabel('Average Particle temperature $(K)$',fontsize=fs)
ax1.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=1,bbox_to_anchor=(1.48, 1.01),fontsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslt+'Rcvr_efficiency_chart.png', bbox_inches='tight')
plt.show()
