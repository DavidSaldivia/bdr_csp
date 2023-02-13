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
import BeamDownReceiver as BDR

newPath = os.path.join(mainDir, '5_SPR_Models')
sys.path.append(newPath)
import SolidParticleReceiver as SPR

# def Get_f_eta():
#     air  = ct.Solution('air.xml')
#     data=[]
#     for (Tp,qi) in [(Tp,qi) for Tp in np.arange(700.,2001,100.) for qi in np.arange(0.25,4.1,0.5)]:
#         eta_th = SPR.HTM_0D_blackbox(Tp, qi, air=air)[0]
#         data.append([Tp,qi,eta_th])
#     data = np.array(data)
#     return spi.interp2d(data[:,0],data[:,1],data[:,2])     #Function to interpolate
# f_eta1 = Get_f_eta()

CSTi = BDR.CST_BaseCase()
air  = ct.Solution('air.xml')

Tps = np.arange(700.,2001.,100.)
qis = [0.25,0.50,1.0,2.0,4.0,0.72]
data = []
Fc = 2.57
for (Tp,qi) in [(Tp,qi) for Tp in Tps for qi in qis]:
    
    eta_th,h_rad,h_conv = SPR.HTM_0D_blackbox(Tp, qi, Fc=Fc, air=air)
    data.append([Tp,qi,eta_th,h_rad,h_conv])
    print(data[-1])
df = pd.DataFrame(data,columns=['Tp','qi','eta','h_rad','h_conv'])

#%% Efficiency plot
fldr_rslt = 'Results_HPR_0D/'
fig, ax1 = plt.subplots(figsize=(9,6))
fs = 18
markers=['o','v','s','d','*','H','P']
i=0
for qi in qis:
    dfq = df[df.qi==qi]
    if i<5:
        ax1.plot(dfq.Tp,dfq.eta,lw=2.0,marker=markers[i],markersize=10,label=r'{:.2f}$[MW/m^2]$'.format(qi))
    else:
        ax1.plot(dfq.Tp,dfq.eta,lw=2.0,ls='--',c='gray',marker=markers[i],markersize=10,label=r'{:.2f}$[MW/m^2](exp)$'.format(qi))
    i+=1
ax1.set_xlim(min(Tps),max(Tps))
ax1.set_ylim(0,1)
ax1.set_xlabel('Average Particle temperature $(K)$',fontsize=fs)
ax1.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=1,bbox_to_anchor=(1.48, 1.01),fontsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslt+'Efficiency_Chart.png', bbox_inches='tight')
plt.show()

#%% Radiative fraction plot

air  = ct.Solution('air.xml')
Tps = np.arange(700.,2001.,100.)
Fcs = [1.0, 2.0, 2.57, 5.0, 10.]
markers=['o','s','v','d','*','H','P']
cs = ['C0','C1','C2','C3','C4','C5']
qi  = 1.0
data = []
for (Tp,Fc) in [(Tp,Fc) for Tp in Tps for Fc in Fcs]:
    
    eta_th,h_rad,h_conv = SPR.HTM_0D_blackbox(Tp, qi, Fc=Fc, air=air)
    data.append([Tp,Fc,eta_th,h_rad,h_conv])
    print(data[-1])
df = pd.DataFrame(data,columns=['Tp','Fc','eta','h_rad','h_conv'])


fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121)
fs = 18
markers=['o','s','v','d','*','H','P']
i=0
for Fc in Fcs:
    dfq = df[df.Fc==Fc]
    rad_frac = dfq.h_rad / (dfq.h_rad+dfq.h_conv)
    ax1.plot(dfq.Tp,rad_frac,c=cs[i],lw=2.0,marker=markers[i],markersize=10,label=r'$F_C={:.2f}$'.format(Fc))
    i+=1
ax1.set_xlim(min(Tps),max(Tps))
ax1.set_ylim(0,1)
ax1.set_xlabel('Particle temperature $(K)$',fontsize=fs)
ax1.set_ylabel(r'Fraction of radiative losses $(-)$',fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
# ax1.legend(loc=1,bbox_to_anchor=(1.25, 1.01),fontsize=fs-2)
ax1.grid()
# fig.savefig(fldr_rslt+'FC_vs_Rad_losses.png', bbox_inches='tight')
# plt.show()

ax2 = fig.add_subplot(122)
fs = 18
markers=['o','s','v','d','*','H','P']
i=0
for Fc in Fcs:
    dfq = df[df.Fc==Fc]
    rad_frac = dfq.h_rad / (dfq.h_rad+dfq.h_conv)
    ax2.plot(dfq.Tp,dfq.eta,c=cs[i],lw=2.0,marker=markers[i],markersize=10,label=r'$F_C={:.1f}$'.format(Fc))
    i+=1
ax2.set_xlim(min(Tps),max(Tps))
ax2.set_ylim(0,1)
ax2.set_xlabel('Particle temperature $(K)$',fontsize=fs)
ax2.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
# ax2.legend(loc=3,bbox_to_anchor=(1.01, 1.01),fontsize=fs-2)
ax1.legend(loc=4,fontsize=fs-2)
ax2.grid()
fig.savefig(fldr_rslt+'FC_vs_Efficiency.png', bbox_inches='tight')
plt.show()


