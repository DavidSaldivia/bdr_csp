# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:00:27 2022

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

fldr_rslt = 'Experiments_Analysis/'

def Q_r_rcv(x,A0,std):
    return A0*np.exp(-0.5*(x/std)**2)

def E_r_arg(r,*args):
    (A0,std) = args
    return  Q_r_rcv(r,A0,std) * (2*np.pi*r)

def get_A_sigma(X,*args):
    r1,r2,Er1,Er2 = args
    A0,std = X
    # print(r1,r2,Er1,Er2,A0,std)
    f1 = quad(E_r_arg,0,r1,args=(A0,std))[0] - Er1
    f2 = quad(E_r_arg,0,r2,args=(A0,std))[0] - Er2
    
    return [f1,f2]

x0 = (3e5,0.05)

r1 = 0.044/2
Er1 = 616.25

r2 = 1.5
Er2 = 729.375

args = r1,r2,Er1,Er2
sol = spo.fsolve(get_A_sigma,x0,args=args)
print(sol)
A0, sigma = sol

###############################
DNI = 870.
A_fr = 1.012
tau_fr = 0.83
E_in =  DNI*A_fr*tau_fr

r1  = 0.044/2
Er1 = 616.25

eta_int1 = Er1 / E_in
sigma = r1 / (-2*np.log(1-eta_int1))**0.5
A0    = E_in / (2*np.pi*sigma**2)

Er1/(np.pi*r1**2)


r2 = 0.021/2
Er2 = quad(E_r_arg,0,r2,args=(A0,sigma))[0]
Er2 / (np.pi*r2**2)

A0 = A0*1e-6
sigma = 11.22
rr = np.arange(0.,50.,0.5)

QQ = A0*np.exp(-0.5*(rr**2)/sigma**2)
# EE = np.array([quad(E_r_arg,0,ri,args=(A0,sigma))[0] for ri in rr]) / (DNI*A_fr*tau_fr)
# QQ_sp = (DNI*A_fr*tau_fr) * np.array([EE[i]/(np.pi*rr[i]**2) for i in range(1,len(rr))])

EE = np.array([quad(E_r_arg,0,ri,args=(A0,sigma))[0] for ri in rr])
QQ_sp = np.array([EE[i]/(np.pi*rr[i]**2) for i in range(1,len(rr))])

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
fs = 16

ax1.plot(rr,QQ,c='C0',lw=2,label=r'$Q_{in}[MW/m^2]$')
ax1.plot([1000*r1,1000*r1],[0,1],ls=':',lw=2,c='gray')
ax2.plot(rr,EE,c='C1',lw=2,label=r'$E_{in}[W]$')
ax1.plot(rr[1:],QQ_sp,c='C0',ls='--',lw=2,label=r'$\overline{Q}_{rcv}=\frac{E_{in}}{A_{rcv}}$')
ax1.plot([],[],c='C1',lw=2,label=r'$E_{in}[W]$')

ax1.set_xlabel(r'Radius, $r_{rcv} [mm]$',fontsize=fs)
ax1.set_ylabel(r'Energy Density, $Q_{in}\; [MW/m^2]$',fontsize=fs)
ax2.set_ylabel(r'Total Energy, $E_{in}\; [W]$',fontsize=fs)

ax2.annotate(r'$E_{in}$',(24,600),c='black',ha='left', fontsize=fs-2)
ax2.arrow(24,600,-1.7,14,color='black')#,width=2.,head_length=0.5,head_width=0.4)

ax1.annotate(r'$Q_{rcv}$',(24,0.42),c='black',ha='left', fontsize=fs-2)
ax1.arrow(24,0.42,-2.0,-0.02,color='black')
# ,
ax1.legend(loc=2,fontsize=fs)
ax1.tick_params(axis='y', colors='C0')
ax2.tick_params(axis='y', colors='C1')
ax1.yaxis.label.set_color('C0')
ax2.yaxis.label.set_color('C1')
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=1,fontsize=fs)
ax1.grid()
plt.show()
fig.savefig(fldr_rslt+'Gaussian_Shape.png', bbox_inches='tight')