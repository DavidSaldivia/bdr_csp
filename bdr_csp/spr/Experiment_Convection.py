# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:56:00 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
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
pd.set_option('display.max_columns', None)

#########################################
Drcv = 0.041       #[m] Diameter of receiver
Arcv = np.pi*Drcv**2/4
prcv = np.pi*Drcv
z1, z2, z3, Hrcv = 0.004,0.014,0.021,0.024

A_mantle = np.pi*Hrcv*Drcv + np.pi*Drcv**2/2
dz1, dz2, dz3 = (z1+z2)/2, (z3-z1)/2, Hrcv - (z2+z3)/2
V1, V2, V3 = Arcv*dz1, Arcv*dz2, Arcv*dz3

dzg = 0.004

L    = Drcv
L    = Arcv/prcv
# L    = 1.5       #[m] Diameter of receiver

#########################################
file_data = 'Experiments_CARBO/Results_Day1.xlsx'
df0 = pd.read_excel(file_data,header=0,index_col=0)
df0 = pd.read_excel(file_data,header=0)

df0['T1'] = df0.Bottom + 273.15
df0['T2'] = df0.Middle + 273.15
df0['T3'] = df0.Top + 273.15
df0['Tamb'] = df0.Ambient + 273.15
df0['Tavg'] = ( (df0.T1*(z2-z1) + df0.T2*(z3-z1) + df0.T3*(z3-z2) )/2. ) / (z3-z1)

# df0['Ts'] = df0.T3 * 1.25       #Initial approximation

df0['Ts'] = (Hrcv-z3)*(df0.T3-df0.T2)/(z3-z2) + df0.T3

# PLOT FOR THE WHOLE PROCESS
fs=16
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)
ax1.plot(df0.time,df0.T3,lw=3,ls='-.',c='C2',label='Top')
ax1.plot(df0.time,df0.T2,lw=3,ls='--',c='C1',label='Middle')
ax1.plot(df0.time,df0.T1,lw=3,ls=':',c='C0',label='Bottom')
ax1.plot([500,500],[0,1700],ls=':',lw=2,c='gray')
ax1.annotate('on-sun\nheating',(490,70),c='gray',ha='right', fontsize=fs-2)
ax1.annotate('cooling\nprocess',(510,70),c='gray',ha='left', fontsize=fs-2)
ax1.arrow(490,225,-50,0,width=4,head_length=10,head_width=25,color='gray')
ax1.arrow(510,225,50,0,width=4,head_length=10,head_width=25,color='gray')
ax1.set_xlim(0,1000)
ax1.set_ylim(0,1700)
ax1.set_xlabel('Time (s)', fontsize=fs)
ax1.set_ylabel('Temperature (K)', fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=0, fontsize=fs-2)
ax1.grid()
fig.savefig(fldr_rslt+'Temperatures_Plot.png', bbox_inches='tight')

#%% DISCHARGING PROCESS

t_ini = 520.
t_fin = 1000.

df = df0[['time','T1','T2','T3','Tamb','Ts','Tavg']]

dfc1 = df[(df.time>t_ini)&(df.time<t_fin)].copy()         #Section with only heat losses (convective mostly)
dfc = dfc1.groupby(dfc1.time // 5).mean()
dfc = dfc1.copy()
air  = ct.Solution('air.xml')

Nroll = 2

dfc['dt'] = df.time.diff(1)

dfc['dt'] = dfc.dt.rolling(Nroll).mean()
dfc['T1r'] = dfc['T1'].rolling(Nroll).mean()
dfc['T2r'] = dfc['T2'].rolling(Nroll).mean()
dfc['T3r'] = dfc['T3'].rolling(Nroll).mean()
dfc['dT1r'] = dfc.T1r.diff(1)
dfc['dT2r'] = dfc.T2r.diff(1)
dfc['dT3r'] = dfc.T3r.diff(1)


dfc['dT1'] = dfc.T1.diff(1)
dfc['dT2'] = dfc.T2.diff(1)
dfc['dT3'] = dfc.T3.diff(1)


dfc['Pr'] = np.nan
dfc['Ra'] = np.nan

dfc = dfc[dfc.dT1r.notna()]

for row in dfc.itertuples():
    
    dt   = row.dt
    Ts   = row.Ts
    Tamb = row.Tamb
    Tsky = Tamb - 8
    T_av = (Ts+Tamb)/2.
    
    air.TP = T_av, ct.one_atm
    mu_a, k_a, rho_a, cp_a = air.viscosity, air.thermal_conductivity, air.density_mass, air.cp_mass
    alpha_a = k_a/(rho_a*cp_a)
    beta = 1./T_av
    visc_a = mu_a/rho_a
    Pr = visc_a/alpha_a
    g = 9.81
    Ra = g * beta * abs(Ts - Tamb) * L**3 * Pr / visc_a**2
    
    #Particle properties
    T1,T2,T3 = row.T1r, row.T2r, row.T3r
    dT1,dT2,dT3 = row.dT1r, row.dT2r, row.dT3r
    
    rho_q, k_q, cp_q, em_g, tau_g = [2200, 1.7, 1000, 0.35, 0.6]
    rho_p = 1810
    em_p = 0.85
    # cp1,cp2,cp3  = 365*T1**0.18, 365*T2**0.18, 365*T3**0.18
    cp1,cp2,cp3  = 148*T1**0.3093, 148*T2**0.3093, 148*T3**0.3093
    
    dE1,dE2,dE3  = rho_p*V1*cp1*dT1/dt, rho_p*V2*cp2*dT2/dt, rho_p*V3*cp3*dT3/dt
    Qloss = dE1 + dE2 + dE3
    
    qloss = Qloss / Arcv
    
    dT_glo = qloss * dzg / k_q
    T_glo  = Ts + dT_glo
    h_rc   = - qloss / (Ts-Tamb)
    
    #!!! FIX THIS PART
    h_rad_g  = em_g * cte.sigma *(T_glo**4-Tsky**4)/(Ts-Tamb) * 1.0
    # h_rad_g  = em_g * cte.sigma *(Ts**4-Tsky**4)/(Ts-Tamb)
    h_conv_side = 5.
    h_cond = (0.01/0.05+1/h_conv_side)**(-1) * A_mantle/Arcv
    # h_rad_p  = em_p * tau_g * cte.sigma *(Ts**4-Tsky**4)/(Ts-Tamb)
    h_conv = h_rc - h_rad_g - h_cond
    
    Nu = h_conv * L / k_a
    
    dfc.loc[row.Index,'k_a']    = k_a
    dfc.loc[row.Index,'Pr']     = Pr
    dfc.loc[row.Index,'Ra']     = Ra
    dfc.loc[row.Index,'qloss']  = qloss
    dfc.loc[row.Index,'h_rc']   = h_rc
    dfc.loc[row.Index,'h_rad_g']  = h_rad_g
    # dfc.loc[row.Index,'h_rad_p']  = h_rad_p
    dfc.loc[row.Index,'h_conv'] = h_conv
    dfc.loc[row.Index,'Nu']     = Nu
    
    h_conv_lit  = SPR.h_conv_Holman(Ts, Tamb, L, air)
    dfc.loc[row.Index,'h_conv_lit']  = h_conv_lit
    
    h_conv_lit2 = SPR.h_conv_NellisKlein(Ts, Tamb, L, air)
    dfc.loc[row.Index,'h_conv_lit2']  = h_conv_lit2
    
    dfc.loc[row.Index,'T_glo']  = T_glo


#%% PLOT FOR ALL COOLING DOWN PROCESS
t_ini = 530
t_fin = 1000

fig = plt.figure(figsize=(14,5), constrained_layout=True)

ax1 = fig.add_subplot(131)

ax1.plot(dfc.time,dfc.T3,lw=2,ls='-.',c='C2',label='Top')
ax1.plot(dfc.time,dfc.T2,lw=2,ls='--',c='C1',label='Middle')
ax1.plot(dfc.time,dfc.T1,lw=2,ls=':',c='C0',label='Bottom')

ax1.plot(dfc.time,dfc.Ts,lw=2,ls='-',c='C3',label='Surface (calc)')
ax1.plot(dfc.time,dfc.Tavg,lw=1,ls='-',c='C4')
ax1.plot(dfc.time,dfc.Tavg,lw=3,ls=':',c='C4')
ax1.plot([],[],lw=2,ls='-',marker='.',c='C4',label='Average (calc)')
ax1.set_xlim(t_ini,t_fin)
ax1.set_ylim(0,1400)
ax1.set_xlabel('Time (s)',fontsize=fs-2)
ax1.set_ylabel('Temperature (K)',fontsize=fs-2)
ax1.legend(loc=0)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()

ax2 = fig.add_subplot(132)
ax2.scatter(dfc.time,dfc.h_rc,s=1,c='C2')
ax2.scatter([],[],s=20,c='C2',label='Combined')
ax2.scatter(dfc.time,dfc.h_rad_g,s=1,c='C1')
ax2.scatter([],[],s=20,c='C1',label='Radiative')
ax2.scatter(dfc.time,dfc.h_conv,s=1,c='C0')
ax2.scatter([],[],s=20,c='C0',label='Convective')
# ax2.plot(dfc.time,dfc.h_conv_c,lw=2,ls='--',c='C3',label='Correlation')

ax2.set_xlim(t_ini,t_fin)
ax2.set_ylim(0,dfc.h_rc.max()*1.1)
ax2.set_xlabel('Time (s)',fontsize=fs-2)
ax2.set_ylabel(r'Heat transfer coefficient $\left ( \frac{W}{m^2-K}\right )$',fontsize=fs-2)
ax2.legend(loc=0)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.grid()

ax3 = fig.add_subplot(133)
ax3b = ax3.twinx()
ax3.scatter(dfc.time,dfc.Ra,c='C0',s=1)
ax3b.scatter(dfc.time,dfc.Nu,c='C1',s=1)
ax3.set_xlim(t_ini,t_fin)
ax3.set_xlabel('Time (s)',fontsize=fs-2)
ax3.set_ylabel('Rayleigh Number (-)',fontsize=fs-2)
ax3b.set_ylabel('Nusselt Number (-)',fontsize=fs-2)
ax3.tick_params(axis='y', colors='C0',size=10)
ax3b.tick_params(axis='y', colors='C1',size=10)
ax3.yaxis.label.set_color('C0')
ax3b.yaxis.label.set_color('C1')
ax3.tick_params(axis='both', which='major', labelsize=fs-2)
ax3.grid()
fig.savefig(fldr_rslt+'Convection_Time.png', bbox_inches='tight')
plt.show()

#####################################
#%% OBTAINING A CORRELATION

t_ini = 530
t_fin = 700

dfc2 = dfc[(dfc.time>t_ini)&(dfc.time<t_fin)]

Xd = dfc2.Ra
Yd = dfc2.Nu

def func_HTC(Ra,C,m):
    return C*Ra**m
popt, pcov = spo.curve_fit(func_HTC, Xd, Yd, p0=(1,0.25))

C = popt[0]
m = popt[1]
Yc = func_HTC(Xd,*popt)
r2 = 1 - (np.sum((Yd - Yc)**2) / np.sum((Yd-np.mean(Yd))**2))

dfc2['h_corr'] = Yc * dfc2.k_a / L

n = 1-3*m
C1 = C/L**n
print(C,m,n)
print(r2)

fig = plt.figure(figsize=(11,6), constrained_layout=True)

ax1 = fig.add_subplot(121)
ax1.scatter(dfc2.Ra,dfc2.Nu,s=2,c='C0')
ax1.plot(Xd,Yc,lw=2,ls='--',c='C3',label=r'$\mathrm{{Nu}}={:.2f}\mathrm{{L}}^{{{:.3f}}}\mathrm{{Ra}}^{{{:.3f}}}$'.format(C1,n,m))
ax1.set_xlabel('Rayleigh Number (-)',fontsize=fs-2)
ax1.set_ylabel('Nusselt Number (-)',fontsize=fs-2)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()
ax1.legend(loc=0)

ax2 = fig.add_subplot(122)
ax2.scatter(dfc2.Ts,dfc2.h_conv,s=1)
ax2.plot(dfc2.Ts,dfc2.h_corr,lw=2,ls='--',c='C3', label='Proposed')
ax2.plot(dfc2.Ts,dfc2.h_conv_lit,lw=2,ls='--',c='C2', label='Holman')
ax2.plot(dfc2.Ts,dfc2.h_conv_lit2,lw=2,ls='--',c='C4', label='Nellis&Klein')
# ax2.plot(dfc2.Ts,dfc2.h_conv_c/dfc.h_conv_lit2,lw=2,ls='--',c='C4', label='Ratio')
ax2.set_ylim(0,100)
ax2.legend(loc=0,fontsize=fs-2)
ax2.set_xlabel('Temperature (K)',fontsize=fs-2)
ax2.set_ylabel(r'Heat transfer coefficient $\left ( \frac{W}{m^2-K}\right )$',fontsize=fs-2)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.grid()
plt.show()
fig.savefig(fldr_rslt+'Convection_Correlation.png', bbox_inches='tight')
sys.exit()


fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(121)
# ax2 = ax1.twinx()
ax3 = fig.add_subplot(133)
# ax1.scatter(dfc.Ts,dfc.h_rc,s=1)
# ax1.scatter(dfc.Ts,dfc.hrad,s=1)
ax1.scatter(dfc2.Ts,dfc.h_conv,s=1)
ax1.plot(dfc2.Ts,dfc2.h_conv_c,lw=2,ls='--',c='C1', label='Proposed')
ax1.plot(dfc2.Ts,dfc2.h_conv_lit,lw=2,ls='--',c='C2', label='Holman')
ax1.plot(dfc2.Ts,dfc2.h_conv_lit2,lw=2,ls='--',c='C4', label='Nellis&Klein')
ax2.plot(dfc2.Ts,dfc2.h_conv_c/dfc.h_conv_lit2,lw=2,ls='--',c='C4', label='Ratio')
# ax1.set_xlim(t_ini,t_fin)
ax1.legend(loc=0)
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel(r'Heat transfer coefficient $\left ( \frac{W}{m^2-K}\right )$')
ax1.grid()

#%%%
def func_HTC_proposed(Ra,L,C,m,Ts,Ta,air):
    
    T_av = (Ts+Ta)/2
    air.TP = T_av, ct.one_atm
    k = air.thermal_conductivity
    n = 1-3*m
    Nu = C * L**n * Ra**m
    h = (k*Nu/L)
    return h

# def func_HTC_Cengel(Ra,L,Ts,Ta,air):
    
#     T_av = (Ts+Ta)/2
#     deltaT =  Ts-Ta
#     air.TP = T_av, ct.one_atm
#     k = air.thermal_conductivity
    
#     if Ra > 1e4 and Ra < 1e7:
#         Nu = 0.54*Ra**0.25
#         h = (k*Nu/L)
#     elif Ra>= 1e7 and Ra < 1e11:
#         Nu = 0.15*Ra**(1./3.)
#         h = (k*Nu/L)
#     else:
#         h = 1.52*deltaT**(1./3.)
#     return h

def func_HTC_Cengel(Ra,L,Ts,Ta,air):
    
    T_av = (Ts+Ta)/2
    deltaT =  Ts-Ta
    air.TP = T_av, ct.one_atm
    k = air.thermal_conductivity
    
    if Ra > 1e4 and Ra < 1e7:
        Nu = 0.54*Ra**0.25
        h = (k*Nu/L)
    elif Ra>= 1e7 and Ra < 1e11:
        Nu = 0.15*Ra**(1./3.)
        h = (k*Nu/L)
    else:
        h = 1.52*deltaT**(1./3.)
    Nu = 0.15*Ra**(1./3.)
    h = (k*Nu/L)
    return h


def func_HTC_Holman(Ra,L,Ts,Ta,air):
    
    T_av = (Ts+Ta)/2
    deltaT =  Ts-Ta
    air.TP = T_av, ct.one_atm
    k = air.thermal_conductivity
    
    if Ra > 2e4 and Ra < 8e6:
        Nu = 0.54*Ra**0.25
        h = (k*Nu/L)
    elif Ra>= 8e6 and Ra < 1e11:
        Nu = 0.15*Ra**(1./3.)
        h = (k*Nu/L)
    else:
        h = 1.52*deltaT**(1./3.)
    return h

def func_HTC_Holman_2(Ra,L,Ts,Ta,air):
    
    # T_av = (Ts+Ta)/2
    deltaT =  Ts-Ta
    h = 1.52*deltaT**(1./3.)
    return h


def func_HTC_NellisKlein(Ra,L,Ts,Ta,air):
    #(Nellis & Klein, 2012)
    
    #Approximated values for desired range
    T_av = (Ts+Ta)/2
    air.TP = T_av, ct.one_atm
    mu, k, rho, cp = air.viscosity, air.thermal_conductivity, air.density_mass, air.cp_mass
    alpha = k/(rho*cp)
    visc = mu/rho
    Pr = visc/alpha
    
    C_lam  = 0.671 / ( 1+ (0.492/Pr)**(9/16) )**(4/9)
    Nu_lam = 1.4/ np.log(1 + 1.4 / (0.835*C_lam*Ra**0.25) ) 
    
    C_tur  = 0.14*(1 + 0.0107*Pr)/(1+0.01*Pr)
    Nu_tur = C_tur * Ra**(1/3)
    
    Nu = (Nu_lam**10 + Nu_tur**10)**(1/10)
    
    h = (k*Nu/L)
    return h


Ras = np.logspace(1.0, 11.5, num=200)
L = 1.0
Ts = 1200.
Tamb = 300.

data = []
for Ra in Ras:    
    h1 = func_HTC_proposed(Ra,L,C1,m,Ts,Tamb,air)
    h2 = func_HTC_Holman(Ra,L,Ts,Tamb,air)
    h3 = func_HTC_NellisKlein(Ra,L,Ts,Tamb,air)
    h4 = func_HTC_Holman_2(Ra,L,Ts,Tamb,air)
    data.append([Ra,h1,h2,h3,h4])

data = pd.DataFrame(data,columns=['Ra','h1','h2','h3','h4'])

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121)
ax1.plot(data.Ra,data.h1,lw=2,ls='--',c='C1',label='Proposed')
ax1.plot(data.Ra,data.h2,lw=4,ls=':', c='C2',label='Holman')
ax1.plot(data.Ra,data.h3,lw=2,ls='-.',c='C4',label='NellisKlein')
# ax1.plot(data.Ra,data.h4,lw=4,ls=':', c='C3',label='Holman_2')
ax1.axvspan(5e7, 2e10, alpha=0.2,label='Operation Range')
ax1.set_xscale('log')
ax1.set_ylim(0,500)
ax1.set_xlabel('Rayleigh (-)')
ax1.set_ylabel(r'Heat transfer coefficient $\left ( \frac{W}{m^2-K}\right )$')
ax1.legend(loc=0)
ax1.grid()

plt.show()