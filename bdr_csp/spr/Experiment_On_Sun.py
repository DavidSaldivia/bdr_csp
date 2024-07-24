# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:05:36 2022

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
#%% DEFINING FIXED PARAMETERS
Drcv = 0.041       #[m] Diameter of receiver
Arcv = np.pi*Drcv**2/4
prcv = np.pi*Drcv
z1, z2, z3, Hrcv = 0.004,0.014,0.021,0.024

DNI     = 894.
A_fr    = 1.012
tau_fr  = 0.83
eta_fr  = 0.701
dzg     = 0.004
Fc      = 2.57
ab_p    = 0.91
em_p    = 0.85
hcond   = 0.833
Fv      = 1.0
rho_p   = 1810
k_p     = 0.7
TSM     = 'CARBO'
air     = ct.Solution('air.xml')

tau_gl  = 0.9

rho_q, k_q, cp_q, em_g, tau_g = [2200, 1.7, 1000, 0.35, 0.6]  #Properties of quartz

t_ini = 100.
t_fin = 400.

A_mantle = np.pi*Hrcv*Drcv + np.pi*Drcv**2/2
dz1, dz2, dz3 = (z1+z2)/2, (z3-z1)/2, Hrcv - (z2+z3)/2
V1, V2, V3 = Arcv*dz1, Arcv*dz2, Arcv*dz3
L    = Arcv/prcv

#########################################
#%%% CHARGING EXPERIMENTAL DATA
file_data = 'Experiments_CARBO/Results_Day1.xlsx'
df0 = pd.read_excel(file_data,header=0)

df0['T1'] = df0.Bottom + 273.15
df0['T2'] = df0.Middle + 273.15
df0['T3'] = df0.Top + 273.15
df0['Tamb'] = df0.Ambient + 273.15
df0['Tavg'] = ( (df0.T1*(z2-z1) + df0.T2*(z3-z1) + df0.T3*(z3-z2) )/2. ) / (z3-z1)
df0['Ts'] = (Hrcv-z3)*(df0.T3-df0.T2)/(z3-z2) + df0.T3


#%%% PLOT FOR THE WHOLE PROCESS
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
# fig.savefig(fldr_rslt+'Temperatures_Plot.png', bbox_inches='tight')
plt.show()

# PLOT FOR THE CHARGING PROCESS
df = df0[['time','T1','T2','T3','Tamb','Ts','Tavg']]
df1 = df[(df.time>t_ini)&(df.time<t_fin)].copy()         #Section with only heat losses (convective mostly)

fs=16
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)
ax1.plot(df1.time,df1.T3,lw=2,ls='-.',c='C2',label='Top')
ax1.plot(df1.time,df1.T2,lw=2,ls='--',c='C1',label='Middle')
ax1.plot(df1.time,df1.T1,lw=2,ls=':',c='C0',label='Bottom')
ax1.plot(df1.time,df1.Ts,lw=2,ls='-',c='C3',label='Surface (calc)')
ax1.set_xlim(t_ini,t_fin)
ax1.set_ylim(0,1700)
ax1.set_xlabel('Time (s)',fontsize=fs-2)
ax1.set_ylabel('Temperature (K)',fontsize=fs-2)
ax1.legend(loc=0)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()
# fig.savefig(fldr_rslt+'Temperatures_Plot.png', bbox_inches='tight')
plt.show()

#########################################

#%%% LOADING WETHER DATA
# file_weather = 'Experiments_CARBO/Weatherdata1min_13April.dat'
# df = pd.read_table(file_weather,header=1,sep=',')
# df = df.iloc[2:][['TIMESTAMP','Direct_Normal_Irradiation_MS56']]
# df.rename({'TIMESTAMP':'time','Direct_Normal_Irradiation_MS56':'DNI'},axis=1,inplace=True)
# df['DNI'] = df[['DNI']].astype(float)
# df['time'] = pd.to_datetime(df.time)
# df.set_index('time', inplace=True)
# df = df[(df.index.day==11)&(df.index.month==4)&(df.index.hour>=12)&(df.index.hour<14)]
# df = df[df.DNI>800.]
# DNI = df.DNI.mean()
# print(df)
# print(DNI)
# plt.scatter(df.index,df.DNI)

#%% SIMULATIONS

tz = Hrcv - z1
# Energy hitting particles
E_sun = DNI * A_fr                       #[W] Energy that hit the Fresnel lense
E_fr  = E_sun * eta_fr                   #[W] Energy delivered by Fresnel lense
E_in  = E_fr * tau_gl                    #[W] Energy that hit the particles (after glass)

q_in  = E_in /Arcv / 1e6                 #[MW/m2]

#Belt vector direction and vector direction
Nz = 10                            #[-] Number of elements in X' axis
dz = tz/Nz                         #[m] Particles z' discr
dt = df1.diff(1)['time'].mean()    #Temporal discretisation [s]

#Positions in vertical direction
P_z = np.array([ (tz+z1) - i*dz for i in range(Nz+1)])

T_p = np.linspace(df1.iloc[0].T1,df1.iloc[0].T1,Nz+1)
T_ini = T_p.mean()
# T_p = np.linspace(df1.iloc[0].T3,df1.iloc[0].T1,Nz+1)
# T_p[0]  = df1.iloc[0].T3
# T_p[-1] = df1.iloc[0].T1  #It is assumed all the particles start with the temperature of the bottom

t = t_ini
t_sim = t_fin

data = []

while t <= t_sim+dt:
    
    #Obtaining the Q_in and efficiency for the top
    # eta = SPR.HTM_0D_blackbox(T_p[0], q_in, Fc=Fc, air=air)[0]
    Ts   = T_p[0]
    idx  = df1.time.sub(t).abs().idxmin()
    Tamb = df1.loc[idx].Tamb
    Tsky = Tamb-15.
    
    air.TP = (T_p[0]+Tamb)/2., ct.one_atm
    hconv = Fc*SPR.h_conv_NellisKlein(Ts, Tamb, 0.01, air)
    hrad  = Fv*em_p*5.67e-8*(Ts**4.-Tsky**4.)/(Ts-Tamb)
    hrc = hconv + hrad + hcond
    qloss = hrc * (Ts - Tamb)
    
    eta = (q_in*1e6*ab_p - qloss)/(q_in*1e6)

    E_abs = E_in * eta
    q_abs = E_abs / Arcv
    
    cp_p    = 148*T_p**0.3093
    alpha_p = k_p/(rho_p*cp_p)
    r_p     = alpha_p * dt / dz**2
    
    #################################
    
    T_k = T_p.copy()        #Getting the previous elements
    
    #Elements on top
    T_p[0] = T_k[0] + q_abs / (cp_p[0] * rho_p * dz / dt) - r_p[0]*(T_k[0] - T_k[1])
    
    T_p[0] = df1.loc[df1.time.sub(t).abs().idxmin()].T3
    
    T_k_prev = np.roll(T_k,+1)[1:-1]
    T_k_next = np.roll(T_k,-1)[1:-1]
    T_p[1:-1] = r_p[1:-1] * (T_k_prev - 2*T_k[1:-1] + T_k_next) + T_k[1:-1]
    
    #Last element, has to be equals to T_bottom
    # T_p[-1] = T_k[-1] + r_p[-1]*(T_k[-2] - T_k[-1])
    T_p[-1] = df1.loc[idx].T1
    
    T_top = T_p[abs(P_z-z3).argmin()]
    T_mid = T_p[abs(P_z-z2).argmin()]
    T_bot = T_p[-1]
    
    row = np.append([t,eta,T_top,T_mid,T_bot],T_p)
    data.append(row)
    
    dT = T_p - T_k
    t += dt

cols = ['time','eta','T_top','T_mid','T_bot'] + ['T'+str(i) for i in range(Nz+1)]
data = pd.DataFrame(data,columns=cols)

plt.scatter(data.time,data.T0,label='Ts mod')
plt.scatter(data.time,data['T_top'],label='T_top mod')
plt.scatter(data.time,data['T_mid'],label='T_mid mod')

# plt.scatter(df1.time,df1.Ts,label='Ts calc')
plt.scatter(df1.time,df1.T1,label='T1 exp')
plt.scatter(df1.time,df1.T2,label='T2 exp')
plt.scatter(df1.time,df1.T3,label='T3 exp')
plt.legend()
plt.show()
plt.scatter(data.time,data.eta)
plt.show()

T_out_av = T_p.mean()
cp_b  = (148*((T_p+T_ini)/2 + 273.15)**0.3093).mean()

print(T_ini,T_p.mean(),T_p.max())


#%% DIFFERENT APPROACH
t_ini = 120
dfc = df1.groupby(df1.time // 5).mean()
dfc = df1.copy()

Nroll = 10
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

E_sun = DNI * A_fr                       #[W] Energy that hit the Fresnel lense
E_fr  = E_sun * eta_fr                   #[W] Energy delivered by Fresnel lense
E_in  = E_fr * tau_gl                    #[W] Energy that hit the particles (after glass)

# Calculating the radiation in focal point
DNI_exp = 870.
A_fr = 1.012
tau_fr = 0.83
E_in =  DNI*A_fr*tau_fr
r1  = 0.044/2
Er1 = 616.25 * (DNI/DNI_exp)
eta_int1 = Er1 / E_in
sigma = r1 / (-2*np.log(1-eta_int1))**0.5
A0    = E_in / (2*np.pi*sigma**2)

A0 = A0*1e-6
sigma = 11.22
rr = np.arange(0.,50.,0.5)
def Q_r_rcv(x,A0,std):
    return A0*np.exp(-0.5*(x/std)**2)
def E_r_arg(r,*args):
    (A0,std) = args
    return  Q_r_rcv(r,A0,std) * (2*np.pi*r)
QQ = A0*np.exp(-0.5*(rr**2)/sigma**2)
EE = np.array([quad(E_r_arg,0,ri,args=(A0,sigma))[0] for ri in rr])
QQ_sp = np.array([EE[i]/(np.pi*rr[i]**2) for i in range(1,len(rr))])

D_fp = 0.016       #[m] Diameter of focal point
r_fp = (D_fp/2)*1000. #[mm]
A_fp = np.pi*D_fp**2/4

E_fp = EE[abs(rr-r_fp).argmin()]
E_in  = E_fp * tau_gl

prcv = np.pi*Drcv
z1, z2, z3, Hrcv = 0.004,0.014,0.021,0.024

q_in  = E_in /A_fp / 1e6                 #[MW/m2]
i=0
for row in dfc.itertuples():
    
    dt   = row.dt
    Ts   = row.Ts
    Tamb = row.Tamb
    Tsky = Tamb - 15
    T_av = (Ts+Tamb)/2.
    
    #Air properties
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
    
    cp1,cp2,cp3  = 148*T1**0.3093, 148*T2**0.3093, 148*T3**0.3093
    
    dE1,dE2,dE3  = rho_p*V1*cp1*dT1/dt, rho_p*V2*cp2*dT2/dt, rho_p*V3*cp3*dT3/dt
    Qgain = dE1 + dE2 + dE3
    
    Qloss = E_in - Qgain
    eta_E   = Qgain/E_in
    # print(row.time,E_in,Qgain,Qloss,eta)
    
    dfc.loc[row.Index,'Qgain']  = Qgain
    dfc.loc[row.Index,'Qloss']  = Qloss
    dfc.loc[row.Index,'eta_E']  = eta_E
    
    air.TP = (Ts+Tamb)/2., ct.one_atm
    hconv = Fc*SPR.h_conv_NellisKlein(Ts, Tamb, 0.01, air)
    hrad  = Fv*em_p*5.67e-8*(Ts**4.-Tsky**4.)/(Ts-Tamb)
    h_conv_side = 5.
    h_cond = (0.01/0.05+1/h_conv_side)**(-1) * A_mantle/Arcv
    
    # qloss = Qloss / Arcv
    # dT_glo = qloss * dzg / k_q
    # T_glo  = Ts + dT_glo
    # hrad_g  = em_g * cte.sigma *(T_glo**4-Tsky**4)/(Ts-Tamb)
    
    hrc = hconv + hrad + hcond
    qloss = hrc * (Ts - Tamb)
    eta_T = (q_in*1e6*ab_p - qloss)/(q_in*1e6)
    dfc.loc[row.Index,'eta_T']  = eta_T
    
    try:
        def func_Ts(Ts_in):
            air.TP = (Ts_in+Tamb)/2., ct.one_atm
            hconv = Fc*SPR.h_conv_NellisKlein(Ts_in, Tamb, 0.01, air)
            hrad  = Fv*em_p*5.67e-8*(Ts_in**4.-Tsky**4.)/(Ts_in-Tamb)
            h_conv_side = 5.
            h_cond = (0.01/0.05+1/h_conv_side)**(-1) * A_mantle/Arcv
            hrc = hconv + hrad + h_cond
            qloss = hrc * (Ts_in - Tamb)
            eta_T = (q_in*1e6*ab_p - qloss)/(q_in*1e6)
            return eta_T-eta_E
        
        Ts_calc = spo.fsolve(func_Ts,T3)[0]
        print(Ts_calc,eta_T,eta_E)
        dfc.loc[row.Index,'Ts_calc']  = Ts_calc
    except:
        dfc.loc[row.Index,'Ts_calc']  = Ts
        
    
    #!!! FIX THIS PART
    
    # h_rad_g  = em_g * cte.sigma *(Ts**4-Tsky**4)/(Ts-Tamb)
    
    # h_rad_p  = em_p * tau_g * cte.sigma *(Ts**4-Tsky**4)/(Ts-Tamb)
    
    # Nu = h_conv * L / k_a
    
    # dfc.loc[row.Index,'k_a']    = k_a
    # dfc.loc[row.Index,'Pr']     = Pr
    # dfc.loc[row.Index,'Ra']     = Ra
    # dfc.loc[row.Index,'qloss']  = qloss
    # dfc.loc[row.Index,'h_rc']   = h_rc
    # dfc.loc[row.Index,'h_rad_g']  = h_rad_g
    # # dfc.loc[row.Index,'h_rad_p']  = h_rad_p
    # dfc.loc[row.Index,'h_conv'] = h_conv
    # dfc.loc[row.Index,'Nu']     = Nu
    
    # h_conv_lit  = SPR.h_conv_Holman(Ts, Tamb, L, air)
    # dfc.loc[row.Index,'h_conv_lit']  = h_conv_lit
    
    # h_conv_lit2 = SPR.h_conv_NellisKlein(Ts, Tamb, L, air)
    # dfc.loc[row.Index,'h_conv_lit2']  = h_conv_lit2
    
    # dfc.loc[row.Index,'T_glo']  = T_glo
    i+=1
    
fig = plt.figure(figsize=(14,5), constrained_layout=True)

ax0 = fig.add_subplot(131)
ax0.plot(dfc.time,dfc.T3,lw=3,ls='--',c='C2',label='T top')
ax0.plot(dfc.time,dfc.T2,lw=3,ls='--',c='C1',label='T mid')
ax0.plot(dfc.time,dfc.T1,lw=3,ls='-.',c='C0',label='T bot')
# ax0.plot(dfc.time,dfc.Ts,lw=2,ls='--',c='C3',label='Ts estimated')
ax0.set_xlim(t_ini,t_fin)
ax0.set_xlabel('Time (s)',fontsize=fs-2)
ax0.set_ylabel('Temperature (K)',fontsize=fs-2)
ax0.legend(loc=0,fontsize=fs-2)
ax0.tick_params(axis='both', which='major', labelsize=fs-2)
ax0.tick_params(axis='both', which='major', labelsize=fs-2)
ax0.grid()

ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)
ax1.plot(dfc.time,dfc.Qgain,lw=2,ls='-',c='C2',label='E stored')
ax1.plot(dfc.time,dfc.Qloss,lw=2,ls='-',c='C1',label='E loss')
ax2.plot(dfc.time,dfc.eta_E,lw=2,ls='-',c='C0',label=r'$\eta_{rcv}$ exp' )
ax2.plot(dfc.time,dfc.eta_T,lw=2,ls='-',c='C3',label=r'$\eta_{rcv}$ calc')
# ax1.scatter(dfc.time,dfc.Qgain,s=2,c='C2',label='E stored')
# ax1.scatter(dfc.time,dfc.Qloss,s=2,c='C1',label='E loss')
# ax2.scatter(dfc.time,dfc.eta_E,s=2,c='C0',label=r'$\eta_{rcv,E}$')
# ax2.scatter(dfc.time,dfc.eta_T,s=2,c='C3',label=r'$\eta_{rcv,T}$')

ax1.set_xlim(t_ini,t_fin)
ax2.set_xlim(t_ini,t_fin)
ax1.set_ylim(0,150)
ax2.set_ylim(0,1)
ax1.set_xlabel('Time (s)',fontsize=fs-2)
ax2.set_xlabel('Time (s)',fontsize=fs-2)
ax1.set_ylabel('Energy flow (W)',fontsize=fs-2)
ax2.set_ylabel('Receiver Efficiency (-)',fontsize=fs-2)
ax1.legend(loc=3,fontsize=fs-2)
ax2.legend(loc=1,fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.grid()
ax2.grid()
fig.savefig(fldr_rslt+'Heating_Efficiency.png', bbox_inches='tight')

plt.show()

fig = plt.figure(figsize=(8,6), constrained_layout=True)
ax0 = fig.add_subplot(111)
ax0.plot(dfc.time,dfc.T1,lw=2,ls='-.',c='C2',label='T bot')
ax0.plot(dfc.time,dfc.T2,lw=2,ls='--',c='C4',label='T mid')
ax0.plot(dfc.time,dfc.T3,lw=2,ls='--',c='C1',label='T top')
ax0.plot(dfc.time,dfc.Ts_calc,lw=2,ls='--',c='C0',label='Ts calc')
ax0.plot(dfc.time,dfc.Ts,lw=2,ls='--',c='C3',label='Ts est')
ax0.set_xlim(t_ini,t_fin)
ax0.set_xlabel('Time (s)',fontsize=fs-2)
ax0.set_ylabel('Temperature (K)',fontsize=fs-2)
ax0.legend(loc=0)
ax0.tick_params(axis='both', which='major', labelsize=fs-2)
ax0.tick_params(axis='both', which='major', labelsize=fs-2)
ax0.grid()
plt.show()