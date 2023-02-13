# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:33:51 2022

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

#%% OBTAIN RESULTS FROM OPTICAL SIMULATION AND GENERATE RADIATION FLUX
#########################################################
def f_Optim1D(X,*args):
    stime_i = time.time()
    CSTi,label = args
    
    CST = CSTi.copy()
    Array, Cg, xrc, yrc, zrc, Arcv = [CST[x] for x in ['Array', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'Arcv']]
    
    #Updating the variables before running the simulation
    # fzv = X if len(X)>1 else X[0]
    if label == 'fzv':
        fzv = X
        fzv = fzv/100. if fzv>1. else fzv
        CST['fzv'] = fzv
    if label == 'Arcv':
        Arcv = X
        CST['Arcv'] = Arcv
    if label == 'Prcv':
        Prcv = X
        CST['P_rcv'] = Prcv
        Tp_avg = (CST['T_ini']+CST['T_out'])/2
        CST['eta_rcv'] = SPR.eta_HPR_ini(Tp_avg,CST['Qavg'])
        Arcv = (Prcv/CST['eta_rcv']) / CST['Qavg']
        CST['Arcv'] = Arcv
    if label == 'zf':
        zf = round(X)
        if zf < 40:  zf=40
        if zf > 100: zf=100
        CST['zf'] = zf
        CST['file_SF'] = 'Datasets_Thesis/Rays_zf_{:.0f}_9m2'.format(zf)
    
    N_TOD,V_TOD = BDR.TOD_Array(Array)
    rO = (Arcv / ( V_TOD*N_TOD*np.tan(np.pi/V_TOD) ) )**0.5
    CST['rO_TOD'] = rO
    
    #Running Optical-Thermal simulation
    _, SF, CST = SPR.Simulation_Coupled(CST)
    
    #Objective function and penalisations
    zf, fzv, Prcv, Arcv, N_hel, eta_rcv, P_rcv_sim, Qstg, Q_av, Q_max,T_p, C  = [CST[x] for x in ['zf','fzv','P_rcv','Arcv', 'N_hel', 'eta_rcv', 'P_rcv_sim', 'Q_stg', 'Q_av', 'Q_max','T_p', 'Costs']]
    eta_SF   = SF[SF.hel_in]['Eta_SF'].mean()
    # R_pen1 =  1 + np.abs(1-np.round(Prcv_sim/CST['P_rcv'],2))*100
    # R_pen2 =  1 + np.floor(Q_max/Q_mx_lim)*10
    fobj = C['LCOH']
    
    print('{:.4f}\t{:.4f}\t'.format(fobj,X)+ '\t'.join('{:.2f}'.format(x) for x in [zf, Prcv, fzv, Arcv,time.time()-stime_i, N_hel, eta_SF, eta_rcv, P_rcv_sim, Qstg, Q_av, Q_max,T_p.max(), C['LCOH'], C['LCOE']])+'\t'+Array)
    return fobj

#################################
#################################
#%% PARAMETRIC STUDY
stime = time.time()
#Parameters for CST
CSTi = BDR.CST_BaseCase()

#Plant related costs
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs
CSTi['type_rcvr'] = 'HPR_0D'

fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
fldr_rslt = 'Results_HPR_2D/'
tol = 1e-4        #Solving the non-linear equation
zf = 50.
Qavg = 1.0
Prcv = 20.

Qavgs = np.array([1.00,0.75,0.50,1.25,1.50])
Prcvs = np.append(np.arange(20,41,5),np.arange(15,4,-5))
tzs   = np.array([0.05])
# Qavgs = np.array([1.5])
# Prcvs = np.array([20])
data = []

for (Qavg,Prcv,tz) in [(Qavg,Prcv,tz) for Qavg in Qavgs for Prcv in Prcvs for tz in tzs]:
    
    #### CHECKING IF THE FILE EXISTED OR CREATE IT
    case = 'case_zf{:.0f}_Q_avg{:.1f}_Prcv{:.1f}'.format(zf, Qavg, Prcv)
    # print(case)
    file_BaseCase = 'Cases/'+case+'.plk'
    
    if isfile(file_BaseCase):
    # if False:
        [CSTo,R2,SF,TOD] = pickle.load(open(file_BaseCase,'rb'))
    
    else:
        
        CSTi = BDR.CST_BaseCase()

        Costs = SPR.Plant_Costs()               #Plant related costs
        CSTi['Costs_i'] = Costs
        CSTi['type_rcvr'] = 'HPR_0D'
        
        CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
        CSTi['zf']    = zf
        CSTi['Qavg']  = Qavg
        CSTi['P_rcv'] = Prcv
        Tp_avg = (CSTi['T_pC']+CSTi['T_pH'])/2
        CSTi['eta_rcv'] = SPR.HTM_0D_blackbox(Tp_avg,CSTi['Qavg'])[0]
        Arcv = (CSTi['P_rcv']/CSTi['eta_rcv']) / CSTi['Qavg']
        CSTi['Arcv'] = Arcv
        Type  = CSTi['Type']
        Array = CSTi['Array']
        N_TOD,V_TOD = BDR.TOD_Array(Array)
        CSTi['rO_TOD'] = (Arcv / ( V_TOD*N_TOD*np.tan(np.pi/V_TOD) ) )**0.5
        
        lbl = 'fzv'
        bracket = (75,84,97)
        res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
        fzv  = res.x/100
        
        CSTi['zf']    = zf
        CSTi['fzv']   = fzv
        CSTi['P_rcv'] = Prcv
        
        R2, SF, CSTo = SPR.Simulation_Coupled(CSTi)
        zf,Type,Array,rO,Cg,xrc,yrc,zrc,Npan = [CSTo[x] for x in ['zf', 'Type', 'Array', 'rO_TOD', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'N_pan']]
        TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},xrc,yrc,zrc)
        
        pickle.dump([CSTo,R2,SF,TOD],open(file_BaseCase,'wb'))

    ######################################################
    #### RECEIVER FUNCTION ITSELF
    ## Input: CST, SF, R2
    ## Output: N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, Mstg, t_res, vel_p
    
    CSTo['tz'] = tz
    T_p, results, Rcvr_full = SPR.Rcvr_HPR_2D_Simple(CSTo, SF, R2, full_output=True)
    N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, Eta_rcv, Mstg, t_res,vel_p, it, Solve_t_res = results
    
    Eta_SF = SF.iloc[:N_hel].mean()['Eta_SF']
    Eta_hel = SF.iloc[:N_hel].mean()['Eta_hel']
    Eta_BDR = SF.iloc[:N_hel].mean()['Eta_BDR']
    Eta_ThS = Eta_SF * Eta_rcv
    
    Tp_max,Tp_min,Tp_std = T_p.max(),T_p.min(),T_p.std()
    # print(N_hel,Q_max,Q_av,eta_rcv,t_res,vel_p)

    ######################################################
    #### GENERATING DETAILED RESULTS
    
    #Retrieven the TOD and CST information
    polygon_i=1
    Type, Array, rO, Cg, zrc = [CSTo[x] for x in ['Type', 'Array','rO_TOD','Cg_TOD','zrc']]
    TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
    N_TOD,V_TOD,H_TOD,rO,rA,x0,y0,Arcv = [TOD[x] for x in ['N', 'V', 'H', 'rO', 'rA', 'x0', 'y0', 'A_rcv']]
    xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
    lims = xO.min(), xO.max(), yO.min(), yO.max()
    # x_fin=lims[0]; x_ini = lims[1]; y_bot=lims[2]; y_top=lims[3]
    
    cols = ['zf', 'Prcv', 'Qavg', 'tz', 'N_hel', 'Arcv', 'Q_max', 'Qstg', 'Q_av', 't_res', 'vel_p', 'Eta_hel', 'Eta_BDR', 'Eta_SF', 'Eta_rcv', 'Eta_ThS', 'Mstg', 'Tp_avg', 'it', 'Solve_t_res', 'Tp_max', 'Tp_min', 'Tp_std']
    
    row = [zf, Prcv, Qavg, tz, N_hel, Arcv, Q_max, Q_av, Qstg, t_res, vel_p, Eta_hel, Eta_BDR, Eta_SF, Eta_rcv, Eta_ThS,Mstg,T_p.mean(), it,Solve_t_res,Tp_max,Tp_min,Tp_std]
    data.append(row)
    print('\t'.join('{:.2f}'.format(x) for x in row))
    ######################################################
    
    ############################################################
    #### Plotting
    for lbl in ['eta','Tp','Q_in']:
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, aspect='equal')
        X = Rcvr_full['x']
        Y = Rcvr_full['y']
        Z = Rcvr_full[lbl]
        f_s = 16
        if lbl == 'eta':
            vmin = 0.7
            vmax = 0.9
            cmap = cm.YlOrRd
        elif lbl =='Tp':
            vmin = (np.floor(Z.min()/100)*100)
            vmax = (np.ceil(Z.max()/100)*100)
            cmap = cm.YlOrRd
        elif lbl =='Q_in':
            vmin = 0.0
            vmax = np.ceil(Z.max())
            cmap = cm.YlOrRd
        # cmap = cm.YlOrRd
        
        # surf = ax.tricontourf(X.flatten(), Y.flatten(), Z.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
        surf = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
        cb = fig.colorbar(surf, shrink=0.5, aspect=4)
        cb.ax.tick_params(labelsize=f_s-2)
        
        xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        lp = len(xA)//2
        fyO = spi.interp1d(xO[:lp],yO[:lp],bounds_error=False,fill_value=0)
        yO2 = fyO(xA[:lp])
        ax.fill_between(xA[:lp],yO2[:lp],2*yA[:lp], color='w')
        ax.fill_between(xA[:lp],-yO2[:lp],-2*yA[:lp], color='w')
        
        i = polygon_i-1
        x0f = x0[i]
        y0f = y0[i]
        radius = rO/np.cos(np.pi/V_TOD)
        ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
        radius = rA/np.cos(np.pi/V_TOD)
        ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
        
        ax.set_xlim(lims[0],lims[1])
        ax.set_ylim(lims[2],lims[3])
        fig.savefig(fldr_rslt+case+'_'+lbl+'.png', bbox_inches='tight')
        # plt.show()
        plt.close()

    # print(time.time()-stime)

df_res = pd.DataFrame(data,columns=cols)
print(df_res)

df_res.to_csv(fldr_rslt+'0_Results_zf_{:.0f}.csv'.format(zf))

fs=16
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(111)

for qi in df_res.Qavg.unique():
    df2 = df_res[df_res.Qavg==qi]
    ax1.plot(df2.Prcv,df2.Eta_SF,lw=3,ls='-.',c='C2',label='Eta_SF')
    ax1.plot(df2.Prcv,df2.Eta_rcv,lw=3,ls='--',c='C1',label='Eta_rcv')
    ax1.plot(df2.Prcv,df2.Eta_ThS,lw=3,ls=':',c='C0',label='Eta_ths')
# ax1.set_xlim(0,1000)
# ax1.set_ylim(0,1700)
ax1.set_xlabel('Receiver Nominal Power (W)', fontsize=fs)
ax1.set_ylabel('Efficiency (-)', fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=0, fontsize=fs-2)
ax1.grid()

#%% ANALYZING THE FUNCTION
CSTi = BDR.CST_BaseCase()

#Plant related costs
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs
CSTi['type_rcvr'] = 'HPR_0D'

fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
fldr_rslt = 'Results_HPR_2D/'
tol = 1e-4        #Solving the non-linear equation
zf = 50.
Qavg = 1.0
Prcv = 20.

case = 'case_zf{:.0f}_Q_avg{:.1f}_Prcv{:.1f}'.format(zf, Qavg, Prcv)
file_BaseCase = 'Cases/'+case+'.plk'
if isfile(file_BaseCase):
    [CSTo,R2,SF,TOD] = pickle.load(open(file_BaseCase,'rb'))

######################################################
## RECEIVER FUNCTION ITSELF
## Input: CST, SF, R2
## Output: N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, Mstg, t_res, vel_p
T_p, results, Rcvr_full = SPR.Rcvr_HPR_2D_Simple(CSTo, SF, R2, full_output=True)
N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, Mstg, t_res,vel_p, it,Solve_t_res = results

# pickle.dump([T_p, results, Rcvr_full],open('Results_HPR_basecase.plk','wb'))

Eta_SF = SF.iloc[:N_hel].mean()['Eta_SF']
Eta_hel = SF.iloc[:N_hel].mean()['Eta_hel']
Eta_BDR = SF.iloc[:N_hel].mean()['Eta_BDR']
# print(N_hel,Q_max,Q_av,eta_rcv,t_res,vel_p)

######################################################
# GENERATING DETAILED RESULTS

#Retrieven the TOD and CST information
polygon_i=1
Type, Array, rO, Cg, zrc = [CSTo[x] for x in ['Type', 'Array','rO_TOD','Cg_TOD','zrc']]
TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
N_TOD,V_TOD,H_TOD,rO,rA,x0,y0,Arcv = [TOD[x] for x in ['N', 'V', 'H', 'rO', 'rA', 'x0', 'y0', 'A_rcv']]
xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
lims = xO.min(), xO.max(), yO.min(), yO.max()
# x_fin=lims[0]; x_ini = lims[1]; y_bot=lims[2]; y_top=lims[3]

row = [zf,Prcv,Qavg,N_hel,Arcv,Q_max,Q_av,t_res,vel_p,Eta_hel,Eta_BDR,Eta_SF,eta_rcv]
# data.append(row)
print('\t'.join('{:.2f}'.format(x) for x in row))

# lims = Rcvr_full['x'].max(), Rcvr_full['x'].min(),Rcvr_full['y'].min(), Rcvr_full['y'].max()
# xrcv_min,xrcv_max = lims[1],lims[0]
# X_tp = np.linspace(xrcv_min,xrcv_max,len(T_p))
# plt.scatter(X_tp,T_p)
# plt.show()

# data = []
# for i in range(Rcvr_full['x'].shape[0]):
#     x_i      = Rcvr_full['x'][i,:]
#     rcv_in_i = Rcvr_full['rcv_in'][i,:]
#     Tp_i     = Rcvr_full['Tp'][i,:]
#     eta_i    = Rcvr_full['eta'][i,:]
#     Qin_i    = Rcvr_full['Q_in'][i,:]
    
#     Tp_av  = Tp_i[rcv_in_i].mean()
#     eta_av = eta_i[rcv_in_i].mean()
#     Qin_av = Qin_i[rcv_in_i].mean()
#     data.append([x_i.mean(),Tp_av,eta_av,Qin_av])
#     print(data[-1],sum(rcv_in_i),eta_i.mean())

# df4 = pd.DataFrame(data,columns=['x','Tp','eta','Qi'])

# Xrcv = (df4.x.max() - df4.x) / (df4.x.max() - df4.x.min())
# plt.scatter(Xrcv, df4.Tp)
# plt.show()
# plt.scatter(Xrcv, df4.eta)
# plt.show()
# plt.scatter(Xrcv, df4.Qi)
# plt.show()
######################################################
############################################################
# Plotting
for lbl in ['eta','Tp','Q_gain']:
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect='equal')
    X = Rcvr_full['x']
    Y = Rcvr_full['y']
    Z = Rcvr_full[lbl]
    f_s = 16
    if lbl == 'eta':
        vmin = 0.7
        vmax = 0.9
        cmap = cm.YlOrRd
        text = r'$\eta_{rcv}[-]$'
    elif lbl =='Tp':
        vmin = (np.floor(Z.min()/100)*100)
        vmax = (np.ceil(Z.max()/100)*100)
        cmap = cm.YlOrRd
        text = r'$T_{p} [K]$'
    elif lbl =='Q_in':
        vmin = 0.0
        vmax = np.ceil(Z.max())
        cmap = cm.YlOrRd
        text = r'$Q_{in}[kW?]$'
    elif lbl =='Q_gain':
        vmin = 0.0
        vmax = np.ceil(Z.max())
        cmap = cm.YlOrRd
        text = r'$Q_{gain}[MW/m^2]$'
    # cmap = cm.YlOrRd
    
    # surf = ax.tricontourf(X.flatten(), Y.flatten(), Z.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
    cb = fig.colorbar(surf, shrink=0.5, aspect=4)
    cb.ax.tick_params(labelsize=f_s-2)
    fig.text(0.82,0.725,text,fontsize=f_s,ha='center')
    
    xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
    xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
    lp = len(xA)//2
    fyO = spi.interp1d(xO[:lp],yO[:lp],bounds_error=False,fill_value=0)
    yO2 = fyO(xA[:lp])
    ax.fill_between(xA[:lp],yO2[:lp],2*yA[:lp], color='w')
    ax.fill_between(xA[:lp],-yO2[:lp],-2*yA[:lp], color='w')
    
    i = polygon_i-1
    x0f = x0[i]
    y0f = y0[i]
    radius = rO/np.cos(np.pi/V_TOD)
    ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
    radius = rA/np.cos(np.pi/V_TOD)
    ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
    
    
    ax.set_xlim(lims[0],lims[1])
    ax.set_ylim(lims[2],lims[3])
    fig.savefig(fldr_rslt+case+'_'+lbl+'.png', bbox_inches='tight')
    plt.show()
    plt.close()



sys.exit()

#%% INFLUENCE OF THICKNESS

sys.exit()
df_rcv = []
for tz in np.arange(0.01,0.11,0.01):
    CSTo['tz'] = tz
    T_p, results = SPR.Rcvr_HPR_2D_Simple(CSTo, SF, R2)
    N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, Mstg, t_res, vel_p = results
    df_rcv.append([tz,N_hel,Q_max,Q_av,eta_rcv,t_res,vel_p,T_p.max(),T_p.min(),T_p.std()])
    
    print('\t'.join('{:.2f}'.format(x) for x in df_rcv[-1]))
    
df_rcv = pd.DataFrame(df_rcv,columns=['tz','N_hel','Q_max','Q_av','eta_rcv','t_res','vel_p', 'Tp_max','Tp_min','Tp_std'])

#Relationship for thickness and flow
h_granflow = np.arange(0.01,0.11,0.01)
C1 = 5.2
vel_granflow1 = np.array([C1*h**1.5 for h in h_granflow])

# C1 = 594
# vel_granflow2 = np.array([C1*h**1.5 for h in h_granflow])

plt.scatter(df_rcv['tz'],df_rcv['t_res'])
plt.show()
plt.scatter(df_rcv['tz'],df_rcv['vel_p'])
plt.scatter(h_granflow,vel_granflow1)
# plt.scatter(h_granflow,vel_granflow2)
plt.show()
plt.scatter(df_rcv['tz'],df_rcv['eta_rcv'])
plt.show()
plt.scatter(df_rcv['tz'],df_rcv['Tp_std'])
plt.show()
plt.scatter(df_rcv['tz'],df_rcv['Tp_max'])
plt.scatter(df_rcv['tz'],df_rcv['Tp_min'])
plt.show()
##################################################
#%%PLOTTING RADIATION MAP

#Parameters for receiver. lims are important here
polygon_i = 1
fzv, rmax = [CSTo[x] for x in ['fzv', 'rmax']]
Type, Array, rO, Cg, zrc = [CSTo[x] for x in ['Type', 'Array','rO_TOD','Cg_TOD','zrc']]
TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
N_TOD,V_TOD,H_TOD,rO,rA,x0,y0,Arcv = [TOD[x] for x in ['N', 'V', 'H', 'rO', 'rA', 'x0', 'y0', 'A_rcv']]
xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)    
lims = xO.min(), xO.max(), yO.min(), yO.max()

hlst = SF.iloc[:N_hel].index
R2a = R2[(R2.hit_rcv)&(R2.Npolygon==polygon_i)].copy()  #Rays into 1 receiver
R2b = R2[(R2.hit_rcv)].copy()                           #Total Rays into receivers
rays   = R2a[R2a.hel.isin(hlst)].copy()
r_i = len(rays)/len(R2b[R2b.hel.isin(hlst)])                      #Fraction of rays that goes into one TOD
P_SF_i = r_i * P_SF2 * 1e6

f_Qrc1,Q_rc1,X_rc1,Y_rc1 = SPR.Get_f_Qrc1(rays['xr'],rays['yr'],lims,P_SF_i)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, aspect='equal')
X, Y = np.meshgrid(X_rc1, Y_rc1)
f_s = 16
vmin = 0
vmax = (np.ceil(Q_rc1.max()/100)*100)
surf = ax.pcolormesh(X, Y, Q_rc1.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
cb = fig.colorbar(surf, shrink=0.5, aspect=4)
cb.ax.tick_params(labelsize=f_s-2)

i = polygon_i-1
x0f = x0[i]
y0f = y0[i]
radius = rO/np.cos(np.pi/V_TOD)
ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
radius = rA/np.cos(np.pi/V_TOD)
ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
    
fig.text(0.77,0.27,'Main Parameters',fontsize=f_s-3)
fig.text(0.77,0.25,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
fig.text(0.77,0.23,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
fig.text(0.77,0.21,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
fig.text(0.77,0.19,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
ax.set_title('b) Radiation flux in receiver aperture',fontsize=f_s)

fig.text(0.77,0.70,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
# fig.savefig(fldr_rslt+case+'_radmap_out.png', bbox_inches='tight')
plt.show()
plt.close()
