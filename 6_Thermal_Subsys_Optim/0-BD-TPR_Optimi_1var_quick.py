# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:35:12 2022

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

#########################################
#########################################
def f_Optim1D(X,*args):
    stime_i = time.time()
    CSTi,label = args
    
    CST = CSTi.copy()
    Array, Cg, xrc, yrc, zrc, Arcv = [CST[x] for x in ['Array', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'Arcv']]
    
    #Updating the variables before running the simulation
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
        eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CST)
        CST['eta_rcv'] = eta_rcv_i
        CST['Arcv'] = Arcv
        CST['rO_TOD'] = rO_TOD
    if label == 'zf':
        zf = round(X)
        if zf < 20:  zf=20
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

#%% OPTIMIZATION WITH ONE VARIABLE QUICK

#Parameters for CST
CSTi = BDR.CST_BaseCase()
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs
CSTi['type_rcvr'] = 'TPR_0D'

plot = False
fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
fldr_rslt = 'Optim_Thesis_Chapter/'

file_CSTs = 'Optim_TPR_0D_quick.plk'
# file_df   = 'Optim_HPR_0D_APSRC_quick_v2.csv'
# file_df   = 'Null'
file_df   = 'Optim_TPR_0D_quick.csv'

stime = time.time()

tol = 1e-3        #Solving the non-linear equation
# zfs = np.arange(30,71,10)
zfs   = np.arange(35,71,5)
Qavgs = [1.35,1.45]
# Qavgs = np.arange(1.1,1.41,0.1)
# Prcvs = np.append(np.arange(10,41,5),np.arange(2,10,2))
Prcvs = np.arange(15,36,1)

for (Qavg,zf) in [(Qavg,zf) for Qavg in Qavgs for zf in zfs]:
    
    CSTi = BDR.CST_BaseCase()
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    
    CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
    CSTi['zf']    = zf
    CSTi['Qavg']  = Qavg
    CSTi['P_rcv'] = zf * 0.35       #Initial guess for Receiver power
    
    #Estimating eta_rcv to obtain the TOD characteristics
    CSTi['type_rcvr'] = 'TPR_0D'
    eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
    CSTi['eta_rcv'] = eta_rcv_i
    CSTi['Arcv'] = Arcv
    CSTi['rO_TOD'] = rO_TOD
    
    #Obtaining initial optimized value for fzv
    lbl = 'fzv'
    bracket = (75 , 84, 97)
    res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
    fzv  = res.x/100
    
    for Prcv in Prcvs:
        
        CSTi = BDR.CST_BaseCase()
        Costs = SPR.Plant_Costs()
        CSTi['Costs_i'] = Costs
        
        CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
        CSTi['zf']    = zf
        CSTi['fzv']   = fzv
        CSTi['Qavg']  = Qavg
        CSTi['P_rcv'] = Prcv
        
        #Estimating eta_rcv to obtain the TOD characteristics
        CSTi['type_rcvr'] = 'TPR_0D'
        eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
        CSTi['eta_rcv'] = eta_rcv_i
        CSTi['Arcv']    = Arcv
        CSTi['rO_TOD']  = rO_TOD
    
        case = 'zf{:.0f}_Q_avg{:.0f}_Prcv{:.1f}'.format(CSTi['zf'], CSTi['Qavg'], CSTi['P_rcv'])
        CSTi['type_rcvr'] = 'TPR_0D'
        R2, SF, CSTo = SPR.Simulation_Coupled(CSTi)

        #Save the results on file    
        date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        CSTo['date_sim'] = date_sim
        if isfile(file_CSTs):
            CSTs = pickle.load(open(file_CSTs,'rb'))
            CSTs.append(CSTo)
            pickle.dump(CSTs,open(file_CSTs,'wb'))
        else:
            CSTs = [CSTo,]
            pickle.dump(CSTs,open(file_CSTs,'wb'))
            
        C   = CSTo['Costs']

        rmin,rmax,S_HB,S_TOD,Arcv,zrc = [CSTo[x] for x in ['rmin','rmax','S_HB','S_TOD','Arcv','zrc']]
        P_SF, Q_max, Q_av, eta_rcv, Qstg, Mstg, t_res = [CSTo[x] for x in ['P_SF', 'Q_max', 'Q_av', 'eta_rcv', 'Q_stg', 'M_p', 't_res']]
        T_p_av = CSTo['T_p'].mean()
        T_p_mx = CSTo['T_p'].max()
        
        Etas = SF[SF['hel_in']].mean()
        hlst = SF[SF.hel_in].index
        N_hel   = len(hlst)
        eta_SF,eta_hbi,eta_BDR  = Etas['Eta_SF'], Etas['Eta_hbi'], Etas['Eta_BDR']
        eta_TOD = Etas['Eta_tdi']*Etas['Eta_tdr']
        eta_StH = eta_SF * eta_rcv
        LCOH = C['LCOH']
        LCOE = C['LCOE']
        land_prod = C['land_prod']
        
        TOD = BDR.TOD_Params({'Type':CSTo['Type'], 'Array':CSTo['Array'],'rO':CSTo['rO_TOD'], 'Cg':CSTo['Cg_TOD']},0.,0.,CSTo['zrc'])
        H_TOD = TOD['H']
        V_stg = C['V_stg']
        H_stg = C['H_stg']
        if isfile(file_df):
            data = pd.read_csv(file_df,index_col=0).values.tolist()
        else:
            data=[]
        
        cols = ('Prcv', 'zf', 'zrc', 'fzv', 'Arcv', 'P_SF', 'rmax', 'N_hel','Q_avg_i', 'eta_rcv_i', 'eta_rcv', 'eta_SF', 'eta_BDR', 'eta_TOD','eta_StH', 'S_HB', 'S_TOD', 'Qstg', 'Q_av', 'Q_max','T_p_mx', 'H_TOD','H_stg','V_stg', 'land_prod', 'LCOH', 'LCOE', 'date')
        data.append([Prcv, zf, zrc, fzv, Arcv, P_SF, rmax, N_hel, Qavg, eta_rcv_i, eta_rcv, eta_SF, eta_BDR, eta_TOD, eta_StH, S_HB, S_TOD, Qstg, Q_av, Q_max, T_p_mx, H_TOD, H_stg, V_stg, land_prod, C['LCOH'], C['LCOE'], date_sim])
        df = pd.DataFrame(data,columns=cols)
        print('\t'.join('{:.3f}'.format(x) for x in data[-1][:-1]))
        
        pd.set_option('display.max_columns', None)
        
        # print(df)
        df.to_csv(file_df)
            
        ##################################
        if plot:
            try:
                ### SOLAR FIELD ##################
                f_s=18
                fig = plt.figure(figsize=(12,8))
                ax1 = fig.add_subplot(111)
                SF2 = SF[SF['hel_in']].loc[hlst]
                N_hel = len(hlst)
                vmin = SF2['Eta_SF'].min()
                vmax = SF2['Eta_SF'].max()
                surf = ax1.scatter(SF2['xi'],SF2['yi'], s=5, c=SF2['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
                cb = fig.colorbar(surf, shrink=0.25, aspect=4)
                cb.ax.tick_params(labelsize=f_s)
                cb.ax.locator_params(nbins=4)
                
                fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']), fontsize=f_s)
                fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
                # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
                ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
                ax1.add_artist(patches.Wedge((0, 0), rmax, 0, 360, width=rmax-rmin,color='C0'))
                for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
                for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
                ax1.grid()
                fig.savefig(fldr_rslt+case+'_SF.png', bbox_inches='tight')
                plt.show()
                plt.close(fig)
                
                # HYPERBOLOID MIRROR
                f_s = 18
                out  = R2[(R2['hel_in'])&(R2['hit_hb'])]
                xmin = out['xb'].min(); xmax = out['xb'].max()
                ymin = out['yb'].min(); ymax = out['yb'].max()
                Nx = 100; Ny = 100
                dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
                Fbin = CSTo['eta_rfl']*Etas['Eta_cos']*Etas['Eta_blk']*(CSTo['Gbn']*CSTo['A_h1']*N_hel)/(1e3*dA*len(out))
                Q_HB,X,Y = np.histogram2d(out['xb'],out['yb'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]],density=False)
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, aspect='equal')
                X, Y = np.meshgrid(X, Y)
                
                vmin = 0
                vmax = (np.ceil(Fbin*Q_HB.max()/10)*10)
                surf = ax.pcolormesh(X, Y, Fbin*Q_HB.transpose(),cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
                ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
                cb = fig.colorbar(surf, shrink=0.25, aspect=4)
                cb.ax.tick_params(labelsize=f_s)
                fig.text(0.77,0.62,r'$Q_{HB}(kW/m^2)$',fontsize=f_s)
                for tick in ax.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
                for tick in ax.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
                
                # from matplotlib import rc
                # rc('text', usetex=True)
                fig.text(0.77,0.35,'Main Parameters',fontsize=f_s-3)
                fig.text(0.77,0.33,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
                fig.text(0.77,0.31,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
                fig.text(0.77,0.29,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
                fig.text(0.77,0.27,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
                
                ax.add_artist(patches.Circle((0.,0.0), rmin, zorder=10, color='black', fill=None))
                ax.add_artist(patches.Circle((0.,0.0), rmax, zorder=10, edgecolor='black', fill=None))
                ax.grid(zorder=20)
                fig.savefig(fldr_rslt+case+'_QHB_upper.png', bbox_inches='tight')
                plt.show()
                plt.close()
                
                ### RECEIVER APERTURE RADIATION
                Nx = 100; Ny = 100
                rO,Cg,zrc,Type,Array = [CSTo[x] for x in ['rO_TOD', 'Cg_TOD', 'zrc', 'Type', 'Array']]
                TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
                N_TOD,V_TOD,rO,rA,x0,y0 = [TOD[x] for x in ['N','V','rO','rA','x0','y0']]
                out   = R2[(R2['hel_in'])&(R2['hit_rcv'])].copy()
                xmin = out['xr'].min(); xmax = out['xr'].max()
                ymin = out['yr'].min(); ymax = out['yr'].max()
                if Array!='N':
                    xmin=min(x0)-rA/np.cos(np.pi/V_TOD) 
                    xmax=max(x0)+rA/np.cos(np.pi/V_TOD)
                    ymin=min(y0)-rA
                    ymax=max(y0)+rA
                
                dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
                Nrays = len(out)
                Fbin  = Etas['Eta_SF'] * (CSTo['Gbn']*CSTo['A_h1']*N_hel)/(1e3*dA*Nrays)
                Q_BDR,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
                Q_BDR = Fbin * Q_BDR
                Q_max = Q_BDR.max()
                
                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(111, aspect='equal')
                X, Y = np.meshgrid(X, Y)
                f_s = 16
                vmin = 0
                vmax = (np.ceil(Q_BDR.max()/100)*100)
                surf = ax.pcolormesh(X, Y, Q_BDR.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
                ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
                cb = fig.colorbar(surf, shrink=0.5, aspect=4)
                cb.ax.tick_params(labelsize=f_s-2)
                
                if Array=='F' or Array=='N':
                    ax.add_artist(patches.Circle((x0[0],y0[0]), rO, zorder=10, color='black', fill=None))
                else:
                    for i in range(N_TOD):
                        radius = rO/np.cos(np.pi/V_TOD)
                        ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
                        radius = rA/np.cos(np.pi/V_TOD)
                        ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
                    
                fig.text(0.77,0.27,'Main Parameters',fontsize=f_s-3)
                fig.text(0.77,0.25,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
                fig.text(0.77,0.23,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
                fig.text(0.77,0.21,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
                fig.text(0.77,0.19,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
                ax.set_title('b) Radiation flux in receiver aperture',fontsize=f_s)
                
                fig.text(0.77,0.70,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
                fig.savefig(fldr_rslt+case+'_radmap_out.png', bbox_inches='tight')
                plt.show()
                plt.close()
                
            except Exception as e:
                print('It was not possible to create figures.')
                print(e)
                
print(df)

#%% OPTIMIZATION OF TWO VARIABLES CONSECUTIVE

#Parameters for CST
CSTi = BDR.CST_BaseCase()
Costs = SPR.Plant_Costs()
CSTi['Costs_i'] = Costs
CSTi['type_rcvr'] = 'TPR_0D'

plot = False
fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
fldr_rslt = 'Optim_Thesis_Chapter/'

file_CSTs = 'Optim_TPR_0D_2var_quick.plk'
# file_df   = 'Optim_HPR_0D_APSRC_quick_v2.csv'
# file_df   = 'Null'
file_df   = 'Optim_TPR_0D_2var_quick.csv'

stime = time.time()

tol = 1e-3        #Solving the non-linear equation
# zfs = np.arange(30,71,10)
zfs   = np.arange(60,71,5)
# zfs   = [50,40,30]
# Qavgs = [1.25]
Qavgs = np.arange(2.0,2.01,0.1)
# Prcvs = np.append(np.arange(10,41,5),np.arange(2,10,2))
# Prcvs = np.arange(15,36,1)

for (Qavg,zf) in [(Qavg,zf) for Qavg in Qavgs for zf in zfs]:
    
    CSTi = BDR.CST_BaseCase()
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    
    CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
    CSTi['zf']    = zf
    CSTi['Qavg']  = Qavg
    
    Prcv_i = zf * 0.35       #Initial guess for Receiver power
    CSTi['P_rcv'] = Prcv_i
    
    #Estimating eta_rcv to obtain the TOD characteristics
    CSTi['type_rcvr'] = 'TPR_0D'
    eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
    CSTi['eta_rcv'] = eta_rcv_i
    CSTi['Arcv'] = Arcv
    CSTi['rO_TOD'] = rO_TOD
    
    ####################################
    #Obtaining initial optimized value for fzv
    print('Obtaining initial optimized value for fzv')
    lbl = 'fzv'
    bracket = (70 , 84, 97)
    res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
    fzv  = res.x/100
    
    ####################################
    #Obtaining initial optimized value for Prcv
    print('Obtaining initial optimized value for Prcv')
    CSTi = BDR.CST_BaseCase()
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
    CSTi['zf']    = zf
    CSTi['fzv']   = fzv
    CSTi['Qavg']  = Qavg
    
    lbl = 'Prcv'
    bracket = (2 , Prcv_i, Prcv_i*2.5)
    res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
    Prcv  = res.x
    CSTi['P_rcv'] = Prcv
    
    ####################################
    #Second optimization on fzv
    print('Second optimization on fzv')
    CSTi['P_rcv'] = Prcv
    CSTi['type_rcvr'] = 'TPR_0D'
    eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
    CSTi['eta_rcv'] = eta_rcv_i
    CSTi['Arcv'] = Arcv
    CSTi['rO_TOD'] = rO_TOD
    lbl = 'fzv'
    bracket = (fzv*100-4, fzv*100, fzv*100+4)
    res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
    fzv  = res.x/100
    ####################################
    #Second optimization for Prcv
    print('Second optimization for Prcv')
    CSTi = BDR.CST_BaseCase()
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
    CSTi['zf']    = zf
    CSTi['fzv']   = fzv
    CSTi['Qavg']  = Qavg
    
    lbl = 'Prcv'
    bracket = (max(Prcv*0.5,2) , Prcv, Prcv+5)
    res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(CSTi,lbl), method='brent', tol=tol)
    Prcv  = res.x
    CSTi['P_rcv'] = Prcv
    
    ####################################
    #Final Situation
    print('Final Simulation')
    CSTi['fzv']   = fzv
    CSTi['P_rcv'] = Prcv
    eta_rcv_i, Arcv, rO_TOD = SPR.Initial_eta_rcv(CSTi)
    CSTi['eta_rcv'] = eta_rcv_i
    CSTi['Arcv'] = Arcv
    CSTi['rO_TOD'] = rO_TOD
    case = 'zf{:.0f}_Q_avg{:.0f}'.format(CSTi['zf'], CSTi['Qavg'])
    CSTi['type_rcvr'] = 'TPR_2D'
    R2, SF, CSTo = SPR.Simulation_Coupled(CSTi)
    
    ####################################
    #Save the results on file    
    date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    CSTo['date_sim'] = date_sim
    if isfile(file_CSTs):
        CSTs = pickle.load(open(file_CSTs,'rb'))
        CSTs.append(CSTo)
        pickle.dump(CSTs,open(file_CSTs,'wb'))
    else:
        CSTs = [CSTo,]
        pickle.dump(CSTs,open(file_CSTs,'wb'))
        
    C   = CSTo['Costs']

    rmin,rmax,S_HB,S_TOD,Arcv,zrc = [CSTo[x] for x in ['rmin','rmax','S_HB','S_TOD','Arcv','zrc']]
    P_SF, Q_max, Q_av, eta_rcv, Qstg, Mstg, t_res = [CSTo[x] for x in ['P_SF', 'Q_max', 'Q_av', 'eta_rcv', 'Q_stg', 'M_p', 't_res']]
    T_p_av = CSTo['T_p'].mean()
    T_p_mx = CSTo['T_p'].max()
    
    Etas = SF[SF['hel_in']].mean()
    hlst = SF[SF.hel_in].index
    N_hel   = len(hlst)
    eta_SF,eta_hbi,eta_BDR  = Etas['Eta_SF'], Etas['Eta_hbi'], Etas['Eta_BDR']
    eta_TOD = Etas['Eta_tdi']*Etas['Eta_tdr']
    eta_StH = eta_SF * eta_rcv
    LCOH = C['LCOH']
    LCOE = C['LCOE']
    land_prod = C['land_prod']
    
    TOD = BDR.TOD_Params({'Type':CSTo['Type'], 'Array':CSTo['Array'],'rO':CSTo['rO_TOD'], 'Cg':CSTo['Cg_TOD']},0.,0.,CSTo['zrc'])
    H_TOD = TOD['H']
    V_stg = C['V_stg']
    H_stg = C['H_stg']
    if isfile(file_df):
        data = pd.read_csv(file_df,index_col=0).values.tolist()
    else:
        data=[]
    
    cols = ('Prcv', 'zf', 'zrc', 'fzv', 'Arcv', 'P_SF', 'rmax', 'N_hel','Q_avg_i', 'eta_rcv_i', 'eta_rcv', 'eta_SF', 'eta_BDR', 'eta_TOD','eta_StH', 'S_HB', 'S_TOD', 'Qstg', 'Q_av', 'Q_max','T_p_mx', 'H_TOD','H_stg','V_stg', 'land_prod', 'LCOH', 'LCOE', 'date')
    data.append([Prcv, zf, zrc, fzv, Arcv, P_SF, rmax, N_hel, Qavg, eta_rcv_i, eta_rcv, eta_SF, eta_BDR, eta_TOD, eta_StH, S_HB, S_TOD, Qstg, Q_av, Q_max, T_p_mx, H_TOD, H_stg, V_stg, land_prod, C['LCOH'], C['LCOE'], date_sim])
    df = pd.DataFrame(data,columns=cols)
    print('\t'.join('{:.3f}'.format(x) for x in data[-1][:-1]))
    
    pd.set_option('display.max_columns', None)
    
    # print(df)
    df.to_csv(file_df)
print(df)