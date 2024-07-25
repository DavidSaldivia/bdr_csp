# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:21:17 2022

@author: z5158936
"""

import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy.interpolate as spi
import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import os
import pickle
import time
import sys

from bdr_csp import BeamDownReceiver as BDR
from bdr_csp import SolidParticleReceiver as SPR

# OBTAIN RESULTS FROM OPTICAL SIMULATION AND GENERATE RADIATION FLUX
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
        Tp_avg = (CST['T_pC']+CST['T_pH'])/2
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
    _, SF, CST = SPR.run_coupled_simulation(CST)
    
    #Objective function and penalisations
    zf, fzv, Prcv, Arcv, N_hel, eta_rcv, P_rcv_sim, Qstg, Q_av, Q_max,T_p, C  = [CST[x] for x in ['zf','fzv','P_rcv','Arcv', 'N_hel', 'eta_rcv', 'P_rcv_sim', 'Q_stg', 'Q_av', 'Q_max','T_p', 'Costs']]
    eta_SF   = SF[SF.hel_in]['Eta_SF'].mean()
    # R_pen1 =  1 + np.abs(1-np.round(Prcv_sim/CST['P_rcv'],2))*100
    # R_pen2 =  1 + np.floor(Q_max/Q_mx_lim)*10
    fobj = C['LCOH']
    
    # print('{:.4f}\t{:.4f}\t'.format(fobj,X)+ '\t'.join('{:.2f}'.format(x) for x in [zf, Prcv, fzv, Arcv,time.time()-stime_i, N_hel, eta_SF, eta_rcv, Prcv_sim, Qstg, Q_av, Q_max,T_p.max(), C['LCOH'], C['LCOE']])+'\t'+Array)
    return fobj



#####################################################
#%% CHARGING BASE CASE DATA

def loading_basecase():
    fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
    fldr_rslt = 'Results_TPR/'
    tol = 1e-4        #Solving the non-linear equation
    zf = 50.
    Qavg = 1.25
    Prcv = 20.

    case = 'case_zf{:.0f}_Q_avg{:.1f}_Prcv{:.1f}'.format(zf, Qavg, Prcv)
    file_BaseCase = 'Cases/'+case+'.plk'
    [CST,R2,SF,TOD] = pickle.load(open(file_BaseCase,'rb'))
    return [CST,R2,SF,TOD]

def running_receiver_function(CST, R2, SF, TOD):

    results, T_p, Rcvr, Rcvr_full = SPR.TPR_2D_model(CST, R2, SF, TOD)
    Rcvr.pop('air')
    # pickle.dump([results, T_p, Rcvr, Rcvr_full],open('Results_TPR_basecase.plk','wb'))

    # PLOTTING THE TEMPERATURE DISTRIBUTION AT THE END
    Tp_out = Rcvr_full['Tp'][-1,:].copy()
    Y_out = Rcvr_full['y'][-1,:]

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    f_s = 18
    markers=['o','s','v','d','*','H','P']
    ax.scatter(Y_out,Tp_out,s=30)
    ax.set_xlabel('Position at receiver outlet (m)',fontsize=f_s);
    ax.set_ylabel('Outlet particle temperature (K)',fontsize=f_s);
    ax.tick_params(axis='both', which='major', labelsize=f_s-2)
    ax.grid()
    fig.savefig(fldr_rslt+'Temp_distribution_Yaxis.png', bbox_inches='tight')
    plt.show()

    Tp_out.sort()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    f_s = 18
    markers=['o','s','v','d','*','H','P']
    ax.scatter(np.linspace(0,100,len(Tp_out)),Tp_out,s=30)
    ax.set_xlabel('Percentile (%)',fontsize=f_s);
    ax.set_ylabel('Outlet particle temperature (K)',fontsize=f_s);
    ax.tick_params(axis='both', which='major', labelsize=f_s-2)
    ax.grid()
    fig.savefig(fldr_rslt+'Temp_distribution_acuum.png', bbox_inches='tight')
    plt.show()

    print(Rcvr)
    lims = Rcvr['x_ini'], Rcvr['x_fin'],Rcvr['y_bot'], Rcvr['y_top'] 

    Rcvr_full['Q_gain'] = Rcvr_full['Q_gain']*1000.

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
            lbl_cb = r'$\eta_{th}[-]$'
        elif lbl =='Tp':
            vmin = (np.floor(Z.min()/100)*100)
            vmax = (np.ceil(Z.max()/100)*100)
            cmap = cm.viridis
            cmap = cm.YlOrRd
            lbl_cb = r'$T_{p}[K]$'
        elif lbl =='Q_gain':
            vmin = 0.0
            vmax = (np.ceil(Z.max()/100)*100)
            cmap = cm.YlOrRd
            lbl_cb = r'$Q_{gain}[kW/m^2]$'
        elif lbl =='Q_abs':
            vmin = 0.0
            vmax = (np.ceil(Z.max()/100)*100)
            cmap = cm.YlOrRd
            lbl_cb = r'$Q_{abs}[J]$'
        elif lbl == 'Fv':
            vmin = 0.0
            vmax = np.ceil(Z.max())
            cmap = cm.YlOrRd
            lbl_cb = r'$F_{v}[-]$'
            
        # surf = ax.tricontourf(X.flatten(), Y.flatten(), Z.flatten(), cmap=cmap, vmin=vmin, vmax=vmax)
        surf = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
        cb = fig.colorbar(surf, shrink=0.5, aspect=4)
        cb.ax.tick_params(labelsize=f_s-2)
        
        fig.text(0.80,0.73,lbl_cb,fontsize=f_s)
        ax.tick_params(axis='both', which='major', labelsize=f_s-2)
        # xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        # xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        # lp = len(xA)//2
        # fyO = spi.interp1d(xO[:lp],yO[:lp],bounds_error=False,fill_value=0)
        # yO2 = fyO(xA[:lp])
        # ax.fill_between(xA[:lp],yO2[:lp],2*yA[:lp], color='w')
        # ax.fill_between(xA[:lp],-yO2[:lp],-2*yA[:lp], color='w')
        
        # i = polygon_i-1
        # x0f = x0[i]
        # y0f = y0[i]
        # radius = rO/np.cos(np.pi/V_TOD)
        # ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
        # radius = rA/np.cos(np.pi/V_TOD)
        # ax.add_artist(patches.RegularPolygon((x0f,y0f), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
        
        ax.set_xlim(lims[1],lims[0])
        ax.set_ylim(lims[2],lims[3])
        fig.savefig(fldr_rslt+case+'_'+lbl+'.png', bbox_inches='tight')
        plt.show()
        plt.close()

def parametric_study():
    stime = time.time()
    #Parameters for CST
    CSTi = BDR.CST_BaseCase()

    #Plant related costs
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    CSTi['type_rcvr'] = 'HPR_0D'

    fldr_data = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
    fldr_rslt = 'Results_TPR/'
    tol = 1e-4        #Solving the non-linear equation
    zf = 50.
    Qavg = 1.0
    Prcv = 20.

    Qavgs = np.array([0.5,0.75,1.0,1.25,1.5])
    Prcvs = np.arange(5,41,5)
    # tzs   = np.arange(0.01,0.06,0.01)
    tzs   = np.array([0.05])
    # Qavgs = np.array([1.5])
    # Prcvs = np.array([20])
    data = []

    for (Qavg,Prcv,tz) in [(Qavg,Prcv,tz) for Qavg in Qavgs for Prcv in Prcvs for tz in tzs]:
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
        ## RECEIVER FUNCTION ITSELF
        ## Input: CST, SF, R2
        ## Output: N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, Mstg, t_res, vel_p
        
        CSTo['tz'] = tz
        results, T_p, Rcvr, Rcvr_full = SPR.TPR_2D_model(CSTo, R2, SF, TOD, full_output=True)
        N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, Eta_rcv, Mstg, t_res,vel_p, it, Solve_t_res = results
        
        Eta_SF = SF.iloc[:N_hel].mean()['Eta_SF']
        Eta_hel = SF.iloc[:N_hel].mean()['Eta_hel']
        Eta_BDR = SF.iloc[:N_hel].mean()['Eta_BDR']
        Eta_ThS = Eta_SF * Eta_rcv
        # print(N_hel,Q_max,Q_av,eta_rcv,t_res,vel_p)
        Tp_max,Tp_min,Tp_std = T_p.max(),T_p.min(),T_p.std()
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
        
        cols = ['zf', 'Prcv', 'Qavg', 'tz', 'N_hel', 'Arcv', 'Q_max', 'Q_av', 'Qstg', 't_res', 'vel_p', 'Eta_hel', 'Eta_BDR', 'Eta_SF', 'Eta_rcv', 'Eta_ThS','Mstg','Tp_avg','it','Solve_t_res','Tp_max','Tp_min','Tp_std']
        
        row = [zf, Prcv, Qavg, tz, N_hel, Arcv, Q_max, Q_av, Qstg, t_res, vel_p, Eta_hel, Eta_BDR, Eta_SF, Eta_rcv, Eta_ThS,Mstg,T_p.mean(), it,Solve_t_res,Tp_max,Tp_min,Tp_std]
        data.append(row)
        print('\t'.join('{:.2f}'.format(x) for x in row))
        ######################################################
        
        ############################################################
        # Plotting
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

def plotting_results_zf():
    zf=50.
    df_res = pd.read_csv(fldr_rslt+'0_Results_zf_{:.0f}.csv'.format(zf),index_col=0)
    fs=16
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    i=0
    Qavgs = df_res.Qavg.unique()
    cs = ['C0','C1','C2','C3','C4']
    ms=['o','s','v','d','*','H','P']

    for qi in Qavgs[:-1]:
        df2 = df_res[df_res.Qavg==qi]
        ax1.plot(df2.Prcv,df2.Eta_rcv,lw=3,ls='--',marker=ms[i],markersize=10,c=cs[i])
        ax1.plot(df2.Prcv,df2.Eta_SF,lw=3,ls=':',marker=ms[i],markersize=10,c=cs[i])
        ax1.plot(df2.Prcv,df2.Eta_ThS,lw=3,ls='-',marker=ms[i],markersize=10,c=cs[i],label=r'$Q_{{avg}}={:.2f}$'.format(qi))
        i+=1
    ax2.plot([],[],lw=3,ls='--',c='grey',label=r'$\eta_{rcv}$')
    ax2.plot([],[],lw=3,ls=':',c='grey',label=r'$\eta_{SF}$')
    ax2.plot([],[],lw=3,ls='-',c='grey',label=r'$\eta_{StH}$')

    ax1.plot([20.,20.],[0.4,0.9],lw=3,ls=':',c='gray')
    ax1.annotate("Base\ncase",(19.05,0.41),rotation='vertical', fontsize=fs-4)

    ax2.set_yticks([],[])
    ax1.set_ylim(0.4,0.9)
    ax1.annotate(r'$\eta_{rcv}$',(10,0.87),c='black', fontsize=fs)
    ax1.annotate(r'$\eta_{SF}$',(20,0.68),c='black', fontsize=fs)
    ax1.annotate(r'$\eta_{StH}$',(14,0.48),c='black', fontsize=fs)

    ax1.add_patch(patches.Ellipse((13,0.82),3,0.11,fill=False,ls='--',lw=2))
    ax1.add_patch(patches.Ellipse((23,0.64),3,0.11,fill=False,ls=':',lw=2))
    ax1.add_patch(patches.Ellipse((17,0.51),2,0.08,fill=False,ls='-',lw=2))

    ax1.set_xlabel(r'Receiver Nominal Power ($MW_{th}$)', fontsize=fs)
    ax1.set_ylabel('Efficiency (-)', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=0, bbox_to_anchor=(1.27, 0.95),fontsize=fs-2)
    ax2.legend(loc=0, bbox_to_anchor=(1.18, 0.6),fontsize=fs-2)
    ax1.grid()
    fig.savefig(fldr_rslt+'Efficiencies.png', bbox_inches='tight')
# sys.exit()

def comparison_with_0D_model():
    file_tpr = fldr_rslt+'0_Results_zf_{:.0f}.csv'.format(zf)
    df_res = pd.read_csv(file_tpr,index_col=0)

    Ly    = 3
    TSM = 'CARBO'
    abs_p = 0.91
    em_p  = 0.85
    em_w  = 0.20
    hcond = 0.833
    T_pC  = 950.
    T_pH = 1200.
    T_w   = 950.
    Fv    = 0.4
    Fc    = 2.57
    beta  = -27.
    tz = 0.05
    air = air=ct.Solution('air.xml')
    df_res['eta_0D_TPR'] = np.nan
    df_res['eta_0D_HPR'] = np.nan

    for idx,row in df_res.iterrows():
        
        Tp_avg = (T_pC + T_pH)/2.
        
        Arcv = row['Arcv']
        Q_avg = row['Qavg']
        
        x_ini = Arcv/Ly
        x_fin = 0.
        y_bot = -Ly/2
        y_top =  Ly/2
        
        
        Rcvr = {'A_rc1':Arcv,
                'em_p':em_p, 'em_w':em_w, 'abs_p':abs_p, 'hcond':hcond,
                'Fc':Fc, 'Fv_avg':Fv, 'Tw':T_w, 'air':air,
                'beta':beta,'x_ini':x_ini,'x_fin':x_fin, 'y_bot':y_bot, 'y_top':y_top, 'tz':tz
                }
        eta1 = SPR.eta_th_tilted(Rcvr, Tp_avg, Q_avg, Fv=Fv)[0]
        eta2 = SPR.HTM_0D_blackbox(Tp_avg, Q_avg,Fc=Fc)[0]
        
        # sys.exit()
        rho_p = 1810.
        c_p = 148*Tp_avg**0.3093
        M_p = Prcv*1e6 / (c_p*(T_pH-T_pC))
        vel_p = M_p / (rho_p * Ly * tz)
        df_res.loc[idx,'eta_0D_TPR'] = eta1
        df_res.loc[idx,'eta_0D_HPR'] = eta2
        # print(row)
    print(df_res)

    from scipy import stats
    X = df_res.eta_0D_TPR
    Y = df_res.Eta_rcv
    m, n, r2, p_value, std_err = stats.linregress(X,Y)
    X_reg = np.linspace(X.min(),X.max(),100)
    Y_reg = m*X_reg+n

    fig = plt.figure(figsize=(14,8), constrained_layout=True)
    ax1 = fig.add_subplot(121,aspect='equal')
    fs=18

    ms=['o','s','v','d','*','H','P']
    i=0
    cmap = cm.viridis
    for Qavg in df_res.Qavg.unique():
        df_aux = df_res[df_res.Qavg==Qavg]
        surf = ax1.scatter(df_aux.eta_0D_TPR, df_aux.Eta_rcv, c=df_aux.Prcv, marker=ms[i], s=100, cmap=cmap, label=r'{:.2f}$MW/m^2$'.format(Qavg))
        i+=1

    ax1.plot([0,1],[0,1],lw=3,c='grey',ls=':')
    ax1.plot(X_reg,Y_reg,lw=3,c='mediumblue',ls='--',label='y={:.2f}x+{:.2f}'.format(m,n))
    ax1.set_xlim(0.78,0.88)
    ax1.set_ylim(0.78,0.88)

    cb = fig.colorbar(surf, shrink=0.3, aspect=6)
    cb.ax.tick_params(labelsize=fs-2)
    fig.text(0.50,0.68,r'$P_{rcv}[MW_{th}]$',fontsize=fs-2)

    ax1.set_xlabel('TPR 0D Model Efficiency (-)', fontsize=fs)
    ax1.set_ylabel('TPR 2D Model Efficiency (-)', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=2,fontsize=fs-4)
    ax1.grid()
    fig.savefig(fldr_rslt+'TPR_0D-2D_Comparison.png', bbox_inches='tight')
    plt.show()


def influence_of_particle_tickness():
    df_rcv = []
    for tz in np.arange(0.01,0.11,0.01):
        CSTo['tz'] = tz
        results, T_p, Rcvr, Rcvr_full = SPR.TPR_2D_model(CSTo, R2, SF, TOD, full_output=True)
        N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, Eta_rcv, Mstg, t_res,vel_p, it, Solve_t_res = results
        df_rcv.append([tz,N_hel,Q_max,Q_av,Eta_rcv,t_res,vel_p,T_p.max(),T_p.min(),T_p.std()])
        
        print('\t'.join('{:.2f}'.format(x) for x in df_rcv[-1]))
        
    df_rcv = pd.DataFrame(df_rcv,columns=['tz','N_hel','Q_max','Q_av','eta_rcv','t_res','vel_p', 'Tp_max','Tp_min','Tp_std'])

    #Relationship for thickness and flow
    h_granflow = np.arange(0.01,0.11,0.01)
    C1 = 5.2
    vel_granflow1 = np.array([C1*h**1.5 for h in h_granflow])

    # PLOTTING INFLUENCE OF THICKNESS
    fs=16
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    ax1.plot(df_rcv['tz'],df_rcv['vel_p'],ls='-',lw=3,marker='s',markersize=10, label=r'$v_p$ from TPR model')
    ax1.plot(h_granflow,vel_granflow1, ls='--',lw=3,marker='o',markersize=10, label=r'$v_p$ from granular flow')
    ax1.set_xlabel(r'Particle layer thickness, $t_{z} [m]$',fontsize=fs)
    ax1.set_ylabel(r'Particle flow velocity, $v_{p} [m/s]$',fontsize=fs)
    ax1.legend(loc=1,fontsize=fs)
    # ax1.tick_params(axis='y', colors='C0')
    # ax1.yaxis.label.set_color('C0')
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.grid()
    # plt.scatter(h_granflow,vel_granflow2)
    plt.show()

    plt.scatter(df_rcv['tz'],df_rcv['t_res'])
    plt.show()

    plt.scatter(df_rcv['tz'],df_rcv['eta_rcv'])
    plt.show()
    plt.scatter(df_rcv['tz'],df_rcv['Tp_std'])
    plt.show()
    plt.scatter(df_rcv['tz'],df_rcv['Tp_max'])
    plt.scatter(df_rcv['tz'],df_rcv['Tp_min'])
    fig.savefig(fldr_rslt+'vel_p_Vs_thickness.png', bbox_inches='tight')
    plt.show()