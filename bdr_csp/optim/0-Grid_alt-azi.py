# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:33:19 2022

@author: z5158936
"""
import pandas as pd
import numpy as np
import scipy.optimize as spo
from scipy.integrate import quad
import scipy.interpolate as spi
import scipy.linalg as sla
import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from os.path import isfile
import os
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

newPath = os.path.join(mainDir, '7_Overall_Optim_Dispatch')
sys.path.append(newPath)
import PerformancePowerCycle as PPC

##################################################
def Optical_Efficiencies(CST,R2,SF):
    """
    Function to obtain the optical efficiencies after a BDR simulation.

    Parameters
    ----------
    CST : dict.
        Characteristics of CST plant
    R2 : pandas DataFrame
        Ray dataset AFTER TOD simulation
    SF : pandas DataFrame
        Solar field heliostats AFTER simulations

    Returns
    -------
    SF : pandas DataFrame
        Same as SF input but including efficiencies

    """
    
    Gbn,A_h1,eta_rfl = [CST[x] for x in ['Gbn', 'A_h1', 'eta_rfl']]
    
    SF2 = R2.groupby('hel')[['hel_in','hit_hb','hit_tod','hit_rcv']].sum()
    SF['Eta_att'] = BDR.Eta_attenuation(R2)
    SF['Eta_hbi'] = SF2['hit_hb']/SF2['hel_in']

    SF['Eta_tdi'] = SF2['hit_tod']/SF2['hit_hb']
    SF['Eta_tdr'] = SF2['hit_rcv']/SF2['hit_tod']

    Nr_cpc = R2[R2['hit_rcv']].groupby('hel')['Nr_tod'].mean()
    SF['Eta_tdr'] = SF['Eta_tdr'] * eta_rfl**Nr_cpc
    
    SF['Eta_hel'] = SF['Eta_cos'] * SF['Eta_blk'] * SF['Eta_att'] * eta_rfl*SF['f_sh']
    SF['Eta_BDR'] = (eta_rfl * SF['Eta_hbi']) * (SF['Eta_tdi'] * SF['Eta_tdr'])
    SF['Eta_SF']  = SF['Eta_hel'] * SF['Eta_BDR']
    SF['Q_h1']  = ( SF['Eta_SF']*Gbn*A_h1*1e-6 )
    SF['Q_pen'] = SF['Q_h1']*SF['f_sh']
    SF.sort_values(by='Q_pen',ascending=False,inplace=True)
    return SF

######################################################################
def TPR_2D_model_fixed(Rcvr,CST,R2,SF,TOD,hlst,file_rad,polygon_i=1,full_output=False):
    
    T_ini, T_out, tz, TSM, Prcv, Qavg = [CST[x] for x in ['T_pC', 'T_pH', 'tz', 'TSM', 'P_rcv', 'Q_av']]
    Type, Array, rO, Cg, zrc = [CST[x] for x in ['Type', 'Array','rO_TOD','Cg_TOD','zrc']]
    if CST['TSM'] == 'CARBO':
        rho_b = 1810
    
    x0,y0,V_TOD,N_TOD,H_TOD,rO,rA,Arcv = [TOD[x] for x in ['x0', 'y0','V','N','H','rO','rA','A_rcv']]
    A_rc1  = Arcv/N_TOD
    
    abs_p, em_p, em_w, hcond, Fc, air, T_w, beta = [Rcvr[x] for x in ['abs_p', 'em_p', 'em_w', 'hcond', 'Fc', 'air', 'Tw', 'beta']]
    
    
    
    #### OBTAINING MCRT IN TILTED SURFACE
    R3  = R2[(R2.Npolygon==polygon_i)&R2.hit_rcv].copy()
    R3b = R2[(R2.hit_rcv)].copy()

    #Point of pivoting belonging to the plane
    xp = R3.xr.max() + 0.1
    yp = 0.                     #This is valid for 1st hexagon, if not:  R3.loc[R3.xr.idxmax()].yr
    zp = R3.zr.mean()
    
    #Obtaining the intersection between the rays and the new plane
    kt = (xp - R3.xr) / ( R3.uzr/np.tan(beta) + R3.uxr)
    R3['xt'] = R3.xr + kt * R3.uxr
    R3['yt'] = R3.yr + kt * R3.uyr
    R3['zt'] = R3.zr + kt * R3.uzr
    
    ##############################################
    #### STARTING THE LOOP HERE
    #Initial guess for number of heliostats
    
    P_SF = CSTo['P_SF'] * 1e6
    R3i  = R3[R3.hel.isin(hlst)].copy()
    r_i  = len(R3i)/len(R3b[R3b.hel.isin(hlst)])
    P_SF_i = r_i * P_SF

    # T_av = (T_ini+T_out)/2
    # eta_rcv = SPR.HTM_0D_blackbox( T_av, Qavg, Fc=2.57, air=air)[0]
    # P_SF  = Prcv / eta_rcv
    
    # Q_acc = SF['Q_h1'].cumsum()
    # N_hel = len(hlst)
    
    # Etas = SF[SF['hel_in']].mean()
    # eta_SF = Etas['Eta_SF']
    # eta_hbi = Etas['Eta_hbi']
    # eta_BDR = Etas['Eta_BDR']
    # eta_TOD = Etas['Eta_tdi']*Etas['Eta_tdr']
        
    ###############################################
    #### GETTING THE RADIATION FLUX FUNCTION
    x_fin, x_ini, y_bot, y_top = [Rcvr[x] for x in ['x_fin', 'x_ini', 'y_bot', 'y_top']]
    lims = x_fin,x_ini,y_bot,y_top
    
    Nx=100; Ny=50
    xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
    xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
    xmin,xmax = max(xA.min(),R3i.xt.quantile(0.01)), min(xA.max(),R3i.xt.quantile(0.99))
    ymin,ymax = max(yA.min(),R3i.yt.quantile(0.01)), min(yA.max(),R3i.yt.quantile(0.99))

    dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Ny; dA=dx*dy
    Nrays = len(R3i.xt[(R3i.xt>xmin)&(R3i.xt<xmax)&(R3i.yt>ymin)&(R3i.yt<ymax)])    #Taking out rays that don't hit in receiver
    
    # P_SF = eta_SF * (CST['Gbn']*CST['A_h1']*N_hel)
    P_SF1 = P_SF * r_i
    Fbin    = P_SF_i/(1e3*dA*Nrays)
    Q_rc1,X_rc1, Y_rc1 = np.histogram2d(R3i.xt, R3i.yt, bins=[Nx,Ny], range=[[xmin, xmax], [ymin, ymax]], density=False)
    Q_rc1 = Fbin * Q_rc1
    Q_max = Q_rc1.max()/1000.
    
    ###########################################
    # REDUCING RECEIVER SIZE
    
    Q_avg_min = 250       #[kW/m2] Minimal average radiation allowed per axis
    ymin_corr = y_bot
    ymax_corr = y_top
    
    Q_avg_min = 250       #[kW/m2] Minimal average radiation allowed per axis
    xmin_corr = x_fin
    xmax_corr = x_ini
    
    X_av = np.array([(X_rc1[i]+X_rc1[i+1])/2 for i in range(len(X_rc1)-1)])
    Y_av = np.array([(Y_rc1[i]+Y_rc1[i+1])/2 for i in range(len(Y_rc1)-1)])
    Q_rcf = Q_rc1[(X_av>xmin_corr)&(X_av<xmax_corr),:][:,(Y_av>ymin_corr)&(Y_av<ymax_corr)]
    
    X_rcf = X_av[(X_av>xmin_corr)&(X_av<xmax_corr)]
    Y_rcf = Y_av[(Y_av>ymin_corr)&(Y_av<ymax_corr)]
    
    eta_tilt = Q_rcf.sum()*dA/(P_SF1/1000)
    eta_tilt_rfl = eta_tilt + (1-eta_tilt)*(1-em_w)
    F_corr = eta_tilt_rfl / eta_tilt
    
    Q_rcf = F_corr * Q_rcf
    
    # lims = xmin_corr,xmax_corr,ymin_corr,ymax_corr
    # x_fin=lims[0]; x_ini = lims[1]; y_bot=lims[2]; y_top=lims[3]
    
    ######################################
    # Plotting of resulting radiation map
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect='equal')
    X, Y = np.meshgrid(X_rcf, Y_rcf)
    f_s = 16
    vmin = 0
    vmax = (np.ceil(Q_rcf.max()/100)*100)
    surf = ax.pcolormesh(X, Y, Q_rcf.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Tiled axis (m)',fontsize=f_s);ax.set_ylabel('Cross axis (m)',fontsize=f_s);
    cb = fig.colorbar(surf, shrink=0.5, aspect=4)
    cb.ax.tick_params(labelsize=f_s-2)
    fig.text(0.79,0.72,r'$Q_{in}[kW/m^2]$',fontsize=f_s)
    plt.show()
    fig.savefig(file_rad, bbox_inches='tight')
    plt.close()
    
    # f_Qrc1= spi.RectBivariateSpline(X_rc1[:-1],Y_rc1[:-1],Q_rc1)     #Function to interpolate
    f_Qrc1 = spi.RectBivariateSpline(X_rcf,Y_rcf,Q_rcf)               #Function to interpolate
    
    ##########################################
    #### OBTAINING VIEWFACTOR_FUNCTION
    # lims = (xmin,xmax,ymin,ymax)
    X1,Y1,Z1,F12 = SPR.View_Factor(CST,TOD,polygon_i,lims,xp,yp,zp,beta)
    # f_ViewFactor = spi.interp2d(X1,Y1,F12)
    X2 = X1[0,:]
    Y2 = Y1[:,0]
    f_ViewFactor = spi.interp2d(X2,Y2,F12)
    Fv = F12.mean()
    
    ##########################################
    #### OBTAINING WALL TEMPERATURE
    # T_pavg = (T_ini+T_out)/2.
    # args = (T_pavg,xmax_corr,xmin_corr,ymax_corr,ymin_corr,beta,Fv,em_p,em_w,A_rc1,hcond)
    # T_w = spo.fsolve(SPR.Getting_T_w, T_w, args=args)[0]
    # print(T_w)
    
    
    ###########################################################
    #### OBTAINING RESIDENCE TIME
  
    #Getting initial estimates
    # x_fin=lims[0]; x_ini = lims[1]; y_bot=lims[2]; y_top=lims[3]
    A_rc1  = Arcv/N_TOD
    xavg   = A_rc1/(y_top-y_bot)
    
    T_av = 0.5*(T_ini+T_out)     #Estimating Initial guess for t_res
    Q_av = P_SF/1e6 / Arcv         #Estimated initial guess for Qavg
    cp = 148*T_av**0.3093
    eta_rcv_g = SPR.eta_th_tilted(Rcvr,T_av,Q_av,Fv=Fv)[0]
    t_res_g = rho_b * cp * tz * (T_out - T_ini ) / (eta_rcv_g * Q_av*1e6)
    t_res = t_res_g
    
    def F_Toutavg(t_res_g,*args):
        Rcvr,f_Qrc1,f_eta,f_ViewFactor = args
        Rcvr['vel_p']  = Rcvr['xavg']/t_res_g                           #Belt velocity magnitud
        Rcvr['t_sim']  = (Rcvr['x_ini']-Rcvr['x_fin'])/Rcvr['vel_p']    #Simulate all the receiver
        T_p, Qstg1, M_stg1 = SPR.HTM_2D_tilted_surface(Rcvr,f_Qrc1,f_eta,f_ViewFactor)
        # print(T_p.mean())
        return T_p.mean() - Rcvr['T_out']
    
    vel_p  = xavg/t_res_g                #Belt velocity magnitud
    t_sim  = (x_ini-x_fin)/vel_p         #Simulate all the receiver
    # inputs = (x_ini,x_fin,y_bot,y_top,vel_p,t_sim,tz,T_ini,TSM,f_Qrc1,f_eta)
    Rcvr['xavg']  = xavg
    Rcvr['vel_p'] = vel_p
    Rcvr['t_sim'] = t_sim
    
    args = (Rcvr, f_Qrc1, SPR.eta_th_tilted,f_ViewFactor)
    sol  = spo.fsolve(F_Toutavg, t_res_g, args=args, xtol=1e-2,full_output=True)
    t_res = sol[0][0]
    n_evals = sol[1]['nfev']
    CST['t_res'] = t_res
    CST['vel_p'] = xavg/t_res
    
    ###################################################
    #Final values for temperature Qstg1 and Mstg1
    vel_p  = xavg/t_res                #Belt velocity magnitud
    t_sim  = (x_ini-x_fin)/vel_p           #Simulation [s]
    Rcvr['xavg']  = xavg
    Rcvr['vel_p'] = vel_p
    Rcvr['t_sim'] = t_sim
    T_p,Qstg1,Mstg1,Rcvr_full = SPR.HTM_2D_tilted_surface(Rcvr, f_Qrc1, SPR.eta_th_tilted, f_ViewFactor, full_output=True)
    Rcvr_full['Q_gain'] = Rcvr_full['Q_in']*Rcvr_full['eta']      #[J]
    
    #Performance parameters
    eta_rcv = Qstg1*1e6/P_SF_i
    Qstg = Qstg1 / r_i
    Mstg = Mstg1 / r_i
    Q_av = P_SF /1e6/ Arcv
    Q_max = Q_rc1.max()/1000.
    
    # print(P_SF,N_new,eta_rcv,Qstg,vel_p)
    results = [ N_hel, Q_max, Q_av, P_SF_i, Qstg, P_SF, eta_rcv, Mstg, t_res, vel_p]
    return results, T_p, Rcvr, Rcvr_full

#%% DEFINING DESIGN CONDITIONS AND CHARGING BASE CASE

#Fixed values
zf   = 50.
Prcv = 19.
Qavg = 1.25
fzv  = 0.818161

# lats = [-25,-10,-20,-30,-40]
lats = [-50]

# lat = -20.

for lat in lats:
    lng = 115.9
    
    CSTi = BDR.CST_BaseCase()
    Costs = SPR.Plant_Costs()
    CSTi['Costs_i'] = Costs
    fldr_data    = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final\SkyGrid_50m')
    
    if lat == -25:
        CSTi['file_SF'] = os.path.join(fldr_data,'Base_height-{:.0f}'.format(zf))
        CSTi['lat'] = lat
    else:
        CSTi['lat'] = lat
        CSTi['lng'] = lng
        alt_des = 90+lat
        azi_des = 0.
        CSTi['file_SF'] = os.path.join(fldr_data,'alt-{:.0f}_azi-{:.0f}'.format(alt_des,azi_des))
    
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
    CSTi['type_rcvr'] = 'TPR_0D'
    
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
    
    results, T_p, Rcvr, Rcvr_full = SPR.TPR_2D_model(CSTo, R2, SF, TOD, full_output=True)
    N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, M_p, t_res, vel_p, it, Solve_t_res = results
    
    print(Rcvr)
    ######################################################
    
    fldr_rslt = 'SkyGrid/'
    # file_df = fldr_rslt+'1-Grid_AltAzi_{:.0f}_vF.csv'.format(abs(lat))
    file_df = fldr_rslt+'1-Grid_AltAzi_vF.csv'
    plot=True
    
    # sys.exit()
    # sys.exit()
    
    ######################################################
    #%% CALCULATIONS
    ######################################################
    alts = np.arange(10,91,10)
    azis = np.append(np.arange(0,31,5),np.arange(45,121,15))
    for (alt,azi) in [(alt,azi) for alt in alts for azi in azis]:
        # break
        case = 'Lat_{:.0f}_Alt_{:d}-Azi_{:d}'.format(abs(lat),alt,azi)
        
        #Getting all rays with direction cosines
        file_SF = fldr_data+'/alt-{:d}_azi-{:d}'.format(alt,azi)
        
        R0, SF = BDR.Rays_Dataset( file_SF, read_plk=True, save_plk=True )
        R0 = R0[R0['hel'].isin(hlst)]
        R1 = BDR.HB_direct( R0 , CSTo, Refl_error=False)
        R1['hel_in'] = R1['hel'].isin(hlst)
        R1['hit_hb'] = (R1['rb']>rmin)&(R1['rb']<rmax)
        
        #Shadowing
        # SF = BDR.Shadow_point(alt,azi,CSTo,SF)
        SF = BDR.Shadow_point_ellipse(alt,azi,CSTo,SF)
        
        #Interceptions with TOD
        R2 = BDR.TOD_NR(R1,TOD,CSTo, Refl_error=False)
        
        #Calculating efficiencies
        SF = Optical_Efficiencies(CSTo,R2,SF)
        
        SF['hel_in'] = SF.index.isin(hlst)
        R2['hel_in'] = R2['hel'].isin(hlst)
        
        Etas = SF[SF['hel_in']].mean()
        eta_SF = Etas['Eta_SF']
        eta_hbi = Etas['Eta_hbi']
        eta_BDR = Etas['Eta_BDR']
        eta_TOD = Etas['Eta_tdi']*Etas['Eta_tdr']
        
        Q_max_BDR = Q_max
        file_rad = fldr_rslt+case+'_TPR_Qin.png'      
        results, T_p, Rcvr, Rcvr_full = TPR_2D_model_fixed(Rcvr, CSTo, R2, SF, TOD, hlst, file_rad, polygon_i=1, full_output=False)
        _, Q_max, Q_av, _, Qstg, _, eta_rcv, Mstg, t_res, vel_p = results
        print(results)
        
        if plot:
            ### RECEIVER APERTURE RADIATION
            Nx = 100; Ny = 100
            TOD = BDR.TOD_Params( {'Type':Type,'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
            N_TOD,V_TOD,rO,rA,x0,y0 = [TOD[x] for x in ['N','V','rO','rA','x0','y0']]
            out   = R2[(R2['hel_in'])&(R2['hit_rcv'])].copy()
            xmin = out['xr'].min(); xmax = out['xr'].max()
            ymin = out['yr'].min(); ymax = out['yr'].max()
            xmin=min(x0)-rA/np.cos(np.pi/V_TOD) 
            xmax=max(x0)+rA/np.cos(np.pi/V_TOD)
            ymin=min(y0)-rA
            ymax=max(y0)+rA
            
            dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
            Nrays = len(out)
            Fbin    = Etas['Eta_SF'] * (CSTo['Gbn']*CSTo['A_h1']*N_hel)/(1e3*dA*Nrays)
            Q_BDR,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
            Q_BDR = Fbin * Q_BDR
            Q_max_BDR = Q_BDR.max()
            
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
            
            if Array=='F':
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
            # plt.show()
            plt.close()
            
            ### SOLAR FIELD ##################
            f_s=18
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(111)
            SF2 = SF[SF['hel_in']].loc[hlst]
            N_hel = len(hlst)
            vmin = SF2['Eta_SF'].min()
            vmax = SF2['Eta_SF'].max()
            surf = ax1.scatter(SF2['xi'],SF2['yi'], s=5, c=SF2['Eta_SF']*SF2['f_sh'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
            cb = fig.colorbar(surf, shrink=0.25, aspect=4)
            cb.ax.tick_params(labelsize=f_s)
            cb.ax.locator_params(nbins=4)
            
            fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']), fontsize=f_s)
            fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
            # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
            ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
            for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
            for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
            ax1.grid()
            fig.savefig(fldr_rslt+case+'_SF.png', bbox_inches='tight')
            # plt.show()
            plt.close(fig)
    
        ########### WRITING RESULTS ###############
        if isfile(file_df):
            data = pd.read_csv(file_df,index_col=0).values.tolist()
        else:
            data=[]
        
        # cols = ['lat','alt', 'azi', 'rO', 'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 'eta_tdi', 'eta_tdr', 'eta_BDR', 'eta_SF', 'Qmax']
        # data.append([lat, alt, azi, rO, Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_tdi'], Etas['Eta_tdr'], Etas['Eta_BDR'], Etas['Eta_SF'], Q_max])
        
        cols = ['lat','alt', 'azi', 'rO', 'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 'eta_tdi', 'eta_tdr', 'eta_BDR', 'eta_SF', 'Qmax_BDR', 'Qmax', 'eta_rcv','Prcv_abs']
        data.append([lat, alt, azi, rO, Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_tdi'], Etas['Eta_tdr'], Etas['Eta_BDR'], Etas['Eta_SF'], Q_max_BDR, Q_max, eta_rcv, Qstg])
        df = pd.DataFrame(data,columns=cols)
        df.to_csv(file_df)
        print('\t'.join('{:.3f}'.format(x) for x in data[-1]))
    
    ######################################
    #%% PLOTING MAP
    
    # df = pd.read_table(file_rslt,sep='\t',header=0)
    # file_df = fldr_rslt+'1-Grid_AltAzi_25_vF.csv'
    # lat=-40
    # file_df = fldr_rslt+'1-Grid_AltAzi_{:.0f}_vF.csv'.format(abs(lat))
    
fldr_rslt = 'SkyGrid/'
file_df = fldr_rslt+'1-Grid_AltAzi_vF.csv'
lng = 115.9

for lat in [0,-10,-20,-30,-40,-50]:
    df = pd.read_csv(file_df,header=0,index_col=0)
    df2 = df[df.lat==lat]
    df2.sort_values(by=['alt','azi'],axis=0,inplace=True)
    print(df2)
    
    for lbl in ['eta_cos','eta_BDR','eta_SF','eta_rcv']:
    # for lbl in ['eta_cos','eta_BDR','eta_SF']:
    # for lbl in ['eta_SF']: 
    # lbl='eta_SF'
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        f_s = 16
        
        alt, azi = df2['alt'].unique(), df2['azi'].unique()
        eta = np.array(df2[lbl]).reshape(len(alt),len(azi))
        f_eta = spi.interp2d(azi, alt, eta, kind='cubic')
        
        alt2,azi2 = np.arange(0,90,0.1), np.arange(0,130,0.1)
        eta2 = f_eta(azi2,alt2)
        azi2,alt2=np.meshgrid(azi2,alt2)
        
        azi,alt = np.meshgrid(azi,alt)
        
        if lbl == 'eta_SF':
            vmin = 0.50
            vmax = 0.64
            lvs = 15
            cbar_label = 'a) Solar Field efficiency'
        elif lbl == 'eta_cos':
            vmin = 0.62
            vmax = 0.82
            lvs = 11
            cbar_label = 'b) Cosine efficiency'
        elif lbl == 'eta_BDR':
            vmin = 0.80
            vmax = 0.86
            lvs = 13
            cbar_label = 'c) BDR efficiency'
        elif lbl == 'eta_rcv':
            vmin = 0.80
            vmax = 0.85
            lvs = 11
            cbar_label = 'Receiver efficiency'
        else:
            vmin=np.floor(df2[lbl].min()*10)/10
            vmax=np.ceil(df2[lbl].max()*10)/10
            lvs = 11
            cbar_label = 'Efficiency'
        
        # vmin=0.38;vmax=0.50
        levels = np.linspace(vmin, vmax, lvs)
        mapx = ax.contour(azi2, alt2, eta2, colors='w', levels=levels , vmin=vmin , vmax=vmax,extend='both')
        ax.clabel(mapx,mapx.levels,inline=True,fmt='%.2f',fontsize=f_s)
        
        mapp = ax.contourf(azi2, alt2, eta2, levels=levels , vmin=vmin , vmax=vmax, cmap=cm.viridis,extend='both')
        
        ax.set_xlabel('Azimuth (dgr)',fontsize=f_s)
        ax.set_ylabel('Elevation (dgr)',fontsize=f_s)
        
        xmin, xmax, ymin, ymax = 0,120,10,90
        
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xticks(np.arange(xmin,xmax+1,10))
        ax.set_yticks(np.arange(ymin,ymax+1,10))
        ax.set_xticklabels(labels=np.arange(xmin,xmax+1,10), fontsize=f_s)
        ax.set_yticklabels(labels=np.arange(ymin,ymax+1,10), fontsize=f_s)
        
        # plt.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.86, 0.10, 0.05, 0.6])
        # cbar = plt.colorbar(mapp,cax=cbar_ax)
        # cbar_ax.set_yticklabels(labels=levels, fontsize=f_s-2)
        # cbar_ticks = np.linspace(vmin,vmax,int(lvs/2)+1)
        # plt.colorbar(mapp,cax=cbar_ax,ticks=cbar_ticks)
        # cbar.set_label(cbar_label, rotation=-270, fontsize=f_s+2)

        
        # lat = -25
        from pvlib.location import Location
        tdelta = .1
        tz = 'Australia/Sydney'
        days = [80,172,355]
        days_length = ['2021-03-21','2021-06-21','2021-12-21']
        lss = [':','--','-']
        lbls = ['Eqquinox','Winter Solstice', 'Summer Solstice']
        
        for i in range(len(days)):
            times = pd.date_range(days_length[i],days_length[i]+' 23:59:00', tz=tz, freq=str(tdelta)+'H')
            sol_pos = Location(lat, lng, tz=tz ).get_solarposition(times)
            sol_pos['azimuth'] = np.where(sol_pos.azimuth>180.,sol_pos.azimuth-360,sol_pos.azimuth)
            # sol_pos.sort_values('azimuth',inplace=True)
            sol_pos = sol_pos[sol_pos.elevation>0]
            
            if abs(lat)<24:
                sol_pos = sol_pos[sol_pos.azimuth>0]
            
            ele  = sol_pos.elevation
            azi  = sol_pos.azimuth
            # print(ele)
            # print(azi)
            N = days[i]
            ax.plot(azi,ele,lw=3, c='tab:red', ls=lss[i], label=lbls[i])
            
        ax.scatter(0,90.+lat,c='darkred',s=120,label='Design Point',zorder=10)
        
        ax.grid()
        
        if lbl == 'eta_SF':
            ax.legend(loc=3, bbox_to_anchor=(0.00, -0.3),fontsize=f_s-2)
            cbar_ax = fig.add_axes([0.40, 0.00, 0.5, 0.05])
        else:
            cbar_ax = fig.add_axes([0.20, 0.00, 0.6, 0.05])
        cbar_ticks = np.linspace(vmin,vmax,int(lvs/2)+1)
        cbar = plt.colorbar(mapp,cax=cbar_ax,ticks=cbar_ticks, orientation='horizontal')
        cbar.ax.tick_params(labelsize=f_s+2,labelrotation=20)
        # cbar.set_label(cbar_label, fontsize=f_s+2)
        
            
        fig.savefig(fldr_rslt+'Fig_lat_{:.0f}_azi-alt_{}.png'.format(abs(lat),lbl), bbox_inches='tight')
        plt.show()