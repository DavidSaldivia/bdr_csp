# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:00:57 2020

@author: z5158936
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from importlib import reload
import gc
import os
import sys
import bdr_csp.BeamDownReceiver as BDR

absFilePath = os.path.abspath(__file__)
fileDir   = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)
# sys.path.append(newPath)

# sys.exit()

def CST_BaseCase_Paper(**kwargs):
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. Anyone can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    ############### STANDARD CASE ####################
    ##################################################
    # Environment conditions
    CST['Gbn']   = 950                # Design-point DNI [W/m2]
    CST['day']   = 80                 # Design-point day [-]
    CST['omega'] = 0.0                # Design-point hour angle [rad]
    CST['lat']   = -23.               # Latitude [°]
    CST['lng']   = 115.9              # Longitude [°]
    CST['T_amb'] = 300.               # Ambient Temperature [K]
    ##################################################
    # Receiver and Power Block
    CST['P_el']    = 10.0               #[MW] Target for Net Electrical Power
    CST['eta_pb']  = 0.50               #[-] Power Block efficiency target 
    CST['eta_sg']  = 0.95               #[-] Storage efficiency target
    CST['eta_rcv'] = 0.75               #[-] Receiver efficiency target
    ##################################################
    # Characteristics of Solar Field
    CST['eta_sfr'] = 0.97*0.95*0.95                # Solar field reflectivity
    CST['eta_rfl'] = 0.95                          # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    CST['A_h1']    = 7.07*7.07                     # Area of one heliostat
    CST['N_pan']   = 16                            # Number of panels per heliostat
    CST['err_x']   = 0.001                    # [rad] Reflected error mirror in X direction
    CST['err_y']   = 0.001                    # [rad] Reflected error mirror in X direction
    
    ##################################################
    # Characteristics of BDR and Tower
    CST['zf']       = 50.               # Focal point (where the rays are pointing originally) will be (0,0,zf)
    CST['fzv']      = 0.83              # Position of HB vertix (fraction of zf)
    CST['eta_hbi']  = 0.95              # Desired hbi efficiency
    
    CST['Type']     = 'PB'              # Type of TOD
    CST['Array']    = 'A'               # Array of polygonal TODs
    CST['xrc']      = 0.                # Second focal point (TOD & receiver)
    CST['yrc']      = 0.                # Second focal point (TOD & receiver)
    CST['fzc']      = 0.20              # Second focal point (Height of TOD Aperture, fraction of zf)
    
    CST['Q_av']     = 0.5               # [MW/m2] Desired average radiation flux on receiver
    CST['Q_mx']     = 2.0               # [MW/m2] Maximum radiation flux on receiver
    
    ##### CHANGING SPECIFIC VARIABLES ###########
    for key, value in kwargs.items():
        CST[key] = value
        
    ####### Variables from calculations #########
    if 'zrc' in CST:                            # Second focal point (CPC receiver)
        CST['fzc'] = CST['zrc']/ CST['zf']
    else:
        CST['zrc']  = CST['fzc']*CST['zf']
    
    if 'zv'  in CST:                            # Hyperboloid vertix height
        CST['fzv'] = CST['zv'] / CST['zf']
    else:
        CST['zv']   = CST['fzv']*CST['zf']
        
    if 'P_SF' in CST:                           #[MW] Required power energy
        CST['P_el'] = CST['P_SF'] * ( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )
    else:
        CST['P_SF'] = CST['P_el']/( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )


    return CST

def CST_BaseCase_Thesis(**kwargs):
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. Anyone can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    ############### STANDARD CASE ####################
    ##################################################
    # Environment conditions
    CST['Gbn']   = 950                # Design-point DNI [W/m2]
    CST['day']   = 80                 # Design-point day [-]
    CST['omega'] = 0.0                # Design-point hour angle [rad]
    CST['lat']   = -23.               # Latitude [°]
    CST['lng']   = 115.9              # Longitude [°]
    CST['T_amb'] = 300.               # Ambient Temperature [K]
    ##################################################
    # Receiver and Power Block
    CST['P_el']    = 10.0               #[MW] Target for Net Electrical Power
    CST['eta_pb']  = 0.50               #[-] Power Block efficiency target 
    CST['eta_sg']  = 0.95               #[-] Storage efficiency target
    CST['eta_rcv'] = 0.75               #[-] Receiver efficiency target
    ##################################################
    # Characteristics of Solar Field
    CST['eta_sfr'] = 0.97*0.95*0.95                # Solar field reflectivity
    CST['eta_rfl'] = 0.95                          # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    CST['A_h1']    = 2.97**2                     # Area of one heliostat
    CST['N_pan']   = 1                            # Number of panels per heliostat
    CST['err_x']   = 0.001                    # [rad] Reflected error mirror in X direction
    CST['err_y']   = 0.001                    # [rad] Reflected error mirror in X direction
    
    ##################################################
    # Characteristics of BDR and Tower
    CST['zf']       = 50.               # Focal point (where the rays are pointing originally) will be (0,0,zf)
    CST['fzv']      = 0.83              # Position of HB vertix (fraction of zf)
    CST['eta_hbi']  = 0.95              # Desired hbi efficiency
    
    CST['Type']     = 'PB'              # Type of TOD
    CST['Array']    = 'A'               # Array of polygonal TODs
    CST['xrc']      = 0.                # Second focal point (TOD & receiver)
    CST['yrc']      = 0.                # Second focal point (TOD & receiver)
    CST['fzc']      = 0.20              # Second focal point (Height of TOD Aperture, fraction of zf)
    
    CST['Q_av']     = 0.5               # [MW/m2] Desired average radiation flux on receiver
    CST['Q_mx']     = 2.0               # [MW/m2] Maximum radiation flux on receiver
    
    ##### CHANGING SPECIFIC VARIABLES ###########
    for key, value in kwargs.items():
        CST[key] = value
        
    ####### Variables from calculations #########
    if 'zrc' in CST:                            # Second focal point (CPC receiver)
        CST['fzc'] = CST['zrc']/ CST['zf']
    else:
        CST['zrc']  = CST['fzc']*CST['zf']
    
    if 'zv'  in CST:                            # Hyperboloid vertix height
        CST['fzv'] = CST['zv'] / CST['zf']
    else:
        CST['zv']   = CST['fzv']*CST['zf']
        
    if 'P_SF' in CST:                           #[MW] Required power energy
        CST['P_el'] = CST['P_SF'] * ( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )
    else:
        CST['P_SF'] = CST['P_el']/( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )


    return CST
################################################
################ MAIN PROGRAM ##################
################################################

#%% SETTING CONDITIONS
op_mode = 3
#   op_mode = 1: paper; 2: thesis; 3: single point; 4: Pvar

if op_mode == 1:
    Type  = 'PB'
    # Type  = 'CPC'
    zfs   = [50]
    # fzvs  = np.arange(0.80,0.910,0.02)
    fzvs  = np.arange(0.750,0.910,0.02)
    Cgs   = [1.5,2.0,3.0,4.0]
    Arrays = ['A','B','C','D','E','F']
    Pels  = [10.]
    
    Npan  = 16
    Ah1   = 7.07*7.07
    lat   = -23.

    fldr_dat    = os.path.join(mainDir, '0_Data\MCRT_Datasets_Paper')
    
    fldr_rslt = 'Results_TOD_NR_paper'
    file_rslt =  fldr_rslt+'/1-TOD_'+Type+'.txt'
    write_f = False
    plot    = True
    
if op_mode == 2:
    zfs   = [50]
    # fzvs  = np.arange(0.80,0.910,0.02)
    fzvs  = np.arange(0.770,0.910,0.02)
    Cgs   = [1.5,2.0,3.0,4.0]
    Arrays = ['A','B','C','D','E','F']
    Types = ['CPC']
    Pels  = [10.]
    fldr_rslt = 'Results_TOD_NR_thesis'
    file_rslt =  fldr_rslt+'/1-CPC_Selection_final.txt'
    write_f = True
    plot    = True
    
    type_shdw='simple'
    Npan  = 1
    Ah1   = 2.92*2.92
    lat   = -23.
    fldr_dat  = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
    file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'

if op_mode == 3:
    zfs   = [100]
    fzvs  = [0.73,]
    Cgs   = [2.0,]
    Arrays = ['A']
    Types = ['PB']
    Pels  = [2]
    plot  = True
    
    Npan  = 1
    Ah1   = 2.92**2
    lat   = -23.
    type_shdw='simple'
    
    fldr_dat  = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
    file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
    write_f = False
    
if op_mode == 4:
    zfs   = [75,100]
    # fzvs  = np.arange(0.80,0.910,0.02)
    fzvs  = np.arange(0.71,0.74,0.02)
    Cgs   = [2.0]
    Arrays = ['A']
    Types = ['PB']
    Pels  = np.arange(3,6,2)
    fldr_rslt = 'Results_Pvar_thesis'
    file_rslt =  fldr_rslt+'/1-Pvar.txt'
    write_f = True
    plot    = True
    
    type_shdw='simple'
    Npan  = 1
    Ah1   = 2.92*2.92
    lat   = -23.
    fldr_dat  = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
    file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
############################################


txt_header  = 'Pel\tzf\tfzv\tCg\tType\tArray\teta_hbi\teta_cos\teta_blk\teta_att\teta_hbi\teta_tdi\teta_tdr\teta_TOD\teta_BDR\teta_SF\tPel_real\tN_hel\tS_hel\tS_HB\tS_TOD\tH_TOD\trO\tQ_max\tstatus\n'
file_cols = 'Pel','zf','fzv','Cg','Type','Array','eta_hbi','eta_cos','eta_blk','eta_att','eta_hbi','eta_tdi', 'eta_tdr', 'eta_TOD', 'eta_BDR', 'eta_SF', 'Pel_real', 'N_hel', 'S_hel', 'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max','status'
if write_f:   f = open(file_rslt,'w'); f.write(txt_header); f.close()

#%%  RUNNING THE LOOP ###############

for (zf,fzv,Cg,Array,Pel,Type) in [(zf,fzv,Cg,Array,Pel,Type) for zf in zfs for fzv in fzvs for Cg in Cgs for Array in Arrays for Pel in Pels for Type in Types]:    
    
    # Defining the conditions for the plant
    case = 'zf_{:d}_fzv_{:.3f}_Cg_{:.1f}_{}-{}_Pel_{:.1f}'.format(zf,fzv,Cg,Type,Array,Pel)
    print(case)
    CST = CST_BaseCase_Paper(zf=zf,fzv=fzv,P_el=Pel,N_pan=Npan,A_h1=Ah1,type_shdw=type_shdw)
    #Files for initial data set and HB intersections dataset
    file_SF = file_SF0.format(zf)
    
    #Getting the receiver area and the characteristics for CPC
    def A_rqrd(rO,*args):
        A_rcv_rq, Array, CST, Cg = args
        xrc,yrc,zrc = CST['xrc'], CST['yrc'], CST['zrc']
        TOD = {'Type':'PB','Array':Array,'rO':rO,'Cg':Cg}
        TOD   = BDR.TOD_Params(TOD, xrc,yrc,zrc)
        A_rcv = TOD['A_rcv']
        return A_rcv - A_rcv_rq
    A_rcv_rq = CST['P_SF'] / CST['Q_av']      #Area receiver
    rO = fsolve(A_rqrd, 1.0, args=(A_rcv_rq,Array,CST,Cg))[0]
    TOD = {'Type':Type, 'Array':Array, 'rO':rO, 'Cg':Cg}
    TOD = BDR.TOD_Params( TOD, CST['xrc'],CST['yrc'],CST['zrc'] )
    
    ### Calling the optimisation function
    ######################################################################
    ######################################################################
    R2, Etas, SF, TOD, HB, hlst, Q_rcv, status = BDR.BDR_Selection(CST,TOD,file_SF)
    ######################################################################
    ######################################################################
    
    #Some postcalculations
    S_TOD, H_TOD, rO = [ TOD[x] for x in ['S_TOD','H','rO'] ]
    N_hel = len(hlst)
    S_hel = N_hel * CST['A_h1']
    Pth_real = SF[SF['hel_in']]['Q_h1'].sum()
    Pel_real = Pth_real * (CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'])
    
    #Printing the result on file
    text_r = '\t'.join('{:.3f}'.format(x) for x in [Pel,zf, fzv, Cg])
    text_r = text_r + '\t'+Type+'\t'+Array+'\t'+ '\t'.join('{:.4f}'.format(x) for x in [CST['eta_hbi'], Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_tdi'], Etas['Eta_tdr'], Etas['Eta_TOD'], Etas['Eta_BDR'], Etas['Eta_SF']])+'\t'
    text_r = text_r + '\t'.join('{:.2f}'.format(x) for x in [ Pel_real, N_hel, S_hel, HB['S_HB'], S_TOD, H_TOD, rO, Q_rcv.max()])+'\t'+status+'\n'
    print(text_r[:-2])
    if write_f:     f = open(file_rslt,'a');     f.write(text_r);    f.close()
    
#####################################################################
#####################################################################
#%% SPECIFIC CONDITIONS PLOTING ################################
#####################################################################
#####################################################################
    
    if plot:
        
        N_TOD,V_TOD,rO,rA,Cg = [ TOD[x] for x in ['N','V','rO','rA','Cg'] ]    
        xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
        
        x0,y0 = BDR.TOD_Centers(Array,rA,xrc,yrc)
        xCA, yCA, xCO, yCO = [],[],[],[]
        for i in range(N_TOD):
            xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)
            xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)    
            xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
        xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
        xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
        
        ##############################################
        ###### Ploting CPC shape
        # Nx = 100; Ny = 100
        # dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Ny
        # dA = dx*dy
        # dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
        
        # fig = plt.figure(figsize=(10,10))
        # # plt.scatter(R2['xr'],R2['yr'],c='b',s=0.01)
        # for N in range(N_TOD):
        #     plt.plot(xCA[N],yCA[N],c='k')
        #     plt.plot(xCO[N],yCO[N],c='k')
        # plt.grid()
        # plt.xlim(xmin,xmax);plt.ylim(ymin,ymax);
        # fig.savefig(fldr_rslt+'/'+case+'_shape.png', bbox_inches='tight')
        # plt.close()
        
        ##############################################
        ###### Ploting CPC efficiency points
        # fig = plt.figure(figsize=(10,10))
        # ax = fig.add_subplot(111, aspect='equal')
        # ax.scatter(R2[(R2['hit_tod']==0)]['xc'], R2[(R2['hit_tod']==0)]['yc'],s=0.1)
        # ax.scatter(R2[(R2['hit_tod']>0)&(~R2['hit_rcv'])]['xc'], R2[(R2['hit_tod']>0)&(~R2['hit_rcv'])]['yc'],s=0.1,c='gray')
        # for N in range(N_TOD):
        #     ax.plot(xCA[N],yCA[N],c='k')
        #     ax.plot(xCO[N],yCO[N],c='k')
        # ax.grid()
        # ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);
        
        # ax.annotate(r'$\eta_{{cpo}}={:.2f}$'.format(Etas['Eta_tdr']),(-2.5,ymin/2),fontsize=18, backgroundcolor='white')
        # ax.annotate(r'$\eta_{{cpi}}={:.2f}$'.format(Etas['Eta_tdi']),(4.,ymin*0.75),fontsize=18, backgroundcolor='white')
        
        # ax.annotate('Design A:',(1.05,0.29),xycoords='axes fraction', fontsize=18)
        # ax.annotate('3-hexagon',(1.05,0.26),xycoords='axes fraction', fontsize=18)
        # ax.annotate(r'$z_{{f}}={:.1f}m$'.format(zf),(1.05,0.20),xycoords='axes fraction', fontsize=18)
        # ax.annotate(r'$f_{{v}}={:.2f}$'.format(fzv),(1.05,0.16),xycoords='axes fraction', fontsize=18)
        # ax.annotate(r'$P_{{el}}={:.1f}MW_e$'.format(Pel),(1.03,0.12),xycoords='axes fraction', fontsize=18)
        # ax.annotate(r'$C_{{CPC}}={:.1f}$'.format(Cg),(1.05,0.08),xycoords='axes fraction', fontsize=18)
        
        
        # kw0 = dict(arrowstyle="Simple, tail_width=2.0, head_width=6, head_length=10", color="k")
        # ax.add_patch(patches.FancyArrowPatch((-1.5, ymin/2-0.5), (-1.0, -0.5), connectionstyle="arc3,rad=-0.2", zorder=10, **kw0))
        # ax.add_patch(patches.FancyArrowPatch((3.9, ymin*0.75-0.5), (2.5, -4.5), connectionstyle="arc3,rad=-0.2", zorder=10, **kw0))
        
        # # plt.show()
        # fig.savefig(fldr_rslt+'/'+case+'_no_hitting.png', bbox_inches='tight')
        # plt.close()
        
        #####################################################
        ######## HYPERBOLOID RADIATION MAP ##################
        
        f_s = 18
        out2  = R2[(R2['hel_in'])&(R2['hit_hb'])]
        xmin = out2['xb'].min(); xmax = out2['xb'].max()
        ymin = out2['yb'].min(); ymax = out2['yb'].max()
        Nx = 100; Ny = 100
        dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
        Fbin = CST['eta_rfl']*Etas['Eta_cos']*Etas['Eta_blk']*(CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*len(out2))
        Q_HB,X,Y = np.histogram2d(out2['xb'],out2['yb'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]],density=False)
        fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(111, title='Ray density on hyperboloid surface (upper view)', aspect='equal')
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
        fig.text(0.77,0.33,r'$z_{f\;}=50 m$',fontsize=f_s-3)
        fig.text(0.77,0.31,r'$f_{zv}=0.83$',fontsize=f_s-3)
        fig.text(0.77,0.29,r'$z_{rc}=10 m$',fontsize=f_s-3)
        fig.text(0.77,0.27,r'$\eta_{hbi}=0.95$',fontsize=f_s-3)
        
        r1 = patches.Circle((0.,0.0), HB['rlims'][0], zorder=10,color='black',fill=None)
        r2 = patches.Circle((0.,0.0), HB['rlims'][1], zorder=10,edgecolor='black',fill=None)
        ax.add_artist(r1)
        ax.add_artist(r2)
        ax.grid(zorder=20)
        # fig.savefig(fldr_rslt+'/'+case+'_QHB_upper.pdf', bbox_inches='tight')
        fig.savefig(fldr_rslt+'/'+case+'_QHB_upper.png', bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        print(Q_HB.sum())
        
        #########################################################
        # #Q BDR Distribution
        # N_TOD,V_TOD,rO,rA,Cg = [ CPC[x] for x in ['N','V','rO','rA','Cg'] ]    
        # xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
        # x0,y0 = BDR.CPC_Centers(Dsgn_TOD,rA,xrc,yrc)
        # xCA, yCA, xCO, yCO = [],[],[],[]
        # for i in range(N_TOD):
        #     #Plotting hexagons
        #     xA,yA = BDR.CPC_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)
        #     xO,yO = BDR.CPC_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)    
        #     xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
        # xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
        # xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
        
        # out   = R2[(R2['hel_in'])&(R2['hit_hb'])]
        # rCPC  = out.rc.quantile(0.95)
        # r_050 = out.rc.quantile(0.50)
        # r_080 = out.rc.quantile(0.80)
        # r_090 = out.rc.quantile(0.90)
        # xmin,xmax,ymin,ymax = -rCPC,rCPC,-rCPC,rCPC
        # Nx = 100; Ny = 100
        # dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
        # Nrays = len(out)
        
        # f_s = 16
        # fig = plt.figure(figsize=(14, 8))
        # ax = fig.add_subplot(111, aspect='equal')
        # # for N in range(N_TOD):
        #     # ax.plot(xCA[N],yCA[N],c='k')
        #     # ax.plot(xCO[N],yCO[N],c='k')
        
        # Eta_aux = Etas['Eta_hel']*Etas['Eta_hbi']*CST['eta_rfl']
        # Fbin    = Eta_aux * (CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*Nrays)
        # Q_BDR,X,Y = np.histogram2d(out['xc'],out['yc'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
        # Q_max   = Fbin * Q_BDR.max()
        # # Q,x,y,surf = ax.hist2d(R2['xc'],R2['yc'], bins=100, range= [[xmin,xmax], [ymin,ymax]] , cmap=cm.YlOrRd)
        # X, Y = np.meshgrid(X, Y)
        # vmin = 0; vmax=2000.
        # surf = ax.pcolormesh(X, Y, Fbin*Q_BDR.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
        # cbar = fig.colorbar(surf, shrink=0.5, aspect=4)
        # cbar.ax.tick_params(labelsize=f_s-2)
        
        # r1 = patches.Circle((0.0,0.0), rCPC , zorder=10,ec='black', ls='-', lw=2,fill=None); ax.add_artist(r1)
        # # r2 = patches.Circle((0.0,0.0), r_050, zorder=10,ec='black', ls='--',lw=2,fill=None); ax.add_artist(r2)
        # r3 = patches.Circle((0.0,0.0), r_080, zorder=10,ec='black', ls=':', lw=2,fill=None); ax.add_artist(r3)
        
        # r4 = patches.Circle((0.0,0.0), r_090, zorder=10,ec='black', ls='--', lw=2,fill=None); ax.add_artist(r4)
        
        # ax.text(0.8,-2.3,r'$\eta_{cpi}=0.80$',fontsize=13)
        # ax.text(1.2,-3.3,r'$\eta_{cpi}=0.90$',fontsize=13)
        # ax.text(3.2,-4.3,r'$\eta_{cpi}=0.95$',fontsize=13)
        
        # ax.text(0.5,-2.55,r'$(r_{{CPC}}={:.1f}m)$'.format(r_080),fontsize=12)
        # ax.text(1.0,-3.55,r'$(r_{{CPC}}={:.1f}m)$'.format(r_090),fontsize=12)
        # ax.text(3.1,-4.55,r'$(r_{{CPC}}={:.1f}m)$'.format(rCPC),fontsize=12)
        
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(1.5)
        #     ax.spines[axis].set_zorder(12)
        # ax.grid(zorder=20)
        
        # fig.text(0.77,0.70,r'$Q_{{BDR}}(kW/m^2)$',fontsize=f_s)
        # ax.set_xlabel('X axis (m)',fontsize=f_s);
        # ax.set_ylabel('Y axis (m)',fontsize=f_s);
        # for tick in ax.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        # for tick in ax.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        # # fig.savefig(fldr_rslt+'/'+case+'_radmap_ap.png', bbox_inches='tight')
        # fig.savefig(fldr_rslt+'/0-Fig5b.png', bbox_inches='tight')
        # # plt.show()
        # plt.close()
    
        ##############################################################
        N_TOD,V_TOD,rO,rA,Cg = [ TOD[x] for x in ['N','V','rO','rA','Cg'] ]    
        xrc, yrc, zrc = [CST[x] for x in ['xrc','yrc','zrc']]
        x0,y0 = BDR.TOD_Centers(Array,rA,xrc,yrc)
        xCA, yCA, xCO, yCO = [],[],[],[]
        for i in range(N_TOD):
            xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)
            xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[i],y0[i],zrc)    
            xCA.append(xA);xCO.append(xO);yCA.append(yA);yCO.append(yO);
        xCA=np.array(xCA);xCO=np.array(xCO);yCA=np.array(yCA);yCO=np.array(yCO)
        xmin,xmax,ymin,ymax = xCA.min(), xCA.max(), yCA.min(), yCA.max()
        
        Nx = 100; Ny = 100
        dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Ny
        dA = dx*dy
        dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
        out   = R2[(R2['hel_in'])&(R2['hit_rcv'])]
        Nrays = len(out)
        Fbin    = Etas['Eta_SF'] * (CST['Gbn']*CST['A_h1']*N_hel)/(1e3*dA*Nrays)
        Q_TOD,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
        Q_max   = Fbin * Q_TOD.max()
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, aspect='equal')
        for N in range(N_TOD):
            ax.plot(xCA[N],yCA[N],c='k')
            ax.plot(xCO[N],yCO[N],c='k')
        X, Y = np.meshgrid(X, Y)
        f_s = 16
        vmin = 0
        vmax = 2000
        surf = ax.pcolormesh(X, Y, Fbin*Q_TOD.transpose(),cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
        ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s-2)
        fig.text(0.77,0.65,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
        fig.savefig(fldr_rslt+'/'+case+'_radmap_out.png', bbox_inches='tight')
        # plt.show()
        plt.grid()
        plt.close(fig)
        
        
        ################## Solar Field #############################
        R2f = pd.merge( R2 , SF[(SF['hel_in'])]['Eta_SF'] , how='inner', on=['hel'] )
        f_s=18
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        
        # vmin = ((np.floor(10*R2f['Eta_SF'].min())-1)/10)
        # vmax = (np.ceil(10*R2f['Eta_SF'].max())/10)
        
        vmin = 0.0
        vmax = 1.0
        
        surf = ax1.scatter(R2f['xi'],R2f['yi'], s=0.5, c=R2f['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s)
        cb.ax.locator_params(nbins=4)
        
        fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']),fontsize=f_s)
        fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
        # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
        ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
        # ax1.set_title(r'Focal point height: {:.0f}m ($\eta_{{avg}}$={:.2f})'.format(zf,Eta_cos.mean()),fontsize=f_s)
        # ax1.set_title('No eta_tdi',fontsize=f_s)
        for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
        ax1.grid()
        fig.savefig(fldr_rslt+'/'+case+'_SF.png', bbox_inches='tight')
        # fig.savefig(fldr_rslt+'/'+case+'_SF.pdf', bbox_inches='tight')
        # plt.show()
        plt.close(fig)
    
    pd.set_option('display.max_columns', None)
    # R2.to_csv(fldr_rslt+'/'+case+'_R2.csv')
    del R2, SF, R2f, Q_HB, Q_TOD
sys.exit()    
######################################################
#%% OVERALL PLOTTING PAPER    ########################
######################################################

import matplotlib.lines as mlines
# from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches

# fldr_rslt='Results_Direct/'
# file_TOD = fldr_rslt+'1-Rebuttal.txt'

Type = 'PB'
fldr_rslt = 'Results_TOD_NR/'
file_rslt =  fldr_rslt+'/1-TOD_'+Type+'.txt'
file_TOD = file_rslt

Array = 'A'
df = pd.read_table(file_TOD,sep="\t",header=0)
df.drop_duplicates(subset=['Cg','Array','fzv'],keep='last',inplace=True)
df = df[df['Array']==Array].sort_values(by=['Cg','Array','fzv'])
print(df)

fig = plt.figure(figsize=(16,14))
fig.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
f_s = 14
i = 0
ms = ['o','H',"D","s"]
Cgs = [1.5,2.0,3.0,4.0]
for Cg in Cgs:
    df2 = df[df['Cg']==Cg]
    H = df[df['Cg']==Cg]['H_TOD'].unique()[0]
    S = df[df['Cg']==Cg]['S_TOD'].unique()[0]
    fzv, eta_tdi, eta_tdr, eta_TOD = df2['fzv'],df2['eta_tdi'],df2['eta_tdr'],df2['eta_TOD']
    # ax1.plot(fzv,eta_tdi,c='C'+str(i),lw=2,ls='--')
    ax1.plot(fzv,eta_tdr,c='C'+str(i),lw=3,ls=':',marker='o',markersize=8,zorder=12-i)
    ax1.plot(fzv,eta_tdi,c='C'+str(i),lw=3,ls='--',marker='o',markersize=8,zorder=12-i)
    ax2.plot(fzv,eta_TOD,c='C'+str(i),lw=3,ls='-',marker='o',markersize=8,label=r'$C_{{TOD}}={:.1f}\; (H_{{TOD}}={:.1f}(m),S_{{TOD}}={:.1f}(m^2))$'.format(Cg,H,S))
    i+=1
ax1.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)
ax2.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)
ax1.set_ylabel('Efficiency (-)',fontsize=f_s)
ax1.set_title(r'a) Intercept and optic efficiencies ($\eta_{tdi}$ and $\eta_{tdo}$) for Design '+Array,fontsize=f_s)
ax2.set_title(r'b) Overall TOD efficiency ($\eta_{TOD}=\eta_{tdi}\times\eta_{tdo}$) for Design '+Array,fontsize=f_s)
xmin=0.75; xmax=0.92
ax1.set_xlim(xmin,xmax)
ax1.set_xticks(np.arange(xmin,xmax,0.02))
ax2.set_xlim(xmin,xmax)
ax2.set_xticks(np.arange(xmin,xmax,0.02))
ymin = 0.5
ax1.set_ylim(ymin,1.0)
ax1.set_yticks(np.arange(ymin,1.01,0.1))
ax2.set_ylim(ymin,1.0)
ax2.set_yticks(np.arange(ymin,1.01,0.1))
ax2.grid()

ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
ax2.tick_params(axis='both', which='major', labelsize=f_s-2)
ax1.grid()

eta_1 = mlines.Line2D([], [], color='black', ls=':', label=r'$\eta_{tdo}$')
eta_2 = mlines.Line2D([], [], color='black', ls='--', label=r'$\eta_{tdi}$')
eta_3 = mlines.Line2D([], [], color='black', ls='-', label=r'$\eta_{TOD}$')

ax1.legend(handles=[eta_1,eta_2,eta_3],loc=7,ncol=2,bbox_to_anchor=(0.50,-0.265),fontsize=f_s)
ax2.legend(loc=8,bbox_to_anchor=(0.20, -0.38),ncol=2,fontsize=f_s-0.5)
# fig.savefig(fldr_TOD+'/0-etaCPC_Dsgn_'+Dsgn+'.pdf', bbox_inches='tight')
# fig.savefig(fldr_TOD+'/0-etaCPC_Dsgn_'+Dsgn+'.png', bbox_inches='tight')
# plt.show()


Cg = 2.0
df = pd.read_table(file_TOD,sep="\t",header=0)
df = df[df['Cg']==Cg]
print(df)


# fig = plt.figure(figsize=(16,6))
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# ax3 = ax2.twinx()
f_s = 14
ms = ['o','H',"D","s",'8','x']
cs = ['C1','purple','teal','yellowgreen','dimgrey','darkturquoise']
i = 1
for Array in ['A','B','C','D','E','F']:
    df2 = df[df['Array']==Array]
    H = df[df['Array']==Array]['H_TOD'].unique()[0]
    S = df[df['Array']==Array]['S_TOD'].unique()[0]
    fzv, eta_tdi, eta_tdr, eta_TOD, S_TOD,eta_SF = df2['fzv'],df2['eta_tdi'],df2['eta_tdr'],df2['eta_TOD'],df2['S_TOD'],df2['eta_SF']
    # ax3.plot(fzv,eta_tdi,c='C'+str(i),lw=2,ls='--')
    ax3.plot(fzv,eta_tdr,c=cs[i-1],lw=3,ls=':',marker=ms[i-1],markersize=8,zorder=12-i)
    ax3.plot(fzv,eta_tdi,c=cs[i-1],lw=3,ls='--',marker=ms[i-1],markersize=8,zorder=12-i)
    ax4.plot(fzv,eta_TOD,c=cs[i-1],lw=3,ls='-',marker=ms[i-1],markersize=8,zorder=12-i,label='Array '+Array+r' $(H_{{TOD}}={:.1f}(m),S_{{TOD}}={:.1f}(m^2))$'.format(H,S))
    # ax4.plot(fzv,eta_SF,c=cs[i-1],lw=3,ls='-',marker=ms[i-1],markersize=8,zorder=12-i,label='Array '+Array+r' $(H_{{TOD}}={:.1f}(m),S_{{TOD}}={:.1f}(m^2))$'.format(H,S))
    # ax3.plot(fzv,S_TOD,c='C'+str(i),lw=3,ls='-.')
    i+=1

ax3.set_xlabel('Ratio between vertix and focal point heights $(f_{zv}=z_v/z_f)$',fontsize=f_s)
ax4.set_xlabel('Ratio between vertix and focal point heights $(f_{zv}=z_v/z_f)$',fontsize=f_s)
ax3.set_ylabel('Efficiency (-)',fontsize=f_s)
ax4.set_ylabel('Efficiency (-)',fontsize=f_s)
# ax3.set_ylabel(r'CPC Surface Area ($m^2$)',fontsize=f_s)
ax3.set_title(r'c) Intercept and optic efficiencies ($\eta_{tdi}$ and $\eta_{tdo}$) for $C_{TOD}=2.0$',fontsize=f_s)
ax4.set_title(r'd) Overall TOD efficiency ($\eta_{TOD}=\eta_{tdi}\times\eta_{tdo}$) for $C_{TOD}=2.0$',fontsize=f_s)
xmin=0.75; xmax=0.92
ax3.set_xlim(xmin,xmax)
ax3.set_xticks(np.arange(xmin,xmax,0.02))
ax4.set_xlim(xmin,xmax)
ax4.set_xticks(np.arange(xmin,xmax,0.02))
ymin = 0.5
ax3.set_ylim(ymin,1.0)
ax3.set_yticks(np.arange(ymin,1.01,0.1))
ax4.set_ylim(ymin,1.0)
ax4.set_yticks(np.arange(ymin,1.01,0.1))
ax4.grid()

ax3.tick_params(axis='both', which='major', labelsize=f_s-2)
ax4.tick_params(axis='both', which='major', labelsize=f_s-2)
ax3.grid()

eta_1 = mlines.Line2D([], [], color='black', ls='--', label=r'$\eta_{tdo}$')
eta_2 = mlines.Line2D([], [], color='black', ls=':', label=r'$\eta_{tdi}$')
eta_3 = mlines.Line2D([], [], color='black', ls='-', label=r'$\eta_{TOD}$')
eta_4 = mlines.Line2D([], [], color='black', ls='-.', label=r'$S_{TOD}$')

ax3.legend(handles=[eta_1,eta_2,eta_3],loc=7,ncol=2,bbox_to_anchor=(0.45,-0.28),fontsize=f_s)
ax4.legend(loc=8,bbox_to_anchor=(0.19, -0.48),ncol=2,fontsize=f_s)

# con =  ConnectionPatch(xyA=(0.76,0.72), xyB=(0.77,0.83),coordsA='data', coordsB='data', axesA=ax2, axesB=ax4, arrowstyle="-|>", color='darkorange',lw=3,ls=':',zorder=12)
# ax4.add_artist(con)

ax4.add_artist(patches.Ellipse((0.81,0.92), 0.1, 0.1 , zorder=10,color='tab:red',lw=3,fill=None))

# fig.savefig(fldr_rslt+'0-etaPB_all.pdf', bbox_inches='tight')
# fig.savefig(fldr_rslt+'0-etaPB_all.png', bbox_inches='tight')
plt.show()
plt.close()

######################################################
#%% OVERALL PLOTTING THESIS   ########################
######################################################

import matplotlib.lines as mlines
# from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches

# fldr_rslt='Results_Direct/'
# file_TOD = fldr_rslt+'1-Rebuttal.txt'

Type = 'PB'
fldr_rslt = 'Results_TOD_NR_thesis/'
file_CPC = fldr_rslt+'/1-CPC_Selection_final.txt'
file_PB  = fldr_rslt+'/1-PB_Selection_final.txt'

Array = 'A'
df = pd.read_table(file_CPC,sep="\t",header=0)
df.drop_duplicates(subset=['Cg','Array','fzv'],keep='last',inplace=True)
df = df[df['Array']==Array].sort_values(by=['Cg','Array','fzv'])

dfx = pd.read_table(file_PB,sep="\t",header=0)
dfx.drop_duplicates(subset=['Cg','Array','fzv'],keep='last',inplace=True)
dfx = dfx[dfx['Array']==Array].sort_values(by=['Cg','Array','fzv'])

print(df)

fig = plt.figure(figsize=(16,14))
fig.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

f_s = 14
i = 0
ms = ['o','H',"D","s"]
Cgs = [1.5,2.0,3.0,4.0]

ms2 = ['o','H',"D","s",'8','x']
cs2 = ['purple','teal','yellowgreen','dimgrey','darkturquoise']

for Cg in Cgs:
    df2 = df[df['Cg']==Cg]
    H = df[df['Cg']==Cg]['H_TOD'].unique()[0]
    S = df[df['Cg']==Cg]['S_TOD'].unique()[0]
    fzv, eta_tdi, eta_tdr, eta_TOD = df2['fzv'],df2['eta_tdi'],df2['eta_tdr'],df2['eta_TOD']
    # ax1.plot(fzv,eta_tdi,c='C'+str(i),lw=2,ls='--')
    ax1.plot(fzv,eta_tdr,c='C'+str(i),lw=3,ls=':',marker='o',markersize=8,zorder=12-i)
    ax1.plot(fzv,eta_tdi,c='C'+str(i),lw=3,ls='--',marker='o',markersize=8,zorder=12-i)
    ax2.plot(fzv,eta_TOD,c='C'+str(i),lw=3,ls='-',marker='o',markersize=8,label=r'$C_{{CPC}}={:.1f}\; (H_{{CPC}}={:.1f}(m),S_{{CPC}}={:.1f}(m^2))$'.format(Cg,H,S))
    
    df2 = dfx[dfx['Cg']==Cg]
    H = dfx[dfx['Cg']==Cg]['H_TOD'].unique()[0]
    S = dfx[dfx['Cg']==Cg]['S_TOD'].unique()[0]
    fzv, eta_tdi, eta_tdr, eta_TOD = df2['fzv'],df2['eta_tdi'],df2['eta_tdr'],df2['eta_TOD']
    # ax1.plot(fzv,eta_tdi,c='C'+str(i),lw=2,ls='--')
    ax3.plot(fzv,eta_tdr,c=cs2[i-1],lw=3,ls=':',marker='H',markersize=8,zorder=12-i)
    ax3.plot(fzv,eta_tdi,c=cs2[i-1],lw=3,ls='--',marker='H',markersize=8,zorder=12-i)
    ax4.plot(fzv,eta_TOD,c=cs2[i-1],lw=3,ls='-',marker='H',markersize=8,label=r'$C_{{PB}}={:.1f}\; (H_{{PB}}={:.1f}(m),S_{{PB}}={:.1f}(m^2))$'.format(Cg,H,S))
    i+=1
    
ax1.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)
ax2.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)
ax3.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)
ax4.set_xlabel('Ratio between vertix and focal point heights $(f_{v}=z_v/z_f)$',fontsize=f_s)

ax1.set_ylabel('Efficiency (-)',fontsize=f_s)
ax3.set_ylabel('Efficiency (-)',fontsize=f_s)
ax1.set_title(r'a) Intercept and optic efficiencies ($\eta_{tdi}$ and $\eta_{tdo}$) for Design '+Array,fontsize=f_s)
ax2.set_title(r'b) Overall CPC efficiency ($\eta_{CPC}=\eta_{tdi}\times\eta_{tdo}$) for Design '+Array,fontsize=f_s)
ax3.set_title(r'c) Intercept and optic efficiencies ($\eta_{tdi}$ and $\eta_{tdo}$) for Design '+Array,fontsize=f_s)
ax4.set_title(r'd) Overall PB efficiency ($\eta_{PB}=\eta_{tdi}\times\eta_{tdo}$) for Design '+Array,fontsize=f_s)

xmin=0.75; xmax=0.92
ax1.set_xlim(xmin,xmax)
ax1.set_xticks(np.arange(xmin,xmax,0.02))
ax2.set_xlim(xmin,xmax)
ax2.set_xticks(np.arange(xmin,xmax,0.02))
ymin = 0.5
ax1.set_ylim(ymin,1.0)
ax1.set_yticks(np.arange(ymin,1.01,0.1))
ax2.set_ylim(ymin,1.0)
ax2.set_yticks(np.arange(ymin,1.01,0.1))
ax2.grid()

ax3.set_xlim(xmin,xmax)
ax3.set_xticks(np.arange(xmin,xmax,0.02))
ax4.set_xlim(xmin,xmax)
ax4.set_xticks(np.arange(xmin,xmax,0.02))
ymin = 0.5
ax3.set_ylim(ymin,1.0)
ax3.set_yticks(np.arange(ymin,1.01,0.1))
ax4.set_ylim(ymin,1.0)
ax4.set_yticks(np.arange(ymin,1.01,0.1))
ax3.grid()
ax4.grid()


ax1.tick_params(axis='both', which='major', labelsize=f_s-2)
ax2.tick_params(axis='both', which='major', labelsize=f_s-2)
ax3.tick_params(axis='both', which='major', labelsize=f_s-2)
ax4.tick_params(axis='both', which='major', labelsize=f_s-2)
ax1.grid()

eta_1 = mlines.Line2D([], [], color='black', ls=':', label=r'$\eta_{tdo}$')
eta_2 = mlines.Line2D([], [], color='black', ls='--', label=r'$\eta_{tdi}$')
eta_3 = mlines.Line2D([], [], color='black', ls='-', label=r'$\eta_{TOD}$')

ax1.legend(handles=[eta_1,eta_2,eta_3],loc=7,ncol=2,bbox_to_anchor=(0.50,-0.265),fontsize=f_s)
ax2.legend(loc=8,bbox_to_anchor=(0.20, -0.38),ncol=2,fontsize=f_s-0.5)
ax1.legend(handles=[eta_1,eta_2,eta_3],loc=7,ncol=2,bbox_to_anchor=(0.50,-0.265),fontsize=f_s)
ax2.legend(loc=8,bbox_to_anchor=(0.20, -0.38),ncol=2,fontsize=f_s-0.5)
ax4.legend(loc=8,bbox_to_anchor=(0.20, -0.38),ncol=2,fontsize=f_s-0.5)

fig.savefig(fldr_rslt+'0-eta_both.pdf', bbox_inches='tight')
fig.savefig(fldr_rslt+'0-eta_both.png', bbox_inches='tight')
plt.show()
plt.close()


#%% SOLAR FIELD PLOT

zf   = 50.
Npan  = 1
Ah1   = 2.92*2.92
lat   = -23.
eta_rfl = 0.95
fldr_dat  = os.path.join(mainDir, '0_Data\MCRT_Datasets_Final')
file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
fldr_plot = 'Results_TOD_NR_thesis'

CST = CST_BaseCase_Paper(zf=zf,N_pan=Npan,A_h1=Ah1)
file_SF = file_SF0.format(zf)
f_s=18
R0, SF  = BDR.Rays_Dataset( file_SF, N_pan=CST['N_pan'], save_plk=True )
N_max   = len(R0['hel'].unique())
hlst    = R0['hel'].unique()

Eta_hel = CST['eta_rfl'] * SF['Eta_cos'] * SF['Eta_blk']

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111, aspect='equal')        
vmin = ((np.floor(10*Eta_hel.min())-1)/10)
vmin = 0.4
vmax = (np.ceil(10*Eta_hel.max())/10)

surf = ax1.scatter(SF['xi'],SF['yi'], s=0.5, c=Eta_hel, cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
cb = fig.colorbar(surf, shrink=0.25, aspect=4, ticks=np.arange(vmin,vmax*1.1,0.2))
cb.ax.tick_params(labelsize=f_s)
fig.text(0.79,0.66,r'$\eta_{hel}=$',fontsize=f_s)
fig.text(0.78,0.63,r'$\rho_{r}\eta_{cos}\eta_{blk}$',fontsize=f_s)
# plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
# ax1.set_title(r'Focal point height: {:.0f}m ($\eta_{{avg}}$={:.2f})'.format(zf,Eta_cos.mean()),fontsize=f_s)
# ax1.set_title('No eta_cpi',fontsize=f_s)
for tick in ax1.xaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
for tick in ax1.yaxis.get_major_ticks():    tick.label.set_fontsize(f_s)
ax1.grid()
fig.savefig(fldr_plot+'/SF_zf_'+str(zf)+'.pdf', bbox_inches='tight')
fig.savefig(fldr_plot+'/SF_zf_'+str(zf)+'.png', bbox_inches='tight')
plt.show()
plt.close(fig)

#%% NO RECUERDO QUE QUERIA HACER AQUI... AWEONAO
fldr_rslt = 'Results_TOD_NR_thesis'
file_PB  = fldr_rslt+'/1-TOD_PB.txt'
file_PB  = fldr_rslt+'/1-PB_Selection_final.txt'

df = pd.read_table(file_PB,sep="\t",header=0)
df = df.sort_values(by=['Array','Cg','fzv'])

pd.set_option('display.max_columns', None)

idx = df.groupby(['Array','Cg'])['eta_SF'].transform(max) == df['eta_SF']
print(df[idx])
df[idx].to_csv(fldr_rslt+'/1-Selection_final_fzv.csv')
