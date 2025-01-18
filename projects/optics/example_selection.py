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

from bdr_csp.BeamDownReceiver import (
    HyperboloidMirror,
    SolarField,
    TertiaryOpticalDevice,
)

DIR_DATA = r"C:\Users\david\OneDrive\academia-pega\2024_25-serc_postdoc\bdr_csp\data"
DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))


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


def plot_hb_rad_map(
        case: str,
        R2: pd.DataFrame,
        Etas: pd.DataFrame,
        CST: dict,
        HB: dict,
        hlsts: np.array,
        fldr_rslt: str
    ) -> None:


    f_s = 18
    out2  = R2[(R2['hel_in'])&(R2['hit_hb'])]
    xmin = out2['xb'].min()
    xmax = out2['xb'].max()
    ymin = out2['yb'].min()
    ymax = out2['yb'].max()
    Nx = 100
    Ny = 100
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Nx
    dA = dx*dy
    N_hel = len(hlsts)
    Fbin = (
        CST['eta_rfl'] * Etas['Eta_cos'] * Etas['Eta_blk']
        * (CST['Gbn'] * CST['A_h1'] * N_hel)
        /( 1e3 * dA * len(out2) )
    ) 
    Q_HB,X,Y = np.histogram2d(
        out2['xb'],out2['yb'],
        bins=[Nx,Ny],
        range=[[xmin, xmax], [ymin, ymax]],
        density=False
    )
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, aspect='equal')
    vmin = 0
    vmax = ( np.ceil(Fbin*Q_HB.max()/10)*10 )
    surf = ax.pcolormesh(
        X, Y, Fbin*Q_HB.transpose(),
        cmap=cm.YlOrRd,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlabel('E-W axis (m)',fontsize=f_s)
    ax.set_ylabel('N-S axis (m)',fontsize=f_s)
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    fig.text(0.77,0.62,r'$Q_{HB}(kW/m^2)$',fontsize=f_s)
    ax.tick_params(axis='both', which='major', labelsize=f_s)
    
    fig.text(0.77,0.35,'Main Parameters',fontsize=f_s-3)
    fig.text(0.77,0.33,r'$z_{f\;}=50 m$',fontsize=f_s-3)
    fig.text(0.77,0.31,r'$f_{zv}=0.83$',fontsize=f_s-3)
    fig.text(0.77,0.29,r'$z_{rc}=10 m$',fontsize=f_s-3)
    fig.text(0.77,0.27,r'$\eta_{hbi}=0.95$',fontsize=f_s-3)
    
    r1 = patches.Circle(
        (0.,0.0), HB.rmin,
        zorder=10, color='black', fill=None
    )
    r2 = patches.Circle(
        (0.,0.0), HB.rmax,
        zorder=10,edgecolor='black',fill=None
    )
    ax.add_artist(r1)
    ax.add_artist(r2)
    ax.grid(zorder=20)
    fig.savefig(fldr_rslt+'/'+case+'_QHB_upper.png', bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    print(Q_HB.sum()*Fbin*dA)

    return None


def plot_receiver_rad_map(
        case: str,
        R2: pd.DataFrame,
        Etas: pd.DataFrame,
        CST: dict,
        TOD: dict,
        hlsts,
        fldr_rslt: str
    ) -> None:
    
    n_tods = TOD.n_tods

    radious_ap = TOD.radious_ap
    radious_out = TOD.radious_out

    N_hel = len(hlsts)
    total_rad = Etas['Eta_SF'] * (CST['Gbn']*CST['A_h1']*N_hel)
    Q_TOD,X,Y = TOD.radiation_flux(R2, total_rad)
    Q_max   = Q_TOD.max()

    print(Q_max, total_rad)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect='equal')
    f_s = 16

    for N in range(n_tods):
        xA,yA = TOD.perimeter_points(radious_ap, tod_index=N)
        xO,yO = TOD.perimeter_points(radious_out, tod_index=N)
        ax.plot(xA,yA,c='k')
        ax.plot(xO,yO,c='k')
    vmin = 0
    vmax = 2000
    surf = ax.pcolormesh(X, Y, Q_TOD, cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
    ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s-2)
    fig.text(0.77,0.65,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
    fig.savefig(fldr_rslt+'/'+case+'_radmap_out.png', bbox_inches='tight')
    # plt.show()
    plt.grid()
    plt.close(fig)
    return None


def plot_solar_field_etas(
        case: str,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        Etas: pd.DataFrame,
        N_hel: int,
        fldr_rslt: str
    ) -> None:
    R2f = pd.merge( R2 , SF[(SF['hel_in'])]['Eta_SF'] , how='inner', on=['hel'] )
    f_s=18
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    vmin = 0.0
    vmax = 1.0
    surf = ax1.scatter(
        R2f['xi'],R2f['yi'],
        s=0.5, c=R2f['Eta_SF'],
        cmap=cm.YlOrRd,
        vmin=vmin, vmax=vmax 
    )
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    cb.ax.locator_params(nbins=4)
    
    fig.text(
        0.76,
        0.70,
        r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF'])
        ,fontsize=f_s
    )
    fig.text(
        0.76,
        0.65,
        r'$N_{{hel}}$'+'={:d}'.format(N_hel),
        fontsize=f_s
    )
    ax1.set_xlabel('E-W axis (m)',fontsize=f_s)
    ax1.set_ylabel('N-S axis (m)',fontsize=f_s)
    ax1.tick_params(axis='both', which='major', labelsize=f_s)
    ax1.grid()
    fig.savefig(fldr_rslt+'/'+case+'_SF.png', bbox_inches='tight')
    plt.close(fig)

    return None

################################################
################ MAIN PROGRAM ##################
################################################

# SETTING CONDITIONS
# op_mode = 3
# #   op_mode = 1: paper; 2: thesis; 3: single point; 4: Pvar
    
# if op_mode == 2:
#     zfs   = [50]
#     # fzvs  = np.arange(0.80,0.910,0.02)
#     fzvs  = np.arange(0.770,0.910,0.02)
#     Cgs   = [1.5,2.0,3.0,4.0]
#     Arrays = ['A','B','C','D','E','F']
#     Types = ['CPC']
#     Pels  = [10.]
#     fldr_rslt = 'Results_TOD_NR_thesis'
#     file_rslt =  fldr_rslt+'/1-CPC_Selection_final.txt'
#     write_f = True
#     plot    = True
    
#     type_shdw='simple'
#     Npan  = 1
#     Ah1   = 2.92*2.92
#     lat   = -23.
#     fldr_dat  = os.path.join(DIR_DATA,'MCRT_Datasets_Final')
#     file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'

# if op_mode == 3:
#     zfs   = [50]
#     fzvs  = [0.83,]
#     Cgs   = [2.0,]
#     Arrays = ['A']
#     Types = ['CPC']
#     Pels  = [10]
    
#     Npan  = 1
#     Ah1   = 2.92**2
#     lat   = -23.
#     type_shdw = 'simple'
    
#     fldr_dat  = os.path.join(DIR_DATA, 'mcrt_datasets_final')
#     file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
    
#     fldr_rslt = os.path.join(DIR_PROJECT, "testing")
#     file_rslt =  os.path.join(fldr_rslt, 'testing.txt')
#     write_f = False
#     plot  = True
    
# if op_mode == 4:
#     zfs   = [75,100]
#     # fzvs  = np.arange(0.80,0.910,0.02)
#     fzvs  = np.arange(0.71,0.74,0.02)
#     Cgs   = [2.0]
#     Arrays = ['A']
#     Types = ['PB']
#     Pels  = np.arange(3,6,2)
#     fldr_rslt = 'Results_Pvar_thesis'
#     file_rslt =  fldr_rslt+'/1-Pvar.txt'
#     write_f = True
#     plot    = True
    
#     type_shdw='simple'
#     Npan  = 1
#     Ah1   = 2.92*2.92
#     lat   = -23.
#     fldr_dat  = os.path.join(DIR_DATA,'MCRT_Datasets_Final')
#     file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
############################################


def run_parametric(
        zfs: list,
        fzvs: list,
        Cgs: list,
        Arrays: list,
        Pels: list,
        Types: list,
        write_f: bool = False,
        plot: bool = True,
):

    Npan  = 1
    Ah1   = 2.92**2
    lat   = -23.
    type_shdw = 'point'
    
    fldr_dat  = os.path.join(DIR_DATA, 'mcrt_datasets_final')
    file_SF0  = fldr_dat+'/Dataset_zf_{:.0f}'
    
    fldr_rslt = os.path.join(DIR_PROJECT, "testing")
    file_rslt =  os.path.join(fldr_rslt, 'testing.txt')

    txt_header  = 'Pel\tzf\tfzv\tCg\tType\tArray\teta_cos\teta_blk\teta_att\teta_hbi\teta_tdi\teta_tdr\teta_TOD\teta_BDR\teta_SF\tPel_real\tN_hel\tS_hel\tS_HB\tS_TOD\tH_TOD\trO\tQ_max\tstatus\n'
    file_cols = 'Pel','zf','fzv','Cg','Type','Array','eta_cos','eta_blk','eta_att','eta_hbi','eta_tdi', 'eta_tdr', 'eta_TOD', 'eta_BDR', 'eta_SF', 'Pel_real', 'N_hel', 'S_hel', 'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max','status'
    if write_f:
        f = open(file_rslt,'w')
        f.write(txt_header)
        f.close()

    for (zf,fzv,Cg,Array,Pel,Type) in [
        (zf,fzv,Cg,Array,Pel,Type) 
        for zf in zfs 
        for fzv in fzvs 
        for Cg in Cgs 
        for Array in Arrays 
        for Pel in Pels 
        for Type in Types
    ]:
        
        # Defining the conditions for the plant
        case = 'zf_{:d}_fzv_{:.3f}_Cg_{:.1f}_{}-{}_Pel_{:.1f}'.format(zf,fzv,Cg,Type,Array,Pel)
        print(case)
        CST = CST_BaseCase_Thesis(zf=zf,fzv=fzv,P_el=Pel,N_pan=Npan,A_h1=Ah1,type_shdw=type_shdw)
        #Files for initial data set and HB intersections dataset
        file_SF = file_SF0.format(zf)
        
        #Getting the receiver area and the characteristics for CPC
        def get_radious_out(*args):
            def A_rqrd(rO,*args):
                A_rcv_rq, Type, Array, xrc, yrc, zrc, Cg = args
                A_rcv = TertiaryOpticalDevice(
                    params={"geometry":Type, "array":Array, "Cg":Cg, "rO":rO},
                    xrc=xrc, yrc=yrc, zrc=zrc,
                ).receiver_area
                return A_rcv - A_rcv_rq
            return fsolve(A_rqrd, 1.0, args=args)[0]
        
        xrc, yrc, zrc = CST["xrc"], CST["yrc"], CST["zrc"]
        A_rcv_rq = CST['P_SF'] / CST['Q_av']      #Area receiver
        rO = get_radious_out(A_rcv_rq,Type,Array,xrc,yrc,zrc,Cg)
        eta_hbi = CST["eta_hbi"]

        HSF = SolarField(zf=zf, A_h1=Ah1, N_pan=Npan, file_SF=file_SF)
        HB = HyperboloidMirror(
            zf=zf, fzv=fzv, xrc=xrc, yrc=yrc, zrc=zrc, eta_hbi=eta_hbi
        )
        TOD = TertiaryOpticalDevice(
            params={"geometry":Type, "array":Array, "Cg":Cg, "rO":rO},
            xrc=xrc, yrc=yrc, zrc=zrc,
        )

        ### Calling the heliostat selection function
        R2, Etas, SF, TOD, HB, hlst, Q_rcv, status = BDR.heliostat_selection(CST,HSF,HB,TOD)
        
        #Some postcalculations
        N_hel = len(hlst)
        S_hel = N_hel * HSF.A_h1
        Pth_real = SF[SF['hel_in']]['Q_h1'].sum()
        Pel_real = Pth_real * (CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'])
        
        #Printing the result on file
        text_r = '\t'.join('{:.3f}'.format(x) for x in [Pel,zf, fzv, Cg])
        text_r = text_r + '\t'+Type+'\t'+Array+'\t'+ '\t'.join('{:.4f}'.format(x) for x in [Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], Etas['Eta_hbi'], Etas['Eta_tdi'], Etas['Eta_tdr'], Etas['Eta_TOD'], Etas['Eta_BDR'], Etas['Eta_SF']])+'\t'
        text_r = text_r + '\t'.join('{:.2f}'.format(x) for x in [ Pel_real, N_hel, S_hel, HB.area, TOD.surface_area, TOD.height, TOD.radious_out, Q_rcv.max()])+'\t'+status+'\n'
        print(text_r[:-2])
        if write_f:
            f = open(file_rslt,'a')
            f.write(text_r)
            f.close()

        # SPECIFIC CONDITIONS PLOTING
        if plot:
            plot_hb_rad_map(case, R2,Etas,CST, HB, hlst, fldr_rslt)

            plot_receiver_rad_map(case, R2, Etas, CST, TOD, hlst, fldr_rslt)

            plot_solar_field_etas(case, R2, SF, Etas, N_hel, fldr_rslt)



def main():

    run_parametric(
        zfs   = [50],
        fzvs  = [0.83,],
        Cgs   = [2.0,],
        Arrays = ['A'],
        Types = ['PB'],
        Pels  = [10],
        write_f = False,
        plot = True
    )

if __name__ == "__main__":
    main()