# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:29:52 2022

@author: z5158936
"""

import os
import sys
from dataclasses import dataclass
from typing import TypedDict

import pandas as pd
import numpy as np
import cantera as ct
import scipy.optimize as spo
import scipy.interpolate as spi

from bdr_csp import BeamDownReceiver as BDR
from bdr_csp import htc

from bdr_csp.BeamDownReceiver import (
    TertiaryOpticalDevice
)


def HTM_0D_blackbox(
        Tp: float,
        qi: float,
        Fc: float = 2.57,
        air: ct.Solution = ct.Solution('air.yaml'),
        HTC: str = 'NellisKlein',
        view_factor: float | None = None
    ) -> tuple[float,float,float]:
    """
    Function that obtains an initial estimation for receiver thermal efficiency

    Parameters
    ----------
    Tp : float. [K] Average temperature particle.
    qi : float. [MW/m2] Average radiation flux on receiver aperture area
    air : cantera air mix, optional. If the function will me used many times, it is better to define a cantera object outside the function, to increase performance. The default is ct.Solution('air.xml').
    HTC : str. 
    Returns
    -------
    eta_rcv : TYPE
        DESCRIPTION.

    """
    Tamb = 300.
    ab_p = 0.91
    em_p = 0.85
    Tp = float(Tp)
    Tsky = Tamb -15.
    air.TP = (Tp+Tamb)/2., ct.one_atm
    if HTC == 'NellisKlein':
        hconv = Fc* htc.h_conv_NellisKlein(Tp, Tamb, 0.01, air)
    elif HTC == 'Holman':
        hconv = Fc* htc.h_conv_Holman(Tp, Tamb, 0.01, air)
    elif HTC == 'Experiment':
        hconv = Fc* htc.h_conv_Experiment(Tp, Tamb, 0.1, air)
    
    if view_factor is None:
        view_factor = 1.0
    
    hrad  = view_factor * em_p*5.67e-8*(Tp**4.-Tsky**4.)/(Tp-Tamb)
    hcond = 0.833
    hrc = hconv + hrad + hcond
    qloss = hrc * (Tp - Tamb)
    if qi<=0.:
        eta_rcv = 0.
    else:
        eta_rcv = (qi*1e6*ab_p - qloss)/(qi*1e6)
    if eta_rcv<0:
        eta_rcv == 0.
    return (eta_rcv,hrad,hconv)


def HTM_2D_moving_particles(
        rcvr_input: dict,
        vel_p: float,
        full_output: bool = False
    ) -> list:

    x_ini = rcvr_input["x_ini"]
    x_fin = rcvr_input["x_fin"]
    y_bot = rcvr_input["y_bot"]
    y_top = rcvr_input["y_top"]
    thickness_parts = rcvr_input["thickness_parts"]
    temp_ini = rcvr_input["temp_ini"]
    material = rcvr_input["material"]
    f_Qrc1 = rcvr_input["func_heat_flux_rcv1"]
    f_eta = rcvr_input["func_eta"]

    #Belt vector direction and vector direction
    B_xi = x_ini
    B_yi = (y_bot + y_top)/2.
    B_ux  = -1.             #Should be unitary vector (B_ux**2+B_uy**2 = 1)
    B_uy  = 0.              #Should be unitary vector
    
    B_Vx  = vel_p*B_ux      #Belt X axis velocity
    B_Vy  = vel_p*B_uy      #Belt Y axis velocity
    
    B_Ny = 100              #[-] Number of elements in Y' axis
    B_Nx = 1                #[-] Number of elements in X' axis
    B_Lx = 0.1
    B_Lz = thickness_parts
    B_Ly = (y_top-y_bot)
    B_dx = B_Lx/B_Nx        #[m] Belt x' discr (If the belt is not moving X axis, x' is different than x)
    B_dy = B_Ly/B_Ny        #[m] Belt y' discr
    
    N_t_min = 30
    t_sim = (x_ini-x_fin) / vel_p
    dt     = min(1.,t_sim/N_t_min)    #Temporal discretisation [s]
    if dt>1.:
        dt=1
    
    # print(dt,t_sim)
    # dt=1
    #Initial positions for Belt elements
    B_y = np.array([ B_yi-B_Ly/2 + (j+0.5)*B_dy for j in range(B_Ny)])
    B_x = np.array([ B_xi for j in range(B_Ny)])
    T_B = np.array([ temp_ini for j in range(B_Ny)])
    
    t = dt
    if full_output:
        P = {'t':[],'x':[],'y':[],
             'Tp':[],'Q_in':[], 'eta':[],
             'Tnew':[], 'rcv_in':[], 'dT':[],
             'Q_gain':[]}
    
    while t <= t_sim+dt:
        
        Q_in = f_Qrc1(np.array([B_x,B_y]).transpose())/1000
        
        eta = np.array([f_eta([T_B[j],Q_in[j]])[0] for j in range(len(T_B))])
        
        Q_in = np.where(Q_in>0.01,Q_in,0.0)
        rcv_in = np.where(Q_in>0.01,True,False)
        
        Q_abs = Q_in * eta
        Q_gain = Q_abs * rcv_in
        if material=='CARBO':
            rho_b = 1810
            if np.isnan(T_B).any():
                print(T_B)
                sys.exit()
            cp_b  = 148*T_B**0.3093
        T_B1 =  Q_abs*1e6 / (cp_b * rho_b * B_Lz / dt) + T_B
        dT  = T_B1-T_B
        
        if full_output:
            P['t'].append(t*np.ones(len(B_x)))
            P['x'].append(B_x)
            P['y'].append(B_y)
            P['Tp'].append(T_B)
            P['Tnew'].append(T_B1)
            P['Q_in'].append(Q_in)
            P['Q_gain'].append(Q_gain)
            P['eta'].append(eta)
            P['rcv_in'].append(rcv_in)
            P['dT'].append(dT)
        
        B_x = B_x + dt*B_Vx
        B_y = B_y + dt*B_Vy
        T_B = T_B1
        # print(t,dt,t_sim)
        t += dt
    
    if full_output:
        for x in P.keys():
            P[x]=np.array(P[x])
    
    temp_out_av = T_B.mean()
    if material =='CARBO':
        rho_b = 1810.
        cp_b = (148*((T_B+temp_ini)/2)**0.3093).mean()
    
    M_stg1 = rho_b * (B_Lz*B_Ly*vel_p)
    Qstg1 = M_stg1 * cp_b * (temp_out_av - temp_ini) /1e6
    
    if full_output:
        return [ T_B, Qstg1, M_stg1, P ]
    else:
        return [ T_B, Qstg1, M_stg1 ]


def HTM_3D_moving_particles(CST,TOD,inputs,f_Qrc1,f_eta,full_output=True):
    
    T_in, tz,TSM,zrc = [CST[x] for x in ['T_pC','tz','TSM','zrc']]
    
    x_ini, x_fin, y_bot, y_top, P_vel, t_sim, T_inlet, Nz = inputs
    #Belt vector direction and vector direction
    P_xi = x_ini
    P_yi = (y_top+y_bot)/2.
    Ux  = -P_vel           #Should be unitary vector (P_ux**2+PP_uy**2 = 1)
    Uy  = 0.             #Should be unitary vector

    Ny = 100              #[-] Number of elements in Y' axis
    Nx = 1                #[-] Number of elements in X' axis
    # Nz = 2                #[-] Number of elements in X' axis
    Lx = 0.1
    Lz = tz
    Ly = (y_top-y_bot)
    dx = Lx/Nx             #[m] Particles x' discr (If the belt is not moving X axis, x' is different than x)
    dy = Ly/Ny             #[m] Particles y' discr
    dz = Lz/Nz             #[m] Particles z' discr
    dt = 0.5               #Temporal discretisation [s]
    
    #Initial positions for Belt elements
    P_y = np.array([ P_yi-Ly/2 + (j+0.5)*dy for j in range(Ny)])
    P_x = np.array([ P_xi for j in range(Ny)])
    T_B = np.array([ T_in for j in range(Ny)])
    
    T_p = T_inlet * np.ones((Ny,Nz))
    
    t=dt
    if full_output:
        P = {'t':[],'x':[],'y':[],'Tp':[],'Q_in':[], 'eta':[], 'Tnew':[], 'rcv_in':[], 'dT':[]}
    
    while t <= t_sim+dt:
        
        #Obtaining the Q_in and efficiency for the top
        Q_in = f_Qrc1.ev(P_x,P_y)/1000              #This is in MW/m2
        eta = np.array([f_eta([T_B[j],Q_in[j]])[0] for j in range(len(T_B))])
        
        Q_in = np.where(Q_in>0,Q_in,0.0)
        rcv_in = np.where(Q_in>0,True,False)
        
        Q_abs = Q_in * eta
        if TSM=='CARBO':
            rho_p   = 1810
            # cp_p    = 365*(T_B+273.15)**0.18
            # cp_p2   = 365*(T_p-273.15)**0.18       #For now
            cp_p    = 148*(T_p)**0.3093
            k_p     = 0.7
            alpha_p = k_p/(rho_p*cp_p)
            r_p     = alpha_p * dt / dz**2
        
        
        T_k = T_p.copy()        #Getting the previous elements
        
        #Elements on top
        T_p[:,0] = T_k[:,0] + Q_abs*1e6 / (cp_p[:,0] * rho_p * dz / dt) - r_p[:,0]*(T_k[:,0] - T_k[:,1])
        
        T_k_prev = np.roll(T_k,+1,axis=1)[:,1:-1]
        T_k_next = np.roll(T_k,-1,axis=1)[:,1:-1]
        T_p[:,1:-1] = r_p[:,1:-1] * (T_k_prev - 2*T_k[:,1:-1] + T_k_next) + T_k[:,1:-1]
        
        T_p[:,-1] = T_k[:,-1] + r_p[:,-1]*(T_k[:,-2] - T_k[:,-1])
        
        T_B1 =  Q_abs*1e6 / (cp_p * rho_p * Lz / dt) + T_B
        dT  = T_B1-T_B
        
        if full_output:
            P['t'].append(t*np.ones(len(P_x)))
            P['x'].append(P_x)
            P['y'].append(P_y)
            P['Tp'].append(T_B)
            P['Tnew'].append(T_B1)
            P['Q_in'].append(Q_in)
            P['eta'].append(eta)
            P['rcv_in'].append(rcv_in)
            P['dT'].append(dT)
        
        P_x = P_x + dt*Ux
        P_y = P_y + dt*Uy
        T_B = T_B1
        t += dt
        
        Py_3D = np.transpose(np.array([P_y for j in range(Nz)]))
    
    if full_output:
        for x in P.keys():  P[x]=np.array(P[x])
    
    T_out_av = T_B.mean()
    # cp_b  = (365*((T_B+T_in)/2 + 273.15)**0.18).mean()
    cp_b  = (148*((T_B+T_in)/2)**0.3093).mean()
    
    M_stg1 = rho_p * (Lz*Ly*P_vel)
    Qstg1 = M_stg1 * cp_b * (T_out_av - T_in) /1e6
    
    print(T_in,T_p.mean(),T_p.max())
    return T_p, Qstg1, M_stg1, Py_3D


def get_func_eta():
    Fc = 2.57
    air  = ct.Solution('air.yaml')
    Tps = np.arange(700.,2001.,100.)
    qis = np.arange(0.10,4.01,0.1)
    eta_ths = np.zeros((len(Tps),len(qis)))
    for (i,j) in [(i,j) for i in range(len(Tps)) for j in range(len(qis))]:
        eta_ths[i,j] = HTM_0D_blackbox(Tps[i], qis[j], Fc=Fc, air=air)[0]
    return spi.RegularGridInterpolator(
        (Tps,qis),
        eta_ths,
        bounds_error = False,
        fill_value = None
        )


def get_func_heat_flux_rcv1(xr,yr,lims,P_SF1):
    
    xmin,xmax,ymin,ymax = lims
    
    Nx = 50; Ny = 50
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Nx
    dA=dx*dy
    Nrays = len(xr)
    Fbin    = P_SF1 /(1e3*dA*Nrays)
    Q_rc1,X_rc1, Y_rc1 = np.histogram2d(
        xr,
        yr,
        bins=[Nx,Ny],
        range=[[xmin, xmax], [ymin, ymax]],
        density=False
    )
    Q_rc1 = Fbin * Q_rc1
    
    # f_Qrc1 = spi.RectBivariateSpline(X_rc1[:-1],Y_rc1[:-1],Q_rc1)     #Function to interpolate
    f_Qrc1 = spi.RegularGridInterpolator(
        [X_rc1[:-1],Y_rc1[:-1]],
        Q_rc1,
        bounds_error = False,
        fill_value = None
        )
    return f_Qrc1, Q_rc1, X_rc1, Y_rc1


class ReceiverOutput(TypedDict, total=False):
    temps_parts: np.array
    n_hels: int | None
    rad_flux_max: float | None
    rad_flux_avg: float | None
    heat_stored: float | None
    eta_rcv: float | None
    mass_stg: float | None
    time_res: float | None
    vel_p: float | None

@dataclass
class HPR0D():
    nom_power:float = 10.          # [MWth] Initial target for Receiver nominal power
    heat_flux_max: float = 3.0     # [MW/m2] Maximum radiation flux on receiver
    heat_flux_avg: float = 0.5     # [MW/m2] Average radiation flux on receiver (initial guess)
    temp_ini: float = 950          # [K] Particle temperature in cold tank
    temp_out: float = 1200         # [K] Particle temperature in hot tank
    material: str = 'CARBO'        # [-] Thermal Storage Material
    thickness_parts: float = 0.05  # [m] Thickness of material on conveyor belt
    factor_htc: float = 2.57       # [-] factor to magnified the HTC

    def run_model(self, SF, air=ct.Solution('air.yaml')) -> ReceiverOutput:
    
        Prcv = self.nom_power
        Qavg = self.heat_flux_avg
        Tin = self.temp_ini
        Tout = self.temp_out
        tz = self.thickness_parts
        Fc = self.factor_htc
    
        Tp = 0.5*(Tin+Tout)
        eta_rcv = HTM_0D_blackbox(Tp, Qavg, Fc=Fc, air=air)[0]
        
        rho_b = 1810
        cp = 148*Tp**0.3093
        t_res = rho_b * cp * tz * (Tout - Tin ) / (Qavg*1e6*eta_rcv)
        m_p = Prcv*1e6 / (cp*(Tout-Tin))
        P_bdr = Prcv / eta_rcv

        Arcv  = P_bdr / Qavg
        Ltot  = Arcv**0.5
        vel_p = Ltot / t_res
        Q_acc    = SF['Q_h1'].cumsum()
        
        rcvr_output = {
            "temps_parts" : np.array([Tp]),
            "n_hels" : len( Q_acc[ Q_acc < P_bdr ] ) + 1,
            "rad_flux_max" : np.nan,            #No calculated
            "rad_flux_avg" : P_bdr/Arcv,
            "heat_stored": np.nan,            #No calculated
            "eta_rcv": eta_rcv,
            "mass_stg": m_p,
            "time_res": t_res,
            "vel_p": vel_p,
        }
        return rcvr_output

@dataclass
class HPR2D():
    nom_power:float = 10.          # [MWth] Initial target for Receiver nominal power
    heat_flux_max:float = 3.0      # [MW/m2] Maximum radiation flux on receiver
    heat_flux_avg: float = 0.5     # [MW/m2] Average radiation flux on receiver (initial guess)
    temp_ini: float = 950          # [K] Particle temperature in cold tank
    temp_out: float = 1200         # [K] Particle temperature in hot tank
    material: str = 'CARBO'        # [-] Thermal Storage Material
    thickness_parts: float = 0.05  # [m] Thickness of material on conveyor belt
    factor_htc: float = 2.57       # [-] factor to magnified the HTC
    x_fin: float | None = None
    x_ini: float | None = None
    x_bot: float | None = None
    x_top: float | None = None

    def get_dimensions(self, TOD: TertiaryOpticalDevice, tod_index: int = 0) -> None:
        xO,yO = TOD.perimeter_points(TOD.radius_out, tod_index=tod_index)
        self.x_fin = xO.min()
        self.x_ini = xO.max()
        self.y_bot = yO.min()
        self.y_top = yO.max()
        self.area = TOD.receiver_area
        return None


    def run_model(
            self,
            TOD: TertiaryOpticalDevice,
            SF: pd.DataFrame,
            R2: pd.DataFrame,
            polygon_i: int = 1,
            full_output: bool = False
        ) -> dict:
        
        def func_temp_out_avg(t_res_g,*args):
            rcvr_input = args[0]
            vel_p  = rcvr_input["x_avg"]/t_res_g[0]
            T_p, _, _ = HTM_2D_moving_particles(rcvr_input, vel_p)
            return T_p.mean()-temp_out
        
        temp_ini = self.temp_ini
        temp_out = self.temp_out
        thickness_parts = self.thickness_parts
        material = self.material
        power_rcv = self.nom_power
        heat_flux_avg = self.heat_flux_avg

        #Parameters for receiver
        xO,yO = TOD.perimeter_points(TOD.radius_out, tod_index=polygon_i-1)
        lims = xO.min(), xO.max(), yO.min(), yO.max()
        x_fin = lims[0]
        x_ini = lims[1]
        y_bot = lims[2]
        y_top = lims[3]
        area_rcv_1 = TOD.receiver_area/TOD.n_tods
        x_avg = area_rcv_1/(y_top-y_bot)

        rcvr_input = {
            "temp_ini": temp_ini,
            "temp_out": temp_out,
            "thickness_parts": thickness_parts,
            "material": material,
            "power_rcv": power_rcv,
            "x_ini": x_ini,
            "x_fin": x_fin,
            "y_bot": y_bot,
            "y_top": y_top,
            "x_avg": x_avg,
            "func_eta": get_func_eta(),
            "func_heat_flux_rcv1": None,
        }

        #Initial guess for number of heliostats
        func_eta = rcvr_input["func_eta"]
        temps_parts_avg = ( temp_ini + temp_out ) / 2
        eta_rcv = func_eta([ temps_parts_avg, heat_flux_avg ])[0]
        Q_acc = SF['Q_h1'].cumsum()
        N_hel = len( Q_acc[ Q_acc < (power_rcv / eta_rcv) ] ) + 1
            
        R2a = R2[(R2["hit_rcv"])&(R2["Npolygon"]==polygon_i)].copy()  #Rays into 1 receiver
        R2_all = R2[(R2["hit_rcv"])].copy()                           #Total Rays into receivers
        
        if material == 'CARBO':
            rho_b = 1810
            cp = 148*temps_parts_avg**0.3093

        it_max = 20
        it = 1
        loop=0
        data = []
        N_hels = []
        solve_t_res = False
        while True:
            
            hlst    = SF.iloc[:N_hel].index
            SF_it   = SF.loc[hlst]              #Array only valid in iteration it
            Etas_it = SF_it.mean()              #Array only valid in iteration it
            P_SF_it = SF_it["Q_pen"].sum()         #Value only valid in iteration it
            eta_SF_it = Etas_it['Eta_SF']       #Value only valid in iteration it
            
            #Getting the rays dataset and the total energy it should represent
            rays   = R2a[R2a["hel"].isin(hlst)].copy()
            r_i = len(rays)/len(R2_all[R2_all["hel"].isin(hlst)])  #Fraction of rays that goes into one TOD
            P_SF_i = r_i * P_SF_it * 1e6
            
            func_heat_flux_rcv1,heat_flux_rcv1,_,_ = get_func_heat_flux_rcv1(
                rays['xr'], rays['yr'], lims, P_SF_i
            )
            rcvr_input["func_heat_flux_rcv1"] = func_heat_flux_rcv1

            #initial estimates for residence time and receiver efficiency
            if it==1:
                heat_flux_avg = P_SF_it / TOD.receiver_area
                eta_rcv_g = func_eta([temps_parts_avg,heat_flux_avg])[0]
            else:      
                eta_rcv_g = eta_rcv
            
            if solve_t_res:
                t_res_g = (
                    rho_b * cp * thickness_parts * (temp_out - temp_ini )
                    / (eta_rcv_g * heat_flux_avg * 1e6 )
                )
                vel_p  = x_avg/t_res_g                #Belt velocity magnitud
                inputs = (rcvr_input)
                sol  = spo.fsolve(func_temp_out_avg, t_res_g, args=inputs, full_output=True)
                t_res = sol[0][0]
                n_evals = sol[1]['nfev']
            else:
                t_res = (
                    rho_b * cp * thickness_parts * (temp_out - temp_ini ) 
                    / (eta_rcv_g * heat_flux_avg * 1e6 )
                )
                
            # Final values for temperature, heat_stored_1 and mass_stg_1
            vel_p  = x_avg/t_res                    #Belt velocity magnitud
            if full_output:
                temps_parts, heat_stored_1, mass_stg_1, Rcvr_full = HTM_2D_moving_particles(
                    rcvr_input, vel_p, full_output=True
                )
            else:
                temps_parts, heat_stored_1, mass_stg_1 = HTM_2D_moving_particles(
                    rcvr_input, vel_p,
                )
            
            #Performance parameters
            eta_rcv = heat_stored_1*1e6/P_SF_i
            heat_stored = heat_stored_1 / r_i
            mass_stg = mass_stg_1 / r_i
            heat_flux_avg = P_SF_it / TOD.receiver_area
            heat_flux_max = heat_flux_rcv1.max()/1000.
            
            P_SF  = power_rcv / eta_rcv
            Q_acc = SF['Q_h1'].cumsum()
            N_new = len( Q_acc[ Q_acc < P_SF ] ) + 1
            
            data.append([TOD.receiver_area, N_hel, heat_flux_max, heat_flux_avg,
                        P_SF_i, heat_stored, P_SF_it, eta_rcv,
                        eta_SF_it, mass_stg, t_res, temps_parts.mean()])
            # print('\t'.join('{:.3f}'.format(x) for x in data[-1]))
            
            conv = abs(heat_stored-power_rcv)<0.1 and (abs(temps_parts.mean()-temp_out)<1.)
            
            if (N_new == N_hel) and conv:
                break
            
            if loop==5:
                N_hel = max(N_hel,N_new)
                break
            
            if N_new in N_hels:
                solve_t_res = True
                N_new = (N_hel + N_new)//2
                if N_new in N_hels:
                    N_new +=1
                loop+=1
            
            if (it==it_max) and conv and (solve_t_res==False):
                # print('System did not converge, t_res will be used')
                it_max = 40
                solve_t_res = True
            
            if it==it_max and solve_t_res:
                print('System did not converge for either method')
                break
            
            N_hel = N_new
            N_hels.append(N_new)
            
            it+=1
        
        rcvr_output = {
            "temps_parts": temps_parts,
            "n_hels" : N_hel,
            "rad_flux_max": heat_flux_max,
            "rad_flux_avg": heat_flux_avg,
            "power_sf_i": P_SF_i,
            "heat_stored": heat_stored,
            "power_sf_total": P_SF_it,
            "eta_rcv": eta_rcv,
            "mass_stg": mass_stg,
            "time_res": t_res,
            "vel_parts": vel_p,
            "iteration": it,
            "solve_t_res": solve_t_res
        }
        if full_output:
            rcvr_output["full"] = Rcvr_full
        
        return rcvr_output


def rcvr_HPR_0D(CST, Fc=2.57, air=ct.Solution('air.yaml')):
    
    Prcv, Qavg, Tin, Tout, tz  = [CST[x] for x in ['P_rcv','Qavg','T_pC','T_pH', 'tz']]
    
    Tp = 0.5*(Tin+Tout)
    eta_rcv = HTM_0D_blackbox(Tp, Qavg,Fc=Fc,air=air)[0]
    
    rho_b = 1810
    # cp = 365*(Tp+273.15)**0.18
    cp = 148*Tp**0.3093
    t_res = rho_b * cp * tz * (Tout - Tin ) / (Qavg*1e6*eta_rcv)
    m_p = Prcv*1e6 / (cp*(Tout-Tin))
    P_SF = Prcv / eta_rcv
       
    Arcv  = P_SF / Qavg
    Ltot  = Arcv**0.5
    vel_p = Ltot / t_res
    
    return  [eta_rcv, P_SF, Tp, t_res, m_p, Arcv, vel_p]


def rcvr_HPR_0D_simple(CST, SF, Fc=2.57, air=ct.Solution('air.yaml')):
    
    Prcv, Qavg, Tin, Tout, tz  = [CST[x] for x in ['P_rcv','Qavg','T_pC','T_pH', 'tz']]
    
    Tp = 0.5*(Tin+Tout)
    eta_rcv = HTM_0D_blackbox(Tp, Qavg,Fc=Fc,air=air)[0]
    
    rho_b = 1810
    cp = 148*Tp**0.3093
    t_res = rho_b * cp * tz * (Tout - Tin ) / (Qavg*1e6*eta_rcv)
    m_p = Prcv*1e6 / (cp*(Tout-Tin))
    P_bdr = Prcv / eta_rcv

    Arcv  = P_bdr / Qavg
    Ltot  = Arcv**0.5
    vel_p = Ltot / t_res
    Q_acc    = SF['Q_h1'].cumsum()
    
    rcvr_output = {
        "temps_parts" : np.array([Tp]),
        "n_hels" : len( Q_acc[ Q_acc < P_bdr ] ) + 1,
        "rad_flux_max" : np.nan,            #No calculated
        "rad_flux_avg" : P_bdr/Arcv,
        "heat_stored": np.nan,            #No calculated
        "eta_rcv": eta_rcv,
        "mass_stg": m_p,
        "time_res": t_res,
        "vel_p": vel_p,
    }
    return rcvr_output


def rcvr_HPR_2D_simple(
        CST: dict,
        TOD: TertiaryOpticalDevice,
        SF: pd.DataFrame,
        R2: pd.DataFrame,
        polygon_i: int = 1,
        full_output: bool = False
    ) -> dict:
    
    def func_temp_out_avg(t_res_g,*args):
        rcvr_input = args[0]
        vel_p  = rcvr_input["x_avg"]/t_res_g[0]
        T_p, _, _ = HTM_2D_moving_particles(rcvr_input, vel_p)
        return T_p.mean()-temp_out
    
    temp_ini, temp_out, thickness_parts, material, power_rcv, heat_flux_avg = [
        CST[x] for x in ['T_pC', 'T_pH', 'tz', 'TSM', 'P_rcv', 'Qavg']
    ]

    #Parameters for receiver
    xO,yO = TOD.perimeter_points(TOD.radius_out, tod_index=polygon_i-1)
    lims = xO.min(), xO.max(), yO.min(), yO.max()
    x_fin = lims[0]
    x_ini = lims[1]
    y_bot = lims[2]
    y_top = lims[3]
    area_rcv_1 = TOD.receiver_area/TOD.n_tods
    x_avg = area_rcv_1/(y_top-y_bot)

    rcvr_input = {
        "temp_ini": temp_ini,
        "temp_out": temp_out,
        "thickness_parts": thickness_parts,
        "material": material,
        "power_rcv": power_rcv,
        "x_ini": x_ini,
        "x_fin": x_fin,
        "y_bot": y_bot,
        "y_top": y_top,
        "x_avg": x_avg,
        "func_eta": get_func_eta(),
        "func_heat_flux_rcv1": None,
    }

    #Initial guess for number of heliostats
    func_eta = rcvr_input["func_eta"]
    temps_parts_avg = ( temp_ini + temp_out ) / 2
    eta_rcv = func_eta([ temps_parts_avg, heat_flux_avg ])[0]
    Q_acc = SF['Q_h1'].cumsum()
    N_hel = len( Q_acc[ Q_acc < (power_rcv / eta_rcv) ] ) + 1
        
    R2a = R2[(R2["hit_rcv"])&(R2["Npolygon"]==polygon_i)].copy()  #Rays into 1 receiver
    R2_all = R2[(R2["hit_rcv"])].copy()                           #Total Rays into receivers
    
    if material == 'CARBO':
        rho_b = 1810
        cp = 148*temps_parts_avg**0.3093

    it_max = 20
    it = 1
    loop=0
    data = []
    N_hels = []
    solve_t_res = False
    while True:
        
        hlst    = SF.iloc[:N_hel].index
        SF_it   = SF.loc[hlst]              #Array only valid in iteration it
        Etas_it = SF_it.mean()              #Array only valid in iteration it
        P_SF_it = SF_it["Q_pen"].sum()         #Value only valid in iteration it
        eta_SF_it = Etas_it['Eta_SF']       #Value only valid in iteration it
        
        #Getting the rays dataset and the total energy it should represent
        rays   = R2a[R2a["hel"].isin(hlst)].copy()
        r_i = len(rays)/len(R2_all[R2_all["hel"].isin(hlst)])  #Fraction of rays that goes into one TOD
        P_SF_i = r_i * P_SF_it * 1e6
        
        func_heat_flux_rcv1,heat_flux_rcv1,_,_ = get_func_heat_flux_rcv1(
            rays['xr'], rays['yr'], lims, P_SF_i
        )
        rcvr_input["func_heat_flux_rcv1"] = func_heat_flux_rcv1

        #initial estimates for residence time and receiver efficiency
        if it==1:
            heat_flux_avg = P_SF_it / TOD.receiver_area
            eta_rcv_g = func_eta([temps_parts_avg,heat_flux_avg])[0]
        else:      
            eta_rcv_g = eta_rcv
        
        if solve_t_res:
            t_res_g = (
                rho_b * cp * thickness_parts * (temp_out - temp_ini )
                / (eta_rcv_g * heat_flux_avg * 1e6 )
            )
            vel_p  = x_avg/t_res_g                #Belt velocity magnitud
            inputs = (rcvr_input)
            sol  = spo.fsolve(func_temp_out_avg, t_res_g, args=inputs, full_output=True)
            t_res = sol[0][0]
            n_evals = sol[1]['nfev']
            CST['t_res'] = t_res
            CST['vel_p'] = x_avg/t_res
        else:
            t_res = (
                rho_b * cp * thickness_parts * (temp_out - temp_ini ) 
                / (eta_rcv_g * heat_flux_avg * 1e6 )
            )
            
        # Final values for temperature, heat_stored_1 and mass_stg_1
        vel_p  = x_avg/t_res                    #Belt velocity magnitud
        if full_output:
            temps_parts, heat_stored_1, mass_stg_1, Rcvr_full = HTM_2D_moving_particles(
                rcvr_input, vel_p, full_output=True
            )
        else:
            temps_parts, heat_stored_1, mass_stg_1 = HTM_2D_moving_particles(
                rcvr_input, vel_p,
            )
        
        #Performance parameters
        eta_rcv = heat_stored_1*1e6/P_SF_i
        heat_stored = heat_stored_1 / r_i
        mass_stg = mass_stg_1 / r_i
        heat_flux_avg = P_SF_it / TOD.receiver_area
        heat_flux_max = heat_flux_rcv1.max()/1000.
        
        P_SF  = power_rcv / eta_rcv
        Q_acc = SF['Q_h1'].cumsum()
        N_new = len( Q_acc[ Q_acc < P_SF ] ) + 1
        
        data.append([TOD.receiver_area, N_hel, heat_flux_max, heat_flux_avg,
                    P_SF_i, heat_stored, P_SF_it, eta_rcv,
                    eta_SF_it, mass_stg, t_res, temps_parts.mean()])
        # print('\t'.join('{:.3f}'.format(x) for x in data[-1]))
        
        conv = abs(heat_stored-power_rcv)<0.1 and (abs(temps_parts.mean()-temp_out)<1.)
        
        if (N_new == N_hel) and conv:
            break
        
        if loop==5:
            N_hel = max(N_hel,N_new)
            break
        
        if N_new in N_hels:
            solve_t_res = True
            N_new = (N_hel + N_new)//2
            if N_new in N_hels:
                N_new +=1
            loop+=1
        
        if (it==it_max) and conv and (solve_t_res==False):
            # print('System did not converge, t_res will be used')
            it_max = 40
            solve_t_res = True
        
        if it==it_max and solve_t_res:
            print('System did not converge for either method')
            break
        
        N_hel = N_new
        N_hels.append(N_new)
        
        it+=1
    
    rcvr_output = {
        "temps_parts": temps_parts,
        "n_hels" : N_hel,
        "rad_flux_max": heat_flux_max,
        "rad_flux_avg": heat_flux_avg,
        "power_sf_i": P_SF_i,
        "heat_stored": heat_stored,
        "power_sf_total": P_SF_it,
        "eta_rcv": eta_rcv,
        "mass_stg": mass_stg,
        "time_res": t_res,
        "vel_parts": vel_p,
        "iteration": it,
        "solve_t_res": solve_t_res
    }
    if full_output:
        rcvr_output["full"] = Rcvr_full
    
    return rcvr_output


# TILTED PARTICLE RECEIVER (TPR)
def eta_th_tilted(Rcvr, Tp, qi, Fv=1):
    """
    Function that obtains an initial estimation for receiver thermal efficiency

    Parameters
    ----------
    Tp : float. [K] Average temperature particle.
    qi : float. [MW/m2] Average radiation flux on receiver aperture area
    air : cantera air mix, optional. If the function will me used many times, it is better to define a cantera object outside the function, to increase performance. The default is ct.Solution('air.xml').
    HTC : str. 
    Returns
    -------
    eta_th : float. thermal specific efficiency

    """
    F_t_rcv = Fv
    Tw  = Rcvr['Tw'] if 'Tw' in Rcvr else Tp
    Fc  = Rcvr['Fc'] if 'Fc' in Rcvr else 2.57
    air = Rcvr['air'] if 'air' in Rcvr else ct.Solution('air.xml')
    HTC = Rcvr['HTC'] if 'HTC' in Rcvr else 'NellisKlein'
    
    T_amb = Rcvr['T_amb'] if 'T_amb' in Rcvr else 300.
    ab_p  = Rcvr['ab_p'] if 'ab_p' in Rcvr else 0.91
    em_p  = Rcvr['em_p'] if 'em_p' in Rcvr else 0.85
    em_w  = Rcvr['em_w'] if 'em_w' in Rcvr else 0.20
    h_cond = Rcvr['hcond'] if 'hcond' in Rcvr else 0.833
    
    xmin = Rcvr['x_fin'] if 'x_fin' in Rcvr else 0.
    xmax = Rcvr['x_ini'] if 'x_ini' in Rcvr else 1.
    ymin = Rcvr['y_bot'] if 'y_bot' in Rcvr else 0.
    ymax = Rcvr['y_top'] if 'y_top' in Rcvr else 1.
    
    beta = Rcvr['beta'] if 'beta' in Rcvr else 30.*np.pi/180.
    Arc1 = Rcvr['A_rc1'] if 'A_rc1' in Rcvr else 0.
    
    sigma = 5.674e-8
    
    Tp = float(Tp)
    Tsky = T_amb - 15.
    air.TP = (Tp+T_amb)/2., ct.one_atm
    if HTC=='NellisKlein':
        hconv = Fc*htc.h_conv_NellisKlein(Tp, T_amb, 0.01, air)
    elif HTC=='Holman':
        hconv = Fc*htc.h_conv_Holman(Tp, T_amb, 0.01, air)
    elif HTC=='Experiment':
        hconv = Fc*htc.h_conv_Experiment(Tp, T_amb, 0.1, air)
    
    hrad = F_t_rcv*em_p*5.67e-8*(Tp**4.-Tsky**4.)/(Tp-T_amb)
    
    F_tw = 1-F_t_rcv
    X_tpr = (xmax-xmin)
    Y_tpr = (ymax-ymin)
    Z_tpr = X_tpr * abs(np.tan(beta))
    A_w = Z_tpr*Y_tpr + (Z_tpr*X_tpr/2)*2 + (X_tpr*Y_tpr - Arc1)
    A_t = Y_tpr*X_tpr/np.cos(beta)
    
    C_tw = (1-em_p)/(em_p*A_t) + 1/(F_tw*A_t) + (1-em_w)/(em_w*A_w)
    
    h_t_wall = (sigma/C_tw)*(Tp**4 - Tw**4)/(Tp-Tw)
    hwall = 1./(1/h_t_wall + 1/h_cond)
    
    hrc = hconv + hrad + hwall
    qloss = hrc * (Tp - T_amb)
    
    if qi<=0.:
        eta_th = 0.
    else:
        eta_th = (qi*1e6*ab_p - qloss)/(qi*1e6)

    if eta_th<0:
        eta_th == 0.
        
    return eta_th,hrad,hconv,hwall


def get_view_factor(CST,TOD,polygon_i,lims1,xp,yp,zp,beta):
    Type, Array, x0, y0, V_TOD, N_TOD, H_TOD, rO, rA, Arcv = [TOD[x] for x in ['Type','Array','x0', 'y0','V','N','H','rO','rA','A_rcv']]
    
    xmin1,xmax1,ymin1,ymax1 = lims1
    n1,n2 = (-np.sin(beta),0.,np.cos(beta)), (0.,0.,-1)
    zrc = CST['zrc']
    x0i,y0i = x0[polygon_i-1], y0[polygon_i-1]
    xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0i,y0i,zrc)
    xmin2,xmax2,ymin2,ymax2 =xO.min(),xO.max(),yO.min(),yO.max()
    
    Nx2 = 100
    Ny2 = 100
    dx2 = (xmax2-xmin2)/Nx2
    dy2 = (ymax2-ymin2)/Nx2
    dA2=dx2*dy2
    Px = np.array([xmin2+dx2/2 + i*dx2 for i in range(Nx2)])
    Py = np.array([ymin2+dy2/2 + j*dy2 for j in range(Ny2)])
    X2, Y2 = np.meshgrid(Px, Py)
    Z2    = (zrc-TOD['H']) * np.ones(X2.shape)
    rcvr_in = BDR.CPC_enter(X2, Y2, rO, Array, V_TOD, x0i, y0i)
    
    Nx1=50
    Ny1=50
    dx1 = (xmax1-xmin1)/Nx1
    dy1 = (ymax1-ymin1)/Nx1
    dA1=dx1*dy1
    Px = np.array([xmin1+dx1/2 + i*dx1 for i in range(Nx1)])
    Py = np.array([ymin1+dy1/2 + j*dy1 for j in range(Ny1)])
    X1, Y1 = np.meshgrid(Px, Py)
    Z1 = (xp-X1)*np.tan(beta) + zp
    
    F12 = np.zeros(Z1.shape)
    for i,j in[(i,j) for i in range(Nx1) for j in range(Ny1)]:
        x1,y1,z1 = X1[i,j],Y1[i,j],Z1[i,j]
        
        R12 = (np.sqrt((X2-x1)**2 + (Y2-y1)**2 + (Z2-z1)**2))
        cos_beta1 =  (n1[0]*(X2-x1) + n1[1]*(Y2-y1) + n1[2]*(Z2-z1))/R12
        cos_beta2 = -(n2[0]*(X2-x1) + n2[1]*(Y2-y1) + n2[2]*(Z2-z1))/R12
        dF12 = rcvr_in * cos_beta1*cos_beta2 / (np.pi*R12**2) * dA2
        F12ij  = dF12.sum()
        if F12ij<0.0:
            F12ij=0.0
        if F12ij>1.0:
            F12ij=1.0
        F12[i,j] = F12ij
        
    return X1,Y1,Z1,F12


def getting_T_w(T_w,*args):
    T_pavg,xmax,xmin,ymax,ymin,beta,F_t_ap,em_p,em_w,A_ap,h_w_amb = args
    Tamb = 300.
    Tsky = Tamb - 15.
    
    X_tpr = (xmax-xmin)
    Y_tpr = (ymax-ymin)
    Z_tpr = np.abs(X_tpr * np.tan(beta))
    
    A_w = Z_tpr*Y_tpr + (Z_tpr*X_tpr/2)*2 + max((X_tpr*Y_tpr - A_ap),0.)
    A_t = Y_tpr*X_tpr/np.cos(beta)
    
    F_t_w   = 1 - F_t_ap
    F_w_ap = (1 - F_t_ap * (A_t/A_ap)) * (A_ap/A_w)
    
    C_tw = (1-em_p)/(em_p*A_t) + 1/(F_t_w*A_t) + (1-em_w)/(em_w*A_w)
    
    sigma = 5.674e-8
    
    q_w_net = (sigma/C_tw)*(T_pavg**4 - T_w**4) - F_w_ap*em_w*sigma*A_w*(T_w**4-Tsky**4) - h_w_amb*A_w*(T_w-Tamb)
    # print(F_w_ap,F_t_w, A_t,A_ap,F_t_ap)
    return q_w_net


def HTM_2D_tilted_surface(Rcvr, f_Qrc1, f_eta, f_ViewFactor, full_output=False):
    
    x_ini = Rcvr['x_ini']
    x_fin = Rcvr['x_fin']
    y_bot = Rcvr['y_bot']
    y_top = Rcvr['y_top']
    vel_p = Rcvr['vel_p']
    t_sim = Rcvr['t_sim']
    tz = Rcvr['tz']
    T_ini = Rcvr['T_ini']
    TSM = Rcvr['TSM']
    
    #Belt vector direction and vector direction
    B_xi = x_ini
    B_yi = (y_bot + y_top)/2.
    B_ux  = -1.             #Should be unitary vector (B_ux**2+B_uy**2 = 1)
    B_uy  = 0.              #Should be unitary vector
    
    B_Vx  = vel_p*B_ux      #Belt X axis velocity
    B_Vy  = vel_p*B_uy      #Belt Y axis velocity
    
    B_Ny = 100              #[-] Number of elements in Y' axis
    B_Nx = 1                #[-] Number of elements in X' axis
    B_Lx = 0.1
    B_Lz = tz
    B_Ly = (y_top-y_bot)
    B_dx = B_Lx/B_Nx        #[m] Belt x' discr (If the belt is not moving X axis, x' is different than x)
    B_dy = B_Ly/B_Ny        #[m] Belt y' discr
    dt     = 1.             #Temporal discretisation [s]
    
    #Initial positions for Belt elements
    B_y = np.array([ B_yi-B_Ly/2 + (j+0.5)*B_dy for j in range(B_Ny)])
    B_x = np.array([ B_xi for j in range(B_Ny)])
    T_B = np.array([ T_ini for j in range(B_Ny)])
    
    t=dt
    if full_output:
        P = {'t':[],'x':[],'y':[],'Tp':[],'Q_in':[], 'eta':[], 'Tnew':[], 'rcv_in':[], 'dT':[], 'Q_abs':[], 'Fv':[]}
    
    while t <= t_sim+dt:
        
        Q_in = f_Qrc1.ev(B_x,B_y)/1000
        
        # eta = np.array([f_eta(T_B[j],Q_in[j])[0] for j in range(len(T_B))])
        
        Fv  = np.array([f_ViewFactor(B_x[j],B_y[j])[0] for j in range(len(T_B))])
        eta = np.array([f_eta(Rcvr,T_B[j],Q_in[j],Fv[j])[0] for j in range(len(T_B))])
        
        Q_in = np.where(Q_in>0.01,Q_in,0.0)
        rcv_in = np.where(Q_in>0.01,True,False)
        
        Q_abs = Q_in * eta
        if TSM=='CARBO':
            rho_b = 1810
            cp_b  = 148*T_B**0.3093
        T_B1 =  Q_abs*1e6 / (cp_b * rho_b * B_Lz / dt) + T_B
        dT  = T_B1-T_B
        
        if full_output:
            P['t'].append(t*np.ones(len(B_x)))
            P['x'].append(B_x)
            P['y'].append(B_y)
            P['Tp'].append(T_B)
            P['Tnew'].append(T_B1)
            P['Q_in'].append(Q_in)
            P['eta'].append(eta)
            P['rcv_in'].append(rcv_in)
            P['dT'].append(dT)
            P['Q_abs'].append(Q_abs)
            P['Fv'].append(Fv)
        
        B_x = B_x + dt*B_Vx
        B_y = B_y + dt*B_Vy
        T_B = T_B1
        t += dt
    
    if full_output:
        for x in P.keys():
            P[x]=np.array(P[x])
    
    T_out_av = T_B.mean()
    if TSM =='CARBO':
        rho_b = 1810.
        cp_b  = (148*((T_B+T_ini)/2)**0.3093).mean()
    
    M_stg1 = rho_b * (B_Lz*B_Ly*vel_p)
    Qstg1 = M_stg1 * cp_b * (T_out_av - T_ini) /1e6
    
    if full_output:
        return [ T_B, Qstg1, M_stg1, P ]
    else:
        return [ T_B, Qstg1, M_stg1 ]


def rcvr_TPR_0D(CST):
    abs_p = 0.91
    em_p  = 0.85
    em_w  = 0.20
    hcond = 0.833
    Fc    = 2.57
    Fv    = 0.4
    air = air=ct.Solution('air.yaml')
    T_w   = CST['T_pC']
    beta  = -27.
    tz    = 0.06
    
    zf,Type,Array,rO,Cg,xrc,yrc,zrc,Npan,T_amb = [CST[x] for x in ['zf', 'Type', 'Array', 'rO_TOD', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'N_pan','T_amb']]
    TOD = BDR.TOD_Params({'Type':Type, 'Array':Array, 'rO':rO, 'Cg':Cg}, xrc, yrc, zrc)
    x0,y0,V_TOD,N_TOD,H_TOD,rO,rA,Arcv = [TOD[x] for x in ['x0', 'y0', 'V', 'N', 'H', 'rO', 'rA', 'A_rcv']]
    xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[0],y0[0],zrc)
    x_fin,x_ini = xO.min(),xO.max()
    y_bot,y_top = yO.min(),yO.max()
    
    Rcvr = {'A_rc1':Arcv, 'em_p':em_p, 'em_w':em_w, 'abs_p':abs_p, 'hcond':hcond, 'Fc':Fc, 'Fv_avg':Fv, 'Tw':T_w, 'air':air, 'beta':beta,'x_ini':x_ini,'x_fin':x_fin, 'y_bot':y_bot, 'y_top':y_top, 'tz':tz, 'T_amb':T_amb}
    
    T_ini  = CST['T_pC']
    T_out  = CST['T_pH']
    Q_avg  = CST['Qavg']
    Prcv   = CST['P_rcv'] 
    Tp_avg = (T_ini+T_out)/2.
    eta_rcv = eta_th_tilted(Rcvr, Tp_avg, Q_avg, Fv=Fv)[0]
    
    rho_b = 1810
    cp = 148*Tp_avg**0.3093
    t_res = rho_b * cp * tz * (T_out - T_ini ) / (Q_avg*1e6*eta_rcv)
    M_p = Prcv*1e6 / (cp*(T_out - T_ini))
    P_SF = Prcv / eta_rcv
    Ltot  = Arcv/(y_top-y_bot)
    vel_p = Ltot / t_res

    return [eta_rcv, P_SF, Tp_avg, t_res, M_p, Arcv, vel_p]


def rcvr_TPR_0D_corr(
        CST: dict,
        TOD: TertiaryOpticalDevice,
        SF: pd.DataFrame | None = None
    ) -> ReceiverOutput:
    abs_p = 0.91
    em_p  = 0.85
    em_w  = 0.20
    hcond = 0.833
    Fc    = 2.57
    Fv    = 0.4
    air = air=ct.Solution('air.yaml')
    T_w   = CST['T_pC']
    T_amb = CST["T_amb"]
    beta  = -27.
    tz    = 0.06
    
    xO,yO = TOD.perimeter_points(TOD.radius_out, tod_index=0)
    Arcv = TOD.receiver_area
    x_fin,x_ini = xO.min(),xO.max()
    y_bot,y_top = yO.min(),yO.max()
    
    Rcvr = {'A_rc1':Arcv, 'em_p':em_p, 'em_w':em_w, 'abs_p':abs_p, 
            'hcond':hcond, 'Fc':Fc, 'Fv_avg':Fv, 'Tw':T_w, 
            'air':air, 'beta':beta,'x_ini':x_ini,'x_fin':x_fin, 
            'y_bot':y_bot, 'y_top':y_top, 'tz':tz, 'T_amb':T_amb}
    
    T_ini  = CST['T_pC']
    T_out  = CST['T_pH']
    Q_avg  = CST['Qavg']
    Prcv   = CST['P_rcv'] 
    Tp_avg = (T_ini+T_out)/2.
    eta_rcv = eta_th_tilted(Rcvr, Tp_avg, Q_avg, Fv=Fv)[0]
    
    rho_b = 1810
    cp = 148*Tp_avg**0.3093
    t_res = rho_b * cp * tz * (T_out - T_ini ) / (Q_avg*1e6*eta_rcv)
    M_p = Prcv*1e6 / (cp*(T_out - T_ini))
    P_bdr = Prcv / eta_rcv
    Ltot  = Arcv/(y_top-y_bot)
    vel_p = Ltot / t_res

    #Correcting 0D model with linear fit
    m = 0.6895
    n = 0.2466
    eta_rcv_corr = eta_rcv*m+n

    if SF is None:
        n_hels = np.nan
    else:
        Q_acc = SF['Q_h1'].cumsum()
        n_hels = len( Q_acc[ Q_acc < P_bdr ] ) + 1

    rcvr_output = {
        "temps_parts" : np.array([Tp_avg]),
        "n_hels" : n_hels,
        "rad_flux_max" : np.nan,            #No calculated
        "rad_flux_avg" : P_bdr/Arcv,
        "heat_stored": np.nan,            #No calculated
        "eta_rcv": eta_rcv_corr,
        "mass_stg": M_p,
        "time_res": t_res,
        "vel_p": vel_p,
    }
    return rcvr_output


def TPR_2D_model(
        CST: dict,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        TOD: dict,
        polygon_i: int = 1,
        full_output: bool = False
    ):
    
    #Working only with 1 receiver
    #Selecting as pivoting point the 'most-right' point in receiver
    
    def F_Toutavg(t_res_g,*args):
        Rcvr,f_Qrc1,f_eta,f_ViewFactor = args
        Rcvr['vel_p']  = Rcvr['xavg']/t_res_g                           #Belt velocity magnitud
        Rcvr['t_sim']  = (Rcvr['x_ini']-Rcvr['x_fin'])/Rcvr['vel_p']    #Simulate all the receiver
        T_p, Qstg1, M_stg1 = HTM_2D_tilted_surface(Rcvr,f_Qrc1,f_eta,f_ViewFactor)
        # print(T_p.mean())
        return T_p.mean() - Rcvr['T_out']
    
    T_ini, T_out, tz, TSM, Prcv, Qavg = [CST[x] for x in ['T_pC', 'T_pH', 'tz', 'TSM', 'P_rcv', 'Qavg']]
    Type, Array, rO, Cg, zrc = [CST[x] for x in ['Type', 'Array','rO_TOD','Cg_TOD','zrc']]
    if CST['TSM'] == 'CARBO':
        rho_b = 1810
    
    x0,y0,V_TOD,N_TOD,H_TOD,rO,rA,Arcv = [TOD[x] for x in ['x0', 'y0','V','N','H','rO','rA','A_rcv']]
    A_rc1  = Arcv/N_TOD
    
    abs_p = 0.91
    em_p  = 0.85
    em_w  = 0.20
    hcond = 0.833
    Fc = 2.57
    air = air=ct.Solution('air.yaml')
    T_w  = 800.                        #initial guess for Tw
    beta = -27. * np.pi/180.
    
    # Calculating MCRT in tilted surface
    R3  = R2[(R2["Npolygon"]==polygon_i)&R2["hit_rcv"]].copy()
    R3b = R2[(R2["hit_rcv"])].copy()

    #Point of pivoting belonging to the plane
    xp = R3["xr"].max() + 0.1
    yp = 0.                     #This is valid for 1st hexagon, if not:  R3.loc[R3.xr.idxmax()].yr
    zp = R3["zr"].mean()
    
    #Obtaining the intersection between the rays and the new plane
    kt = (xp - R3["xr"]) / ( R3["uzr"]/np.tan(beta) + R3["uxr"])
    R3['xt'] = R3["xr"] + kt * R3["uxr"]
    R3['yt'] = R3["yr"] + kt * R3["uyr"]
    R3['zt'] = R3["zr"] + kt * R3["uzr"]
    
    # STARTING THE LOOP HERE
    # Initial guess for number of heliostats
    air=ct.Solution('air.yaml')
    T_av = (T_ini+T_out)/2
    eta_rcv = HTM_0D_blackbox( T_av, Qavg, Fc=2.57, air=air)[0]
    Q_acc = SF['Q_h1'].cumsum()
    N_hel = len( Q_acc[ Q_acc < (Prcv / eta_rcv) ] ) + 1
    
    it_max = 20;
    it = 1
    loop=0
    data = []
    N_hels = []
    solve_t_res = False
    Loop_break = False
    while True:
        
        #Defining number of heliostats
        hlst    = SF.iloc[:N_hel].index
        SF_it   = SF.loc[hlst]              #Array only valid in iteration it
        Etas_it = SF_it.mean()              #Array only valid in iteration it
        P_SF_it = SF_it["Q_pen"].sum()         #Value only valid in iteration it
        eta_SF_it = Etas_it['Eta_SF']       #Value only valid in iteration it
        
        R3i   = R3[R3["hel"].isin(hlst)].copy()
        r_i = len(R3i)/len(R3b[R3b["hel"].isin(hlst)])                #Fraction of rays that goes into one TOD
        P_SF_i = r_i * P_SF_it * 1e6

        #### getting the radiation flux function        
        Nx=100
        Ny=50
        xO,yO = BDR.TOD_XY_R(rO,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        xA,yA = BDR.TOD_XY_R(rA,H_TOD,V_TOD,N_TOD,x0[polygon_i-1],y0[polygon_i-1],zrc)
        xmin = max(xA.min(),R3i.xt.quantile(0.01))
        xmax = min(xA.max(),R3i.xt.quantile(0.99))
        ymin = max(yA.min(),R3i.yt.quantile(0.01))
        ymax = min(yA.max(),R3i.yt.quantile(0.99))

        dx = (xmax-xmin)/Nx
        dy = (ymax-ymin)/Ny
        dA=dx*dy
        Nrays = len(R3i.xt[(R3i.xt>xmin)&(R3i.xt<xmax)&(R3i.yt>ymin)&(R3i.yt<ymax)])    #Taking out rays that don't hit in receiver
        
        P_SF = eta_SF_it * (CST['Gbn']*CST['A_h1']*N_hel)
        P_SF1 = P_SF * r_i
        Fbin    = P_SF1/(1e3*dA*Nrays)
        Q_rc1,X_rc1, Y_rc1 = np.histogram2d(R3i.xt, R3i.yt, bins=[Nx,Ny], range=[[xmin, xmax], [ymin, ymax]], density=False)
        Q_rc1 = Fbin * Q_rc1
        Q_max = Q_rc1.max()/1000.
        
        # reducing receiver size
        Q_avg_min = 250       #[kW/m2] Minimal average radiation allowed per axis
        ymin_corr = Y_rc1[:-1][ Q_rc1.mean(axis=0) > Q_avg_min][0]
        ymax_corr = Y_rc1[:-1][ Q_rc1.mean(axis=0) > Q_avg_min][-1]
        
        Q_avg_min = 250       #[kW/m2] Minimal average radiation allowed per axis
        xmin_corr = X_rc1[:-1][ Q_rc1.mean(axis=1) > Q_avg_min][0]
        xmax_corr = X_rc1[:-1][ Q_rc1.mean(axis=1) > Q_avg_min][-1]
        
        X_av = np.array([(X_rc1[i]+X_rc1[i+1])/2 for i in range(len(X_rc1)-1)])
        Y_av = np.array([(Y_rc1[i]+Y_rc1[i+1])/2 for i in range(len(Y_rc1)-1)])
        Q_rcf = Q_rc1[(X_av>xmin_corr)&(X_av<xmax_corr),:][:,(Y_av>ymin_corr)&(Y_av<ymax_corr)]
        
        X_rcf = X_av[(X_av>xmin_corr)&(X_av<xmax_corr)]
        Y_rcf = Y_av[(Y_av>ymin_corr)&(Y_av<ymax_corr)]
        
        eta_tilt = Q_rcf.sum()*dA/(P_SF1/1000)
        eta_tilt_rfl = eta_tilt + (1-eta_tilt)*(1-em_w)
        F_corr = eta_tilt_rfl / eta_tilt
        
        Q_rcf = F_corr * Q_rcf
        
        lims = xmin_corr,xmax_corr,ymin_corr,ymax_corr
        x_fin = lims[0]
        x_ini = lims[1]
        y_bot = lims[2]
        y_top = lims[3]
        

        # # Plotting of resulting radiation map
        # fig = plt.figure(figsize=(14, 8))
        # ax = fig.add_subplot(111, aspect='equal')
        # X, Y = np.meshgrid(X_rcf, Y_rcf)
        # f_s = 16
        # vmin = 0
        # vmax = (np.ceil(Q_rcf.max()/100)*100)
        # surf = ax.pcolormesh(X, Y, Q_rcf.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
        # ax.set_xlabel('Tiled axis (m)',fontsize=f_s);ax.set_ylabel('Cross axis (m)',fontsize=f_s);
        # cb = fig.colorbar(surf, shrink=0.5, aspect=4)
        # cb.ax.tick_params(labelsize=f_s-2)
        # fig.text(0.79,0.72,r'$Q_{in}[kW/m^2]$',fontsize=f_s)
        # plt.show()
        # fig.savefig(fldr_rslt+case+'_Q_in.png', bbox_inches='tight')
        # plt.close()
        
        # f_Qrc1= spi.RectBivariateSpline(X_rc1[:-1],Y_rc1[:-1],Q_rc1)     #Function to interpolate
        f_Qrc1 = spi.RectBivariateSpline(X_rcf,Y_rcf,Q_rcf)               #Function to interpolate
        
        

        #### Obtaining view factor function
        # lims = (xmin,xmax,ymin,ymax)
        X1,Y1,Z1,F12 = get_view_factor(CST,TOD,polygon_i,lims,xp,yp,zp,beta)
        X2 = X1[0,:]
        Y2 = Y1[:,0]
        f_ViewFactor = spi.interp2d(X2,Y2,F12)
        Fv = F12.mean()
        
        #### Obtaining wall temperature
        if it==1:
            T_pavg = (T_ini+T_out)/2.
            args = (T_pavg,xmax_corr,xmin_corr,ymax_corr,ymin_corr,beta,Fv,em_p,em_w,A_rc1,hcond)
            T_w = spo.fsolve(getting_T_w, T_w, args=args)[0]
        # print(T_w)
        
        #Rcvr conditions for simulation
        Rcvr = {'Qavg':Qavg, 'A_rc1':A_rc1, 'P_SF1':P_SF1,'T_ini':T_ini,'T_out':T_out,
                'TSM':TSM, 'em_p':em_p, 'em_w':em_w, 'abs_p':abs_p, 'hcond':hcond,
                'Fc':Fc, 'Fv_avg':Fv, 'Tw':T_w, 'air':air,
                'beta':beta,'x_ini':x_ini,'x_fin':x_fin, 'y_bot':y_bot, 'y_top':y_top, 'tz':tz
                }
        
        # OBTAINING RESIDENCE TIME
        x_fin = lims[0]
        x_ini = lims[1]
        y_bot = lims[2]
        y_top = lims[3]
        A_rc1  = Arcv/N_TOD
        xavg   = A_rc1/(y_top-y_bot)
        
        if it==1:
            T_av = 0.5*(T_ini+T_out)
            Q_av = P_SF_it / Arcv
            cp = 148*T_av**0.3093
            eta_rcv_g = eta_th_tilted(Rcvr,T_av,Qavg,Fv=Fv)[0]
        else:      
            eta_rcv_g = eta_rcv
        
        if solve_t_res:
            t_res_g = rho_b * cp * tz * (T_out - T_ini ) / (eta_rcv_g * Q_av*1e6)
            
            vel_p  = xavg/t_res_g                #Belt velocity magnitud
            t_sim  = (x_ini-x_fin)/vel_p         #Simulate all the receiver
            # inputs = (x_ini,x_fin,y_bot,y_top,vel_p,t_sim,tz,T_ini,TSM,f_Qrc1,f_eta)
            Rcvr['xavg']  = xavg
            Rcvr['vel_p'] = vel_p
            Rcvr['t_sim'] = t_sim
            
            args = (Rcvr, f_Qrc1, eta_th_tilted,f_ViewFactor)
            sol  = spo.fsolve(F_Toutavg, t_res_g, args=args, xtol=1e-2,full_output=True)
            t_res = sol[0][0]
            n_evals = sol[1]['nfev']
            CST['t_res'] = t_res
            CST['vel_p'] = xavg/t_res
        else:
            t_res_g = rho_b * cp * tz * (T_out - T_ini ) / (eta_rcv_g * Q_av*1e6)
            t_res = t_res_g
        
        #Final values for temperature Qstg1 and Mstg1
        vel_p  = xavg/t_res                #Belt velocity magnitud
        t_sim  = (x_ini-x_fin)/vel_p           #Simulation [s]
        Rcvr['xavg']  = xavg
        Rcvr['vel_p'] = vel_p
        Rcvr['t_sim'] = t_sim
        T_p,Qstg1,Mstg1,Rcvr_full = HTM_2D_tilted_surface(Rcvr, f_Qrc1, eta_th_tilted, f_ViewFactor, full_output=True)
        
        Rcvr_full['Q_gain'] = Rcvr_full['Q_in']*Rcvr_full['eta']      #[J]
        
        #Performance parameters
        eta_rcv = Qstg1*1e6/P_SF_i
        Qstg = Qstg1 / r_i
        Mstg = Mstg1 / r_i
        Q_av = P_SF_it / Arcv
        Q_max = Q_rc1.max()/1000.
        
        P_SF  = Prcv / eta_rcv
        Q_acc = SF['Q_h1'].cumsum()
        N_new = len( Q_acc[ Q_acc < P_SF ] ) + 1
        
        repeated = N_new in N_hels
        # print(solve_t_res,it,loop,N_hel,N_new,repeated,T_w,Qstg,T_p.mean(),eta_rcv)
        
        #CHECKING CONVERGENCE
        conv = abs(Qstg-Prcv)<0.1 and (abs(T_p.mean()-T_out)<1.)
        
        if (N_new == N_hel) and conv:
            print('System converged')
            break
        
        # if (abs(N_new-N_hel) == 1):
        #     loop+=1
        if Loop_break:
            print('System enters in a loop.')
            break
        
        if loop==5:
            N_new = (N_hel+N_new)//2
            Loop_break = True
        
        if N_new in N_hels:
            solve_t_res=True
            # print(loop,N_new)
            N_new = (N_hel + N_new)//2
            if N_new in N_hels:
                N_new +=1
            loop+=1
        
        if (it==it_max) and conv and (solve_t_res==False):
            print('System did not converge, solve_t_res will be used')
            it_max = 40
            solve_t_res = True
        
        if it==it_max and solve_t_res:
            print('System did not converge for either method')
            break
        
        # N_hel_its.append(N_hel)
        N_hel = N_new
        N_hels.append(N_new)
        
        it+=1
    
    # print(P_SF,N_new,eta_rcv,Qstg,vel_p)
    results = [ N_hel, Q_max, Q_av, P_SF_i, Qstg, P_SF_it, eta_rcv, Mstg, t_res, vel_p, it, solve_t_res]
    # results = [ N_hel, Q_max, Q_av, P_SF1, Qstg, eta_rcv, Mstg, t_res, vel_p ]
    
    return results, T_p, Rcvr, Rcvr_full

# COUPLED FUNCTIONS
def get_plant_costs():
    Ci = {}
    Ci['R_ftl']  = 1.3                   # Field to land ratio, SolarPilot
    Ci['C_land'] = 2.47                  # USD per m2 of land. SAM/SolarPilot (10000 USD/acre)
    Ci['C_site'] = 16.                   # USD per m2 of heliostat. SAM/SolarPilot
    Ci['C_hel']  = 100.                  # USD per m2 of heliostat. Kurup et al. (2022)
    
    Ci['F_HB']   = 1.                   # [-] Factor for cost in HB mirror
    Ci['F_TOD']  = 1.                   # [-] Factor for cost in TOD mirror
    Ci['F_rcv']  = 1.                   # [-] Factor for cost in receiver
    Ci['F_tow']  = 1.                   # [-] Factor for cost in tower
    Ci['F_pb']   = 1.                   # [-] Factor for cost in power block
    Ci['F_stg']  = 1.                   # [-] Factor for cost in storage material
    
    # Ci['e_rcv']  = 0.7                   # Scale factor in receiver
    Ci['C_tpr']  = 37400.                # [USD/m2] Same value from FPR (Albrecht et al., 2019)
    
    Ci['tow_type'] = 'Antenna'           #'Antenna' or 'Conv'
    Ci['C_tow'] = 3e6 * 0.75             # USD fixed. SAM/SolarPilot, assumed tower 25% cheaper. Conventional
    Ci['e_tow'] = 0.0113                 # Scaling factor for tower. Conventional
    Ci['C_OM'] = 0.02                    # % of capital costs.
    Ci['DR']   = 0.05                    # Discount rate
    Ci['Ny']   = 30                      # Horizon project
    
    Ci['C_parts'] = 1.0                  # USD/kg (Gonzalez-Portillo et al. 2022). CARBO HSP
    
    Ci['C_pb']   = 1e6                   # USD/MWe
    Ci['C_pb_est'] = 'Sandia'            # Source to estimate PB costs.
                                         # Two options: 'Neises' or 'Sandia'
                                         # Neises is based on python module
                                         # Sandia is based on Gonzalez-Portillo et al. (2021)
                                         
    Ci['C_xtra'] = 1.3                   # Engineering and Contingency
    
    Ci['CF_sf']  = 0.2                   # Capacity Factor (Solar field) if not calculated
    
    return Ci

######################################
def BDR_cost(SF,CST):
    
    zmax,zrc,A_h1, S_HB = [CST[x] for x in ['zmax', 'zrc', 'A_h1', 'S_HB']]
    
    T_stg   = CST['T_stg']                   # hr storage
    SM      = CST['SM']                   # (-) Solar multiple (P_rcv/P_pb)
    eta_rcv = CST['eta_rcv']
    eta_pb  = CST['eta_pb']
    T_pH    = CST['T_pH']
    T_pC    = CST['T_pC']
    HtD_stg = CST['HtD_stg']
    
    S_HB    = CST['S_HB']
    S_TOD   = CST['S_TOD']
    Arcv    = CST['Arcv']
    M_p     = CST['M_p']
    
    Ci      = CST['costs_in']
    
    #Solar field related costs
    R_ftl  = Ci['R_ftl']                   # Field to land ratio, SolarPilot
    C_land = Ci['C_land']                  # USD per m2 of land. SAM/SolarPilot (10000 USD/acre)
    C_site = Ci['C_site']                 # USD per m2 of heliostat. SAM/SolarPilot
    C_hel  = Ci['C_hel']                  # USD per m2 of heliostat. Projected by Pfal et al (2017)
    C_tpr  = Ci['C_tpr']                  # [USD/m2]
    C_tow = Ci['C_tow']             # USD fixed. SAM/SolarPilot, assumed tower 25% cheaper
    e_tow = Ci['e_tow']                 # Scaling factor for tower
    C_parts = Ci['C_parts']
    tow_type = Ci['tow_type']
    
    #Factors to apply on costs. Usually they are 1.0.
    F_HB   = Ci['F_HB']
    F_TOD  = Ci['F_TOD']
    F_rcv  = Ci['F_rcv']
    F_tow  = Ci['F_tow']
    F_stg  = Ci['F_stg']
    
    C_OM = Ci['C_OM']                    # % of capital costs.
    DR   = Ci['DR']                    # Discount rate
    Ny   = Ci['Ny']                      # Horizon project
    
    C_pb    = Ci['C_pb']                                #USD/MWe
    C_PHX   = Ci['C_PHX'] if 'C_PHX' in Ci else 0.      #USD/MWe
    
    C_xtra = Ci['C_xtra']                   # Engineering and Contingency
    
    P_rcv = CST['P_rcv_sim'] if 'P_rcv_sim' in CST else CST['P_rcv'] #[MW] Nominal power stored by receiver
    P_pb  = P_rcv / SM              #[MWth]  Thermal energy delivered to power block
    Q_stg = P_pb * T_stg            #[MWh]  Energy storage required to meet T_stg
    
    #######################################
    Co = {}      #Everything in MM USD, unless explicitely indicated
    
    #######################################
    # CAPACITY FACTOR
    # CF_sf = Capacity Factor (Solar field)
    # CF_pb = Capacity Factor (Power block)
    
    if CST['type_weather'] == 'TMY':
        if 'file_weather' in CST:
            file_weather = CST['file_weather']
            df_w = pd.read_csv(file_weather,header=1)
            DNI_tag = 'DNI (W/m^2)'
            DNI_des = CST['Gbn']
            DNI_min = 400.
            F_sf    = 0.9       #Factor to make it conservative
            Qacc    = df_w[df_w[DNI_tag]>=DNI_min][DNI_tag].sum()  #In Wh/m2
            CF_sf  = F_sf * Qacc / (DNI_des*len(df_w))
            CF_pb = CF_sf*SM
        else:
            print('Weather File not found. Default value used for CF_sf')
            CF_sf  = Ci['CF_sf']
            CF_pb = CF_sf*SM                      # Assumed all is converted into electricity
            
    elif CST['type_weather'] == 'MERRA2':  # Not implemented yet.
        CF_sf  = Ci['CF_sf']               # Assumes CF_sf is calculated outside and stored
        CF_pb  = Ci['CF_pb']               # Assumes CF_pb is calculated outside and stored
        # print('MERRA2')
    else:       #If nothing is given, assumed default values
        CF_sf = Ci['CF_sf']
        CF_pb = CF_sf*SM
    
    Co['CF_sf'] = CF_sf
    Co['CF_pb'] = CF_pb
    #######################################
    # LAND CALCULATIONS
    SF_in = SF[SF.hel_in].copy()
    S_hel = len(SF_in) * A_h1
    
    N_sec = 8
    x_c,y_c = SF_in.xi.mean(),SF_in.yi.mean()
    SF_in['ri_c']  = ((SF_in["xi"] - x_c)**2 + (SF_in["yi"] - y_c)**2)**0.5
    SF_in['theta'] = np.arctan2(SF_in["yi"],SF_in["xi"])
    SF_in['bin'] = SF_in['theta']//(2*np.pi/N_sec) + N_sec/2
    points = pd.DataFrame([],columns=['xi','yi','ri_c'])
    for n in range(N_sec):
        SF_n = SF_in[SF_in["bin"]==n]
        set_n = SF_n.loc[SF_n["ri_c"].nlargest(10).index][['xi','yi','ri_c']]
        if n==0:
            points = set_n
        else:
            points = pd.concat([points,set_n])
    S_land = np.pi * (points["ri_c"].mean())**2
    
    Co['land'] = ( C_land*S_land*R_ftl + C_site*S_hel )  / 1e6
    
    #######################################
    # MIRROR MASSES
    M_HB_fin  = CST['M_HB_fin']          # Fins components of HB weight
    M_HB_t    = CST['M_HB_tot']          # Total weight of HB
    
    M_TOD_mirr  = (S_TOD * 15.1 /1000)
    F_TOD_pipe  = 2.0                       # Extra factor due to fins (must be corrected)
    M_TOD_t     = M_TOD_mirr*(1 + F_TOD_pipe)
    
    #######################################
    # MIRROR COSTS
    F_struc_hel = 0.17
    zhel        = 4.0
    F_HB_fin    = M_HB_fin / (M_HB_t - M_HB_fin)
    F_mirr_HB   = 0.34               # Double than heliostat's
    F_struc_HB  = F_struc_hel * np.log((zmax-3.0)/0.003) / np.log((zhel-3.0)/0.003)
    F_cool_HB   = F_struc_HB * F_HB_fin
    F_oth_HB    = 0.20               # Double than heliostat's
    
    F_mirr_TOD  = 0.68              # Fourfold than heliostat's
    F_struc_TOD = F_struc_hel * np.log((zrc-3.0)/0.003) / np.log((zhel-3.0)/0.003)
    F_cool_TOD  = (8000 + 252.9*S_TOD**0.91) / (C_hel*S_TOD)  # Normalized Hall's correlation
    F_oth_TOD   = 0.20               # Double than heliostats
    
    C_HB  = C_hel * (F_mirr_HB + F_struc_HB + F_cool_HB + F_oth_HB)
    C_TOD = C_hel * (F_mirr_TOD + F_struc_TOD + F_cool_TOD + F_oth_TOD)
    
    Co['hel'] = C_hel * S_hel /1e6
    Co['HB']  = F_HB * C_HB * S_HB /1e6
    Co['TOD'] = F_TOD * C_TOD * S_TOD /1e6
    
    #######################################
    # TOWER COST
    if tow_type == 'Conv':
        Co['tow'] = C_tow * np.exp(e_tow*zmax) /1e6
    elif 'Antenna':
        M_HB_1   = M_HB_t / 4.          #Weight per column
        M_TOD_1  = M_TOD_t / 4.
        C_tow_HB = ( (123.21 + 362.6*M_HB_1) * np.exp(0.0224*zmax) / 1e6 ) * 4
        C_tow_TOD = ( (123.21 + 362.6*M_TOD_1) * np.exp(0.0224*zrc) / 1e6 ) * 4
        Co['tow'] = F_tow * (C_tow_HB + C_tow_TOD)
    
    #######################################
    # STORAGE COST
    dT_stg  = T_pH - T_pC                   # K
    Tp_avg  = (T_pH + T_pC)/2.              # K
    cp_stg  = 148*Tp_avg**0.3093
    rho_stg = 1810.
    
    M_stg = (Q_stg * 3600 * 1e6) / (cp_stg * dT_stg)            #[kg]
    V_stg = M_stg / rho_stg                                     #[m3]
    C_stg_p  = 1.1 * C_parts * M_stg  /1e6                      #[MM USD]
    
    C_stg_H = 1000. * (1 + 0.3*(1 + (T_pH - 273.15 - 600)/400.))
    C_stg_C = 1000. * (1 + 0.3*(1 + (T_pC - 273.15 - 600)/400.))
    D_stg   = ( 4. * V_stg / np.pi / HtD_stg) **(1./3.)
    H_stg   = HtD_stg*D_stg
    A_tank  = np.pi * ( D_stg**2/2. + D_stg * H_stg)
    C_stg_t = A_tank * (C_stg_H + C_stg_C) / 1e6            #[MM USD]
    
    Co['stg'] = F_stg*C_stg_p + C_stg_t      #[MM USD]
    Co['H_stg'] = H_stg
    Co['V_stg'] = V_stg
    #######################################
    # RECEIVER COST
    # Co['rcv'] = (C_rcv * Prcv) * (50/Prcv)**e_rcv /1e6
    C_tpr = 37400           #[USD/m2]
    C_rcv_tpr = C_tpr * Arcv
    hlift_CH = H_stg + 2.
    hlift_HC = H_stg + 5.
    C_lift = ( 58.37*hlift_CH + 58.37*hlift_HC/SM ) * M_p
    Co['rcv'] = F_rcv * ( C_rcv_tpr + C_lift ) /1e6
    
    #######################################
    # TOTAL HEAT COSTS
    Co['Heat'] = (Co['land'] + Co['hel'] + Co['HB'] + Co['TOD'] + Co['rcv'] + Co['tow'] + Co['stg']) * C_xtra
    Co['SCH'] = Co['Heat'] / P_rcv if P_rcv>0 else np.inf      #(USD/Wp) Specific cost of Heat (from receiver)
    
    #######################################
    # LEVELIZED COST OF HEAT
    #Levelised cost of heat (sun-to-storage)
    TPF  = (1./DR)*(1. - 1./(1.+DR)**Ny)
    P_yr = CF_sf * 8760 * P_rcv  /1e6            #[TWh_th/yr]
    Co['LCOH'] =  Co['Heat'] * (1. + C_OM*TPF) / (P_yr * TPF) if P_yr>0 else np.inf #USD/MWh delivered from receiver
    
    Co['land_prod'] = P_rcv/(S_land/1e4)            #MW/ha
    
    #######################################
    # LEVELIZED COST OF ELECTRICITY
    # Levelised cost of electricity (sun-to-electricity)
    
    Ntower     = CST['Ntower'] if 'Ntower' in CST else 1
    Pel        = Ntower * eta_pb * P_pb           #[MWe]  Nominal power of power block
    Co['PB']   = C_pb * Pel / 1e6
    Co['PHX']  = C_PHX * P_pb / 1e6
    Co['Elec'] = Ntower*Co['Heat'] +  (Co['PHX']+Co['PB']) * C_xtra
    E_yr       = CF_pb * 8760 * Pel   /1e6            #[TWh_e/yr]
    Co['LCOE'] =  Co['Elec'] * (1. + C_OM*TPF) / (E_yr * TPF)  if E_yr>0 else np.inf
    
    
    # Receiver and HX costs
    # C_rcvbase  = 1e8
    # A_rcvbase  = 1571
    # e_rcv      = 0.7
    # A_rcv      = np.pi * CST['DM_rcv']**2/4. * CST['H_rcv']
    # Cost_rcv = C_rcvbase * (A_rcv/A_rcvbase)**e_rcv
    
    return Co

def initial_eta_rcv(CSTi):
    Tp_avg = (CSTi['T_pC']+CSTi['T_pH'])/2
    geometry  = CSTi['Type']
    array = CSTi['Array']
    Cg = CSTi["Cg_TOD"]
    xrc, yrc, zrc = (CSTi["xrc"],CSTi["yrc"],CSTi["zrc"])

    eta_rcv1 = HTM_0D_blackbox(Tp_avg,CSTi['Qavg'])[0]
    Arcv = (CSTi['P_rcv']/eta_rcv1) / CSTi['Qavg']
    TOD = BDR.TertiaryOpticalDevice(
            params={"geometry":geometry, "array":array, "Cg":Cg, "Arcv":Arcv},
            xrc=xrc, yrc=yrc, zrc=zrc,
        )
    CSTi['rO_TOD'] = TOD.radius_out
    results = rcvr_TPR_0D_corr(CSTi, TOD)
    eta_rcv = results["eta_rcv"]

    return eta_rcv, Arcv, TOD.radius_out

##################################

from bdr_csp.BeamDownReceiver import (
    SolarField,
    HyperboloidMirror,
    TertiaryOpticalDevice
)
def run_coupled_simulation(
        CST: dict,
        HSF: SolarField,
        HB: HyperboloidMirror,
        TOD: TertiaryOpticalDevice,
    **kwargs
) -> list[pd.DataFrame, pd.DataFrame, dict]:

    #Getting the RayDataset
    R0, SF = HSF.load_dataset(save_plk=True)

    #Getting interceptions with HB
    R1 = HB.mcrt_direct(R0, refl_error=True)
    R1['hel_in'] = True
    HB.rmin = R1['rb'].quantile(0.0001)
    HB.rmax = R1['rb'].quantile(0.9981)
    R1['hit_hb'] = (R1['rb']>HB.rmin)&(R1['rb']<HB.rmax)
    CST['rmin'] = HB.rmin
    CST['rmax'] = HB.rmax
    
    SF = HB.shadow_simple( lat=CST['lat'], lng=CST["lng"], type_shdw="simple", SF=SF)
    
    #Interceptions with TOD
    R2 = TOD.mcrt_solver(R1, refl_error=False)
    
    ### Optical Efficiencies
    SF = BDR.optical_efficiencies(CST,R2,SF)
    
    ### Running receiver simulation and getting the results
    type_rcvr = CST['type_rcvr'] if 'type_rcvr' in CST else 'None'
    
    if type_rcvr=='HPR_2D':
        receiver = HPR2D(
            nom_power=CST["P_rcv"],
            heat_flux_avg= CST["Qavg"],
            temp_ini= CST["T_pC"],
            temp_out= CST["T_pH"],
            material= CST["TSM"],
            thickness_parts= CST["tz"],
            factor_htc = 2.57
        )
        rcvr_output = receiver.run_model(TOD,SF,R2)
        # rcvr_output = rcvr_HPR_2D_simple(CST, TOD, SF, R2)
        T_p = rcvr_output["temps_parts"]
        N_hel = rcvr_output["n_hels"]
        Q_max = rcvr_output["rad_flux_max"]
        Q_av = rcvr_output["rad_flux_avg"]
        Qstg = rcvr_output["heat_stored"]
        eta_rcv = rcvr_output["eta_rcv"]
        M_p = rcvr_output["mass_stg"]
        t_res = rcvr_output["time_res"]

    elif type_rcvr=='TPR_2D':
        results, T_p, Rcvr, Rcvr_full = TPR_2D_model(CST, R2, SF, TOD, full_output=True)
        N_hel, Q_max, Q_av, P_rc1, Qstg, P_SF2, eta_rcv, M_p, t_res, vel_p, it, solve_t_res = results

    elif type_rcvr=='HPR_0D':
        receiver = HPR0D(
            nom_power=CST["P_rcv"],
            heat_flux_avg= CST["Qavg"],
            temp_ini= CST["T_pC"],
            temp_out= CST["T_pH"],
            material= CST["TSM"],
            thickness_parts= CST["tz"],
            factor_htc = 2.57
        )
        rcvr_output = receiver.run_model(SF)
        # rcvr_output = rcvr_HPR_0D_simple(CST, SF)
        T_p = rcvr_output["temps_parts"]
        N_hel = rcvr_output["n_hels"]
        Q_max = rcvr_output["rad_flux_max"]
        Q_av = rcvr_output["rad_flux_avg"]
        Qstg = rcvr_output["heat_stored"]
        eta_rcv = rcvr_output["eta_rcv"]
        M_p = rcvr_output["mass_stg"]
        t_res = rcvr_output["time_res"]
    
    elif type_rcvr=='TPR_0D':
        rcvr_output = rcvr_TPR_0D_corr(CST, TOD, SF)
        T_p = rcvr_output["temps_parts"]
        N_hel = rcvr_output["n_hels"]
        Q_max = rcvr_output["rad_flux_max"]
        Q_av = rcvr_output["rad_flux_avg"]
        Qstg = rcvr_output["heat_stored"]
        eta_rcv = rcvr_output["eta_rcv"]
        M_p = rcvr_output["mass_stg"]
        t_res = rcvr_output["time_res"]
    
    elif type_rcvr in ['SEVR','None']:
        Prcv, eta_rcv, Qavg = CST['P_rcv'], CST['eta_rcv'], CST["Qavg"]
        P_bdr = Prcv / eta_rcv
        Q_acc    = SF['Q_h1'].cumsum()
        N_hel    = len( Q_acc[ Q_acc < P_bdr ] ) + 1

        rcvr_output = {
            "temps_parts" : CST["T_pH"],
            "n_hels" : len( Q_acc[ Q_acc < P_bdr ] ) + 1,
            "rad_flux_max" : np.nan,            #No calculated
            "rad_flux_avg" : Qavg,
            "heat_stored": np.nan,            #No calculated
            "eta_rcv": eta_rcv,
            "mass_stg": np.nan,
            "time_res": np.nan,
            "vel_p": np.nan,
        }

    # Outputs

    #Receiver Parameters
    CST['T_p']     = T_p
    CST['eta_rcv'] = eta_rcv
    CST['Q_max']   = Q_max
    CST['Q_av']    = Q_av
    CST['t_res']   = t_res
    
    #Storage parameters
    CST['M_p']   = M_p
    CST['Q_stg'] = Qstg
    
    #General parameters
    Prcv,eta_pb,SM = [CST[x] for x in ['P_rcv', 'eta_pb', 'SM']]
    CST['P_pb']  = (Prcv / SM)
    CST['P_el']  = (Prcv / SM) * eta_pb
    CST['P_SF']  = Prcv / eta_rcv
    
    A_h1,eta_pb = [CST[x] for x in ['A_h1', 'eta_pb']]
    
    # Heliostat selection
    Q_acc    = SF['Q_h1'].cumsum()
    hlst     = Q_acc.iloc[:N_hel].index
    SF['hel_in'] = SF.index.isin(hlst)
    R2['hel_in'] = R2['hel'].isin(hlst)
    CST['N_hel'] = N_hel
    CST['P_SF_sim']  = SF[SF.hel_in]['Q_h1'].sum()
    CST['P_rcv_sim'] = CST['P_SF_sim'] * eta_rcv
    CST['P_el_sim']  = CST['P_rcv_sim'] * eta_pb
    
    # Calculating HB surface
    HB.rmin = R2[R2.hel_in]['rb'].quantile(0.0001)
    HB.rmax = R2[R2.hel_in]['rb'].quantile(0.9981)
    R2['hit_hb'] = (R2['rb']>HB.rmin)&(R2['rb']<HB.rmax)
    
    HB.get_surface_area(R2)
    HB.height_range()
    
    #Masses of mirrors
    M_HB_fin, M_HB_mirr, M_HB_str, M_HB_tot = BDR.HB_mass_cooling(R2, CST, SF)
    
    # Final parameters for mirrors
    CST['rmin'] = HB.rmin
    CST['rmax'] = HB.rmax
    CST['zmin'] = HB.zmin
    CST['zmax'] = HB.zmax
    CST['S_HB']  = HB.area
    CST['S_TOD'] = TOD.surface_area
    CST['S_SF']  = N_hel*A_h1
    CST['S_tot'] = CST['S_HB'] + CST['S_TOD'] + CST['S_SF']
    
    CST['M_HB_fin'] = M_HB_fin
    CST['M_HB_tot'] = M_HB_tot
    
    #Costing calculation
    CST['Costs'] = BDR_cost(SF,CST)
    
    return R2, SF, CST