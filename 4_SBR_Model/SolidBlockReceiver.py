# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:16:55 2019

@author: z5158936
"""
import numpy as np
import pandas as pd
from scipy import constants as cte, interpolate as spi
#import scipy.linalg as spl
from scipy.sparse import linalg as spsl
import scipy.sparse as sps
import cantera as ct
#import pickle
#import os

######################################################################
######################################################################

#def Conditions():
#    variables = mat, H_t, D_t, D_r, N_t, N_r, t_sim, dt, eps_t, eps_r, T_ini, T_amb, q_sol, h_conv, q_out
#    file_vars = str(uuid.uuid4())
#    with open(file_vars, 'wb') as fp:      pickle.dump(variables, fp)
#    return file_vars

def Conditions_std():
    #COMPLETE PLANT
    plant = {
        ############ OVERALL PLANT #############
        'G_bd'     : 800.     ,              #[W/m2] Nominal DNI
        'C_so'     : 1000.    ,              #[-] Concentrated Ratio
        'A_sun_nom': 1000.    ,              #[m2] Surface Area
        'P_sun_nom': 1.0e6    ,              #[W] Solar thermal power
        'SM_nom'   : 2.       ,              #[-] Solar multiple
        'St_nom'   : 8.0      ,              #[hr] Storage capacity (hours of nominal capacity)
        ############ POWER BLOCK ###############
        'P_pel'    : 10.0     ,              #[MW] Nominal electrical Power
        'eta_el'   : 0.50     ,              #[-] Efficiency power cycle
        'm_CO2'    : 5.       ,              #[kg/s] Mass flow rate in Power Block
        'T_CO2_hi' : 950.     ,              #[K] High CO2 Temp for power cycle (in HP turbine)
        'T_CO2_lo' : 775.     ,              #[K] Low CO2 Temp for power cycle (out regenerator)
        't_dis'    : 48.      ,              #[ht] Time max of discharging process
        ############ SOLAR FIELD ###############
        'q_sun'    : 1e6      ,              #[W/m2] Solar radiation over receiver after optic elements
        'T_amb'    : 300.     ,              #[K] Ambient temperature
        'Tst_max'  : 1000.    ,              #[K] Maximum storage temperature
        'N_tanks'  : 4        ,              #[-] Number of storage units
        't_sun'    : 8.       ,              #[hr] Time of solar radiation (idealised by the moment)
        ############## RESULTS #################
        'P_th '    : 0.       ,              #[]
        'Q_st'     : 0.       ,              #[]
        'E_sun'    : 0.       ,              #[]
        'SED'      : 0.       ,              #[]
        'A_sun'    : 0.       ,              #[]
        'SM'       : 0.       ,              #[]
        't_pr_ch'  : 0.       ,              #[]
        't_pr_ds'  : 0.       ,              #[]
        }
    
    #BASE CONDITIONS FOR SOLID TANK (CHARGE AND DISCHARGE)
    tank = {
        ##### STORAGE MEDIUM PARAMETERS ######
        'mat'      : 'Cast steel'          ,        #[-] Material of storage
        'str_typ'  : 'Cylinder'            ,        #[-] type of storage
        'HDR'      : 0.25                  ,        #[-] Ratio height-diameter (or height-lenght) in storage
        'RSSR'     : 0.20                  ,        #[-] Receiver-Storage Surface Ratio
        'ER'       : 1.10                  ,        #[m] Eccentricity respect storage center
        'R_nu'     : 0.10                  ,        #[m] Percentage of no use (covered by pipes)
        'H_st'     : 1.00                  ,        #[m] Tank height(if HDR is not given)
        ####### Cylinder ######
        'D_st'     : 2.00                  ,        #[m] Tank diameter (if RSSR is not given)
        ###### Parallelepiped ####
        'L_st'     : 1.00                  ,        #[m] Block lenght
        'W_st'     : 0.50                  ,        #[m] Block width (if not considered equal to H_st)
        ####### Ring ######
        'Di_st'    : 1.00                  ,        #[m] Tank inner diameter (if RSSR is not given)
        ##### RECEIVER GEOMETRY #####
        'rcv_typ'  : 'Circ'                ,         #[-] type of receiver
        'D_rc'     : 0.80                  ,         #[m] Receiver diameter
        'exc'      : 1.75                  ,         #[m] eccentricity (if ER is not given)
        'w_vel'    : 2.                    ,         #[rph] or [m/h] Angular speed or linear speed (according to str_typ)
        'dx_sg'    : 0.30                  ,         #[m] Space between storage and glass
        'dx_gl'    : 0.10                  ,         #[m] Thickness of glass
        ####### OPTIC PROPERTIES #######
        'eps_st'   : 0.90                  ,         #[-] tank material emisivity
        'alpha_st' : 0.98                  ,         #[-] tank material absortivity
        'eps_rd'   : 0.90                  ,         #[-] rods emisivity
        'alpha_rd' : 0.90                  ,         #[-] rods absortivity
        'eps_gl'   : 0.40                  ,         #[-] glass emisivity
        'tau_gl'   : 0.90                  ,         #[-] glass transmisivity
        'alpha_gl' : 0.05                  ,         #[-] glass absortivity
        ### CHARGING DISCRETISATION ##### WARNING: ONLY USE EVEN NUMBERS AS N's, DUE TO RECEIVER FUNCTION!!!!
        'N_x'      : 30                    ,         #[-] Horizontal x discretisation
        'N_y'      : 30                    ,         #[-] Horizontal y discretisation
        'N_z'      : 20                    ,         #[-] Vertical discretisation
        't_ch_mx'  : 1.0                   ,         #[hr] Simulation time
        'dt_ch'    : 120.                  ,         #[s] Temporal discretisation
        'limit_ch' : True                  ,         #Limit charging time to temp?
        ##### BOUNDARY CONDITIONS ######
        'T_use'    : plant['T_CO2_hi']     ,         #[K] Power plant use temperature
        'T_ini'    : plant['Tst_max']-150. ,         #[K] Initial tank temperature
        'T_gl'    : plant['T_amb']         ,         #[K] Initial temp of glass cover
        'T_amb'    : plant['T_amb']        ,         #[K] Ambient temperature
        'q_sol'    : plant['q_sun']        ,         #[W/m2] inlet Concentrated radiation
        'h_conv'   : 20.                   ,         #[W/m2] Natural HTC (if not calculated)
        'q_out'    : 0.                    ,         #[W/m2] Bottom losses
        'T_avmx'   : plant['Tst_max']      ,         #[K] Max average temperature if controled
        'T_rcvmx'  : 1800.                 ,         #[K] Max temperature allowed
        
        #   DISCHARGING PROCESS
        'D_p'      : 0.05                  ,         #[m] Pipe diameter
        'F_enh'    : 2.                    ,         #[-] Factor of HTC inhencement
        'h_c1'     : 10.                   ,         #[W/m2K] HTC (if not calculated)
        'T_fin'    : plant['T_CO2_lo']     ,         #[K] Target of Temperature input
        'T_fm'     : plant['T_CO2_lo']     ,         #[K] Initial temp of fluid
        'fluid'    : ct.Solution('gri30.xml','gri30_mix') ,
        'm_f'      : plant['m_CO2']        ,         #[kg/s] Mass flow rate
        'min_m'    : 0.1                   ,         #[-] Minimum mass flow rate
        'N_h'      : 100                   ,         #[-] Spatial discretisation of discharge
        'dt_ds'    : 30.                   ,         #[-] Temporal discretisation of discharge
        
        #TEMPERATURE PROFILES (CHARGING AND DISCHARGING)
        'T_ch'     : plant['Tst_max']-150. ,         #[K] Charge profile temperature
        'T_ds'     : plant['Tst_max']      ,         #[K] Discharge profile temperature
        
        #OUTPUT VARIABLES (CHARGING AND DISCHARGING)
        'T_av'     : plant['Tst_max']-150. ,         #[K] Storage average temperature
        'eta_r'    : 0.0                   ,         #[-] Receiver efficiency
        'eff_r'    : 0.0                   ,         #[-] Receiver effectiveness
        'Q_ch'     : 0.0                   ,         #[-] Charging process
        }
    return plant, tank

###########################################################
########  3D MODDEL WITH CARTESIAN COORDINATES ############
###########################################################

def Charge_3D_sparse(tnk):
    ######################  GEOMETRY ######################
    aux         = 'dx_sg', 'mat', 'rcv_typ', 'str_typ', 'w_vel', 'dx_gl'
    dx_sg, mat, rcv_typ, str_typ, w_vel, dx_gl = [tnk[x] for x in aux]
    ######################  OPTICS ######################
    aux         = 'eps_st', 'alpha_st', 'eps_gl', 'tau_gl', 'alpha_gl'
    eps_st, alpha_st, eps_gl, tau_gl, alpha_gl = [tnk[x] for x in aux]
    ################ BOUNDARY CONDITIONS ################
    aux         = 'T_gl', 'T_amb', 'T_use', 'T_avmx', 'T_rcvmx'
    T_g, T_amb, T_use, T_avmx, T_rcvmx = [tnk[x] for x in aux]
    T_sky       = T_amb - 15.
    air         = ct.Solution('air.xml')
    ############### DISCRETISATION ######################
    aux         = 'N_x', 'N_y', 'N_z', 't_ch_mx', 'limit_ch'
    Nx, Ny, Nz, t_sim, limit_ch = [tnk[x] for x in aux]
    Nt          =  Nx * Ny * Nz
    
    #Creating indexes
    P  = {i : np.zeros(Nt) for i in ['x','y','z','fs']}
    mm = np.zeros( shape=(Nx,Ny,Nz) , dtype=int )
    
    ######################################################
    ##########   INICIALIZING THE VARIABLES ##############
    ######################################################
    
    ######################################################
    ####### CONDITIONS FOR CYLINDER STORAGE ##############
    ######################################################
    if str_typ == 'Cylinder' :
        
        #Geometry
        D_st     = tnk['D_st']
        H_st     = tnk['HDR'] * D_st
        D_rc     = np.sqrt(tnk['RSSR']) * tnk['D_st']
        exc      = tnk['ER'] * D_rc / 2.
        
        Vol_st   = np.pi * D_st**2. * H_st / 4.
        A_rcv    = np.pi * D_rc**2. / 4.
        
        #Initial conditions
        w_v      = w_vel * (2.*np.pi/3600.)                     #Rotation speed [rad/s]
        w_c      = tnk['w_c'] if ('w_c' in tnk) else 0.         #Angular position of receiver
        xr       = exc * np.cos(w_c)                            #Position of receiver center
        yr       = exc * np.sin(w_c)
        
        #Discretisation
        X_d , Y_d, Z_d = 1.2*D_st, 1.2*D_st, H_st
        dx,dy,dz = [ X_d / (Nx-1) , Y_d / (Ny-1) , Z_d / Nz ]
        dt       = min( tnk['dt_ch'] , (np.pi/15.)/w_v ) if w_v>0. else tnk['dt_ch']
        difs     = [ dx , dy , dz ]
        geom     = [ D_st/2. , H_st ]
        
        #Saving variables
        tnk['D_st'] , tnk['H_st'] , tnk['exc'] = D_st, H_st, exc
        tnk['X_d']  , tnk['Y_d']  , tnk['Z_d'] = X_d, Y_d, Z_d
        
        #Creating indexes
        Xo, Yo, Zo = -X_d/2, -Y_d/2, Z_d
    
    #########################################################
    ####### CONDITIONS FOR RECTANGULAR STORAGE ##############
    #########################################################    
    elif str_typ == 'Rect':
        
        #Geometry
        L_st        = tnk['L_st']
        W_st        = tnk['W_st']
        H_st        = tnk['L_st'] * tnk['HDR']
        D_rc        = np.sqrt( tnk['RSSR'] * L_st * W_st / np.pi )
        
        Vol_st, A_rcv   = L_st * W_st * H_st , np.pi * D_rc**2.
        
        #Discretisation
        X_d , Y_d, Z_d = 1.2*L_st, 1.2*W_st, H_st
        dx,dy,dz       = [ X_d / (Nx-1) , Y_d / (Ny-1) , Z_d / Nz ]
        dt             = tnk['dt_ch']
        difs           = [ dx , dy , dz ]
        geom           = [L_st, W_st, H_st, X_d, Y_d, Z_d]
        
        tnk['L_st']  , tnk['W_st']  , tnk['H_st'], tnk['D_rc'] = L_st, W_st, H_st, D_rc
        tnk['X_d']  , tnk['Y_d']  , tnk['Z_d']  = X_d, Y_d, Z_d
        
        #Initial conditions
        w_v         = w_vel * L_st / 3600.              #[m/s]
        xr, yr      = (X_d-L_st)/2. if 'xr' not in tnk else tnk['xr'], 0.      #Position of receiver center
        
        #Creating indexes
        Xo, Yo, Zo = 0, -Y_d/2, Z_d
            
    #########################################################
    ########### CONDITIONS FOR RING STORAGE #################
    #########################################################    
    elif str_typ == 'Ring':
        
        #Geometry
        D_st, Di_st, D_rc = tnk['D_st'], tnk['Di_st'], tnk['D_rc']
        exc      = (D_st + Di_st)/4.
        H_st     = tnk['HDR'] * D_st              if ('HDR'  in tnk) else tnk['H_st']
        
        Vol_st, A_rcv   = np.pi * ( D_st**2 - Di_st**2 ) * H_st / 4., np.pi * D_rc**2. /4.
        
        #Initial conditions
        w_v      = w_vel * (2.*np.pi/3600.)                 #Rotation speed [rad/s]
        w_c      = tnk['w_c'] if ('w_c' in tnk) else 0.     #Angular position of receiver
        xr, yr   = exc * np.cos(w_c)  , exc * np.sin(w_c)   #Position of receiver center
        
        #Discretisation
        X_d , Y_d, Z_d = 1.2*D_st, 1.2*D_st, H_st
        dx,dy,dz = [ X_d / (Nx-1) , Y_d / (Ny-1) , Z_d / Nz ]
        dt       = min( tnk['dt_ch'] , (np.pi/15.)/w_v ) if w_v>0. else tnk['dt_ch']
        difs     = [ dx , dy , dz ]
        geom     = [ D_st/2. , Di_st/2., H_st ]
        
        tnk['Di_st'], tnk['H_st'] , tnk['exc']  = Di_st, H_st, exc
        tnk['X_d']  , tnk['Y_d']  , tnk['Z_d']  = X_d, Y_d, Z_d
        #Creating indexes
        Xo, Yo, Zo = -X_d/2, -Y_d/2, Z_d

    ######################################################            
    else:
        print("Storage type not identified")
    
    
    #######################################################
    #Creating indexes
    m = 0
    for z,y,x in [(z,y,x) for z in range(Nz) for y in range(Ny) for x in range(Nx)]:
        mm[x,y,z]  = m
        P['x'][m]  = (Xo + x*dx)
        P['y'][m]  = (Yo + y*dy)
        P['z'][m]  = (z+0.5)*dz
        pos        = [ P['x'][m] , P['y'][m], P['z'][m] ]
        P['fs'][m] = Storage_part( pos , difs, geom, str_typ )[0]
        m+=1
    
    #########################################################
    #Defining the receiver geometry
    if rcv_typ=='Circ':
        Az_gl, pz_gl, q_sol    = np.pi * D_rc**2 / 4. , np.pi * D_rc , tnk['q_sol']
    elif rcv_typ == 'None':
        Az_gl, pz_gl, q_sol, dt   = 1e-10 , 1e-10, 0.0, tnk['dt_ch']
    else:
        print("Receiver type not identified")
        
    ######################################################
    ############# BEGIN OF CYCLE SIMULATION ##############
    ######################################################
    Tp       = tnk['T_ch']*np.ones(Nt)
    T_ini    = Tp
    T_iniav  = np.average(T_ini)
    tt       = 0
    
    
    dt   = tnk['dt_ch']
    
    while tt <= t_sim*3600 - dt*0.1:
        
        ###################################################
        #Defining the matrix system to fill
        if rcv_typ == 'Circ':
            A       = sps.dok_matrix((Nt+1,Nt+1))
            ind     = np.zeros(Nt+1)
            hint    = np.append(Tp,T_g)
        elif rcv_typ == 'None':
            A       = sps.dok_matrix((Nt,Nt))
            ind     = np.zeros(Nt)
            hint    = Tp
        else:
            print("Receiver type not identified")
        
        # Glass cover properties
        rho_gl, k_gl, cp_gl = props_cte('Glass',T_g)
        r_gl                = dt / (dx_gl * rho_gl * cp_gl)
        hrc_gs              = 0.
        
        Tav     = np.sum(Tp*P['fs'])/np.sum(P['fs'])
        #####################################################
        #####################################################
        #Filling Matrix System
        #Rectangular coordinates elements
        for z,y,x in [(z,y,x) for z in range(Nz) for y in range(Ny) for x in range(Nx)]:
    
            #Defining element, volume and surface
            m                   = mm[x,y,z]
            V_m                 = dx * dy * dz
            Az_m                = dx * dy
            
            pos                 = [ P['x'][m] , P['y'][m], P['z'][m] ]
            difs                = [ dx , dy , dz ]
            fs, bE, bN, bW, bS  = Storage_part( pos , difs, geom, str_typ )
            
            rcvr                = [ D_rc, xr , yr, H_st ]
            fr                  = Receiver( pos , difs , rcv_typ, rcvr )         # Under receiver??
            
            rho_m , k_m , cp_m  = props_cte(mat,Tp[m])
            
            alpha_m = k_m / ( rho_m * cp_m ) if fs > 0. else 0.
            rx, ry, rz = [ alpha_m * dt / i**2 for i in [dx,dy,dz]]
            
            A [ m , m ]  = 1.
            ind [m]      = Tp[m]
            
            if x>0        :       A [m , mm[x-1,y,z] ] = -rx * bW
            if x<Nx-1     :       A [m , mm[x+1,y,z] ] = -rx * bE
            
            if y>0        :       A [m , mm[x,y-1,z] ] = -ry * bS
            if y<Ny-1     :       A [m , mm[x,y+1,z] ] = -ry * bN

            if z>0        :       A [m , mm[x,y,z-1] ] = -rz
            if z<Nz-1     :       A [m , mm[x,y,z+1] ] = -rz
            
            ################# VERTICAL DIRECTION #########################
            if z==Nz-1    :                                                         ##### UPPER BOUNDARY #####
                
                T_s          = ( 3*Tp[m] - Tp[mm[x,y,z-1]] ) / 2.
                
                if fr > 0.:
                    hr_sg    = cte.sigma*(T_s+T_g)*(T_s**2+T_g**2) / ( 1/eps_st + (fr*Az_m/Az_gl) * (1/eps_gl-1.))
                    air.TP   = (T_s+T_g)/2., ct.one_atm
                    hc_sg    = h_conv_nat_conf( T_s, T_g, dx_sg, air )
                    hrc      = ( hr_sg + hc_sg ) * fr
                    qi       = (q_sol*alpha_st*tau_gl) * fr
                else:
                    hr_sg, hc_sg, hrc, qi = 0., 0., 0., 0.
                    
                #For upper elements
                A [ m , m ]  += rz * (1 + hrc * dz / k_m )
                ind [m]      +=   (rz*dz/k_m) * qi
                
                #Conditions for glass
                if rcv_typ == 'Circ':
                    hrc_gs            += hrc * Az_m / Az_gl
                    A [ Nt , m ]       = - r_gl * hrc * Az_m / Az_gl
                    A [ m , Nt ]       = - (rz*dz/k_m) * hrc
            
            if (0 < z < Nz-1)   :     A [ m , m ] += 2*rz               ##### INNER ELEMENTS ####
                
            if z==0             :     A [ m , m ] += rz                 ##### BOTTOM BOUNDARY ####
            
            ################### X DIRECTION ###########################
            if x==0             :     A [ m , m ] += rx * bE            ##### LEFT BOUNDARY #####
                
            if (0 < x < Nx-1)   :     A [ m , m ] += rx * bE + rx * bW  ##### INNER ELEMENTS ####
                
            if x==Nx-1          :     A [ m , m ] += rx * bW            ##### RIGHT BOUNDARY ####
                 
            ################### Y DIRECTION ###########################
            if y==0             :     A [ m , m ] += ry * bN            ##### LOWER BOUNDARY ####
                
            if (0 < y < Ny-1)   :     A [ m , m ] += ry * bN + ry * bS  ##### INNER ELEMENTS ####
                
            if y==Ny-1          :     A [ m , m ] += ry * bS            ##### UPPER BOUNDARY ####

            ####################
            m += 1
        
        ########### GLASS COVER CONDITIONS #####################
        hr_ga               = eps_gl * cte.sigma * ( T_g + T_sky ) * ( T_g**2 + T_sky**2 )
        air.TP              = (T_g+T_amb)/2., ct.one_atm
        hc_ga               = h_conv_nat_amb(T_g, T_amb, (Az_gl/pz_gl), air)
        hrc_ga              = hr_ga + hc_ga
        
        ################### SOLVING THE SYSTEM ##########################
        if rcv_typ == 'Circ':
            A[ Nt , Nt ]    = 1.  + r_gl * (hrc_ga + hrc_gs)
            ind[Nt]         = T_g + r_gl * (q_sol*alpha_gl + hc_ga*T_amb + hr_ga*T_sky)
            solve           = spsl.bicgstab(A,ind,hint,tol=1e-5)
            Tp              = solve[0][0:Nt]
            T_g             = solve[0][Nt]
        elif rcv_typ == 'None':
            solve           = spsl.bicgstab(A,ind,hint,tol=1e-5)
            Tp              = solve[0]
        else:
            print("Receiver type not identified")
        
        ############### Average and Global parameters ###################
        Tav_ant  = Tav
        Tav      = np.sum(Tp*P['fs'])/np.sum(P['fs'])
        Tmax     = np.max(Tp)
        Tsd,Tmin = 0, Tav
        for m in range(Nt):
            Tsd  += (Tp[m]*P['fs'][m] - Tav)**2
            if P['fs'][m]>0.98 and Tp[m]<Tmin: Tmin = Tp[m]
            
        Tsd    = np.sqrt(Tsd /np.sum(P['fs']))
        Q_st   = rho_m * cp_m * Vol_st * (Tav - Tav_ant) / dt
        Q_loss = q_sol * A_rcv - Q_st
        U_t    = Q_loss / ( A_rcv * ( Tmax - T_amb ) )
        
        #Printing preliminar results
#        print('Tiempo: {0:.2f}. T_gl = {1:.2f}. T_av = {2:.2f}. T_max = {3:.2f}. T_sd = {4:.2f}. T_min = {5:.2f}. xr = {6:.3f} yr = {7:.3f}'.format(tt,T_g,Tav,Tmax,Tsd, Tmin, xr, yr))
        print('{0:7.2f}\t{1:7.2f}\t{2:7.2f}\t{3:7.2f}\t{4:7.2f}'.format(tt, T_g, Tav, Tmax, Tmin))
#        text = '{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\n'.format(tt,T_g,Tav,Tmax,Tsd, Tmin)
#        f = open('3DModel_standby/perfil_tiempo.txt','a'); f.write(text); f.close()
        
        # Check if storage is above desired max temperature
        if limit_ch and (Tav  > T_avmx or Tmax > T_rcvmx):    break
    
        # Move receiver to the next time step
        if (str_typ == 'Cylinder' or str_typ == 'Ring') and rcv_typ != 'None':
            w_c       += dt*w_v
            xr, yr     = exc * np.cos(w_c)  , exc * np.sin(w_c)
        elif str_typ == 'Rect' and rcv_typ != 'None':
            xr, yr     = (xr + dt*w_v) , yr
            
        tt  += dt
        
    ####################################################
    ############# END OF TEMPORAL CYCLE ################
    ####################################################
    
    ########### PERFORMANCE VARIABLES ##################
    Q_in, Q_ch, eff, Vol = 0., 0., 0., 0.
    Tav       = np.sum(Tp*P['fs'])/np.sum(P['fs'])
    Tsd, Tmin = 0., Tav
    
    for m in range(Nt):
        rho_m , k_m , cp_m = props_cte(mat,Tp[m])
        fs                 = Storage_part( [P['x'][m],P['y'][m],P['z'][m]] , difs, geom, str_typ )[0]
        fr                 = Receiver( [P['x'][m],P['y'][m],P['z'][m]], difs, rcv_typ, [D_rc,xr,yr,H_st] )
        Az_m, V_m          = (dx * dy * fr) , (dx * dy * dz * fs)
        
        Q_in              += q_sol * Az_m * ((tt/3600)*1e-3)
        Q_ch              += V_m * ( rho_m * cp_m ) * ( Tp[m]-T_ini[m] ) / 3.6e6
        eff               = ( eff + fs ) if Tp[m]>T_use else eff
        Tsd               += (Tp[m]*P['fs'][m] - Tav)**2
        Tmin              = Tp[m] if P['fs'][m]>0.98 and Tp[m]<Tmin else Tmin
    
    Vol   = (dx*dy*dz) * np.sum(P['fs'])
    eta_r = Q_ch / Q_in if Q_in > 0. else 0.         # Efficiency
    eff   = eff / np.sum(P['fs'])                    # Effectiveness
    SED   = Q_ch/Vol                                 # Stored Energy density
    Teq   = Q_ch * 3.6e6 / (Vol*rho_m*cp_m) + T_iniav
    Tmax  = np.max(Tp)
    Tsd   = np.sqrt(Tsd /np.sum(P['fs']))
    
    ########### NON-DIMENSIONAL PARAMETERS #############
    Tmxav = (Tmax + T_iniav) / 2.
    rho_m , k_m , cp_m  = props_cte(mat,Tp[m]); alpha_m = k_m / ( rho_m * cp_m )
    h_ch  = (Q_in - Q_ch) / (A_rcv * (Tmxav - T_amb ) )
    if str_typ == 'Cylinder':   L_ch = (np.pi * D_st**2/4. * H_st ) / A_rcv
    elif str_typ == 'Rect':     L_ch = L_st * W_st * H_st / A_rcv
    elif str_typ == 'Ring':     L_ch = np.pi * ( D_st**2-Di_st**2)/4. * H_st  / A_rcv

    Bi , Fo = h_ch * L_ch / k_m , alpha_m * tt / L_ch**2
    
    ##################### OUTPUT #######################
    
    tnk['T_ch']  , tnk['T_gl']  , tnk['T_eq']                 = Tp, T_g, Teq
    tnk['T_av']  , tnk['T_sd']  , tnk['T_mx']  , tnk['T_mn']  = Tav, Tsd, Tmax, Tmin
    tnk['Bi_ch'] , tnk['Fo_ch'] , tnk['h_ch']  , tnk['L_ch']  = Bi, Fo, h_ch, L_ch
    tnk['eta_r'] , tnk['eff_r'] , tnk['SED']                  = eta_r, eff, SED
    tnk['V_st']  , tnk['A_rcv']                               = Vol, A_rcv
    tnk['Q_in']  , tnk['Q_ch']  , tnk['tt']                   = Q_in, Q_ch, tt
    tnk['xr']    , tnk['yr']    , tnk['w_c'], tnk['t_ch']     = xr, yr, w_c, tt/3600.
    tnk['xyz']   , tnk['difs']                                = [ mm , P['x'] , P['y'] , P['z'] , P['fs'] ] , [dx, dy, dz]
#    tnk['Q_st']  , tnk['Q_loss'], tnk['U_t']                  = Q_st, Q_loss, U_t
    return

##################################################
############### CHARGE STRATEGY ##################
##################################################

def Charging_process(Case, N_pr, plnt, tnk, Tank, file_ch):
    
    Ntnks, t_total   = plnt['N_tanks'], plnt['t_sun']
    t_ch_def         = min(t_total/Ntnks,1.)

    #
    aux              = {x:Tank[0][x] for x in Tank[0]}
    
    # CHARGING PROCESS (EACH STORAGE UNIT ONE HOUR MAX)
    t_step, t_proc   = t_ch_def, 0.0
#    for i in range(Ntnks):
#        
#        #= t_step at beginning
#        aux['t_ch_mx'] = min( t_total-t_proc, t_step )
#        aux['rcv_typ'] = 'Circ' if i == 0 else 'None'
#        
#        Charge_3D_sparse(aux)
#        t_step     =    aux['t_ch']+aux['dt_ch']/7200.;  t_proc     +=   t_step
#        
#        for j in range(i-1,0,-1):   Tank[j+1]   = {x:Tank[j][x] for x in Tank[j]}       #Uploading the tanks status
#        Tank[0]     = {x:aux[x] for x in aux}
#        
#        print('Simulated Active Tank: '+str(i)+'. Time of step: '+str(t_step)+'. Time of process: '+str(t_proc))
#        print([Tank[x]['T_av'] for x in range(Ntnks)])
#        
#        if t_proc >= t_total: break
#    
#    Q_st  = sum([Tank[x]['Q_ch'] for x in range(Ntnks)])              #[kWh] Energy acummulated from the sun
#    
    Q_st = 0.0; 
    # COMPLETING PROCESS
    i_act = 0
    while t_proc < t_total :
        
        print('Simulating Active Tank: '+str(i_act)+'.')
        for i in range(Ntnks):  Tank[i]['rcv_typ'] = 'Circ' if i == i_act else 'None'
        
        #Active process
        t_step = 0.
        for dd in range(4):
            if Tank[i_act]['T_av']>Tank[i_act]['T_avmx']:   break
        
            Tank[i_act]['t_ch_mx'] = min( t_total-t_proc, t_ch_def )
            Charge_3D_sparse(Tank[i_act])
            Tank[i_act]['w_c'] += np.pi /2.
            t_step += Tank[i_act]['t_ch'] + (aux['dt_ch']/7200.)*3/2.
            Q_st   += Tank[i_act]['Q_ch']
        
        #Stand-by processes
        print('Time of step: '+str(t_step))
        t_proc += t_step
        for i in range(Ntnks):
            if i != i_act:
                Tank[i]['t_ch_mx'] = t_step                
                
                if abs(np.max(Tank[i]['T_ch'])-Tank[i]['T_av'])>5.:
                    Charge_3D_sparse(Tank[i])
                    Tank[i]['dt_ch'] = tnk['dt_ch']
                    print('Simulating Stand-by Tank: '+str(i)+'. Time of step: '+str(t_step)+'. Time of process: '+str(t_proc))
                else:
                    print('No required to simulate Stand-by Tank: '+str(i)+'.')
#                print('Temps: '+[Tank[x]['T_av'] for x in range(Ntnks)])
                text = 'Temps:\t'+'\t'.join('{:.2f}'.format(Tank[x]['T_av']) for x in range(Ntnks))+'. Goals:\t'+'\t'.join('{:.2f}'.format(Tank[x]['T_avmx']) for x in range(Ntnks))
                print(text)
        
        #Break if necessary
#        if t_step <= 0.1: break
        if all(Tank[x]['T_av']>Tank[x]['T_avmx'] for x in range(Ntnks)): break
        i_act = i_act + 1 if i_act < (Ntnks-1) else 0
    
    # PRINTING RESULTS
    CO2          = ct.Solution('gri30.xml','gri30_mix')
    CO2.TPX      = 0.5*(plnt['T_CO2_hi']+plnt['T_CO2_lo']), 8e6, 'CO2:1.00'
    P_th         = plnt['m_CO2'] * CO2.cp * (plnt['T_CO2_hi'] - plnt['T_CO2_lo'])
    
    V_st, A_rcv = Tank[0]['V_st'], Tank[0]['A_rcv']
    Vol     = Ntnks * V_st                                              #[m3]
    E_sun   = tnk['q_sol'] * A_rcv * t_proc / 1e3                       #[kWh] Energy acummulated from the sun
    SED     = Q_st/Vol                                                  #[kWh/m3] Energy Density
    A_sun   = plnt['C_so'] * A_rcv                                      #[m2] heliostat area (assuming eta_hf=1.0)
    SM      = tnk['q_sol']*A_rcv/P_th if P_th>0. else 0.                #[-] Solar multiple
     
    plnt['P_th'], plnt['Q_st'],  plnt['E_sun']                    = P_th, Q_st, E_sun 
    plnt['SED'] , plnt['A_sun'], plnt['SM'],   plnt['t_pr_ch']    = SED, A_sun, SM, t_proc
    
    text = str(int(Case))+'\t'+str(N_pr).zfill(2)+'\t'
    text = text+'\t'.join('{:.2f}'.format(x) for x in [Vol, Q_st, E_sun, SED, A_sun,SM,t_proc])+'\t'
    text = text+'\t'.join('{:.2f}'.format(Tank[x]['T_av']) for x in range(Ntnks))+'\n'
    print([Tank[x]['T_av'] for x in range(Ntnks)])
    f = open(file_ch,'a'); f.write(text); f.close()

    return

###############################################
############# DISCHARGE PROCESS ###############
###############################################
    
def Discharge_1D(tnk):
    
    def TDMAsolver(a, b, c, d): #TDMA solver, a b c d can be NumPy array type or Python list type.
        for i in range(1, len(d)): b[i], d[i] = ( b[i]-a[i-1]/b[i-1]*c[i-1]) , (d[i]-a[i-1]/b[i-1]*d[i-1])
        x = b; x[-1] = d[-1]/b[-1]
        for i in range(len(d)-2,-1,-1):  x[i] = (d[i]-c[i]*x[i+1])/b[i]
        return x    

    ##################################################
    aux         = 'm_f', 'T_fin', 'HDR', 'D_st', 'R_nu', 'D_p', 'mat', 'F_enh', 'N_h', 'dt_ds', 'fluid'
    m_f, T_fin, HDR, D_st, R_nu, D_p, mat, F_enh, Nz, dt, CO2 = [tnk[x] for x in aux]

    Tst         = tnk['T_ds']*np.ones(Nz)
    Tsta        = Tst
    Tfm         = tnk['T_fin']*np.ones(Nz)
    h_c         = np.zeros(Nz)
    Tav         = Tst.mean()
    
    h_c_aux    = 0
    
    #Velocity y ratio of use    
    H_st            = HDR * D_st
    dz              = H_st/Nz
    z               = [(n+0.5)*dz for n in range(Nz) ]
    N_p             = np.ceil(R_nu * (D_st/D_p)**2)
    m_fp            = m_f / N_p                      #[kg/s] Flow in each pipe
    dA_st, dA_hz    = (np.pi*(D_st/2)**2/N_p - np.pi*D_p**2/4) , ( np.pi * D_p * dz )
    A_c             = N_p * np.pi * D_p * H_st
    
    a , b           = D_p/2 , np.sqrt( (np.pi*(D_st/2)**2/N_p) /np.pi)
    L_c, eta        = (b**2-a**2)/(2*a) , b/a
    k_st            = props_cte(mat, Tst[0] )[1]
    
#    try:
    #Equations for fluid
    Tfi         = T_fin
    UA          = 0.
    for n in range(Nz):
        CO2.TPY     = Tfi, 8e6, 'CO2:1.00'
        cp_f        = CO2.cp
        
        h_c2        = h_conv_tube(CO2, m_fp, Tfi, Tst[n], D_p, H_st , F_enh, z[n] )[0] if m_f>0. else 0.
        
        if m_f>0.:
            Bi_LC       = h_c2 * L_c/k_st
            h_eff       = (1./h_c2 + (a**3*(4.*b**2-a**2)+a*b**4.*(4.*np.log(b/a)-3.))/( 4.*k_st*(b**2-a**2)**2 ))**(-1.)
            Bi_eff      = Bi_LC / ( 1 + Bi_LC * (eta**4*(4*np.log(eta)-3) + 4*eta**2 - 1) / ( 2*(eta**2-1 )**3 ) )
        else:
            Bi_eff, h_eff, h_c2 = 0.,0.,0.
        
        h_c_aux     += h_c2
        h_c[n]       = h_eff
        UA          += h_c[n]*dA_hz
    
        s1, s2      = 2*m_fp*cp_f , h_c[n]*dA_hz
        Tfm[n]      = ( s1 * Tfi + s2 * Tst[n] ) / ( s1 + s2 ) if m_f>0. else Tfi
        Tfo         = 2 * Tfm[n] - Tfi
        Tfi         = Tfo
                
    CO2.TPY     = np.mean(Tfm), 8e6, 'CO2:1.00'
    Q_out       = m_f * CO2.cp * (Tfo - T_fin)
        
    #Equation for storage discharge
    ( a1, a2, a3, b )  = ( np.zeros(Nz-1), np.zeros(Nz), np.zeros(Nz-1), np.zeros(Nz) )
    for n in range(Nz):
        
        rho_st, k_st, cp_st  = props_cte(mat, Tst[n] )
        r1 , r2              = k_st*dt / ( dz**2*rho_st*cp_st ) , h_c[n]*np.pi*D_p*dt / ( dA_st*rho_st*cp_st )
        
        if n == 0       :  a2[n]  , a3[n]           = ( 1+r1+r2 ) ,      -r1
        elif n < Nz-1   :  a1[n-1], a2[n]  , a3[n]  =    -r1      , ( 1+2*r1+r2 ) ,    -r1
        else            :           a1[n-1], a2[n]  =                    -r1      , ( 1+r1+r2 )
        b[n] = Tst[n] + r2*Tfm[n]

    Tst     = TDMAsolver(a1,a2,a3,b)
    Q_out2  = (N_p*dA_st*dz) * rho_st*cp_st * sum(Tst-Tsta) / dt
    
    h_av    = h_c.mean()
    L_c     = dA_st * dz / dA_hz
    Bi      = h_av*L_c/k_st
    Tav     = Tst.mean()
    
    h_c_aux = h_c_aux/Nz
    
    Re  = h_conv_tube(CO2, m_fp, Tfm.mean(), Tav, D_p, H_st , F_enh, H_st )[2] if m_f>0. else 0.
    
    out = dict()
    out['T_ds']  , out['T_fm'], out['z'] =  Tst, Tfm, z
    out['T_av']  , out['Tfo']            =  Tav, Tfo
    out['Q_out'] , out['Q_out2']         =  Q_out, Q_out2
    out['N_p']   , out['A_c']            =  N_p, A_c
    out['UA']    , out['UA2']            =  h_av * A_c * F_enh, UA * N_p
    out['Bi']    , out['h_c']            =  Bi, h_c_aux
    out['h_av']  , out['Bi_eff']         =  h_av, Bi_eff
    out['Re']                            =  Re
    return out

########################################################
########################################################

##################################################
##################################################
    
def Storage_part(pos,difs,geom,typ):
    #INPUTS:
    #   xp, yp, zp           = Coordinates of element central point (in this order inside pos)
    #   dx, dy, dz           = Spatial discretisation (in this order inside difs)
    #   geom                 = Geometry of Storage
    #   typ                  = Type of Storage
    #OUTPUTS:
    #   fs                  = Factor of element under storage (0<fs<1, 0 none, 1 full)
    #   bE , bN , bW , bS   = Portion of boundaries under storage
    xp,yp,zp = pos
    dx,dy,dz = difs
    xi       = [ xp+dx/2 , xp-dx/2 , xp+dx/2 , xp-dx/2 ]
    yi       = [ yp+dy/2 , yp+dy/2 , yp-dy/2 , yp-dy/2 ]
        
    if typ == 'Cylinder' or typ == 'Ring':
        
        if typ == 'Cylinder'    :   r , H  = geom
        else                    :   r,ri,H = geom
        
        d        = [ np.sqrt( (xi[i])**2 + (yi[i])**2 ) <= r for i in range(4)]
        
        if all(di for di in d)        : fs, bE, bN, bW, bS   = [1. for i in range(5)]         #### 1 = STORAGE
        elif all(not(di) for di in d) : fs, bE, bN, bW, bS   = [0. for i in range(5)]         #### 0 = NOT STORAGE
        else:                                                                                 #### BOUNDARIES
            fs, bE, bN, bW, bS   = [1. for i in range(5)]
            
            if ( not(d[0]) and not(d[2]) ):     bE = 0.
            if ( not(d[1]) and not(d[3]) ):     bW = 0.
            if ( not(d[0]) and not(d[1]) ):     bN = 0.
            if ( not(d[3]) and not(d[2]) ):     bS = 0.
            
            #EAST BORDER IS BOUNDARY
            if (d[0] and not(d[2])): bE = min(max(abs( np.sign(yp)*np.sqrt(abs(r**2 - xi[0]**2 )) - yi[0] ) / dy , 1e-10),1.)
            if (d[2] and not(d[0])): bE = min(max(abs( np.sign(yp)*np.sqrt(abs(r**2 - xi[2]**2 )) - yi[2] ) / dy , 1e-10),1.)
            #WEST BORDER IS BOUNDARY
            if (d[1] and not(d[3])): bW = min(max(abs( np.sign(yp)*np.sqrt(abs(r**2 - xi[1]**2 )) - yi[1] ) / dy , 1e-10),1.)
            if (d[3] and not(d[1])): bW = min(max(abs( np.sign(yp)*np.sqrt(abs(r**2 - xi[3]**2 )) - yi[3] ) / dy , 1e-10),1.)
            #NORTH BORDER IS BOUNDARY
            if (d[0] and not(d[1])): bN = min(max(abs( np.sign(xp)*np.sqrt(abs(r**2 - yi[0]**2 )) - xi[0] ) / dx , 1e-10),1.)
            if (d[1] and not(d[0])): bN = min(max(abs( np.sign(xp)*np.sqrt(abs(r**2 - yi[1]**2 )) - xi[1] ) / dx , 1e-10),1.)
            #SOUTH BORDER IS BOUNDARY
            if (d[3] and not(d[2])): bS = min(max(abs( np.sign(xp)*np.sqrt(abs(r**2 - yi[3]**2 )) - xi[3] ) / dx , 1e-10),1.)
            if (d[2] and not(d[3])): bS = min(max(abs( np.sign(xp)*np.sqrt(abs(r**2 - yi[2]**2 )) - xi[2] ) / dx , 1e-10),1.)
            
            ## Defining the portion of storage element
            bnds = [ bE, bN, bW, bS ]; bnds.sort()
            if   d.count(True) == 1:    fs =  bnds[-1]*bnds[-2]/2.
            elif d.count(True) == 2:    fs =  (bnds[1]+bnds[2])/2.
            elif d.count(True) == 3:    fs =   1 - (1 - bnds[0])*(1-bnds[1])/2.
            
    elif typ == 'Rect':
        
        L_st , W_st , H_st, X_d, Y_d, Z_d  = geom
        
        XE, XW, YN, YS  = (X_d+L_st)/2 , (X_d-L_st)/2,    W_st/2  , -W_st/2
        d               = [ xi[0] < XE ,  xi[1] > XW , yi[1] < YN , yi[2] > YS ]
        
        fs, bE, bN, bW, bS   = [1. for i in range(5)]
        #EAST BORDER IS OUTSIDE
        if not(d[0]):
            bE = 0.
            bN = max ( 1 - (xi[0] - XE) / dx, 0. )
            bS = bN
            bW =  1 if bN > 0 else 0
            
        #WEST BORDER IS OUTSIDE
        if not(d[1]):
            bW = 0.
            bN = max ( 1 - (XW - xi[1]) / dx, 0. )
            bS = bN
            bE =  1 if bN > 0 else 0
            
        if not(d[2]):
            bN = 0.
            bW = max ( 1 - (yi[1] - YN) / dy, 0. )
            bE = bW
            bS =  1 if bW > 0 else 0
            
        if not(d[3]):
            bS = 0.
            bW = max ( 1 - (YS - yi[2] ) / dy, 0. )
            bE = bW
            bN =  1 if bW > 0 else 0
            
        bnds = [ bE, bN, bW, bS ]; bnds.sort()
        fs   = bnds[2]*bnds[3]
            
    else:
        fs, bE, bN, bW, bS   = [0. for i in range(5)]
        print('Storage type does not recognized')
    
    #For ring verifying the inner diameter conditions
    if typ == 'Ring':
        
        d        = [ np.sqrt( (xi[i])**2 + (yi[i])**2 ) >= ri for i in range(4)]
        
        if   all(not(di) for di in d)   : fs, bE, bN, bW, bS   = [0. for i in range(5)]     #### 0 = NOT STORAGE
        elif any(d) and not(all(d))     :                                                   #### 1 = BOUNDARY
            
            if ( not(d[0]) and not(d[2]) ):     bE = 0.
            if ( not(d[1]) and not(d[3]) ):     bW = 0.
            if ( not(d[0]) and not(d[1]) ):     bN = 0.
            if ( not(d[3]) and not(d[2]) ):     bS = 0.
            
            #EAST BORDER IS BOUNDARY
            if (d[0] and not(d[2])): bE = min(max(abs( np.sign(yp)*np.sqrt(abs(ri**2 - xi[0]**2 )) - yi[0] ) / dy , 1e-10),1.)
            if (d[2] and not(d[0])): bE = min(max(abs( np.sign(yp)*np.sqrt(abs(ri**2 - xi[2]**2 )) - yi[2] ) / dy , 1e-10),1.)
            #WEST BORDER IS BOUNDARY
            if (d[1] and not(d[3])): bW = min(max(abs( np.sign(yp)*np.sqrt(abs(ri**2 - xi[1]**2 )) - yi[1] ) / dy , 1e-10),1.)
            if (d[3] and not(d[1])): bW = min(max(abs( np.sign(yp)*np.sqrt(abs(ri**2 - xi[3]**2 )) - yi[3] ) / dy , 1e-10),1.)
            #NORTH BORDER IS BOUNDARY
            if (d[0] and not(d[1])): bN = min(max(abs( np.sign(xp)*np.sqrt(abs(ri**2 - yi[0]**2 )) - xi[0] ) / dx , 1e-10),1.)
            if (d[1] and not(d[0])): bN = min(max(abs( np.sign(xp)*np.sqrt(abs(ri**2 - yi[1]**2 )) - xi[1] ) / dx , 1e-10),1.)
            #SOUTH BORDER IS BOUNDARY
            if (d[3] and not(d[2])): bS = min(max(abs( np.sign(xp)*np.sqrt(abs(ri**2 - yi[3]**2 )) - xi[3] ) / dx , 1e-10),1.)
            if (d[2] and not(d[3])): bS = min(max(abs( np.sign(xp)*np.sqrt(abs(ri**2 - yi[2]**2 )) - xi[2] ) / dx , 1e-10),1.)
            
            ## Defining the portion of storage element
            bnds = [ bE, bN, bW, bS ]; bnds.sort()
            if   d.count(True) == 1:    fs =  bnds[-1]*bnds[-2]/2.
            elif d.count(True) == 2:    fs =  (bnds[1]+bnds[2])/2.
            elif d.count(True) == 3:    fs =   1 - (1 - bnds[0])*(1-bnds[1])/2.
    
    out = [ fs , bE , bN , bW , bS ]
    return out

######################################################
######################################################

def Receiver( pos , difs , typ, receiver ):
    #INPUTS:
    #   xp, yp, zp           = Coordinates of element central point (in this order inside pos)
    #   dx, dy, dz           = Spatial discretisation (in this order inside difs)
    #   typ                  = Type of receiver (by the moment only 'Circle')
    #   do, xo, yo, ho       = Characteristic of receiver (in this order inside 'receiver')
    #OUTPUTS:
    #   fr                   = Factor of element under receiver (0<fs<1, 0 out, 1 full)
    
    xp, yp, zp = pos
    dx, dy, dz = difs
    xi       = [ xp+dx/2 , xp-dx/2 , xp+dx/2 , xp-dx/2 ]
    yi       = [ yp+dy/2 , yp+dy/2 , yp-dy/2 , yp-dy/2 ]
    
    fr = 0.
    
    if typ == 'Circ':       do, xo, yo, ho     = receiver               # Circular receiver
    elif typ == 'None':     do, xo, yo, ho, fr = 0.,0.,0.,0.,0.         # Without Receiver
    
    if zp > ho - dz:
        d  = [ np.sqrt( (xi[i]-xo)**2 + (yi[i]-yo)**2 ) <= do/2. for i in range(4)]
        
        if all(di for di in d):             fr, bE, bN, bW, bS = [1. for i in range(5)]  #### 1 = RECEIVER
        elif all(not(di) for di in d):      fr, bE, bN, bW, bS = [0. for i in range(5)]  #### 0 = NOT RECEIVER
        else:                                                                            #### BOUNDARIES
            fr, bE, bN, bW, bS   = [1. for i in range(5)]
            ro = do/2.
            
            if ( not(d[0]) and not(d[2]) ):     bE = 0.
            if ( not(d[1]) and not(d[3]) ):     bW = 0.
            if ( not(d[0]) and not(d[1]) ):     bN = 0.
            if ( not(d[3]) and not(d[2]) ):     bS = 0.
            
            #EAST BORDER IS BOUNDARY
            if (d[0] and not(d[2])):    bE = min(max(abs( yo + np.sign(yp-yo)*np.sqrt(abs(ro**2 - (xi[0]-xo)**2 )) - yi[0] ) / dy , 1e-10),1.)
            if (d[2] and not(d[0])):    bE = min(max(abs( yo + np.sign(yp-yo)*np.sqrt(abs(ro**2 - (xi[2]-xo)**2 )) - yi[2] ) / dy , 1e-10),1.)
            #WEST BORDER IS BOUNDARY
            if (d[1] and not(d[3])):    bW = min(max(abs( yo + np.sign(yp-yo)*np.sqrt(abs(ro**2 - (xi[1]-xo)**2 )) - yi[1] ) / dy , 1e-10),1.)
            if (d[3] and not(d[1])):    bW = min(max(abs( yo + np.sign(yp-yo)*np.sqrt(abs(ro**2 - (xi[3]-xo)**2 )) - yi[3] ) / dy , 1e-10),1.)
            #NORTH BORDER IS BOUNDARY
            if (d[0] and not(d[1])):    bN = min(max(abs( xo + np.sign(xp-xo)*np.sqrt(abs(ro**2 - (yi[0]-yo)**2 )) - xi[0] ) / dx , 1e-10),1.)
            if (d[1] and not(d[0])):    bN = min(max(abs( xo + np.sign(xp-xo)*np.sqrt(abs(ro**2 - (yi[1]-yo)**2 )) - xi[1] ) / dx , 1e-10),1.)
            #SOUTH BORDER IS BOUNDARY
            if (d[3] and not(d[2])):    bS = min(max(abs( xo + np.sign(xp-xo)*np.sqrt(abs(ro**2 - (yi[3]-yo)**2 )) - xi[3] ) / dx , 1e-10),1.)
            if (d[2] and not(d[3])):    bS = min(max(abs( xo + np.sign(xp-xo)*np.sqrt(abs(ro**2 - (yi[2]-yo)**2 )) - xi[2] ) / dx , 1e-10),1.)
            
            ## Defining the portion of storage element
            bnds = [ bE, bN, bW, bS ]; bnds.sort()
            if d.count(True) == 1:      fr =  bnds[-1]*bnds[-2]/2.
            elif d.count(True) == 2:    fr =  (bnds[1]+bnds[2])/2.
            elif d.count(True) == 3:    fr =  1 - (1 - bnds[0])*(1-bnds[1])/2.
        
    return fr

##############################################
##############################################
def Converting_chds(tnk):
    ########### CONVERTING CHARGE (3D) TO DISCHARGE (1D) TEMPERATURES #######
    Nx, Ny, Nz, Nh = tnk['N_x'] , tnk['N_y'], tnk['N_z'], tnk['N_h']
    T_ch, T_ds     = tnk['T_ch'], tnk['T_ds']
    
    H_st, dz = tnk['H_st'], tnk['difs'][2]
    P    = dict()
    mm, P['x'], P['y'], P['z'], P['fs']   = tnk['xyz']
        
    z_ds         = np.array([(n+0.5)*(H_st/Nh) for n in range(Nh)])
    Tz, zz, T_ds = np.zeros(Nz+2), np.zeros(Nz+2) , np.zeros(Nh)
    for z in range(0,Nz):
        mmz     = mm[:,:,z].reshape(Nx*Ny)
        Tz[z+1] = sum(T_ch[mmz]*P['fs'][mmz]) / sum(P['fs'][mmz])
        zz[z+1] = (z+0.5)*dz
    zz[0], zz[Nz+1], Tz[0], Tz[Nz+1] = 0., H_st, Tz[1], Tz[Nz]
    
    func = spi.interp1d(zz,Tz)
    T_ds = func(z_ds)
    
    return T_ds

########################################################
def Converting_dsch(tnk):
    
    #####  STILL NOT WORKING!!!! ######
    ########### CONVERTING DISCHARGE (1D) TO CHARGE (3D) TEMPERATURES #######
    Nx, Ny, Nz, Nh = tnk['N_x'] , tnk['N_y'], tnk['N_z'], tnk['N_h']
    T_ch, T_ds     = tnk['T_ch'], tnk['T_ds']
    
    H_st = tnk['H_st']
    
    dx,dy,dz =  tnk['difs']
    P = dict()
    mm, P['x'], P['y'], P['z'], P['fs']   = tnk['xyz']
    Nt = Nx*Ny*Nz
    
#    z_ch         = np.array([H_st - (n+0.5)*(H_st/Nh) for n in range(Nh)])
    Tz, zz, T_ch = np.zeros(Nz+2), np.zeros(Nz+2) , np.zeros(Nt)
    for z in range(0,Nz):
        mmz     = mm[:,:,z].reshape(Nx*Ny)
        Tz[z+1] = sum(T_ch[mmz]*P['fs'][mmz]) / sum(P['fs'][mmz])
        zz[z+1] = H_st - (z+0.5)*dz
    zz[0], zz[Nz+1], Tz[0], Tz[Nz+1] = H_st, 0., Tz[1], Tz[Nz]
    func = spi.interp1d(zz,Tz)
    T_ch = func(z_ds)
    
    return T_ch
#############################################################
#############################################################
def tnk_save_excel(tnk,file):
    
    Nx , Ny , Nz    = tnk['N_x'] , tnk['N_y'] , tnk['N_z']
    df              = pd.DataFrame(tnk['T_ch'].transpose(), columns=['Tp'])
    df['m']         = pd.Series(list(range(Nx*Ny*Nz)),index=df.index)
    df['x']         = pd.Series(tnk['xyz'][1].transpose(),index=df.index)
    df['y']         = pd.Series(tnk['xyz'][2].transpose(),index=df.index)
    df['z']         = pd.Series(tnk['xyz'][3].transpose(),index=df.index)
    df['fs']        = pd.Series(tnk['xyz'][4].transpose(),index=df.index)
    df.to_excel(file,index=False)
    return

#############################################################
def tnk_save_pickle(tnk,file):
    import pickle

    tnk.pop('fluid', None)
    tnk_save = open(file,'wb')
    pickle.dump(tnk,tnk_save)
    tnk_save.close()

    return

#############################################################
#############################################################
def plot_cond_charg( file_pickle, file_output) :
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Rectangle
    from scipy import interpolate
    import pickle
    
    tnk_data = open(file_pickle, "rb"); tnk = pickle.load(tnk_data); tnk_data.close()

    #######################################################################
    X_plot, Y_plot = 10, 8*tnk['Y_d']/tnk['X_d']
    typ = tnk['str_typ']
    
    fig,ax = plt.subplots(1,1,figsize=(X_plot,Y_plot))
    f_s = 14
    
    z_mx = max(tnk['xyz'][3])
    x_out  = tnk['xyz'][1][ tnk['xyz'][3] == z_mx ]
    y_out  = tnk['xyz'][2][ tnk['xyz'][3] == z_mx ]
    Tp_out = tnk['T_ch']  [ tnk['xyz'][3] == z_mx ]
    
    xx, yy = np.meshgrid(x_out,y_out)
    ff = interpolate.griddata((x_out,y_out),Tp_out,(xx,yy), method='linear')
    
    vmin=min(min(Tp_out),850.); vmax=max(max(Tp_out),1600.); lvs = 31; levels = np.linspace(np.floor(vmin), np.ceil(vmax), lvs)
    
    im = ax.contourf( xx , yy , ff , lvs , levels=levels , vmin=vmin , vmax=vmax, cmap='YlOrRd')
    im.ax.tick_params(labelsize=f_s)
    ax.set_xlabel('X direction (m)', fontsize=f_s); ax.set_ylabel('Y direction (m)', fontsize=f_s)
    
    ax.grid(ls='--'); fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]); fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_yticklabels(labels=cbar_ax.get_yticklabels(), fontsize=f_s)
    ax.set_title('Upper view with z=0. time = {0:.3f}[hr], RSR = {1:.3f}[-], HDR={2:.2f}[-]. Q_st={3:.2f}[kWh], eta= {4:.2f} %, Tave= {5:.2f}[K], Tsd= {6:.2f}[K].'.format(tnk['t_ch'],tnk['RSSR'],tnk['HDR'],tnk['Q_in'],tnk['eta_r']*100,tnk['T_av'],tnk['T_sd']))
    
#    ax.set_title('Upper view with z=0. D_rc = {0:.1f}[-], D = {1:.1f}[-]. H = {2:.1f}[kWh]'.format( tnk['D_rc'], tnk['D_st'], tnk['H_st']), fontsize=f_s )
    
    if typ == 'Cylinder' or typ == 'Ring':
        circle1 = plt.Circle((tnk['xr'], tnk['yr']), tnk['D_rc']/2., color='black', fill=False)
        circle2 = plt.Circle((0, 0), tnk['D_st']/2, color='black', fill=False)
        p0 = Wedge((0., 0.), tnk['D_st']/2 + 2, 0, 360, width=2., color='white',facecolor='white')
        ax.add_artist(p0); ax.add_artist(circle1); ax.add_artist(circle2)
    elif typ == 'Rect':
        circle1 = plt.Circle((tnk['xr'], tnk['yr']), tnk['D_rc']/2., color='black', fill=False)
        st_pos  = [ (tnk['X_d']-tnk['L_st'])/2. , -tnk['W_st']/2. ]
        rect1   = Rectangle(st_pos, tnk['L_st'], tnk['W_st'], color='black', lw = 3, fill=False)
        ax.add_artist(circle1); ax.add_artist(rect1)
    if typ == 'Ring':
        circle3 = plt.Circle((0, 0), tnk['Di_st']/2, color='white', fill=True)
        circle4 = plt.Circle((0, 0), tnk['Di_st']/2, color='black', fill=False)
        ax.add_artist(circle3); ax.add_artist(circle4)
        
    plt.show()
    
    fig.savefig( file_output+'_up.pdf' , bbox_inches='tight' )
    fig.savefig( file_output+'_up.png' , bbox_inches='tight' )
    
    #######################################################################
    #######################################################################
    
    fig, ax1 = plt.subplots( nrows=2 , ncols=2 , figsize=(14,8) )
    axes = [ax1[0,0] , ax1[0,1] , ax1[1,0] , ax1[1,1]]
    m_out,x_out,y_out,z_out  = tnk['xyz'][0:4]
    Tp_out   = tnk['T_ch']
    dx,dy,dz = tnk['difs']
    Nx,Ny,Nz = tnk['N_x'], tnk['N_y'], tnk['N_z']
    
    x_aux, z_aux = list(), list()
    Tp_aux = [ list(), list(), list(), list() ]
    
    ttle   = [ 'a) y=0' , 'b) y=x' , 'c) x=0' , 'd) y=-x' ]
    mm = np.zeros( shape=(Nx,Ny,Nz) , dtype=int )
    m=0
    for z,y,x in [(z,y,x) for z in range(Nz) for y in range(Ny) for x in range(Nx)]:
        mm[x,y,z]  = m; m+=1
    
    for x,z in [(x,z) for x in range(Nx) for z in range(Nz)]:
        m = mm[x][int(Ny/2)][z]
        x_aux.append(x_out[m]); z_aux.append(z_out[m])
        Tp_aux[0].append(Tp_out[m])
    
    for x,z in [(x,z) for x in range(Nx) for z in range(Nz)]:
        m = mm[x][x][z]
        Tp_aux[1].append(Tp_out[m])
    
    for x,z in [(x,z) for x in range(Nx) for z in range(Nz)]:
        m = mm[int(Nx/2)][x][z]
        Tp_aux[2].append(Tp_out[m])
    
    for x,z in [(x,z) for x in range(Nx) for z in range(Nz)]:
        m = mm[x][(Ny-1)-x][z]
        Tp_aux[3].append(Tp_out[m])
        
    vmin=min(min(Tp_out),850.); max(max(Tp_out),1600.); lvs = 31; levels = np.linspace(np.floor(vmin), np.ceil(vmax), lvs)
    xx, zz = np.meshgrid( x_aux , z_aux )
    for j in range(4):
        ax = axes[j]        
        TT = interpolate.griddata(( x_aux , z_aux ),Tp_aux[j] , (xx,zz), method='linear')
        im = ax.contourf( xx , zz , TT , lvs , levels=levels , vmin=vmin , vmax=vmax, cmap='YlOrRd') #cmap='YlOrRd'
        im.ax.tick_params(labelsize=f_s)
        ax.set_title(ttle[j], size=f_s+4)
        ax.grid(ls='--'); fig.subplots_adjust(right=0.8)
    
    axes[0].set_ylabel('Height (m)', fontsize= f_s)
    axes[2].set_ylabel('Height (m)', fontsize= f_s)
    axes[2].set_xlabel('Radius (m)', fontsize= f_s)
    axes[3].set_xlabel('Radius (m)', fontsize= f_s)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]); fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_yticklabels(labels=cbar_ax.get_yticklabels(), fontsize=f_s)
    fig.suptitle('Side view with. Time = {0:.3f}[hr], RSR = {1:.3f}[-], HDR={2:.2f}[-]. Q_st={3:.2f}[kWh], eta= {4:.2f} %, Tave= {5:.2f}[K], Tsd= {6:.2f}[K].'.format(tnk['t_ch'],tnk['RSSR'],tnk['HDR'],tnk['Q_in'],tnk['eta_r']*100,tnk['T_av'],tnk['T_sd']))
    
#    fig.suptitle('time = {0:.2f}[hr], Tave= {1:.1f}[K], Tmax= {2:.1f}[K].'.format( tnk['t_tot'], tnk['T_av'], tnk['T_mx']), fontsize=f_s+6 )
    plt.show()
    fig.savefig(file_output+'_side.pdf', bbox_inches='tight')
    fig.savefig(file_output+'_side.png', bbox_inches='tight')
    
    return

#############################################################
############### HEAT TRANSFER  UTILITIES ####################
#############################################################
def props_cte(mat,T):    
    prop = {
            'Sand rock-mineral oil' : [1700,    1.0,    1300 ],
            'Reinforced concrete'   : [2200,    1.5,     850 ],
            'NaCl'                  : [2160,    7.0,     850 ],
            'Cast iron'             : [7200,   37.0,     560 ],
            'Cast steel'            : [7800,   40.0,     600 ],
            'Silica fire bricks'    : [1820,    1.5,    1000 ],
            'Magnesia fire bricks'  : [3000,    5.0,    1150 ],
            'HT concrete'           : [2700,    1.0,     910 ],
            'Castable Ceramic'      : [3500,    1.4,     866 ],
            'Copper'                : [8960,  401.0,     385 ],
            'Glass'                 : [2490,    0.8,     837 ],
            'Quartz'                : [2200,    1.7,    1000 ]
            }
    
    return prop[mat]

#############################################################
#############################################################

def h_conv_nat_conf( T_1, T_2, dx, f ):
    """
    Correlation for natural convection in upper hot surface horizontal plate
    T_1, T_2             : lower hot and upper cold temperatures [K]
    dx                   : Space [m]
    P                    : Pressure [-]
    """
    
    T_av = ( T_1 + T_2 )/2
    mu, k, rho, cp = f.viscosity, f.thermal_conductivity, f.density_mass, f.cp_mass
    alpha = k/(rho*cp)
    beta = 1./T_av
    visc = mu/rho
    Pr = visc/alpha
    
    Ra = cte.g * beta * abs(T_1 - T_2) * dx**3 * Pr / visc**2
    
    if Pr>0.5 and Pr<2:
        if Ra>1e4 and Ra<4e5:
            Nu = 0.195*Ra**0.25
        elif Ra>=4e5 and Ra<1e7:
            Nu = 0.068*Ra**(1./3.)
        else:
            Nu=1.       #Conduction
    else:
        Nu = 0.069*Ra**(1/3)*Pr**0.074
    
    h = (k*Nu/dx)
    return h

#############################################################
#############################################################
    
def h_conv_nat_amb(T_s, T_inf, L, f):
    """
    Correlation for natural convection in upper hot surface horizontal plate
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = ( T_s + T_inf )/2
    mu, k, rho, cp = f.viscosity, f.thermal_conductivity, f.density_mass, f.cp_mass
    alpha = k/(rho*cp)
    beta = 1./T_av
    visc = mu/rho
    Pr = visc/alpha
    
    Ra = cte.g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
    if Ra > 1e4 and Ra < 1e7:
        Nu = 0.54*Ra**0.25
        h = (k*Nu/L)
    elif Ra>= 1e7 and Ra < 1e9:
        Nu = 0.15*Ra**(1./3.)
        h = (k*Nu/L)
    else:
        h = 1.52*(T_s-T_inf)**(1./3.)   #Simplified expression for air, Turbulent (range)
#        print('fuera de Ra range: Ra= '+str(Ra))
    return h

##############################################################
##############################################################
def h_conv_tube(fluid, m_f, T_f, T_st, D_p, L_p ,F_enh, x):

    """ Source: Kakac et al. (1987) """
    
    fluid.TPY     = T_f, 8e6, 'CO2:1.00'
    rho_f, cp_f, k_f, visc_f = fluid.density_mass, fluid.cp, fluid.thermal_conductivity, fluid.viscosity
    
    Re, Pr  = 4 * m_f / (visc_f * np.pi * D_p), cp_f * visc_f / k_f
    Pe, F   = Re * Pr , F_enh if Re>2300. else 1.
    
    eps          = 0.26/(D_p*1000.)
    f            = ( - 1.8 * np.log10(6.9/Re + (eps/3.7)**1.11) )**(-2)
    xx, xL, vel  = x/(D_p*Pe) , L_p/(D_p*Pe), m_f/(rho_f * np.pi * D_p**2/4)

    if Re>2300:
        NuL = 0.023 * Re**0.8 * Pr**0.4                                               #Dittus-Boelter
        Nux = (f/8) * (Re-1000.) * Pr / ( 1 + 12.7 * (f/8)**0.5 * ( Pr**(2/3) - 1 ) ) #Gnielinski
    else:
        Nux = 3.66 + 0.0018 /(xx**(1./3) * (0.04 + xx**(2./3))**2 )          #Laminar flow in entry region (local)
        NuL = 3.66 + 0.0668 /(xL**(1./3) * (0.04 + xL**(2./3)) )             #Laminar flow in entry region (mean)
    
    hx, hL  = F*Nux*k_f/D_p, F*NuL*k_f/D_p
    
    return hx, hL, Re, Pr, f, vel


##############################################################

def CO2_k(T,Pa):
    
    CO2 = ct.CarbonDioxide()
    CO2.TP = T , Pa
    rho = CO2.density_mass
    
    P = Pa/1e6
    if Pa<20e6:
        k = 0.0936*T - 0.448*P + 0.0739*rho - 0.244*np.log(rho) - 10.8*P/T
        k = k + 0.00753*P**2 + 1.85e-5 * rho**2 + 94.7*rho/(T*P) - 16.7
    else:
        k = 0.0575*T - 0.0151*P + 0.0372*rho + 4.96e-5**np.log(np.log(rho)) - 0.00695*np.log(rho)
        k = k + 1.41e-5*T**2 + 7.2e-8 * rho**3 + 1.78
        
    return k
