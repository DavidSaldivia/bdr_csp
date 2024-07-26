# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:35:16 2019

@author: z5158936
"""

import SBM as sbm
import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import scipy.optimize as spo
import cantera as ct

    
#######################################################
#######################################################    
def Conditions_discharge(plant,block):
    D             = dict()
    D['m']        = [0. for x in range(10)]
    D['T']        = [block['T_fm'] for x in range(10)]
    D['m'][0]     = plant['m_CO2']
    D['T'][0]     = plant['T_CO2_lo']
    D['T_fout']   = plant['T_CO2_hi']
    D['eta_el']   = plant['eta_el']
    D['min_m']    = block['min_m']
    D['ini'], D['conv'], D['plot'], D['PH_enter'], D['ex_sol'], D['PreHT'], D['change_tnk'] = True, True, True, False, True, True, False
    D['dm_tol'], D['dT_tol'], D['max_it'], D['N_noc'], D['Npr']              = 1e-3*D['m'][0], 0.5, 1e4, 0, 2
    D['R']        = {'HL':0,'ps':0,'rL':0,'rH':0}
    
    out  = sbm.Discharge_1D(block)
    T_fo      = out['Tfo']
    D['N_p']  = out['N_p']
    D['UA']   = out['UA']
    
    if T_fo < D['T_fout']:  D['op_ini'] = 1; D['c_tnk'] = 2
    else:                   D['op_ini'] = 5; D['c_tnk'] = 2         #!!!!

    D['op_mode'], D['op_sty']  = D['op_ini'], 2
    CO2          = ct.Solution('gri30.xml','gri30_mix')
    CO2.TPX      = 0.5*(plant['T_CO2_hi']+plant['T_CO2_lo']), 8e6, 'CO2:1.00'
    D['P_th']    = plant['m_CO2'] * CO2.cp * (plant['T_CO2_hi'] - plant['T_CO2_lo'])
    D['P_el']    = D['eta_el'] * D['P_th']                                #[W] Target of electrical power
    return D

#######################################################
#######################################################
def Discharge_process(D,HT,LT,PH):

    aux = 'm','T','T_fout','op_mode','op_ini','op_sty','N_tks','Npr','R','N_noc'
    m, T, T_fout, op_mode, op_ini, op_sty, N_tks, Npr, R, N_noc = [D[x] for x in aux]
    aux = 'dm_tol', 'dT_tol', 'max_it','ini','conv','plot','PH_enter','PreHT','ex_sol','change_tnk'
    dm_tol, dT_tol,max_it, ini, conv, plot, PH_enter, PreHT, ex_sol, change_tnk = [D[x] for x in aux]
    l0 , l1  = 0.-dm_tol , 1.+dm_tol
    
    #SELECTING INITIAL GUESSES FOR OPERATION MODE
    if op_mode == 0 and ini: # All the flow goes through HT. No parallel flow, only recirculation in HT.
       (m[1],m[3]) = (0., 2*m[0])           ; R = {'HL':1., 'ps':0., 'rL':0., 'rH':(m[3]-m[0])/m[3] }
    if op_mode == 1 and ini:       # Parallel operation. Both flows LT and HT without recirculation
        (m[1],m[3]) = (0.95*m[0], 0.05*m[0]); R = {'HL':m[3]/m[0], 'ps':1., 'rL':0., 'rH':0. }
    if op_mode == 2 and ini:       # Serie/parallel flow, no recirculating LT nor HT
        (m[1],m[3]) = (m[0], 0.5*m[0])      ; R = {'HL':0., 'ps':(m[0]-m[3])/m[1], 'rL':0., 'rH':0. }
    if op_mode == 3 and ini:       # Serie flow, recirculating in LT
        (m[1],m[3]) = (2*m[0], m[0])        ; R = {'HL':0., 'ps':0., 'rL':m[0]/m[1], 'rH':0. }
    if op_mode == 4 and ini:       # Serie flow, recirculating in HT
        (m[1],m[3]) = (m[0], 1.1*m[0])      ; R = {'HL':0., 'ps':0., 'rL':0., 'rH':(m[3]-m[0])/m[0] }
    if op_mode == 5 and ini:       # Partial flow throw HT. 
        (m[1],m[3]) = (0.5*m[0], 0.5*m[0])  ; R = {'HL':m[3]/m[0], 'ps':1., 'rL':0., 'rH':0. }
    if ini: ini = False
    
    ################################################################################
    PH_enter = True  if  ( PreHT and N_tks>Npr )  else  False
    it = 1;   conv=True;      auxA = np.zeros(10)
    while conv:                        # While loop to get solution
        # Conditions of Decisions and Nodes Mass flow rates
        m[2] = m[1]                                             # LT Model Discharge mass flow rate
        m_HT = m[0] * R['HL']   ; m_LT = m[0] - m_HT            # Ratio between HT and LT; General eq mass flow rate
        m[6] = m[2] * R['ps']   ; m[5] = m[2] - m[6]            # Ratio between paralell and serie; Node iv) mass flow rate
        m[7] = m[5] * R['rL']   ; m[8] = m[5] - m[7]            # Ratio of recirculation in LT; Node v) mass flow rate
        m[4] = m[3]             ; m[9] = (m[6]+m[4]) * R['rH']  # HT Model Discharge; Ratio of recirculation in HT
        
        no_ds = 0 if op_mode==5 else 1
        # Obtaining partial answers in functions
        PH['m_f'], PH['T_fin'] = m[0]*PH_enter , T[0]    ; out_PH = sbm.Discharge_1D( PH )
        LT['m_f'], LT['T_fin'] = m[1]*no_ds    , T[1]    ; out_LT = sbm.Discharge_1D( LT )
        HT['m_f'], HT['T_fin'] = m[3]          , T[3]    ; out_HT = sbm.Discharge_1D( HT )
        
        ( T_ph , T[2] , T[4] ) = (out_PH['T_fm'][-1] , out_LT['T_fm'][-1] , out_HT['T_fm'][-1] )   #Model Discharge Temps
        ( T[5] , T[6] )        = ( T[2] , T[2] )                                                #Node iv)a) and b) Temp
        ( T[7] , T[8] )        = ( T[5] , T[5] )                                                #Node  v)a) and b) Temp
        T[9] = ( m[6]*T[6] + m[4]*T[4] ) / ( m[0] + m[9] )                                      #Node iii) Temp
        T[1] = ( m_LT*T_ph + m[7]*T[7] ) / m[1]              if ( m[1]>0. ) else T_ph           #Node  i) Temp
        T[3] = ( m_HT*T_ph + m[8]*T[8] + m[9]*T[9] ) / m[3]  if ( m[3]>0. ) else T_ph           #Node ii) Temp
        
        # Comparing with decision rules, repeat or break
        if abs(T[9] - T_fout) > dT_tol and (it < max_it) :
            
            if op_mode == 0:
                m[3] = ( m[3] + dm_tol ) if ( T[9] > T_fout) else ( m[3] - dm_tol )
                m[4] = m[3]               ; m[9] = m[3] - m_HT - m[8]
                
            if op_mode == 1:
                m_LT = ( m_LT + dm_tol ) if ( T[9] > T_fout) else ( m_LT - dm_tol )
                m_HT = m[0] - m_LT        ; m[1] = m_LT + m[7]          ; m[3] = m_HT + m[8] + m[9]

            if op_mode == 2:
                m[3] = ( m[3] + dm_tol ) if ( T[9] < T_fout) else ( m[3] - dm_tol )
                m[8] = m[3] - m_HT - m[9] ; m[5] = m[8]                 ; m[6] = m[2]-m[5]
                
            if op_mode == 3:
                m[1] = ( m[1] + dm_tol ) if ( T[9] > T_fout) else ( m[1] - dm_tol )
                m[2] = m[1]               ; m[3] = m_HT + m[8] + m[9]   ; m[7] = m[1] - m_LT

            if op_mode == 4:
                m[3] = ( m[3] + dm_tol ) if ( T[9] > T_fout) else ( m[3] - dm_tol )
                m[4] = m[3]               ; m[9] = m[3] - m_HT - m[8]

            if op_mode == 5:
                m_LT = ( m_LT + dm_tol ) if ( T[9] > T_fout) else ( m_LT - dm_tol )
                m_HT = m[0] - m_LT        ; m[1] = m_LT + m[7]          ; m[3] = m_HT + m[8] + m[9]
            
            R['HL'] = ( m_HT / m[0] )
            R['ps'] = ( m[6] / m[2] ) if m[2]>0 else 0.
            R['rL'] = ( m[7] / m[5] ) if m[5]>0 else 0.
            R['rH'] =   m[9] / ( m[4] + m[6] )
            
            txt = str(op_mode)+' '+str(it)+' '+' '.join('{:.4f}'.format(x) for x in m) + ' | '
            txt = txt + ' '.join('{:.1f}'.format(x) for x in T[1:] ) + ' | ' + ' '.join('{:.2f}'.format(R[x]) for x in R )
#            print(txt)
            
            res    = [ (l0<=R[x]<=l1) for x in R ] + [ x>0. for x in T ] + [ x>l0 for x in m ]
            if not(all(res)):   (conv, PH_enter) = (False, False)
            else:               it += 1
        
            if it>10 and PH_enter and out_LT['Tfo']>T_fout:      PH_enter = False   #If Preheater doesn't work
            
            auxA = np.concatenate(([abs(T[9] - T_fout),],auxA[:-1]))
            if it>100 and np.array_equal(auxA,auxA[np.argsort(-auxA)]):   conv = False      #If conv method doesn't work
        
        else:
            if (it>= max_it):   ex_sol = False
            break
    ################################################################################
    
    if not(conv):            # Decide if change operation mode
        change_tnk = True if op_mode != op_ini else False
        ini        = True
        op_mode = op_sty ; print("System do not converged with "+str(it)+" iterations"); print(txt)
    else:                      # Go next timestep
        change_tnk = False
        for x in out_HT:    ( PH[x] , LT[x] , HT[x] ) = ( out_PH[x] , out_LT[x] , out_HT[x] )
    
    if change_tnk:                 N_tks       += 1
    if N_noc>5:                    ex_sol      = False
    
    plot  = True if conv else False
    N_noc = 0    if conv else N_noc+1
    
    D['ini'], D['conv'], D['plot'], D['PreHT'], D['PH_enter']       = ini, conv, plot, PreHT, PH_enter
    D['N_noc'], D['ex_sol'], D['change_tnk']                        = N_noc, ex_sol, change_tnk
    D['m'], D['T'], D['op_mode'], D['op_sty'], D['N_tks'] , D['R']  = m, T, op_mode, op_sty, N_tks, R
    
    return D, HT, LT, PH

######################################################
##############DISCHARGING PROCESS ####################
######################################################

Case    = 100
N_tanks = 5
Tst_mx  = 1000.
HDR     = 0.25
RSSR    = 0.20
R_nu    = 0.100
D_p     = 0.050
w_vel   = 2.0
q_sun   = 1e6
m_CO2   = 10.
folder = 'Discharge'
file = folder+'/Cases.txt'

f = open(file,'w'); f.write('Case\tN_tnks\tm_CO2\tTst_mx\tD_st\tV_st\tt_ds_tot\tT_min\n'); f.close()

for (N_tanks,m_CO2,D_st) in [(N_tanks,m_CO2,D_st) for N_tanks in [4,] for m_CO2 in [10,] for D_st in [4,]]:
    
    print('Discharging process with HDR={0:.2f}\tRSSR={1:.2f}\tR_nu={2:.2f}\tTst_mx={3:.2f}\tD_st={4:.2f}\t'.format(HDR,RSSR,R_nu,Tst_mx,D_st)+'. Aguante el bulla giles culiaos. Indio sapo y la conchetumare.')
    print('Discharging conditions: N_tnks={0:.3f}\tm_CO2={1:.3f}\tD_st={2:.3f}'.format(N_tanks,m_CO2,D_st))
    
    ##################################################
    ########### SETTINGS PARAMETERS ##################
    ##################################################
    plant, tnk = sbm.Conditions_std()
    plant['N_tanks']= N_tanks
    plant['q_sun']  = q_sun
    plant['m_CO2']  = m_CO2
    plant['Tst_max']= 1000

    tnk['q_sol']    = q_sun
    tnk['m_f']      = m_CO2
    tnk['D_st']     = D_st
    tnk['HDR']      = HDR
    tnk['RSSR']     = RSSR
    tnk['D_p']      = D_p
    tnk['R_nu']     = R_nu
    tnk['w_vel']    = w_vel
    tnk['D_rc']     = D_st * np.sqrt(tnk['RSSR'])
    tnk['ER']       = max(np.sqrt(max(0.5/tnk['RSSR'] - 1.,0.)) if tnk['D_st']>2*tnk['D_rc'] else tnk['D_st']/tnk['D_rc']-1.,0.)
    tnk['T_ds']     = Tst_mx
    tnk['T_avmx']   = Tst_mx
    
    Tank            = [{x:tnk[x] for x in tnk} for X in range(N_tanks+1)]
    
    
    ############ DISCHARGING PROCESS ##################
    HT, LT, PH      = [{x:X[x] for x in X} for X in Tank[0:3]]
    print('Using tanks: 0, 1, and 2.')
    
    ##################################################
    ############  DISCHARGING PROCESS  ###############
    ##################################################
    D     = Conditions_discharge(plant,tnk)
    c_tnk = D['N_tks']
    pics  = []; n_pic   = 1    
    data  = pd.DataFrame([],columns=['t','c_tnk','mLT','mHT','T_PH','T_LT','T_HT'])
    
    tt    = 0.
    while tt <= (plant['t_dis']*3600.):
        
        ################################################
        D, HT, LT, PH = Discharge_process(D,HT,LT,PH)
        ################################################
        
        # Exit if it is necessary
        if not(D['ex_sol']):        print('No solution in current state');  break
        if D['N_tks'] > N_tanks:    print('All tanks already used');        break
        
        if D['conv']:
            data = data.append(pd.Series([tt,c_tnk,D['m'][1],D['m'][3],PH['T_av'],LT['T_av'],HT['T_av']],index=data.columns), ignore_index=True)    
            tt += tnk['dt_ds']

        ################################################
        if D['change_tnk']:
            c_tnk += 1; print('Using tanks: '+str(c_tnk-2)+', '+str(c_tnk-1)+', and '+str(c_tnk)+'.')
            (Tank[c_tnk-3],Tank[c_tnk-2],Tank[c_tnk-1])  = [{ x:X[x] for x in X } for X in [PH, LT, HT]]
            (PH, LT, HT)                                 = [{ x:X[x] for x in X } for X in [LT, HT, Tank[c_tnk]]]

        ################################################
        #Plot tanks temp profiles
        if D['plot'] and (tt%720==0 or n_pic==1):
            f = plt.figure(figsize=(9,6))
            plt.plot( HT['z'] , HT['T_fm'] , color='red'  , ls='-'  , label='HT T CO2')
            plt.plot( HT['z'] , HT['T_ds'] , color='red'  , ls='--' , label='LT T block (N='+str(c_tnk)+')')
            plt.plot( LT['z'] , LT['T_fm'] , color='blue' , ls='-'  , label='LT T CO2')
            plt.plot( LT['z'] , LT['T_ds'] , color='blue' , ls='--' , label='LT block (N='+str(c_tnk-1)+')')
            
            if D['PH_enter']:
                plt.plot( PH['z'] , PH['T_fm'] , color='green'  , ls='-'  , label='PH CO2')
                plt.plot( PH['z'] , PH['T_ds'] , color='green'  , ls='--' , label='PH block (c='+str(c_tnk-2)+')')
            
            titulo = 'time: {0:.3f}[hrs]'.format(tt/3600.)
            
            plt.title(titulo); plt.legend(loc=4)
            plt.xlabel('Height [m]'); plt.ylabel('Temperature [K]'); plt.ylim(plant['T_CO2_lo'],plant['Tst_max']+1.)
            plt.grid(); plt.show()
            name_file = folder+"/pic_"+str(Case).zfill(3)+"_"+str(n_pic).zfill(3)+".png"
            f.savefig(name_file, bbox_inches='tight')
            pics.append(name_file)
            n_pic+=1
        
    ####################################################
    ########### PERFORMANCE AND PRINT ##################
    ####################################################
    f = open(file,'a')
    try:
        t_ds_tot = tt/3600.
        V_st = np.pi * D_st**3 * HDR / 4
        T_min = np.min(data['T_PH'])
        data.to_excel(folder+'/Cases_'+str(Case).zfill(3)+'.xlsx', sheet_name='time', index=False )
        if D['ex_sol']:
            text=str(Case).zfill(4)+'\t'+'\t'.join('{:.3f}'.format(x) for x in [N_tanks, m_CO2, Tst_mx, D_st, V_st, t_ds_tot,T_min])+'\n'
            print(text)
            f.write(text)
        else:
            f.write(str(Case).zfill(4)+'\t'+'\t'.join('{:.3f}'.format(x) for x in [N_tanks, m_CO2, Tst_mx, D_st])+'System without solution\n')
    except:
        f.write(str(Case).zfill(4)+'\t'+'\t'.join('{:.3f}'.format(x) for x in [N_tanks, m_CO2, Tst_mx, D_st])+'System without solution. CTM!\n')
    f.close()
    
    tnk.clear(); plant.clear(); D.clear(); HT.clear(); LT.clear(); PH.clear(); Tank.clear()
    
    Case +=1
#############################################################
#############################################################