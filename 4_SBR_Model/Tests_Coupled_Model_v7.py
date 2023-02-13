# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:31:21 2020

@author: z5158936
"""


import SBM as sbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cantera as ct
import time
import imageio
import os
import pygifsicle

#######################################################
#######################################################

def Discharge_process_all(Case, N_pr, plant, tnk, PH, LT, HT, file_ds):
    ###################################################
    ### INITIALISING CONDITIONS
    ###################################################
    N_tanks         = plant['N_tanks']
    
    m        = [plant['m_CO2'],]    + [0. for x in range(9)]
    T        = [plant['T_CO2_lo'],] + [tnk['T_fm'] for x in range(9)]
    T_fout   = plant['T_CO2_hi']
    R        = {'HL':0,'ps':0,'rL':0,'rH':0}
    
    ini, conv, plot, ex_sol = True, True, True, True
    dm_tol , dT_tol          =   1e-3*m[0] , 0.5                    #!!!!
    max_it , N_noc           =   1e4 , 1
    l0     , l1              = 0.-dm_tol , 1.+dm_tol
    
    op_mode = 0; PH_enter = True
    
    #This is to let LT max temp be diff than HT max temp. Avoid to change everything inside the loop.
    if op_mode==0:
        aux = {x:HT[x] for x in HT}
        HT  = {x:LT[x] for x in LT}
        LT  = {x:aux[x] for x in aux}
    ##########################################################################
    n_pic   = 1
    data  = pd.DataFrame([],columns=['t','mLT','mHT','T_avPH','T_avLT','T_avHT'])
    
    print('D_st={0:.1f}. N={1:.0f}. HDR={2:.2f}. mCO2={3:.1f}'.format(HT['D_st'],N_tanks,HT['HDR'],plant['m_CO2']))
    print('Using tanks: 0, 1, and 2.')
    print('Tanks average temperatures: TavHT={0:8.2f}\tTavLT={1:8.2f}\tTavPH={2:8.2f}'.format(HT['T_av'], LT['T_av'], PH['T_av']))
    
    tt    = 0.
    while True:
        
        ################################################################################
        ################################################################################
        
        #SELECTING INITIAL GUESSES FOR OPERATION MODE
        if op_mode == 0 and ini:        # All the flow goes through HT. No parallel flow, only recirculation in HT.
            (m[1],m[3]) = (0.5*m[0], 0.5*m[0])  ; R = {'HL':m[3]/m[0], 'ps':1., 'rL':0., 'rH':0. }
        if op_mode == 1 and ini:        # Serie/parallel flow, no recirculating LT nor HT
            (m[1],m[3]) = (m[0], 0.5*m[0])      ; R = {'HL':0., 'ps':(m[0]-m[3])/m[1], 'rL':0., 'rH':0. }
        if ini: ini = False
        
        ################################################################################
        it = 1;   conv=True;      auxA = np.zeros(10)
        while conv:                        # While loop to get solution
            # Conditions of Decisions and Nodes Mass flow rates
            m[2] = m[1]                                             # LT Model Discharge mass flow rate
            m_HT = m[0] * R['HL']   ; m_LT = m[0] - m_HT            # Ratio between HT and LT; General eq mass flow rate
            m[6] = m[2] * R['ps']   ; m[5] = m[2] - m[6]            # Ratio between paralell and serie; Node iv) mass flow rate
            m[7] = m[5] * R['rL']   ; m[8] = m[5] - m[7]            # Ratio of recirculation in LT; Node v) mass flow rate
            m[4] = m[3]             ; m[9] = (m[6]+m[4]) * R['rH']  # HT Model Discharge; Ratio of recirculation in HT
            
            no_ds = 0 if op_mode==0 else 1
            # Obtaining partial answers in functions
            PH['m_f'], PH['T_fin'] = m[0]*PH_enter , T[0]    ; out_PH = sbm.Discharge_1D( PH )
            LT['m_f'], LT['T_fin'] = m[1]*no_ds    , T[1]    ; out_LT = sbm.Discharge_1D( LT )
            HT['m_f'], HT['T_fin'] = m[3]          , T[3]    ; out_HT = sbm.Discharge_1D( HT )
            
            ( T_ph , T[2] , T[4] ) = (out_PH['T_fm'][-1] , out_LT['T_fm'][-1] , out_HT['T_fm'][-1] )   #Model Discharge Temps
            ( T[5] , T[6] )        = ( T[2] , T[2] )                                                   #Node iv)a) and b) Temp
            ( T[7] , T[8] )        = ( T[5] , T[5] )                                                   #Node  v)a) and b) Temp
            T[9] = ( m[6]*T[6] + m[4]*T[4] ) / ( m[0] + m[9] )                                         #Node iii) Temp
            T[1] = ( m_LT*T_ph + m[7]*T[7] ) / m[1]              if ( m[1]>0. ) else T_ph              #Node  i) Temp
            T[3] = ( m_HT*T_ph + m[8]*T[8] + m[9]*T[9] ) / m[3]  if ( m[3]>0. ) else T_ph              #Node ii) Temp
            
            # Comparing with decision rules, repeat or break
            if abs(T[9] - T_fout) > dT_tol and (it < max_it) :
                
                if op_mode == 0:
                    m_LT = ( m_LT + dm_tol ) if ( T[9] > T_fout) else ( m_LT - dm_tol )
                    m_HT = m[0] - m_LT        ; m[1] = m_LT + m[7]          ; m[3] = m_HT + m[8] + m[9]
                    
                if op_mode == 1:
                    m[3] = ( m[3] + dm_tol ) if ( T[9] < T_fout) else ( m[3] - dm_tol )
                    m[8] = m[3] - m_HT - m[9] ; m[5] = m[8]                 ; m[6] = m[2]-m[5]
                
                R['HL'] = ( m_HT / m[0] )
                R['ps'] = ( m[6] / m[2] ) if m[2]>0 else 0.
                R['rL'] = ( m[7] / m[5] ) if m[5]>0 else 0.
                R['rH'] =   m[9] / ( m[4] + m[6] )
                
                txt = str(op_mode)+' '+str(it)+' '+' '.join('{:.4f}'.format(x) for x in m) + ' | '
                txt = txt + ' '.join('{:.1f}'.format(x) for x in T[1:] ) + ' | ' + ' '.join('{:.2f}'.format(R[x]) for x in R )
#                print(txt)
                
                res    = [ (l0<=R[x]<=l1) for x in R ] + [ x>0. for x in T ] + [ x>l0 for x in m ]
                if not(all(res)):   conv = False
                else:               it += 1
            
                auxA = np.concatenate(([abs(T[9] - T_fout),],auxA[:-1]))
                if it>100 and np.array_equal(auxA,auxA[np.argsort(-auxA)]):   conv = False      #If conv method doesn't work
            
            else:
                if (it>= max_it):   ex_sol = False
                break
        ################################################################################
        
        if conv:                          # Go next timestep
            for x in out_HT:    ( PH[x] , LT[x] , HT[x] ) = ( out_PH[x] , out_LT[x] , out_HT[x] )
            data = data.append(pd.Series([tt, m[1], m[3],PH['T_av'], LT['T_av'], HT['T_av']], index=data.columns ), ignore_index=True)
            tt += tnk['dt_ds']
        
        else:                             # Decide if change operation mode
            print("System do not converged with "+str(it)+" iterations"); print(txt)
            ini = True
            if op_mode == 0 and PH_enter == True:
                op_mode = 1
                PH_enter = True
                aux = {x:HT[x] for x in HT}
                HT  = {x:LT[x] for x in LT}
                LT  = {x:aux[x] for x in aux}
            else:
                print('Aguante el bulla!')
                break
            
        if N_noc>4:   ex_sol  = False
        plot  = True  if conv else False
        N_noc = 1     if conv else N_noc+1

        
        ##############################################
        ##### Ploting tanks temp profiles
        ##############################################
        if plot and tt%900==0:
            print('D_st={0:.1f}. N={1:.0f}. HDR={2:.1f}. mCO2={3:.1f}'.format(HT['D_st'], N_tanks, HT['HDR'], plant['m_CO2']))
            f1 = plt.figure(figsize=(12,8))
            if op_mode==0:
                plt.plot( HT['z'] , HT['T_fm'] , color='blue' , ls='-'  , lw=3 , label=r'LT $T_f$')
                plt.plot( HT['z'] , HT['T_ds'] , color='blue' , ls='--' , lw=3 , label=r'LT $T_{st}$')
                plt.plot( PH['z'] , PH['T_fm'] , color='green', ls='-'  , lw=3 , label=r'PH $T_{f}$')
                plt.plot( PH['z'] , PH['T_ds'] , color='green', ls='--' , lw=3 , label=r'PH $T_{st}$')
            
            if op_mode==1:
                plt.plot( HT['z'] , HT['T_fm'] , color='red'  , ls='-'  , lw=3 , label=r'HT $T_f$')
                plt.plot( HT['z'] , HT['T_ds'] , color='red'  , ls='--' , lw=3 , label=r'HT $T_{st}$')
                plt.plot( PH['z'] , PH['T_fm'] , color='green', ls='-'  , lw=3 , label=r'PH $T_{f}$')
                plt.plot( PH['z'] , PH['T_ds'] , color='green', ls='--' , lw=3 , label=r'PH $T_{st}$')
                plt.plot( LT['z'] , LT['T_fm'] , color='blue' , ls='-'  , lw=3 , label=r'LT $T_f$')
                plt.plot( LT['z'] , LT['T_ds'] , color='blue' , ls='--' , lw=3 , label=r'LT $T_{st}$')    
            
            f_s = 18
#            titulo = 'tiempo: {0:.2f}[hrs]. m_LT={1:.1f}[kg/s], m_HT={2:.1f}[kg/s],Tfo={3:.1f}[K], THT_av={4:.1f}[K], TLT_av={5:.1f}[K],TLT_av={6:.1f}[K].'.format(tt/3600., m[1],m[3], T[9],HT['T_av'],LT['T_av'],PH['T_av'])
            
            aux_text=''
            if n_pic == 2: aux_text='a) '
            if n_pic == 4: aux_text='b) '
            if n_pic == 6: aux_text='c) '
            if n_pic == 8: aux_text='d) '
            
#            plt.title('time: {0:.2f}(hr)'.format(tt/3600.), fontsize=f_s)
            plt.title(aux_text+'time: {0:.2f}(hr)'.format(tt/3600.), fontsize=f_s)
            
            plt.legend(loc=4, fontsize=f_s)
            plt.xlabel('Height (m)', fontsize=f_s-2); plt.ylabel('Temperature (K)', fontsize=f_s)
            plt.xticks(fontsize=f_s); plt.yticks(fontsize=f_s)
            plt.ylim(plant['T_CO2_lo'],plant['Tst_max']+1.)
            plt.grid(ls='--'); plt.show()
            name_file = folder+"/Case_"+str(int(Case))+"_"+str(n_pic).zfill(3)+"_ds7.png"
            name_file = "Case_"+str(int(Case))+"_"+str(n_pic).zfill(3)+"_ds7.png"
            f1.savefig(name_file, bbox_inches='tight')
#            pics.append(name_file)
            n_pic+=1
        ################################################
        ################################################
        # Breaking if no solutions
        if not(ex_sol)          :        print('No solution in the system') ; break
        ###############################################
        
        ###############################################
        ########## END OF TEMPORAL LOOP ###############
        ###############################################
    
    for x in out_HT:    ( PH[x] , LT[x] , HT[x] ) = ( out_PH[x] , out_LT[x] , out_HT[x] )
    
    data.to_csv(folder+'/Case_'+str(int(Case))+'_ds7.csv')
        
    CO2          = ct.Solution('gri30.xml','gri30_mix')
    CO2.TPX      = 0.5*(plant['T_CO2_hi']+plant['T_CO2_lo']), 8e6, 'CO2:1.00'
    P_th         = plant['m_CO2'] * CO2.cp * (plant['T_CO2_hi'] - plant['T_CO2_lo'])
    E_sun        = plant['E_sun']
    
    t_ds_tot = tt/3600.                                     #[hr] Total time of discharge
    E_th     = P_th*t_ds_tot/1e3 if P_th>0. else 0.         #[kWh] Energy delivered to Power Block
    eta_st   = E_th / E_sun if E_sun>0. else 0.             #[-] Efficiency of dual receiver-storage
    
    plant['t_pr_ds'] , plant['E_th'] , plant['eta_st'] = t_ds_tot, E_th, eta_st
    
    text = str(Case).zfill(4)+'\t'+str(N_pr).zfill(2)+'\t'
    text = text+'\t'.join('{:.2f}'.format(x) for x in [E_th, eta_st, t_ds_tot])+'\t'
    text = text+'\t'.join('{:.2f}'.format(x['T_av']) for x in [PH, LT, HT])+'\n'
    print([x['T_av'] for x in [PH, LT, HT]])
    f = open(file_ds,'a'); f.write(text); f.close()
    
    return PH, LT, HT


#########################################################
#########################################################
    
folder     = 'Discharge'

file_cs    = 'Coupled_cs7.txt'
file_ch    = 'Coupled_ch7.txt'
file_ds    = 'Coupled_ds7.txt'

#f = open(file_cs,'w')    ; f.write('Case\tN_pr\tD_st\tN_tanks\tHDR\tRSSR\tm_f\tTavmx\tT_phmx\tR_nu\n'); f.close()
#f = open(file_ch,'w')    ; f.write('Case\tN_pr\tVol\tQ_st\tE_sun\tSED\tA_sun\tSM\tt_proc\tTavs\n'); f.close()
#f = open(file_ds,'w')    ; f.write('Case\tN_pr\tE_th\teta_st\tt_ds_tot\tTavs\n'); f.close()

##############################################################
##############################################################
plnt,tnk         = sbm.Conditions_std()

v1    = [10.,]                                             #[m] D_st
v2    = [3, ]                                              #[-] Ntnks
v3    = [0.25, ]                                           #[-] HDR
v4    = [0.2, ]                                            #[-] RSSR
v5    = [ 10., 25., 50., 100., ]                        #[kg/s] mass flow rate
v6    = [1050., ]                                    #[K] max average temp.
v7    = [0.10,]                                            #[-] R_nu

v1    = [4.,]                                             #[m] D_st
v2    = [3, ]                                              #[-] Ntnks
v3    = [0.5,  ]                                      #[-] HDR
v4    = [0.2,  ]                                            #[-] RSSR
v5    = [10, ]                        #[kg/s] mass flow rate
v6    = [1000.,]                                     #[K] max average temp.
v7    = [0.10, ]                                            #[-] R_nu

Case = 00
for (a1,a2,a3,a4,a5,a6,a7) in [(a1,a2,a3,a4,a5,a6,a7) for a1 in v1 for a2 in v2 for a3 in v3 for a4 in v4 for a5 in v5 for a6 in v6 for a7 in v7]:

    tnk['D_st'], plnt['N_tanks'], tnk['HDR'], tnk['RSSR'] = a1, a2, a3, a4
    tnk['m_f'],  tnk['T_avmx'] , tnk['R_nu']              = a5, a6, a7
    
    plnt['m_CO2'], plnt['Tst_max'] = tnk['m_f'], tnk['T_avmx']
    
    tnk['D_rc']      = np.sqrt(tnk['RSSR'])*tnk['D_st']
    tnk['ER']        = max(np.sqrt(max(0.5/tnk['RSSR'] - 1.,0.)) if tnk['D_st']>2*tnk['D_rc'] else tnk['D_st']/tnk['D_rc']-1.,0.)

    T_phmx = (tnk['T_avmx'] +  tnk['T_ch'])/2.
    T_ltmx = tnk['T_avmx'] - (tnk['T_avmx'] -  tnk['T_ch'])/8.
#    T_ltmx = 990
    plnt['t_sun'] = 12.0
    
    ##########################################################
    ########## SIMULATION FIRST CHARGING PROCESS #############
    ##########################################################
    N_pr = 1

#   DISCHARGE PROCESSS
    for N_pr in [1,2,3,4,5]:
        PH,LT,HT = [{x:tnk[x] for x in tnk} for X in range(3)]
        PH['T_av'],LT['T_av'],HT['T_av'] = T_phmx, T_ltmx, tnk['T_avmx']
        PH['T_ds'],LT['T_ds'],HT['T_ds'] = T_phmx, T_ltmx, tnk['T_avmx']
    
        PH,LT,HT = Discharge_process_all(Case, N_pr, plnt, tnk, PH, LT, HT, file_ds)
        sbm.tnk_save_pickle(PH,folder+"/Case_"+str(int(Case))+"_PH_"+str(N_pr)+"_ds6")
        sbm.tnk_save_pickle(LT,folder+"/Case_"+str(int(Case))+"_LT_"+str(N_pr)+"_ds6")
        sbm.tnk_save_pickle(HT,folder+"/Case_"+str(int(Case))+"_HT_"+str(N_pr)+"_ds6")
        
        print(PH['h_c'],LT['h_c'],HT['h_c'])
        print(PH['h_av'],LT['h_av'],HT['h_av'])
        aux1, aux2 = LT['T_av'], HT['T_av']
    #   RECHARGE PROCESSS
        if plnt['t_pr_ds'] > 0.0:
            PH['T_ch'], LT['T_ch'], HT['T_ch'] = PH['T_ds'].mean(), LT['T_ds'].mean(), HT['T_ds'].mean()
            
            PH['T_avmx'] = T_phmx
            LT['T_avmx'] = T_ltmx
            
            sbm.Charging_process(Case, N_pr, plnt, tnk, [PH, LT, HT], file_ch)
            sbm.tnk_save_pickle(PH,folder+"/Case_"+str(int(Case))+"_PH_"+str(N_pr)+"_ch6")
            sbm.tnk_save_pickle(LT,folder+"/Case_"+str(int(Case))+"_LT_"+str(N_pr)+"_ch6")
            sbm.tnk_save_pickle(HT,folder+"/Case_"+str(int(Case))+"_HT_"+str(N_pr)+"_ch6")
        
        T_phmx = aux1
        T_ltmx = aux2
        
    ######################################################
    ######################################################
    
    text = str(int(Case))+'\t'+str(N_pr).zfill(2)+'\t'
    text = text+'\t'.join('{:.2f}'.format(x) for x in [tnk['D_st'], plnt['N_tanks'], tnk['HDR'], tnk['RSSR'],tnk['m_f'],  tnk['T_avmx'], T_phmx, tnk['R_nu']])+'\n'
    f = open(file_cs,'a') ; f.write(text); f.close()
    
    text = text+'\t'.join('{:.2f}'.format(x) for x in [a1,a2,a3,a4,a5,a6,a7])+'\n'
    print(text)
    
    PH.clear();     LT.clear();     HT.clear()
    Case +=1
