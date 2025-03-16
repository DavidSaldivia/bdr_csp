# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:21:20 2022

@author: z5158936
"""
import pandas as pd
import numpy as np
import xarray as xr
import scipy.optimize as spo
import sys
import time
import os
from os.path import isfile
import pickle

import cantera as ct
from pvlib.location import Location

from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
pd.set_option('display.max_columns', None)

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
DIR_MAIN = os.path.dirname(fileDir)

import bdr_csp.BeamDownReceiver as BDR
import bdr_csp.SolidParticleReceiver as SPR
from bdr_csp.dir import DIRECTORY

DIR_DATA = DIRECTORY.DIR_DATA


def Getting_BaseCase(zf,Prcv,Qavg,fzv):
    
    case = 'zf{:.0f}_Q_avg{:.2f}_Prcv{:.0f}'.format(zf, Qavg, Prcv)
    file_BaseCase = os.path.join(DIR_MAIN, '7_Overall_Optim_Dispatch','Cases',case+'.plk')
    # file_BaseCase = 'Cases/'+case+'.plk'
    
    if isfile(file_BaseCase):
    # if False:
        [CSTo,R2,SF,TOD] = pickle.load(open(file_BaseCase,'rb'))
    else:
        
        CSTi = BDR.CST_BaseCase()
        Costs = SPR.get_plant_costs()
        CSTi['Costs_i'] = Costs
    
        fldr_data = os.path.join(DIR_MAIN, '0_Data\MCRT_Datasets_Final')
        CSTi['file_SF'] = os.path.join(fldr_data,'Dataset_zf_{:.0f}'.format(zf))
        CSTi['file_weather'] = 'Preliminaries/Alice_Springs_Real2000_Created20130430.csv'
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
        CSTi['type_rcvr'] = 'TPR_2D'
        
        R2, SF, CSTo = SPR.Simulation_Coupled(CSTi)
        
        zf,Type,Array,rO,Cg,xrc,yrc,zrc,Npan = [CSTo[x] for x in ['zf', 'Type', 'Array', 'rO_TOD', 'Cg_TOD', 'xrc', 'yrc', 'zrc', 'N_pan']]
        TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},xrc,yrc,zrc)
        
        pickle.dump([CSTo,R2,SF,TOD],open(file_BaseCase,'wb'))
    return CSTo,R2,SF,TOD

def getting_basecase(
        zf: float,
        Prcv: float,
        Qavg: float,
        fzv: float,
        save_detailed_results: bool = False,
        dir_cases: str = ""
    ) -> tuple[dict,pd.DataFrame, pd.DataFrame, BDR.TertiaryOpticalDevice]:

    # constants
    xrc, yrc, zrc = (0.,0.,10.)
    Ah1 = 2.92**2
    Npan = 1
    geometry = "PB"
    array = "A"
    Cg = 2

    # checking if the file existed
    case = 'case_zf{:.0f}_Q_avg{:.2f}_Prcv{:.1f}'.format(zf, Qavg, Prcv)
    # print(case)
    file_base_case = os.path.join(dir_cases,f'{case}.plk')
    
    if isfile(file_base_case):
        [CSTo,R2,SF,TOD] = pickle.load(open(file_base_case,'rb'))
    
    else:
        
        CSTi = BDR.CST_BaseCase()
        CSTi['costs_in'] = SPR.get_plant_costs()         #Plant related costs
        CSTi['type_rcvr'] = 'HPR_0D'
        file_SF = os.path.join(
            DIR_DATA,
            'mcrt_datasets_final',
            'Dataset_zf_{:.0f}'.format(zf)
        )
        CSTi['file_weather'] = os.path.join(
            DIR_DATA,
            'weather',
            "Alice_Springs_Real2000_Created20130430.csv"
        )

        CSTi['zf'] = zf
        CSTi['Qavg'] = Qavg
        CSTi['P_rcv'] = Prcv
        Tp_avg = (CSTi['T_pC']+CSTi['T_pH'])/2
        CSTi['eta_rcv'] = SPR.HTM_0D_blackbox( Tp_avg, CSTi['Qavg'] )[0]
        Arcv = (CSTi['P_rcv']/CSTi['eta_rcv']) / CSTi['Qavg']
        CSTi['Arcv'] = Arcv

        HSF = BDR.SolarField(zf=zf, A_h1=Ah1, N_pan=Npan, file_SF=file_SF)
        HB = BDR.HyperboloidMirror(
            zf=zf, fzv=fzv, xrc=xrc, yrc=yrc, zrc=zrc, eta_hbi=CSTi["eta_rfl"]
        )
        TOD = BDR.TertiaryOpticalDevice(
            params={"geometry":geometry, "array":array, "Cg":Cg, "Arcv":Arcv},
            xrc=xrc, yrc=yrc, zrc=zrc,
        )

        CSTi['type_rcvr'] = 'HPR_2D'
        R2, SF, CSTo = SPR.run_coupled_simulation(CSTi, HSF, HB, TOD)
        print(CSTo)
        
        if save_detailed_results:
            pickle.dump([CSTo,R2,SF,TOD],open(file_base_case,'wb'))

    return (CSTo,R2,SF,TOD)


#%% Optical efficiency of Solar Field
def eta_optical_SF(file_ae): 
    df_ae = pd.read_table(file_ae,sep='\t',header=0)
    df_ae.sort_values(by=['alt','azi'],axis=0,inplace=True)
    
    lbl='eta_SF'
    alt, azi = df_ae['alt'].unique(), df_ae['azi'].unique()
    eta = np.array(df_ae[lbl]).reshape(len(alt),len(azi))
    return interp2d(azi, alt, eta, kind='linear')


def eta_optical(
        file_alt_azi: str,
        lbl: str ='eta_SF'
    ) -> RegularGridInterpolator:
    df = pd.read_csv(file_alt_azi,header=0,index_col=0)
    df.sort_values(by=['lat','alt','azi'],axis=0,inplace=True)
    lat = df['lat'].unique()
    alt = df['alt'].unique()
    azi = df['azi'].unique()
    eta = np.array(df[lbl]).reshape(len(lat),len(alt),len(azi))
    f_int = RegularGridInterpolator(
        (lat,alt,azi),
        eta,
        method='linear',
        bounds_error=False,fill_value=None
    )
    return f_int


def eta_power_block(file_PB : str) -> RegularGridInterpolator:

    df_sCO2 = pd.read_csv(file_PB,index_col=0)
    df_sCO2 = df_sCO2[df_sCO2.part_load_od==1.0]
    
    lbl_x='T_htf_hot_des'
    lbl_y='T_amb_od'
    lbl_f='eta_thermal_net_less_cooling_od'
    df_sCO2.sort_values(by=[lbl_x,lbl_y],axis=0,inplace=True)
    X, Y = df_sCO2[lbl_x].unique(), df_sCO2[lbl_y].unique()
    F = np.array(df_sCO2[lbl_f]).reshape(len(X),len(Y))
    
    # return interp2d(Y, X, F, kind='linear')
    return RegularGridInterpolator(
        (X,Y),
        F,
        method='linear',
        bounds_error=False,
        fill_value=None
    )


def eta_HPR_0D_2vars(Tstg):
    F_c  = 5.
    Tp   = Tstg + 273.
    air  = ct.Solution('air.xml')
    #Particles characteristics
    ab_p = 0.91    
    em_p = 0.75
    rho_b = 1810
    df_eta=[]
    for (T_amb,qi) in [(T_amb,qi) for T_amb in np.arange(5,56,5) for qi in np.arange(0.25,4.1,0.5)]:
        Tamb = T_amb + 273
        air.TP = (Tp+Tamb)/2., ct.one_atm
        Tsky = Tamb-15
        hcov = F_c*SPR.h_conv_NellisKlein(Tp, Tamb, 5.0, air)
        hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
        hrc  = hcov + hrad
        qloss = hrc * (Tp - Tamb)
        eta_th = (qi*1e6*ab_p - qloss)/(qi*1e6)
        df_eta.append( [Tp,qi,T_amb,hcov,hrad,qloss,eta_th])    
    df_eta = pd.DataFrame(df_eta,columns=['Tp','qi','T_amb','hcov', 'hrad', 'qloss', 'eta'])
    return interp2d(df_eta['T_amb'],df_eta['qi'],df_eta['eta'])  #Function to interpolate


def eta_HPR_0D() -> RegularGridInterpolator:
    F_c  = 5.
    air  = ct.Solution('air.xml')
    #Particles characteristics
    ab_p = 0.91    
    em_p = 0.75
    rho_b = 1810
    
    Tps   = np.arange(600,2001,100, dtype='int64')
    Tambs = np.arange(5,56,5, dtype='int64') + 273.15
    qis   = np.arange(0.25,4.1,0.5)
    eta_ths = np.zeros((len(Tps),len(Tambs),len(qis)))
    for (i,j,k) in [
        (i,j,k)
        for i in range(len(Tps))
        for j in range(len(Tambs))
        for k in range(len(qis))
    ]:
        Tp    = Tps[i]
        T_amb = Tambs[j]
        qi    = qis[k]
        
        Tamb = T_amb #+ 273
        air.TP = (Tp+Tamb)/2., ct.one_atm
        Tsky = Tamb-15
        hcov = F_c*SPR.h_conv_NellisKlein(Tp, Tamb, 5.0, air)
        hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
        hrc  = hcov + hrad
        qloss = hrc * (Tp - Tamb)
        eta_th = (qi*1e6*ab_p - qloss)/(qi*1e6)
        eta_ths[i,j,k] = eta_th
    return RegularGridInterpolator((Tps,Tambs,qis),eta_ths)  #Function to interpolate


def eta_TPR_0D(CSTo: dict) -> RegularGridInterpolator:
    CST = CSTo.copy()
    dT = CST['T_pH'] - CST['T_pC']
    Tps   = np.arange(600,2001,100, dtype='int64')
    Tambs = np.arange(5,56,5, dtype='int64') + 273.15
    Qavgs   = np.arange(0.25,4.1,0.5)
    eta_ths = np.zeros((len(Tps),len(Tambs),len(Qavgs)))
    # data = []
    for (i,j,k) in [
        (i,j,k)
        for i in range(len(Tps))
        for j in range(len(Tambs))
        for k in range(len(Qavgs))
    ]:
        Tp    = Tps[i]
        T_amb = Tambs[j]
        Qavg    = Qavgs[k]
        CST['T_pC']  = Tp - dT/2
        CST['T_pH']  = Tp + dT/2
        CST['T_amb'] = T_amb
        CST['Qavg'] = Qavg
        eta_rcv = SPR.initial_eta_rcv(CST)[0]
        eta_ths[i,j,k] = eta_rcv

    return RegularGridInterpolator(
        (Tps,Tambs,Qavgs),
        eta_ths,
        bounds_error=False,
        fill_value=None
    )


def generating_dataFrame_1year(lat,lon,state,year,dT=0.5):
    
    fldr_SP = DIR_MAIN+'\\0_Data\\SpotPrices_old\\'
    file_SP  = 'SPOT_PRICES_{:d}.csv'.format(year)
    df_SP = pd.read_csv(fldr_SP+file_SP)
    df_SP['time'] = pd.to_datetime(df_SP['time'])
    df_SP.set_index('time', inplace=True)
    
    fldr_weather = DIR_MAIN+'\\0_Data\\MERRA-2_Data\\'
    file_weather  = 'MERRA2_Processed_{:d}.nc'.format(year)
    data_weather = xr.open_dataset(fldr_weather+file_weather)
    
    lats = np.array(data_weather.lat)
    lons = np.array(data_weather.lon)
    
    #Selecting closest value for plant latitud and longitude
    lon_a = lons[(abs(lons-lon)).argmin()]
    lat_a = lats[(abs(lats-lat)).argmin()]
    
    # Checking states and selecting Demand and SP
    df_SP_st = df_SP[['Demand_'+state,'SP_'+state]]
    
    # Getting the weather file only for the given (lon,lat) and merging data
    df_w = data_weather.sel(lon=lon_a,lat=lat_a).to_dataframe()
    df_w.index = df_w.index.tz_localize('UTC')
    dT_str = '{:.1f}H'.format(dT)
    df_w = df_w.resample(dT_str).interpolate()       #Getting the data in half hours
    
    # Merging the data
    df = df_w.merge(df_SP_st,how='inner',left_index=True,right_index=True)
    
    del data_weather            #Cleaning memory
    del df_w
    
    #Calculating  solar position
    df.drop(['elevation','azimuth'],axis=1,inplace=True)
    tz = 'Australia/Brisbane'
    df.index = df.index.tz_convert(tz)
    sol_pos = Location(lat, lon, tz=tz).get_solarposition(df.index)
    df['azimuth'] = sol_pos['azimuth'].copy()
    df['elevation'] = sol_pos['elevation'].copy()
    
    #Setting the initial dataframe for dispatch model
    df = df[['DNI','T10M','SLP','WS','azimuth','elevation','Demand_'+state,'SP_'+state]]
    df.rename(columns={'T10M':'Tamb','SLP':'Pr','azimuth':'azi','elevation':'ele'},inplace=True)
    df.rename(columns={ 'Demand_'+state:'Demand', 'SP_'+state:'SP' },inplace=True)
    df = df[df.index.year==year]
    return df


def generating_df_aus(
        lat: float,
        lon: float,
        state: str,
        year_i: int,
        year_f: int,
        dT: float = 0.5) -> pd.DataFrame:
    
    DIR_WEATHER_DATA = os.path.join(DIR_MAIN,"data","weather")
    DIR_SPOTPRICE = os.path.join( DIR_MAIN,'data','NEM_spotprice')
    FILE_SPOTPRICE = os.path.join(DIR_SPOTPRICE, 'NEM_TRADINGPRICE_2010-2020.PLK')
    FILE_WEATHER  = os.path.join(DIR_WEATHER_DATA, 'MERRA2_Processed_All.nc')

    #Spot price data for state
    df_SP = pd.read_pickle(FILE_SPOTPRICE)
    df_SP_st = (
        df_SP[
        (df_SP.index.year>=year_i) & (df_SP.index.year<=year_f)
        ][['SP_'+state]]
    ) 
    
    data_weather = xr.open_dataset(FILE_WEATHER)
    lats = np.array(data_weather.lat)
    lons = np.array(data_weather.lon)
    lon_a = lons[(abs(lons-lon)).argmin()]
    lat_a = lats[(abs(lats-lat)).argmin()]
    df_weather = data_weather.sel(lon=lon_a,lat=lat_a).to_dataframe()
    
    del data_weather #Cleaning memory
    
    df_weather['DNI_dirint'] = df_weather['DNI_dirint'].fillna(0)
    df_weather.index = df_weather.index.tz_localize('UTC')
    df_weather = df_weather.resample(f'{dT:.1f}h').interpolate()       #Getting the data in half hours
    df_weather['DNI'] = df_weather['DNI_dirint']
    
    # Merging the data
    df_data = df_weather.merge(
        df_SP_st,
        how='inner',
        left_index=True,
        right_index=True
    )
    del df_weather   #Cleaning memory
    
    #Calculating  solar position
    tz = 'Australia/Brisbane'
    df_data.index = df_data.index.tz_convert(tz)
    sol_pos = Location(lat, lon, tz=tz).get_solarposition(df_data.index)
    df_data['azimuth'] = sol_pos['azimuth'].copy()
    df_data['elevation'] = sol_pos['elevation'].copy()
    
    #Setting the initial dataframe for dispatch model
    df_data = df_data[['DNI','T2M','WS','azimuth','elevation','SP_'+state]]
    df_data.rename(
        columns={'T2M':'Tamb','azimuth':'azi','elevation':'ele'},
        inplace=True
    )
    df_data.rename(
        columns={ 'Demand_'+state:'Demand', 'SP_'+state:'SP' },
        inplace=True
    )
    return df_data


def PHX_LMTD(
        P_sCO2: float,
        T_sCO2_in: float,
        T_sCO2_out: float,
        T_p_in: float,
        T_p_out: float,
        U_HX: float) -> dict[str,float]:
    
    #### Tube and shell hx using log-mean temp difference    
    dT_sCO2  = T_sCO2_out - T_sCO2_in
    CO2 = ct.Solution('gri30.yaml','gri30')
    T_sCO2_avg = (T_sCO2_out+T_sCO2_in)/2
    CO2.TPY  = T_sCO2_avg, 25e6, 'CO2:1.00'
    cp_sCO2  = CO2.cp
    
    #This is a guess, I have to figure out how the number of pass influence
    Npass = 4
    m_sCO2 = P_sCO2 / (cp_sCO2 * dT_sCO2) / Npass
    C_sCO2 = m_sCO2 * cp_sCO2
    Q_HX = C_sCO2 * dT_sCO2
    
    dT_p    = T_p_in - T_p_out
    T_p_avg = (T_p_in + T_p_out)/2
    cp_p = 365*T_p_avg**0.18
    C_p = Q_HX / dT_p
    m_p = C_p / cp_p
    
    dT_lm = (
        ((T_p_out-T_sCO2_in) - (T_p_in-T_sCO2_out))
        /np.log((T_p_out-T_sCO2_in)/(T_p_in-T_sCO2_out))
    )
    C_min = min(C_p, C_sCO2)
    C_max = max(C_p, C_sCO2)
    Q_max = C_max * (T_p_in - T_sCO2_in)
    epsilon = Q_HX  / Q_max
    
    A_HX = Q_HX / ( U_HX * dT_lm)
    
    Ntubes = 400  #[tubes/m2]
    Dtubes = 0.03 #[m]
    dAdV = (np.pi*Dtubes*1)*Ntubes
    
    V_HX   = A_HX/dAdV
    
    results = {
        "heat_hx": Q_HX,
        "epsilon": epsilon,
        "LMTD": dT_lm,
        "area_hx": A_HX,
        "volume_hx": V_HX,
        "mass_flowrate": m_p
    }
    return results


def PHX_cost(
        P_pb_th: float,
        T_sCO2_in: float,
        T_sCO2_out: float,
        T_pH: float,
        T_pC: float,
        U_HX: float) -> float:
    # Result is obtained in MM USD / MW_th
    results = PHX_LMTD(
        P_pb_th*1e6, T_sCO2_in+273., T_sCO2_out+273., T_pH, T_pC, U_HX
    )
    A_HX = results["area_hx"]
    T_ps   = np.linspace(T_pH,T_pC,10)
    f_HXTs = [
        1 if (T_pi < 550+273.) else (1+ 5.4e-5 * (T_pi-273.15-550)**2)
        for T_pi in T_ps
    ]
    C_PHX  = (np.mean(f_HXTs) * 1000 * A_HX / P_pb_th)
    return C_PHX


def dispatch_model_simple(
        df: pd.DataFrame,
        Q_stg: float
    ) -> pd.DataFrame:

    #For simplicity the day starts with sunrise and lasts until next sunrise
    df['NewDay']  = np.where(
        (df["ele"]*df["ele"].shift(1)<0) & (df["ele"]>0),
        True,
        False
    )
    df.loc[df.index[0],'NewDay'] = True    #The first day is manually set as 1
    df['DaySet'] = df['NewDay'].cumsum()   # creating day index
    df["DaySet"] = df["DaySet"].ffill()
    
    df['E_rcv'] = df["E_rcv"].fillna(0)
    
    #Sorting dispatch periods by their potential revenue
    #First the periods with the solar field operating are sorted
    df['Rank_Disp_Op'] = (
        df[ df["E_rcv"]>df["E_th_pb"] ]
        .groupby( df["DaySet"] )["R_gen"]
        .rank('first', ascending=False)
    )
    #Accumulated thermal energy extracted from Storage System
    # for dispatch hours during solar field operation time, 
    # sorted by Dispatch Ranking
    df['E_th_pb_Cum_Op'] = (
        df
        .sort_values(['DaySet', 'Rank_Disp_Op'])
        .groupby([ df["DaySet"] ])["E_th_pb"]
        .cumsum()
    )
    df['E_th_pb_Cum_Op'] = (
        np.where(
            df["Rank_Disp_Op"].isnull(),
            np.nan,
            df["E_th_pb_Cum_Op"])
        )
    
    #Total Energy delivered by Solar Field per day
    df['E_collected'] = (
        df.groupby(df["DaySet"])['E_rcv']
        .transform('sum')
    )
    
    #Energy delivered by Solar Field that excede the storage capacity
    df['E_excess']    = df['E_collected'] - Q_stg
    
    #Selecting the hours that should dispatch during operating time
    df['Dispatch_Op'] = (
        np.where(
            (df["E_th_pb_Cum_Op"] < df["E_excess"]) & (df["SP"]>0) ,
            1,
            0
        )
    )
    df['E_stg_cum_Op'] =(
        (df["SF_Op"] * df["E_rcv"] - df["E_th_pb"] * df["Dispatch_Op"] )
        .groupby(df["DaySet"])
        .cumsum()
    )
    
    #If E_stg_cum > Q_stg, then we need to redirect part of solar field
    df['SF_Op'] = (
        np.where(
            (df["E_stg_cum_Op"] > Q_stg ) & (df["SF_Op"]>0),
            1 - (df["E_stg_cum_Op"] - Q_stg )/df["E_rcv"],
            df["SF_Op"]
        )
    )
    df['SF_Op'] = np.where((df["SF_Op"]<0), 0 , df["SF_Op"])
    df['E_stg_cum_Op'] = (
        (df["SF_Op"]*df["E_rcv"] - df["E_th_pb"]*df["Dispatch_Op"])
        .groupby(df["DaySet"])
        .cumsum()
    )
    df['E_used_Op'] = (
        (df["E_th_pb"]*df["Dispatch_Op"])
        .groupby(df["DaySet"])
        .transform('sum')
    )
    
    #Energy available to generate electricity after the first group dispatched
    df['E_avail_NoOp'] = (
        df['E_stg_cum_Op']
        .groupby(df["DaySet"])
        .transform('last')
    )
    
    # Ranking for all remaining hours that haven't dispatched yet
    df['Rank_Disp_NoOp'] = (
        df[(df["Dispatch_Op"]<=0) & (df['E_stg_cum_Op'] > df['E_th_pb'])]
        .groupby(df["DaySet"])
        ["R_gen"]
        .rank('first', ascending=False)
    )
    
    #Selecting the hours that should dispatch at anytime,
    # excluding those who already dispatched
    df['E_th_pb_Cum_NoOp'] = (
        df[(df["Dispatch_Op"]<=0) & (df['E_stg_cum_Op']>df['E_th_pb'])]
        .sort_values(['DaySet', 'Rank_Disp_NoOp'])
        .groupby(df["DaySet"])
        ["E_th_pb"]
        .cumsum()
    )
    df['Dispatch_NoOp'] = (
        np.where(
            (df["E_th_pb_Cum_NoOp"]<=df["E_avail_NoOp"])
            & (df["Dispatch_Op"]<=0) & (df["SP"]>0),
            1,
            0
        )
    )
    df['Dispatch_Both'] = (
        np.where(
            (df["Dispatch_Op"]>0) | (df["Dispatch_NoOp"]>0),
            1,
            0
        )
    )    
    df['E_stg_cum_both'] = (
        (df["SF_Op"] * df["E_rcv"] - df["E_th_pb"]*df["Dispatch_Both"])
        .groupby(df["DaySet"])
        .cumsum()
    )
    
    #Net energy on each time after Both
    df['E_stg_both']  = (
        df["SF_Op"] * df["E_rcv"] - df["E_th_pb"]*df["Dispatch_Both"]
    )
    df['E_remaining'] = (
        df['E_stg_cum_both']
        .groupby(df["DaySet"])
        .transform('last')
    )
    df['disp_remaining'] = (df["E_remaining"]/df["E_th_pb"])
    
    df['Rank_Disp_Extra'] = (
        df[(df["Dispatch_Both"]<=0) & (df["SP"]>0)]
        .groupby(df["DaySet"])
        ["R_gen"]
        .rank('first', ascending=False)
    )
    
    #Selecting the hours that should dispatch at anytime, excluding those who already dispatched
    # df['E_th_pb_Cum_Extra'] = df[df['Rank_Disp_Extra']>0].sort_values(['DaySet', 'Rank_Disp_Extra'] ).groupby(df.DaySet).E_th_pb.cumsum()
    
    #All the full periods are included
    df['E_avail_Extra'] = (
        np.where(
            (
                (df["Rank_Disp_Extra"]>0)
                & (df["disp_remaining"]<1) 
                & (df["disp_remaining"]>0.01)
            ),
            df["E_stg_both"]/df["E_th_pb"],
            0.
        )
    )
    
    df['E_avail_Extra'] = (
        np.where(
            (df["Rank_Disp_Extra"]>0),
            df["E_stg_both"]/df["E_th_pb"],
            0.
        )
    )
    df['Dispatch_Extra'] = np.where((df.Rank_Disp_Extra>0)&(df.Rank_Disp_Extra-1<df.disp_remaining), df[['disp_remaining', 'E_avail_Extra']].min( axis=1 ), 0.)
    
    df['Dispatch_Full'] = (df["Dispatch_Both"] + df["Dispatch_Extra"])
    
    df['E_stg_net'] = df["SF_Op"] * df["E_rcv"] - df["E_th_pb"] * df["Dispatch_Full"]
    df['E_stg_cum_tot'] = df["E_stg_net"].cumsum()
    df['E_stg_cum_day'] = df.groupby(df["DaySet"])['E_stg_net'].cumsum()
    
    df['E_gen_final'] = df["Dispatch_Full"] * df["E_gen"]      # [MWh]
    df['Rev_final'] = df["E_gen_final"] * df["SP"]           # [AUD]
    return df


def annual_performance(df,CSTo,SF,args):
    f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min = args
    
    T_pb_max = CSTo['T_pb_max']
    T_pavg = (CSTo['T_pH']+CSTo['T_pC'])/2.
    A_SF   = CSTo['S_SF']
    A_rcv  = CSTo['Arcv']
    Prcv   = CSTo['P_rcv']
    P_el   = CSTo['P_el']           #It already considers the number of towers
    Ntower = CSTo['Ntower']
    Q_stg  = CSTo['Q_stg']
    P_pb   = CSTo['P_pb']
    lat    = CSTo['lat']
    
    # Obtaining the efficiencies in hourly basis
    Npoints = len(df)
    df['date'] = df.index.date
    Ndays = len(df["date"].unique())
    years = Ndays/365
    
    lats = lat*np.ones(Npoints)
    azis = df["azi"].to_numpy()
    azis = np.where(azis>180,360-azis,azis)
    eles = df["ele"].to_numpy()
    df['eta_SF'] = f_eta_opt(np.array([lats,eles,azis]).transpose())
    T_pavgs = T_pavg * np.ones(Npoints)
    TambsC = df["Tamb"].to_numpy() - 273.15         #Ambient temperature in Celcius
    TambsK = df["Tamb"].to_numpy()         #Ambient temperature in Kelvin
    T_maxC = (T_pb_max-273.15) * np.ones(Npoints)
    Qavgs = df["DNI"] * df["eta_SF"] * (A_SF / A_rcv) * 1e-6
    
    df['eta_rcv'] = f_eta_TPR(np.array([T_pavgs,TambsK,Qavgs]).transpose())
    df['eta_pb'] = f_eta_pb(np.array([T_maxC,TambsC]).transpose())
    
    #Calculations for Collected Energy and potential Dispatched energy
    # Is the Solar Field operating?
    df['SF_Op']  = np.where( df["DNI"]>DNI_min, 1, 0)
    df['E_SF']   = df["SF_Op"] * df["eta_SF"] * df["DNI"] * A_SF * 1e-6 * dT   #[MWh]
    df['E_rcv']  = df["E_SF"] * df["eta_rcv"]                                  #[MWh]
    
    #It is assumed the same amount of energy is extracted from storage, so particle-sCO2 HX always try to work at design capacity. The energy dispatched is lower then.
    df['E_th_pb'] = P_pb * dT                              #[MWh_th]
    df['E_gen'] = Ntower * df['eta_pb'] * df['E_th_pb']    #[MWh_e] Including the Number of towers
    df['R_gen'] = df['E_gen'] * df['SP']                   #[$]
    
    df = dispatch_model_simple(df, Q_stg)

    E_stored = df["E_rcv"].sum() / years                    #[MWh/yr] Annual energy stored per tower
    E_dispatch = df["E_gen_final"].sum()  / years           #[MWh/yr] Annual energy dispatched per power block
    Rev_tot = df["Rev_final"].sum()/1e6  / years            #[MM USD/yr] Total annual revenue per power block
    Rev_day = df["Rev_final"].sum() / 1000 / Ndays          #kAUD/day Daily revenue
    CF_sf = df["E_rcv"].sum() / (Prcv * dT * len(df))       # Solar Field Capacity Factor
    CF_pb = df["E_gen_final"].sum() / (P_el * dT * len(df)) # Solar Field Capacity Factor
    TimeOp_sf = df["SF_Op"].sum()*dT / years                #hrs
    TimeOp_pb = df["Dispatch_Full"].sum()*dT / years        #hrs
    
    Stg_min = df["E_stg_cum_day"].min()
    
    #Updating costs and converting USD to AUD
    # Obtaining the costs for the power block an PHX
    USDtoAUD = 1.4
    Costs_i = CSTo['costs_in']
    Costs_i['CF_sf'] = CF_sf
    Costs_i['CF_pb'] = CF_pb
    
    Fr_pb = Costs_i['Fr_pb']
    file_pb = os.path.join(DIR_DATA,'sCO2_cycle','sCO2_lookuptable_costs.csv')
    df_pb = pd.read_csv(file_pb,index_col=0)
    lbl_x='P_el'
    lbl_y='T_htf_hot_des'
    if Costs_i['C_pb_est']=='Neises':
        lbl_f='cycle_spec_cost'
    elif Costs_i['C_pb_est']=='Sandia':
        lbl_f='cycle_spec_cost_Sandia'
    
    df_pb.sort_values(by=[lbl_x,lbl_y],axis=0,inplace=True)
    X = df_pb[lbl_x]
    Y = df_pb[lbl_y]
    F = df_pb[lbl_f]
    f_pb = LinearNDInterpolator(list(zip(X,Y)), F)
    C_pb = (1-Fr_pb) * f_pb([P_el,T_pb_max-273.15])[0]
    
    #Obtaining the CO2 temperatures
    lbl_f = 'T_co2_PHX_in'
    F = df_pb[lbl_f]
    f_sCO2_in = LinearNDInterpolator(list(zip(X,Y)), F)
    T_sCO2_in  = f_sCO2_in([P_el,T_pb_max-273.15])[0]
    
    lbl_f = 'T_turb_in'
    F = df_pb[lbl_f]
    f_sCO2_out = LinearNDInterpolator(list(zip(X,Y)), F)
    T_sCO2_out  = f_sCO2_out([P_el,T_pb_max-273.15])[0]
    
    U_HX = 240.                             #Value from DLR design
    C_PHX = PHX_cost(P_pb, T_sCO2_in, T_sCO2_out, CSTo['T_pH'], CSTo['T_pC'], U_HX)
    
    Costs_i['C_pb']  = C_pb * 1e3            #[USD/MW]
    Costs_i['C_PHX'] = C_PHX                 #[USD/MW]
    
    CSTo['costs_in'] = Costs_i
    CSTo['type_weather'] = 'MERRA2'         #This is to use CF_sf calculated here
    Costs_o = SPR.BDR_cost(SF, CSTo)
    CSTo['Costs'] = Costs_o
    LCOH = Costs_o['LCOH'] * USDtoAUD          #AUD/MWh
    LCOE = Costs_o['LCOE'] * USDtoAUD          #AUD/MWh
    C_C  = Costs_o['Elec']  * USDtoAUD         #Total Capital Cost [AUD]
    
    DR = Costs_i['DR']
    Ny = Costs_i['Ny']
    C_OM = Costs_i['C_OM']
    
    TPF = (1./DR)*(1. - 1./(1.+DR)**Ny)
    RoI = TPF*Rev_tot/C_C         #Return over investment
    NPV = Rev_tot*TPF - C_C*(1. + C_OM*TPF)
    RoI = (Rev_tot*TPF - C_C*(1. + C_OM*TPF)) / C_C
    def fNPV(i,*args):
        DR,Rev_tot,C_C,C_OM = args
        TPF = (1./DR)*(1. - 1./(1.+DR)**i)
        NPV = Rev_tot*TPF - C_C*(1. + C_OM*TPF)
        return NPV
    sol = spo.fsolve(fNPV,Ny,args=(DR,Rev_tot,C_C,C_OM),full_output=True)
    if sol[2]==1:
        PbP = sol[0][0]
    else:
        # PbP = np.nan
        PbP = sol[0][0]
        print(sol)
        
    results = {
        'LCOH':LCOH,
        'LCOE':LCOE,
        'C_C':C_C,
        'C_pb':C_pb,
        'E_stored':E_stored,
        'E_dispatch':E_dispatch,
        'Rev_tot':Rev_tot,
        'Rev_day':Rev_day,
        'CF_sf':CF_sf,
        'CF_pb':CF_pb,
        'RoI':RoI,
        'PbP':PbP,
        'NPV':NPV,
        'TimeOp_sf':TimeOp_sf,
        'TimeOp_pb':TimeOp_pb,
        'Stg_min':Stg_min
    }
    return results