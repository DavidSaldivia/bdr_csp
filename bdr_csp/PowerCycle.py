from __future__ import annotations
from dataclasses import dataclass, field
from os.path import isfile
from typing import Callable
import os
import pickle
import time

import pandas as pd
import numpy as np
import xarray as xr
import scipy.optimize as spo
import cantera as ct
from pvlib.location import Location
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

from antupy.units import Variable
import bdr_csp.bdr as bdr
import bdr_csp.spr as spr
import bdr_csp.htc as htc
from bdr_csp.dir import DIRECTORY

pd.set_option('display.max_columns', None)

DIR_MAIN = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DIR_DATA = DIRECTORY.DIR_DATA

COLS_INPUT = [
    'location',
    'n_tower',
    'temp_stg',
    'solar_multiple',
    'receiver_power'
]
COLS_OUTPUT = [
    'stg_heat_stored',
    'pb_energy_dispatched',
    'sf_cf', 
    'pb_cf',
    'revenue_tot',
    'revenue_daily',
    'lcoh',
    'lcoe', 
    'pb_capital',
    'cost_capital',
    'roi',
    'payback_period',
    'npv',
    'stg_min_stored',
]

@dataclass
class PlantCSPBeamDownParticle():
    zf: Variable = Variable(50, "m")
    fzv: Variable  = Variable(0.818161, "-")
    receiver_power: Variable = Variable(19.,"MW")
    flux_avg: Variable = Variable(1.25,"MW/m2")
    xrc: Variable = Variable(0., "m")
    yrc: Variable = Variable(0., "m")
    zrc: Variable = Variable(10., "m")
    Ah1: Variable = Variable(2.92**2, "m2")
    Npan: int = 1
    geometry: str = "PB"
    array: str = "A"
    Cg: Variable = Variable(2, "-")
    receiver_area: Variable = Variable(20., "m2")

    cost_input: dict[str, float|str] = field(default_factory=spr.get_plant_costs)

    Gbn: Variable = Variable(950, "W/m2")                # Design-point DNI [W/m2]
    day: int = 80                                        # Design-point day [-]
    omega: Variable = Variable(0.0, "rad")               # Design-point hour angle [rad]
    lat: Variable = Variable(-23., "deg")                # Latitude [°]
    lng: Variable = Variable(115.9, "deg")               # Longitude [°]
    state: str = "SA"
    temp_amb: Variable = Variable(300., "K")             # Ambient Temperature [K]
    type_weather: str | None = 'TMY'                     # Weather source (for CF) Accepted: TMY, MERRA2, None
    file_weather: str | None = None                      # Weather source (for CF)
    DNI_min: Variable = Variable(400., "W/m2")           # minimum allowed DNI to operate

    # Characteristics of Solar Field
    eta_rfl: Variable = Variable(0.95, "-")              # Includes mirror refl, soiling and refl. surf. ratio. Used also for HB and TOD
    err_tot: Variable = Variable(0.002, "-")             # [rad] Total reflected error
    type_shadow = 'simple'                               # Type of shadow modelling

    ### Receiver and Storage Tank characteristics
    receiver_type: str = 'HPR_0D'                            # model for Receiver
    receiver_flux_max: Variable = Variable(3.0, "MW/m2")     # Maximum radiation flux on receiver
    stg_temp_cold: Variable = Variable(950,"K")              # particle temperature in cold tank
    stg_temp_hot: Variable = Variable(1200,"K")              # particle temperature in hot tank
    stg_material: str  = 'CARBO'                        # Thermal Storage Material
    stg_thickness: Variable = Variable(0.05, "m")  # Thickness of material on conveyor belt
    stg_cap: Variable = Variable(8., "hr")          # Hours of storage
    stg_h_to_d: Variable = Variable(0.5, "-")                 # [-] height to diameter ratio for storage tank
    solar_multiple: Variable = Variable(2.0, "-")             # [-] Initial target for solar multiple
    
    # Receiver and Power Block efficiencies
    pb_temp_max: Variable = Variable(875 + 273.15,"K")         #[K] Maximum temperature un power block cycle
    Ntower: int = 1                                            #[-] Number of towers feeding one power block 
    pb_eta_des: Variable = Variable(0.50,"-")                  #[-] Power Block efficiency target (initial value) 
    stg_eta_des: Variable = Variable(1.00,"-")                  #[-] Storage efficiency target (assumed 1 for now)
    receiver_eta_des: Variable = Variable(0.75, "-")                #[-] Receiver efficiency target (initial value)


    # Outputs
    sf_n_hels = Variable(None, "-")                     # Number of heliostats
    sf_area = Variable(None, "m2")                     # Solar Field area
    sf_power = Variable(None, "MW")                     # Solar Field power (output)

    pb_power_th: Variable = Variable(None, "MW")                       # Power Block thermal power (input)
    pb_power_el: Variable = Variable(None, "MW")                       # Power Block electrical power (output)
    storage_heat: Variable = Variable(None, "MWh")                     # Storage heat (output)

    receiver_massflowrate: Variable = Variable(None, "kg/s")  # Receiver mass flow rate (to be calculated)
    costs_out : dict[str, float] = field(default_factory=dict)  # Costs output from simulation

    def run_thermal_subsystem(
            self,
            save_detailed_results: bool = False,
    ) -> tuple[dict,pd.DataFrame, pd.DataFrame]:
        
        zf = self.zf
        Ah1 = self.Ah1
        Npan = self.Npan
        fzv = self.fzv
        xrc = self.xrc
        yrc = self.yrc
        zrc = self.zrc
        eta_hbi = self.eta_rfl
        geometry = self.geometry
        array = self.array
        Cg = self.Cg

        flux_avg = self.flux_avg.get_value("MW/m2")
        receiver_power = self.receiver_power.get_value("MW")
        temp_avg = (self.stg_temp_cold.get_value("K") + self.stg_temp_hot.get_value("K")) / 2
        
        eta_receiver = Variable(
            spr.HTM_0D_blackbox( temp_avg, flux_avg )[0], "-"
        )
        receiver_area = Variable((receiver_power / eta_receiver.v) / flux_avg, "m2")

        self.eta_rcv_des = eta_receiver
        self.receiver_area = receiver_area

        file_SF = os.path.join(
            DIR_DATA,
            'mcrt_datasets_final',
            'Dataset_zf_{:.0f}'.format(zf.v)
        )
        self.file_weather = os.path.join(
            DIR_DATA,
            'weather',
            "Alice_Springs_Real2000_Created20130430.csv"
        )
        HSF = bdr.SolarField(zf=zf, A_h1=Ah1, N_pan=Npan, file_SF=file_SF)
        HB = bdr.HyperboloidMirror(
            zf=zf, fzv=fzv, xrc=xrc, yrc=yrc, zrc=zrc, eta_hbi=eta_hbi
        )
        TOD = bdr.TertiaryOpticalDevice(
            geometry = geometry, array = array, Cg=Cg, receiver_area= receiver_area,
            xrc=xrc, yrc=yrc, zrc=zrc,
        )

        self.type_receiver = 'HPR_0D'
        R2, SF, costs_out = spr.run_coupled_sim(self, HSF, HB, TOD)
        
        self.costs_out = costs_out

        if save_detailed_results:
            pickle.dump([costs_out,R2,SF,TOD],open(file_base_case,'wb'))

        return costs_out, R2, SF

    def eta_optical_hourly(
            self,
            df: pd.DataFrame,
            file_alt_azi: str = os.path.join(DIR_DATA,'annual_perf','1-Grid_AltAzi_vF.csv'),
            lbl: str ='eta_SF'
        ) -> np.ndarray:

        latitude = self.lat.get_value("deg")
        # getting the efficiency grid from the file
        df_grid = pd.read_csv(file_alt_azi,header=0,index_col=0)
        df_grid.sort_values(by=['lat','alt','azi'],axis=0,inplace=True)
        lat = df_grid['lat'].unique()
        alt = df_grid['alt'].unique()
        azi = df_grid['azi'].unique()
        eta = np.array(df_grid[lbl]).reshape(len(lat),len(alt),len(azi))
        f_int = RegularGridInterpolator(
            (lat,alt,azi),
            eta,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # applying the interpolator to the dataframe
        Npoints = len(df)
        lats = latitude*np.ones(Npoints)
        azis = df["azimuth"].to_numpy()
        azis = np.where(azis>180,360-azis,azis)
        eles = df["elevation"].to_numpy()
        eta_SF = f_int(np.array([lats,eles,azis]).transpose())
        return eta_SF
    
    def eta_receiver_hourly(
            self,
            df: pd.DataFrame,
            func_receiver: Callable | None = None,
    ) -> pd.DataFrame:
        
        if func_receiver is None:
            raise ValueError("Receiver efficiency function not provided.")
        
        A_SF = self.sf_area.get_value("m2")
        temp_parts_avg = (self.stg_temp_cold.get_value("K") + self.stg_temp_hot.get_value("K")) / 2
        area_rcv = self.receiver_area.get_value("m2")
        Npoints = len(df)

        T_pavgs = temp_parts_avg * np.ones(Npoints)
        TambsK = df["temp_amb"].to_numpy()         #Ambient temperature in Kelvin
        Qavgs = df["DNI"] * df["sf_eta"] * (A_SF / area_rcv) * 1e-6
        eta_rcv = func_receiver(np.array([T_pavgs,TambsK,Qavgs]).transpose())
        return eta_rcv


    def eta_pb_hourly(
            self,
            df: pd.DataFrame,
            func_pb: Callable | None = None,
    ) -> pd.DataFrame:

        if func_pb is None:
            raise ValueError("Power block efficiency function not provided.")
        temp_pb_max = self.pb_temp_max.get_value("K")
        Npoints = len(df)
        TambsC = df["temp_amb"].to_numpy() - 273.15         #Ambient temperature in Celcius
        T_maxC = (temp_pb_max-273.15) * np.ones(Npoints)
        eta_pb = func_pb(np.array([T_maxC,TambsC]).transpose())
        return eta_pb


def getting_basecase(
        zf: Variable,
        Prcv: Variable,
        Qavg: Variable,
        fzv: Variable,
        save_detailed_results: bool = False,
        dir_cases: str = ""
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame, bdr.TertiaryOpticalDevice]:

    # constants
    xrc = Variable(0., "m")
    yrc = Variable(0., "m")
    zrc = Variable(10., "m")
    Ah1 = Variable(2.92**2, "m2")
    Cg = Variable(2, "-")
    Npan = 1
    geometry = "PB"
    array = "A"

    # checking if the file existed
    zf_value = zf.get_value("m")
    Prcv_value = Prcv.get_value("MW")
    Qavg_value = Qavg.get_value("MW/m2")
    case = 'case_zf{:.0f}_Q_avg{:.2f}_Prcv{:.1f}'.format(zf_value, Qavg_value, Prcv_value)
    # print(case)
    file_base_case = os.path.join(dir_cases,f'{case}.plk')
    
    if isfile(file_base_case):
        [CSTo,R2,SF,TOD] = pickle.load(open(file_base_case,'rb'))
    else:
        CSTi = bdr.CST_BaseCase()
        CSTi['costs_in'] = spr.get_plant_costs()         #Plant related costs
        CSTi['type_rcvr'] = 'HPR_0D'
        file_SF = os.path.join(
            DIR_DATA,
            'mcrt_datasets_final',
            'Dataset_zf_{:.0f}'.format(zf_value)
        )
        CSTi['file_weather'] = os.path.join(
            DIR_DATA,
            'weather',
            "Alice_Springs_Real2000_Created20130430.csv"
        )
        CSTi['zf'] = zf.get_value("m")
        CSTi['Qavg'] = Qavg.get_value("MW/m2")
        CSTi['P_rcv'] = Prcv.get_value("MW")
        Tp_avg = (CSTi['T_pC']+CSTi['T_pH'])/2
        CSTi['eta_rcv'] = spr.HTM_0D_blackbox( Tp_avg, CSTi['Qavg'] )[0]
        CSTi['Arcv'] = (CSTi['P_rcv']/CSTi['eta_rcv']) / CSTi['Qavg']
        Arcv = Variable(CSTi["Arcv"], "m2")

        HSF = bdr.SolarField(zf=zf, A_h1=Ah1, N_pan=Npan, file_SF=file_SF)
        HB = bdr.HyperboloidMirror(
            zf=zf, fzv=fzv, xrc=xrc, yrc=yrc, zrc=zrc, eta_hbi=CSTi["eta_rfl"]
        )
        TOD = bdr.TertiaryOpticalDevice(
            geometry = geometry, array = array, Cg=Cg, receiver_area= Arcv,
            xrc=xrc, yrc=yrc, zrc=zrc,
        )
        CSTi['type_rcvr'] = 'HPR_2D'
        R2, SF, CSTo = spr.run_coupled_simulation(CSTi, HSF, HB, TOD)
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
    return RegularGridInterpolator(
        (lat,alt,azi),
        eta,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )


def eta_power_block(
        file_PB : str = os.path.join(DIR_DATA, "sCO2_cycle", "sCO2_OD_lookuptable.csv")
        ) -> RegularGridInterpolator:

    df_sCO2 = pd.read_csv(file_PB,index_col=0)
    df_sCO2 = df_sCO2[df_sCO2["part_load_od"]==1.0]
    
    lbl_x='T_htf_hot_des'
    lbl_y='T_amb_od'
    lbl_f='eta_thermal_net_less_cooling_od'
    df_sCO2.sort_values(by=[lbl_x,lbl_y],axis=0,inplace=True)
    X, Y = df_sCO2[lbl_x].unique(), df_sCO2[lbl_y].unique()
    F = np.array(df_sCO2[lbl_f]).reshape(len(X),len(Y))
    
    return RegularGridInterpolator(
        (X,Y),
        F,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )


def eta_HPR_0D_2vars(Tstg):
    F_c  = 5.
    Tp   = Tstg + 273.
    air  = ct.Solution('air.yaml')
    #Particles characteristics
    ab_p = 0.91    
    em_p = 0.75
    rho_b = 1810
    df_eta=[]
    for (T_amb,qi) in [(T_amb,qi) for T_amb in np.arange(5,56,5) for qi in np.arange(0.25,4.1,0.5)]:
        Tamb = float(T_amb + 273.15)
        air.TP = (Tp+Tamb)/2., ct.one_atm
        Tsky = Tamb-15
        hcov = F_c*htc.h_conv_NellisKlein(Tp, Tamb, 5.0, air)
        hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
        hrc  = hcov + hrad
        qloss = hrc * (Tp - Tamb)
        eta_th = (qi*1e6*ab_p - qloss)/(qi*1e6)
        df_eta.append( [Tp,qi,T_amb,hcov,hrad,qloss,eta_th])    
    df_eta = pd.DataFrame(df_eta,columns=['Tp','qi','T_amb','hcov', 'hrad', 'qloss', 'eta'])
    return interp2d(df_eta['T_amb'],df_eta['qi'],df_eta['eta'])  #Function to interpolate


def eta_HPR_0D() -> RegularGridInterpolator:
    F_c  = 5.
    air  = ct.Solution('air.yaml')
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
        hcov = F_c * htc.h_conv_NellisKlein(Tp, Tamb, 5.0, air)
        hrad = em_p*5.67e-8*(Tp**4-Tsky**4)/(Tp-Tamb)
        hrc  = hcov + hrad
        qloss = hrc * (Tp - Tamb)
        eta_th = (qi*1e6*ab_p - qloss)/(qi*1e6)
        eta_ths[i,j,k] = eta_th
    return RegularGridInterpolator(
        (Tps,Tambs,qis),
        eta_ths,
        bounds_error=False,
        fill_value=np.nan
    )


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
        eta_rcv = spr.initial_eta_rcv(CST)[0]
        eta_ths[i,j,k] = eta_rcv

    return RegularGridInterpolator(
        (Tps,Tambs,Qavgs),
        eta_ths,
        bounds_error=False,
        fill_value=np.nan
    )

def load_weather_data(
        lat: float,
        lng: float,
        year_i: int,
        year_f: int,
        dT: float = 0.5,
        file_weather: str | None = None
) -> pd.DataFrame:
    
    tz = 'Australia/Brisbane'
    if file_weather is None:
        # dir_data = os.path.join(DIR_MAIN,"data","weather")
        file_weather  = os.path.join(
            DIR_DATA,"weather", 'MERRA2_Processed_All.nc'
        )
    lat_v = lat
    lon_v = lng
    data_weather = xr.open_dataset(file_weather, engine="netcdf4")
    lats = np.array(data_weather.lat)
    lons = np.array(data_weather.lon)
    lon_a = lons[(abs(lons-lon_v)).argmin()]
    lat_a = lats[(abs(lats-lat_v)).argmin()]
    df_weather = data_weather.sel(lon=lon_a,lat=lat_a).to_dataframe()

    del data_weather #Cleaning memory
    
    df_weather['DNI_dirint'] = df_weather['DNI_dirint'].fillna(0)
    df_weather.index = pd.to_datetime(df_weather.index).tz_localize('UTC')
    df_weather = df_weather.resample(f'{dT:.1f}h').interpolate()       #Getting the data in half hours
    df_weather['DNI'] = df_weather['DNI_dirint']

    #Calculating  solar position
    tz = 'Australia/Brisbane'
    df_weather.index = pd.to_datetime(df_weather.index).tz_convert(tz)
    sol_pos = Location(lat_v, lon_v, tz=tz).get_solarposition(df_weather.index)
    df_weather['azimuth'] = sol_pos['azimuth'].copy()
    df_weather['elevation'] = sol_pos['elevation'].copy()    

    year = pd.to_datetime(df_weather.index).year
    df_weather = df_weather[(year >= year_i) & (year <= year_f)]

    #Finishing the weather data
    df_weather = df_weather[['DNI','T2M','WS','azimuth','elevation']].copy()
    df_weather.rename(
        columns={'T2M':'temp_amb'}, inplace=True
    )
    return df_weather


def load_spotprice_data(
        state: str = "NSW",
        year_i: int = 2019,
        year_f: int = 2019,
        dT: float = 0.5,
        file_data: str | None = None
) -> pd.DataFrame:
    
    if file_data is None:
        dir_spotprice = os.path.join( DIR_DATA,'NEM_spotprice')
        file_data = os.path.join(dir_spotprice, 'NEM_TRADINGPRICE_2010-2020.PLK')
    df_SP = pd.read_pickle(file_data)
    df_sp_state = (
        df_SP[
        (df_SP.index.year>=year_i) & (df_SP.index.year<=year_f)
        ][['SP_'+state]]
    )
    df_sp_state.rename(
        columns={ 'Demand_'+state:'Demand', 'SP_'+state:'SP' },
        inplace=True
    )
    return df_sp_state


# def generating_df_aus(
#         lat: float,
#         lon: float,
#         state: str,
#         year_i: int,
#         year_f: int,
#         dT: float = 0.5) -> pd.DataFrame:
    
#     DIR_SPOTPRICE = os.path.join( DIR_MAIN,'data','NEM_spotprice')
#     FILE_SPOTPRICE = os.path.join(DIR_SPOTPRICE, 'NEM_TRADINGPRICE_2010-2020.PLK')
#     DIR_WEATHER_DATA = os.path.join(DIR_MAIN,"data","weather")
#     FILE_WEATHER  = os.path.join(DIR_WEATHER_DATA, 'MERRA2_Processed_All.nc')

#     #Spot price data for state
#     df_SP = pd.read_pickle(FILE_SPOTPRICE)
#     df_SP_st = (
#         df_SP[
#         (df_SP.index.year>=year_i) & (df_SP.index.year<=year_f)
#         ][['SP_'+state]]
#     )
    
#     data_weather = xr.open_dataset(FILE_WEATHER)
#     lats = np.array(data_weather.lat)
#     lons = np.array(data_weather.lon)
#     lon_a = lons[(abs(lons-lon)).argmin()]
#     lat_a = lats[(abs(lats-lat)).argmin()]
#     df_weather = data_weather.sel(lon=lon_a,lat=lat_a).to_dataframe()
    
#     del data_weather #Cleaning memory
    
#     df_weather['DNI_dirint'] = df_weather['DNI_dirint'].fillna(0)
#     df_weather.index = df_weather.index.tz_localize('UTC')
#     df_weather = df_weather.resample(f'{dT:.1f}h').interpolate()       #Getting the data in half hours
#     df_weather['DNI'] = df_weather['DNI_dirint']
    
#     # Merging the data
#     df_data = df_weather.merge(
#         df_SP_st,
#         how='inner',
#         left_index=True,
#         right_index=True
#     )
#     del df_weather   #Cleaning memory
    
#     #Calculating  solar position
#     tz = 'Australia/Brisbane'
#     df_data.index = df_data.index.tz_convert(tz)
#     sol_pos = Location(lat, lon, tz=tz).get_solarposition(df_data.index)
#     df_data['azimuth'] = sol_pos['azimuth'].copy()
#     df_data['elevation'] = sol_pos['elevation'].copy()
    
#     #Setting the initial dataframe for dispatch model
#     df_data = df_data[['DNI','T2M','WS','azimuth','elevation','SP_'+state]]
#     df_data.rename(
#         columns={'T2M':'Tamb','azimuth':'azi','elevation':'ele'},
#         inplace=True
#     )
#     df_data.rename(
#         columns={ 'Demand_'+state:'Demand', 'SP_'+state:'SP' },
#         inplace=True
#     )
#     return df_data


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
        1 if (T_pi < 550+273.) else float(1+ 5.4e-5 * (T_pi-273.15-550)**2)
        for T_pi in T_ps
    ]
    C_PHX  = float(np.mean(f_HXTs) * 1000 * A_HX / P_pb_th)
    return C_PHX


def dispatch_model_simple(
        df: pd.DataFrame,
        plant: PlantCSPBeamDownParticle,
    ) -> pd.DataFrame:


    Q_stg = plant.stg_cap.get_value("hr") * plant.pb_power_th.get_value("MW")
    DNI_min = plant.DNI_min.get_value("W/m2")
    Ntower = plant.Ntower
    P_pb = plant.pb_power_el.get_value("MW")
    A_SF = plant.sf_area.get_value("m2")

    dT = 0.5
    #Calculations for Collected Energy and potential Dispatched energy
    # Is the Solar Field operating?
    df['SF_Op']  = np.where( df["DNI"]>DNI_min, 1, 0)
    df['E_SF']   = df["SF_Op"] * df["sf_eta"] * df["DNI"] * A_SF * 1e-6 * dT   #[MWh]
    df['E_rcv']  = df["E_SF"] * df["receiver_eta"]                                  #[MWh]
    
    #It is assumed the same amount of energy is extracted from storage, so particle-sCO2 HX always try to work at design capacity. The energy dispatched is lower then.
    df['E_th_pb'] = P_pb * dT                              #[MWh_th]
    df['E_gen'] = Ntower * df['pb_eta'] * df['E_th_pb']    #[MWh_e] Including the Number of towers
    df['R_gen'] = df['E_gen'] * df['SP']                   #[$]
    
    #For simplicity the day starts with sunrise and lasts until next sunrise
    df['NewDay']  = np.where(
        (df["elevation"]*df["elevation"].shift(1)<0) & (df["elevation"]>0),
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
        .groupby(df["DaySet"])["R_gen"]
        .rank('first', ascending=False)
    )
    #Selecting the hours that should dispatch at anytime, excluding those who already dispatched
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
    df['Dispatch_Extra'] = np.where(
        (df.Rank_Disp_Extra>0) & (df.Rank_Disp_Extra-1<df.disp_remaining), 
        df[['disp_remaining', 'E_avail_Extra']].min( axis=1 ), 
        0.
    )
    df['Dispatch_Full'] = (df["Dispatch_Both"] + df["Dispatch_Extra"])
    
    df['E_stg_net'] = df["SF_Op"] * df["E_rcv"] - df["E_th_pb"] * df["Dispatch_Full"]
    df['E_stg_cum_tot'] = df["E_stg_net"].cumsum()
    df['E_stg_cum_day'] = df.groupby(df["DaySet"])['E_stg_net'].cumsum()
    df['E_gen_final'] = df["Dispatch_Full"] * df["E_gen"]      # [MWh]
    df['Rev_final'] = df["E_gen_final"] * df["SP"]           # [AUD]
    return df


def calc_financial_metrics(
        revenue_tot: float,
        cost_capital: float,
        discount_rate: float,
        n_years: int,
        cost_om: float
    ) -> tuple[float, float, float]:
    """
    Calculate financial metrics including ROI, NPV, and payback period.
    """
    tpf = (1./discount_rate)*(1. - 1./(1.+discount_rate)**n_years)
    roi2 = tpf * revenue_tot / cost_capital         # Return over investment
    npv = revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)
    roi = (revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)) / cost_capital
    
    def fNPV(i):
        tpf = (1./discount_rate)*(1. - 1./(1.+discount_rate)**i)
        return revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)
    
    sol = spo.fsolve(fNPV, n_years, full_output=True)
    payback_period = sol[0][0]
    if sol[2] != 1:
        print(sol)

    return npv, roi, payback_period


def annual_performance(
        plant: PlantCSPBeamDownParticle,
        df: pd.DataFrame,
        ):
    
    dT = 0.5
    receiver_power = plant.receiver_power.get_value("MW")
    pb_power_el = plant.pb_power_el.get_value("MW")
    pb_power_th = plant.pb_power_th.get_value("MW")
    pb_temp_max = plant.pb_temp_max.get_value("K")
    stg_temp_hot = plant.stg_temp_hot.get_value("K")
    stg_temp_cold = plant.stg_temp_cold.get_value("K")
    
    # Obtaining the efficiencies in hourly basis
    df["date"] = pd.to_datetime(df.index).date
    Ndays = len(df["date"].unique())
    years = Ndays/365
    
    df["sf_eta"] = plant.eta_optical_hourly(df)
    df["receiver_eta"] = plant.eta_receiver_hourly(df, eta_HPR_0D())
    df["pb_eta"] = plant.eta_pb_hourly(df, eta_power_block())
    
    df_out = dispatch_model_simple(df, plant)

    stg_heat_stored = Variable(
        df_out["E_rcv"].sum()/years, "MWh/yr"
    )                                           # annual energy stored per tower
    pb_energy_dispatched = Variable(
        df_out["E_gen_final"].sum()/years, "MWh/yr"
    )                                           # annual energy dispatched per power block
    revenue_tot = Variable(
        df_out["Rev_final"].sum()/1e6/years, "MM USD/yr"
    )    # total annual revenue per power block
    revenue_daily = Variable(
        df_out["Rev_final"].sum()/1000/Ndays, "k AUD/day"
    )   #daily revenue
    sf_cf = Variable(
        df_out["E_rcv"].sum() / (receiver_power * dT * len(df_out)), "-"
    )
    pb_cf = Variable(
        df_out["E_gen_final"].sum() / (pb_power_el * dT * len(df_out)), "-"
    )
    sf_time_operation = Variable(
        df_out["SF_Op"].sum()*dT / years, "-"
    )                #hrs
    pb_time_operation = Variable(
        df_out["Dispatch_Full"].sum()*dT/years, "-"
    )        #hrs
    stg_min_stored = Variable(
        df_out["E_stg_cum_day"].min(), "MWh"
    )  # minimum energy stored in the storage system
    
    #Updating costs and converting USD to AUD
    # Obtaining the costs for the power block an PHX
    USDtoAUD = 1.4
    costs_in = spr.get_plant_costs()
    costs_in['CF_sf'] = sf_cf.v
    costs_in['CF_pb'] = pb_cf.v
    
    pb_red_cost_factor = float(costs_in['F_pb'])
    file_pb = os.path.join(DIR_DATA,'sCO2_cycle','sCO2_lookuptable_costs.csv')
    df_pb = pd.read_csv(file_pb,index_col=0)
    lbl_x='P_el'
    lbl_y='T_htf_hot_des'
    df_pb.sort_values(by=[lbl_x,lbl_y],axis=0,inplace=True)
    X = df_pb[lbl_x]
    Y = df_pb[lbl_y]
    XY = list(zip(X,Y))

    if costs_in['C_pb_est']=='Neises':
        lbl_f='cycle_spec_cost'
    elif costs_in['C_pb_est']=='Sandia':
        lbl_f='cycle_spec_cost_Sandia'
    else:
        raise ValueError('Invalid value for C_pb_est')
    f_pb = LinearNDInterpolator(XY,df_pb[lbl_f])
    pb_capital = (1-pb_red_cost_factor) * f_pb([pb_power_el,pb_temp_max-273.15])[0]
    
    #Obtaining the CO2 temperatures
    lbl_f = 'T_co2_PHX_in'
    f_sCO2_in = LinearNDInterpolator(XY,df_pb[lbl_f])
    T_sCO2_in  = f_sCO2_in([pb_power_el,pb_temp_max-273.15])[0]
    
    lbl_f = 'T_turb_in'
    f_sCO2_out = LinearNDInterpolator(XY,df_pb[lbl_f])
    T_sCO2_out  = f_sCO2_out([pb_power_el,pb_temp_max-273.15])[0]
    
    U_HX = 240.                             #Value from DLR design
    C_PHX = PHX_cost(pb_power_th, T_sCO2_in, T_sCO2_out, stg_temp_hot, stg_temp_cold, U_HX)
    
    costs_in['C_pb']  = pb_capital * 1e3            #[USD/MW]
    costs_in['C_PHX'] = C_PHX                 #[USD/MW]
    
    # CSTo = {}
    # SF = pd.DataFrame()
    # CSTo['costs_in'] = costs_in
    # CSTo['type_weather'] = 'MERRA2'         #This is to use CF_sf calculated here
    # Costs_o = spr.BDR_cost(SF, CSTo)
    # Costs_o = spr.plant_costing_calculations(plant, HB, TOD, costs_in, SF)

    cost_capital  = plant.costs_out["Elec"] * USDtoAUD         #Total Capital Cost [AUD]
    discount_rate = float(costs_in['DR'])
    n_years = int(costs_in['Ny'])
    cost_om = float(costs_in['C_OM'])
    
    lcoh = spr.get_lcoh(
            plant.costs_out["Heat"], cost_om, receiver_power, sf_cf.v, discount_rate, n_years,
        )* USDtoAUD
    lcoe = spr.get_lcoe(
            plant.costs_out["Elec"], cost_om, pb_power_el, pb_cf.v, discount_rate, n_years,
        )* USDtoAUD
    npv, roi, payback_period = calc_financial_metrics(
        revenue_tot.get_value("MM USD/yr"),
        cost_capital,
        discount_rate,
        n_years,
        cost_om
    )
    results = {
        "stg_min_stored": stg_min_stored,
        "stg_heat_stored": stg_heat_stored,
        'pb_energy_dispatched':pb_energy_dispatched,
        'sf_cf':sf_cf,
        'pb_cf':pb_cf,
        'revenue_tot':revenue_tot,
        'revenue_daily':revenue_daily,
        'lcoh': Variable(lcoh, "AUD/MWh"),
        'lcoe':Variable(lcoe, "AUD/MWh"),
        'pb_capital': Variable(pb_capital, "MM AUD"),
        'cost_capital': Variable(cost_capital, "MM AUD"),
        'roi':Variable(roi, "-"),
        'payback_period': Variable(payback_period, "yr"),
        'npv': Variable(npv, "MM AUD"),
        'time_operation_sf':sf_time_operation,
        'time_operation_pb':pb_time_operation,
        'date_simulation':time.time(),
    }
    return results
# %%
