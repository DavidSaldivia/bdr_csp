from __future__ import annotations
from dataclasses import dataclass, field
from os.path import isfile
from typing import Callable, Union
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

from antupy.units import Variable, conversion_factor as CF
import bdr_csp.bdr as bdr
import bdr_csp.spr as spr
import bdr_csp.htc as htc
from bdr_csp.dir import DIRECTORY

ParticleReceiver = Union[spr.HPR0D, spr.HPR2D]

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
    'rcv_power'
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
    'pb_cost',
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
    rcv_power: Variable = Variable(19.,"MW")
    flux_avg: Variable = Variable(1.25,"MW/m2")
    xrc: Variable = Variable(0., "m")
    yrc: Variable = Variable(0., "m")
    zrc: Variable = Variable(10., "m")

    # BDR parameters
    geometry: str = "PB"
    array: str = "A"
    Cg: Variable = Variable(2, "-")
    rcv_area: Variable = Variable(20., "m2")

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
    Ah1: Variable = Variable(2.92**2, "m2")
    Npan: int = 1

    ### Receiver and Storage Tank characteristics
    rcv_type: str = 'HPR_0D'                            # model for Receiver
    rcv_flux_max: Variable = Variable(3.0, "MW/m2")     # Maximum radiation flux on receiver
    stg_temp_cold: Variable = Variable(950,"K")              # particle temperature in cold tank
    stg_temp_hot: Variable = Variable(1200,"K")              # particle temperature in hot tank
    stg_material: str  = 'CARBO'                             # Thermal Storage Material
    stg_thickness: Variable = Variable(0.05, "m")            # Thickness of material on conveyor belt
    stg_cap: Variable = Variable(8., "hr")                   # Hours of storage
    stg_h_to_d: Variable = Variable(0.5, "-")                # [-] height to diameter ratio for storage tank
    solar_multiple: Variable = Variable(2.0, "-")            # [-] Initial target for solar multiple
    
    # Heat exchanger and Power Block efficiencies
    hx_u = Variable(240., "W/m2-K")                             #Value from DLR design
    pb_temp_max: Variable = Variable(875 + 273.15,"K")         #[K] Maximum temperature un power block cycle
    Ntower: int = 1                                            #[-] Number of towers feeding one power block 
    pb_eta_des: Variable = Variable(0.50,"-")                  #[-] Power Block efficiency target (initial value) 
    stg_eta_des: Variable = Variable(1.00,"-")                 #[-] Storage efficiency target (assumed 1 for now)
    # rcv_eta_des: Variable = Variable(0.75, "-")           #[-] Receiver efficiency target (initial value)


    def __post_init__(self):
        # Outputs
        self.sf_n_hels = Variable(None, "-")                     # Number of heliostats
        self.sf_area = Variable(None, "m2")                     # Solar Field area
        self.sf_power = Variable(None, "MW")                     # Solar Field power (output)

        self.pb_power_th: Variable = Variable(None, "MW")        # Power Block thermal power (input)
        self.pb_power_el: Variable = Variable(None, "MW")        # Power Block electrical power (output)
        self.storage_heat: Variable = Variable(None, "MWh")      # Storage heat (output)

        self.rcv_massflowrate: Variable = Variable(None, "kg/s")  # Receiver mass flow rate (to be calculated)
        self.costs_out : dict[str, float] = {}  # Costs output from simulation

        # Creating the derived objects and calculating derived attributes
        flux_avg = self.flux_avg.u("MW/m2")
        rcv_power = self.rcv_power.u("MW")
        temp_avg = (self.stg_temp_cold.u("K") + self.stg_temp_hot.u("K")) / 2

        self.rcv_eta_des = Variable( spr.HTM_0D_blackbox( temp_avg, flux_avg )[0], "-" )
        self.rcv_area = Variable((rcv_power / self.rcv_eta_des.u("-")) / flux_avg, "m2")

        file_SF = os.path.join(
            DIR_DATA,
            'mcrt_datasets_final',
            'Dataset_zf_{:.0f}'.format(self.zf.u("m"))
        )
        self.file_weather = os.path.join(
            DIR_DATA, 'weather', "Alice_Springs_Real2000_Created20130430.csv"
        )
        self.HSF = bdr.SolarField(zf=self.zf, A_h1=self.Ah1, N_pan=self.Npan, file_SF=file_SF)
        self.HB = bdr.HyperboloidMirror(
            zf=self.zf, fzv=self.fzv, xrc=self.xrc, yrc=self.yrc, zrc=self.zrc, eta_hbi=self.eta_rfl
        )
        self.TOD = bdr.TertiaryOpticalDevice(
            geometry = self.geometry, array = self.array, Cg=self.Cg, receiver_area= self.rcv_area,
            xrc=self.xrc, yrc=self.yrc, zrc=self.zrc,
        )

        self.receiver = self.select_receiver(self.rcv_type)
    

    def select_receiver(self, type_: str) -> ParticleReceiver:
        """
        Factory method to select the receiver type.
        """
        if type_ == 'HPR_2D':
            return spr.HPR2D(
                rcv_nom_power=self.rcv_power,
                heat_flux_avg=self.flux_avg,
                temp_ini=self.stg_temp_cold,
                temp_out=self.stg_temp_hot,
                thickness_parts=self.stg_thickness,
            )
        elif type_ == 'HPR_0D':
            return spr.HPR0D(
                rcv_nom_power=self.rcv_power,
                heat_flux_avg=self.flux_avg,
                temp_ini=self.stg_temp_cold,
                temp_out=self.stg_temp_hot,
                thickness_parts=self.stg_thickness,
            )
        elif type_ == "TPR_0D":
            raise ValueError("Receiver type {type_} is not implemented yet.")
        elif type_ == "TPR_2D":
            raise ValueError("Receiver type {type_} is not implemented yet.")
        else:
            raise ValueError(f"Receiver type '{type_}' not recognized. Use 'HPR_2D' or 'HPR_0D'.")

    def run_thermal_subsystem(
            self,
            # save_detailed_results: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        

        # R2, SF, costs_out = spr.run_coupled_sim(self) #, self.HSF, self.HB, self.TOD)
        # self.costs_out = costs_out

        HSF = self.HSF
        HB = self.HB
        TOD = self.TOD

        #Getting the RayDataset
        R0, SF = HSF.load_dataset(save_plk=True)

        #Getting interceptions with HB
        R1 = HB.mcrt_direct(R0, refl_error=True)
        R1['hel_in'] = True
        HB.rmin = Variable( R1['rb'].quantile(0.0001), "m")
        HB.rmax = Variable( R1['rb'].quantile(0.9981), "m")
        R1['hit_hb'] = (R1['rb']>HB.rmin.u("m")) & (R1['rb']<HB.rmax.u("m"))
        
        SF = HB.shadow_simple(
            lat=self.lat.v, lng=self.lng.v, type_shdw="simple", SF=SF
        )
        
        #Interceptions with TOD
        R2 = TOD.mcrt_solver(R1, refl_error=False)
        
        ### Optical Efficiencies
        SF = bdr.optical_efficiencies(self,R2,SF)
        
        ### Running receiver simulation and getting the results
        receiver = self.receiver
        if isinstance(receiver, spr.HPR2D):
            receiver.run_model(TOD,SF,R2)
        elif isinstance(receiver, spr.HPR0D):
            receiver.run_model(SF)
        else:
            raise ValueError(f"Receiver type '{self.rcv_type}' not recognized or not implemented.")

        T_p = receiver.temps_parts
        N_hel = receiver.n_hels.u("-")
        Qstg = receiver.heat_stored.u("MWh")
        M_p = receiver.mass_stg.u("kg/s")
        eta_rcv = receiver.eta_rcv.u("-")

        # Heliostat selection
        Gbn = self.Gbn.u("W/m2")
        A_h1 = self.Ah1.u("m2")
        Q_acc    = SF['Q_h1'].cumsum()
        hlst     = Q_acc.iloc[:N_hel].index
        SF['hel_in'] = SF.index.isin(hlst)
        R2['hel_in'] = R2['hel'].isin(hlst)
        
        # Outputs
        self.sf_n_hels = Variable(N_hel, "-")
        self.sf_power  = Variable( SF[SF.hel_in]['Q_h1'].sum(), "MW")
        self.rcv_power = Variable(self.sf_power.v * eta_rcv, "MW")
        self.sf_area = Variable(N_hel * A_h1, "m2")
        self.rcv_massflowrate = Variable(M_p, "kg/s")
        
        # Calculating HB surface
        HB.rmin = Variable(R2[R2.hel_in]['rb'].quantile(0.0001), "m")
        HB.rmax = Variable(R2[R2.hel_in]['rb'].quantile(0.9981), "m")
        R2['hit_hb'] = (R2['rb']>HB.rmin.v)&(R2['rb']<HB.rmax.v)
        
        HB.update_geometry(R2)
        HB.height_range()
        HB.calculate_mass(R2, SF, Gbn, A_h1)
        
        #Costing calculation
        self.costs_out = spr.plant_costing_calculations(self, SF) # HB, TOD, costs_in, SF)
        
        # if save_detailed_results:
        #     pickle.dump([costs_out,R2,SF,TOD],open(file_base_case,'wb'))

        print(HB.surface_area.u("m2"))
        print(self.HB.surface_area.u("m2"))

        return R2, SF


    def eta_optical_hourly(
            self,
            df: pd.DataFrame,
            file_alt_azi: str = os.path.join(DIR_DATA,'annual_perf','1-Grid_AltAzi_vF.csv'),
            lbl: str ='eta_SF'
        ) -> np.ndarray:

        latitude = self.lat.u("deg")
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
        
        sf_area = self.sf_area.u("m2")
        temp_parts_avg = (self.stg_temp_cold.u("K") + self.stg_temp_hot.u("K")) / 2
        rcv_area = self.rcv_area.u("m2")
        Npoints = len(df)

        temp_parts = temp_parts_avg * np.ones(Npoints)
        temp_ambsK = df["temp_amb"].to_numpy()         #Ambient temperature in Kelvin
        rcv_flux_avg = df["DNI"] * df["sf_eta"] * (sf_area / rcv_area) * 1e-6
        return func_receiver(np.array([temp_parts, temp_ambsK, rcv_flux_avg]).transpose())


    def eta_pb_hourly(
            self,
            df: pd.DataFrame,
            func_pb: Callable | None = None,
    ) -> pd.DataFrame:

        if func_pb is None:
            raise ValueError("Power block efficiency function not provided.")
        temp_pb_max = self.pb_temp_max.u("K")
        Npoints = len(df)
        TambsC = df["temp_amb"].to_numpy() - 273.15         #Ambient temperature in Celcius
        T_maxC = (temp_pb_max-273.15) * np.ones(Npoints)
        return func_pb(np.array([T_maxC,TambsC]).transpose())


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
    case = 'case_zf{:.0f}_Q_avg{:.2f}_Prcv{:.1f}'.format(zf.u("m"), Qavg.u("MW/m2"), Prcv.u("MW"))
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
            'Dataset_zf_{:.0f}'.format(zf.u("m"))
        )
        CSTi['file_weather'] = os.path.join(
            DIR_DATA,
            'weather',
            "Alice_Springs_Real2000_Created20130430.csv"
        )
        CSTi['zf'] = zf.u("m")
        CSTi['Qavg'] = Qavg.u("MW/m2")
        CSTi['P_rcv'] = Prcv.u("MW")
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
        file_weather  = os.path.join(
            DIR_DATA, "weather", 'MERRA2_Processed_All.nc'
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

class CO2:
    _ct_solution = ct.Solution('gri30.yaml','gri30')
    def cp(self, temp: float, pressure: float) -> float:
        # Calculate specific heat capacity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.cp

    def rho(self, temp: float, pressure: float) -> float:
        # Calculate specific heat capacity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.density_mass


@dataclass
class PrimaryHeatExchanger:
    fluid = CO2()
    parts = spr.Carbo()
    
    pb_power_th: Variable = Variable(20., "MW")
    temp_sco2_in: Variable = Variable(550 + 273.15, "K")
    temp_sco2_out: Variable = Variable(650 + 273.15, "K")
    temp_parts_in: Variable = Variable(950, "K")
    temp_parts_out: Variable = Variable(1200, "K")
    U: Variable = Variable(240., "W/m2-K")               # Heat transfer coefficient
    press_sco2: Variable = Variable(25e6, "Pa")        # Pressure of the sCO2 in the HX
    n_pass: Variable = Variable(4, "-")              # Number of passes in the HX
    n_tubes: Variable = Variable(400, "tubes/m2")           # Number of tubes per m2
    d_tubes: Variable = Variable(0.03, "m")         # Diameter of the tubes in the HX


    def __post_init__(self):
        # Outputs
        self.heat_exchanged: Variable = Variable(None, "MW")
        self.epsilon: Variable = Variable(None, "-")
        self.lmtd: Variable = Variable(None, "K")
        self.surface_area: Variable = Variable(None, "m2")
        self.volume: Variable = Variable(None, "m3")
        self.parts_mfr: Variable = Variable(None, "kg/s")
        self.sco2_mfr: Variable = Variable(None, "kg/s")

    @property
    def cost(self) -> Variable:         # Result is obtained in MM USD/MW
        self.run_model()
        T_ps   = np.linspace(self.temp_parts_in.u("K"), self.temp_parts_out.u("K"),10)
        f_HXTs = [
            1 if (T_pi < 550+273.) else float(1+ 5.4e-5 * (T_pi-273.15-550)**2)
            for T_pi in T_ps
        ]
        return Variable(
            float(np.mean(f_HXTs) * 1000 * self.surface_area.u("m2") / self.pb_power_th.u("MW")), "MM USD/MW"
        ) 

    def run_model(self) -> None:
    
        #### Tube and shell hx using log-mean temp difference
        temp_sco2_in = self.temp_sco2_in.u("K")
        temp_sco2_out = self.temp_sco2_out.u("K")
        temp_parts_in = self.temp_parts_in.u("K")
        temp_parts_out = self.temp_parts_out.u("K")
        pb_power_th = self.pb_power_th.u("MW")
        U_HX = self.U.u("W/m2-K")
        press_sco2 = self.press_sco2.u("Pa")
        Npass = int(self.n_pass.u("-"))
        n_tubes = int(self.n_tubes.u("tubes/m2"))
        d_tubes = self.d_tubes.u("m")

        dT_sco2  = temp_sco2_out - temp_sco2_in
        temp_sco2_avg = (temp_sco2_out+temp_sco2_in)/2


        cp_sco2  = self.fluid.cp(temp_sco2_avg, press_sco2)
        
        sco2_mfr = pb_power_th / (cp_sco2 * dT_sco2) / Npass
        sco2_C = sco2_mfr * cp_sco2
        heat_exchanged = sco2_C * dT_sco2
        
        dtemp_parts    = temp_parts_in - temp_parts_out
        temp_parts_avg = (temp_parts_in + temp_parts_out)/2
        parts_cp = self.parts.cp(temp_parts_avg)
        parts_C = heat_exchanged / dtemp_parts
        parts_mfr = parts_C / parts_cp
        
        lmtd = (
            ((temp_parts_out-temp_sco2_in) - (temp_parts_in-temp_sco2_out))
            /np.log((temp_parts_out-temp_sco2_in)/(temp_parts_in-temp_sco2_out))
        )
        C_min = min(parts_C, sco2_C)
        C_max = max(parts_C, sco2_C)
        heat_max = C_max * (temp_parts_in - temp_sco2_in)
        epsilon = heat_exchanged  / heat_max
        
        surface_area = heat_exchanged / ( U_HX * lmtd)
        dAdV = (np.pi*d_tubes*1)*n_tubes
        
        volume   = surface_area/dAdV
        
        self.heat_exchanged = Variable(heat_exchanged, "MW")
        self.epsilon = Variable(epsilon, "-")
        self.lmtd = Variable(lmtd, "K")
        self.surface_area = Variable(surface_area, "m2")
        self.volume = Variable(volume, "m3")
        self.parts_mfr = Variable(parts_mfr, "kg/s")
        self.sco2_mfr = Variable(sco2_mfr, "kg/s")
        return None

 


def dispatch_model_simple(
        df: pd.DataFrame,
        plant: PlantCSPBeamDownParticle,
    ) -> pd.DataFrame:


    stg_cap_heat = plant.storage_heat.u("MWh")
    pb_power_el = plant.pb_power_el.u("MW")
    pb_power_th = plant.pb_power_th.u("MW")
    stg_cap = plant.stg_cap.u("hr")
    sf_area = plant.sf_area.u("m2")
    Ntower = plant.Ntower
    DNI_min = plant.DNI_min.u("W/m2")


    dT = 0.5 #[hr]
    #Calculations for Collected Energy and potential Dispatched energy
    # Is the Solar Field operating?
    df["rcv_op_fraction"]  = np.where( df["DNI"]>DNI_min, 1, 0)
    df["sf_heat_to_rcv"]   = Ntower * df["rcv_op_fraction"] * df["sf_eta"] * df["DNI"] * sf_area * CF("W", "MW") * dT   #[MWh]
    df['rcv_heat']  = df["sf_heat_to_rcv"] * df["rcv_eta"]                                  #[MWh]
    df["rcv_heat"] = (
        Ntower * df["DNI"] * sf_area * CF("W", "MW") * dT
         * df["sf_eta"] * df["rcv_eta"] * df["rcv_op_fraction"]
    )
    df['rcv_heat'] = df["rcv_heat"].fillna(0)

    #It is assumed the same amount of energy is extracted from storage, so particle-sCO2 HX always try to work at design capacity. The energy dispatched is lower then.
    df["pb_heat_req"] = pb_power_th * dT              #[MWh_th]
    df["energy_gen_pot"] = df['pb_eta'] * df["pb_heat_req"]    #[MWh_e]
    df["revenue_gen_pot"] = df["energy_gen_pot"] * df['SP']          #[$]
    
    #For simplicity the day starts with sunrise and lasts until next sunrise
    df["new_day"]  = (df["elevation"]*df["elevation"].shift(1)<0) & (df["elevation"]>0)
    df.loc[df.index[0],"new_day"] = True    #The first day is manually set as 1
    df["day_index"] = df["new_day"].cumsum()   # creating day index
    df["day_index"] = df["day_index"].ffill()
    
    #Total heat delivered by receiver(s) per day and heat that exceeds the storage capacity
    df["rcv_heat_collected"] = df.groupby(df["day_index"])['rcv_heat'].transform('sum')
    df["heat_in_excess"] = df["rcv_heat_collected"] - stg_cap_heat
    
    #Ranking dispatch periods by their potential revenue
    #First the periods when receiver generates more energy than extracted by power block
    df["rank_dispatch_while_rcv_op"] = (
        df[ df["rcv_heat"] > df["pb_heat_req"] ]
        .groupby( df["day_index"] )["revenue_gen_pot"]
        .rank('first', ascending=False)
    )
    #Accumulated thermal energy extracted from Storage System
    # for dispatch hours during solar field operation time,
    # sorted by Dispatch Ranking
    # df["pb_heat_req_cum_while_rcv_op"] = (
    #     df.sort_values(["day_index", "rank_dispatch_while_rcv_op"])
    #     .groupby([ df["day_index"] ])["pb_heat_req"]
    #     .cumsum()
    # )
    df["pb_heat_req_cum_while_rcv_op"] = (
        np.where(
            df["rank_dispatch_while_rcv_op"].notnull(),
            (
                df.sort_values(["day_index", "rank_dispatch_while_rcv_op"])
                .groupby([ df["day_index"] ])["pb_heat_req"]
                .cumsum()
            ),
            np.nan
        )
    )
    #Selecting the hours that should dispatch during operating time
    df["dispatch_while_rcv_op"] = (
        np.where(
            (df["pb_heat_req_cum_while_rcv_op"] < df["heat_in_excess"]) & (df["SP"]>0) ,
            1,
            0
        )
    )
    df["stg_stored_cum_while_rcv_op"] =(
        (df["rcv_op_fraction"] * df["rcv_heat"] - df["pb_heat_req"] * df["dispatch_while_rcv_op"] )
        .groupby(df["day_index"])
        .cumsum()
    )
    #If E_stg_cum > stg_cap_heat, then we need to redirect part of solar field
    df["rcv_op_fraction"] = (
        np.where(
            (df["stg_stored_cum_while_rcv_op"] > stg_cap_heat ) & (df["rcv_op_fraction"]>0),
            1 - (df["stg_stored_cum_while_rcv_op"] - stg_cap_heat )/df["rcv_heat"],
            df["rcv_op_fraction"]
        )
    )
    df["rcv_op_fraction"] = np.where((df["rcv_op_fraction"]<0), 0 , df["rcv_op_fraction"])
    df["stg_stored_cum_while_rcv_op"] = (
        (df["rcv_op_fraction"]*df["rcv_heat"] - df["pb_heat_req"]*df["dispatch_while_rcv_op"])
        .groupby(df["day_index"])
        .cumsum()
    )
    df["heat_used_while_rcv_op"] = (
        (df["pb_heat_req"]*df["dispatch_while_rcv_op"])
        .groupby(df["day_index"])
        .transform('sum')
    )
    #Energy available to generate electricity after the first group dispatched
    df["heat_avail_after_first_dispatch"] = (
        df["stg_stored_cum_while_rcv_op"]
        .groupby(df["day_index"])
        .transform('last')
    )
    # Ranking for all remaining hours that haven't dispatched yet
    df["rank_dispatch_after_first"] = (
        df[(df["dispatch_while_rcv_op"]<=0) & (df["stg_stored_cum_while_rcv_op"] > df["pb_heat_req"])]
        .groupby(df["day_index"])
        ["revenue_gen_pot"]
        .rank('first', ascending=False)
    )
    #Selecting the hours that should dispatch at anytime,
    # excluding those who already dispatched
    df["pb_heat_req_cum_after_first"] = (
        df[(df["dispatch_while_rcv_op"]<=0) & (df["stg_stored_cum_while_rcv_op"]>df["pb_heat_req"])]
        .sort_values(["day_index", "rank_dispatch_after_first"])
        .groupby(df["day_index"])
        ["pb_heat_req"]
        .cumsum()
    )
    df["dispatch_second"] = (
        np.where(
            (
                (df["pb_heat_req_cum_after_first"]<=df["heat_avail_after_first_dispatch"])
                & (df["dispatch_while_rcv_op"]<=0)
                & (df["SP"]>0)
            ),
            1,
            0
        )
    )
    df["dispatch_both"] = (
        np.where(
            (df["dispatch_while_rcv_op"]>0) | (df["dispatch_second"]>0),
            1,
            0
        )
    )    
    #Net energy on each time after Both
    df["heat_stored_after_both"]  = (
        df["rcv_op_fraction"] * df["rcv_heat"] - df["pb_heat_req"]*df["dispatch_both"]
    )
    df["heat_stored_cum_after_both"] = (
        df["heat_stored_after_both"]
        .groupby(df["day_index"])
        .cumsum()
    )
    df["heat_remaining_after_both"] = (
        df['heat_stored_cum_after_both']
        .groupby(df["day_index"])
        .transform('last')
    )
    df["dispatch_remaining"] = (df["heat_remaining_after_both"]/df["pb_heat_req"])
    df["rank_dispatch_extra"] = (
        df[(df["dispatch_both"]<=0) & (df["SP"]>0)]
        .groupby(df["day_index"])["revenue_gen_pot"]
        .rank('first', ascending=False)
    )
    #Selecting the hours that should dispatch at anytime, excluding those who already dispatched
    #All the full periods are included
    df["heat_avail_extra"] = (
        np.where(
            (
                (df["rank_dispatch_extra"]>0)
                & (df["dispatch_remaining"]<1) 
                & (df["dispatch_remaining"]>0.01)
            ),
            df["heat_stored_after_both"]/df["pb_heat_req"],
            0.
        )
    )
    df["heat_avail_extra"] = (
        np.where(
            (df["rank_dispatch_extra"]>0),
            df["heat_stored_after_both"]/df["pb_heat_req"],
            0.
        )
    )
    df["dispatch_extra"] = np.where(
        (df["rank_dispatch_extra"]>0) & (df["rank_dispatch_extra"]-1<df["dispatch_remaining"]), 
        df[['dispatch_remaining', 'heat_avail_extra']].min( axis=1 ), 
        0.
    )
    df["dispatch_all"] = (df["dispatch_both"] + df["dispatch_extra"])
    
    df["heat_stored_in"] = df["rcv_op_fraction"] * df["rcv_heat"]
    df["heat_stored_out"] = df["dispatch_all"] * df["pb_heat_req"]
    df['heat_stored_net'] =  df["heat_stored_in"] - df["heat_stored_out"]  # [MWh]
    df['heat_stored_cum_total'] = df["heat_stored_net"].cumsum()
    df['heat_stored_cum_day'] = df.groupby(df["day_index"])['heat_stored_net'].cumsum()
    df['energy_gen_final'] = df["dispatch_all"] * df["energy_gen_pot"]      # [MWh]
    df['revenue_final'] = df["energy_gen_final"] * df["SP"]                 # [AUD]
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
    rcv_power = plant.rcv_power.u("MW")
    pb_power_el = plant.pb_power_el.u("MW")
    Ntower = plant.Ntower
    pb_temp_max = plant.pb_temp_max.u("K")

    
    # Obtaining the efficiencies in hourly basis
    df["date"] = pd.to_datetime(df.index).date
    Ndays = len(df["date"].unique())
    years = Ndays/365
    
    df["sf_eta"] = plant.eta_optical_hourly(df)
    df["rcv_eta"] = plant.eta_receiver_hourly(df, eta_HPR_0D())
    df["pb_eta"] = plant.eta_pb_hourly(df, eta_power_block())
    
    df_out = dispatch_model_simple(df, plant)

    stg_heat_stored = Variable(
        df_out["rcv_heat"].sum()/years, "MWh/yr"
    )                                           # annual energy stored per tower
    pb_energy_dispatched = Variable(
        df_out["energy_gen_final"].sum()/years, "MWh/yr"
    )                                           # annual energy dispatched per power block
    revenue_tot = Variable(
        df_out["revenue_final"].sum()/1e6/years, "MM USD/yr"
    )    # total annual revenue per power block
    revenue_daily = Variable(
        df_out["revenue_final"].sum()/1000/Ndays, "k USD/day"
    )   #daily revenue
    sf_cf = Variable(
        df_out["rcv_heat"].sum() / (Ntower * rcv_power * dT * len(df_out)), "-"
    )
    pb_cf = Variable(
        df_out["energy_gen_final"].sum() / (pb_power_el * dT * len(df_out)), "-"
    )
    sf_time_operation = Variable(
        df_out["rcv_op_fraction"].sum()*dT / years, "-"
    )                #hrs
    pb_time_operation = Variable(
        df_out["dispatch_all"].sum()*dT/years, "-"
    )        #hrs
    stg_min_stored = Variable(
        df_out["heat_stored_cum_day"].min(), "MWh"
    )  # minimum energy stored in the storage system
    
    #Updating costs and converting USD to AUD
    # Obtaining the costs for the power block an PHX
    USDtoAUD = 1.4
    costs_in = plant.cost_input
    costs_in['CF_sf'] = sf_cf.u("-")
    costs_in['CF_pb'] = pb_cf.u("-")
    
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
    pb_cost = pb_red_cost_factor * f_pb([pb_power_el,pb_temp_max-273.15])[0]
    
    #Obtaining the CO2 temperatures
    lbl_f = 'T_co2_PHX_in'
    f_sCO2_in = LinearNDInterpolator(XY,df_pb[lbl_f])
    temp_sco2_in  = f_sCO2_in([pb_power_el,pb_temp_max-273.15])[0]
    
    lbl_f = 'T_turb_in'
    f_sCO2_out = LinearNDInterpolator(XY,df_pb[lbl_f])
    temp_sco2_out  = f_sCO2_out([pb_power_el,pb_temp_max-273.15])[0]
    
    phx = PrimaryHeatExchanger(
        pb_power_th=plant.pb_power_th,
        temp_sco2_in=Variable(temp_sco2_in + 273.15, "K"),
        temp_sco2_out=Variable(temp_sco2_out + 273.15, "K"),
        temp_parts_in=plant.stg_temp_hot,
        temp_parts_out= plant.stg_temp_cold,
        U=plant.hx_u,
    )
    
    costs_in['C_pb']  = pb_cost * 1e3            #[USD/MW]
    costs_in['C_PHX'] = phx.cost.u("USD/MW")                 #[USD/MW]

    cost_capital  = plant.costs_out["Elec"]
    discount_rate = float(costs_in['DR'])
    n_years = int(costs_in['Ny'])
    cost_om = float(costs_in['C_OM'])
    
    lcoh = spr.get_lcoh(
            plant.costs_out["Heat"], cost_om, rcv_power, sf_cf.v, discount_rate, n_years,
        )
    lcoe = spr.get_lcoe(
            plant.costs_out["Elec"], cost_om, pb_power_el, pb_cf.v, discount_rate, n_years,
        )
    npv, roi, payback_period = calc_financial_metrics(
        revenue_tot.u("MM USD/yr"),
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
        'lcoh': Variable(lcoh, "USD/MWh"),
        'lcoe':Variable(lcoe, "USD/MWh"),
        'pb_cost': Variable(pb_cost, "MM USD"),
        'cost_capital': Variable(cost_capital, "MM USD"),
        'roi':Variable(roi, "-"),
        'payback_period': Variable(payback_period, "yr"),
        'npv': Variable(npv, "MM USD"),
        'sf_time_operation':sf_time_operation,
        'pb_time_operation':pb_time_operation,
        'date_simulation':time.time(),
    }
    return results
# %%
