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
from pvlib.location import Location
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

from antupy import Var, Array, CF
from antupy import Plant, component, constraint, derived
from antupy.props import Air, CO2, Carbo
from antupy.htc import h_horizontal_surface_upper_hot

from bdr_csp import bdr, spr
from bdr_csp.dir import DIRECTORY

ParticleReceiver = spr.HPR0D | spr.HPR2D | spr.TPR2D
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
class PlantCSPBeamDownParticle(Plant):
    """Enhanced CSP Beam Down Receiver Plant with smart component management."""
    
    # Core plant parameters
    zf: Var = Var(50, "m")                        # Tower focal height
    fzv: Var = Var(0.818161, "-")                 # HB vertex height fraction
    
    # Receiver parameters
    rcv_power: Var = Var(19., "MW")               # Receiver thermal power
    flux_avg: Var = Var(1.25, "MW/m2")           # Average flux on receiver
    
    # Receiver position
    xrc: Var = Var(0., "m")                      # Receiver center X
    yrc: Var = Var(0., "m")                      # Receiver center Y  
    zrc: Var = Var(10., "m")                     # Receiver center Z

    # BDR optical system parameters
    geometry: str = "PB"                          # TOD geometry type
    array: str = "A"                              # TOD array configuration
    Cg: Var = Var(2, "-")                        # Geometric concentration
    rcv_area: Var = Var(20., "m2")               # Receiver area (will be recalculated)

    # Economic parameters
    cost_input: dict[str, float|str] = field(default_factory=spr.get_plant_costs)

    # Environmental conditions
    Gbn: Var = Var(950, "W/m2")                  # Design-point DNI
    day: int = 80                                 # Design-point day
    omega: Var = Var(0.0, "rad")                 # Design-point hour angle
    lat: Var = Var(-23., "deg")                  # Latitude
    lng: Var = Var(115.9, "deg")                 # Longitude
    state: str = "SA"                             # State for electricity pricing
    temp_amb: Var = Var(300., "K")               # Ambient temperature
    type_weather: str | None = 'TMY'             # Weather data source
    file_weather: str | None = None              # Weather file path
    DNI_min: Var = Var(400., "W/m2")             # Minimum DNI for operation

    # Solar field characteristics
    eta_rfl: Var = Var(0.95, "-")                # Mirror reflectivity
    err_tot: Var = Var(0.002, "-")               # Total reflection error
    type_shadow = 'simple'                        # Shadow modeling type
    Ah1: Var = Var(2.92**2, "m2")               # Heliostat area
    Npan: Var = Var(1, "-")                     # Panels per heliostat

    # Receiver and storage characteristics
    rcv_type: str = 'HPR_0D'                     # Receiver model type
    rcv_flux_max: Var = Var(3.0, "MW/m2")       # Maximum flux on receiver
    stg_temp_cold: Var = Var(950, "K")          # Cold tank temperature
    stg_temp_hot: Var = Var(1200, "K")          # Hot tank temperature
    stg_material: str = 'CARBO'                  # Storage material
    stg_thickness: Var = Var(0.05, "m")         # Particle layer thickness
    stg_cap: Var = Var(8., "hr")                # Storage capacity
    stg_h_to_d: Var = Var(0.5, "-")             # Storage tank height/diameter ratio
    solar_multiple: Var = Var(2.0, "-")         # Solar multiple

    # Power block parameters
    hx_u = Var(240., "W/m2-K")                  # Heat exchanger U-value
    pb_temp_max: Var = Var(875 + 273.15, "K")   # Max power block temperature
    Ntower: int = 1                              # Number of towers per power block
    pb_eta_des: Var = Var(0.50, "-")            # Power block design efficiency
    stg_eta_des: Var = Var(1.00, "-")           # Storage efficiency

    # Output dictionary
    out: dict[str, Var|Array|float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived parameters and output variables."""
        # Initialize output variables
        self.sf_n_hels = Var(None, "-")           # Number of heliostats
        self.sf_area = Var(None, "m2")            # Solar field area
        self.sf_power = Var(None, "MW")           # Solar field power output

        self.pb_power_th: Var = Var(None, "MW")   # Power block thermal input
        self.pb_power_el: Var = Var(None, "MW")   # Power block electrical output
        self.storage_heat: Var = Var(None, "MWh") # Storage capacity

        self.rcv_massflowrate: Var = Var(None, "kg/s")  # Receiver mass flow rate
        self.costs_out: dict[str, float] = {}     # Cost calculation outputs

        # Set default weather file if not provided
        if self.file_weather is None:
            self.file_weather = os.path.join(
                DIR_DATA, 'weather', "Alice_Springs_Real2000_Created20130430.csv"
            )

    def _compute_file_SF(self, zf: Var) -> str:
        """Compute solar field dataset filename based on tower height."""
        return os.path.join(
            DIR_DATA,
            'mcrt_datasets_final',
            f'Dataset_zf_{zf.gv("m"):.0f}'
        )

    def _compute_rcv_eta_des(self, stg_temp_cold: Var, stg_temp_hot: Var, flux_avg: Var) -> Var:
        """Compute receiver design efficiency."""
        temp_avg = (stg_temp_cold.gv("K") + stg_temp_hot.gv("K")) / 2
        eta_des = spr.HTM_0D_blackbox(temp_avg, flux_avg.gv("MW/m2"))[0]
        return Var(eta_des, "-")

    def _compute_rcv_area(self, rcv_power: Var, rcv_eta_des: Var, flux_avg: Var) -> Var:
        """Compute required receiver area."""
        area = (rcv_power.gv("MW") / rcv_eta_des.gv("-")) / flux_avg.gv("MW/m2")
        return Var(area, "m2")

    @property
    def HSF(self) -> bdr.SolarField:
        """Solar Field component with smart dependency tracking."""
        return component(bdr.SolarField(
            zf=constraint(self.zf),
            A_h1=constraint(self.Ah1),
            N_pan=constraint(self.Npan),
            file_SF=derived(self._compute_file_SF, self.zf)
        ))

    @property
    def HB(self) -> bdr.HyperboloidMirror:
        """Hyperboloid Mirror component with smart dependency tracking."""
        return component(bdr.HyperboloidMirror(
            zf=constraint(self.zf),
            fzv=constraint(self.fzv),
            eta_hbi=constraint(self.eta_rfl),
            xrc=constraint(self.xrc),
            yrc=constraint(self.yrc),
            zrc=constraint(self.zrc)
        ))

    @property
    def TOD(self) -> bdr.TertiaryOpticalDevice:
        """Tertiary Optical Device component with smart dependency tracking."""
        # Compute derived receiver parameters with proper dependency tracking
        rcv_eta_des = derived(self._compute_rcv_eta_des, 
                             self.stg_temp_cold, self.stg_temp_hot, self.flux_avg)
        rcv_area = derived(self._compute_rcv_area, 
                          self.rcv_power, rcv_eta_des, self.flux_avg)
        
        return component(bdr.TertiaryOpticalDevice(
            geometry=constraint(self.geometry),
            array=constraint(self.array),
            Cg=constraint(self.Cg),
            receiver_area=rcv_area,
            xrc=constraint(self.xrc),
            yrc=constraint(self.yrc),
            zrc=constraint(self.zrc)
        ))

    @property
    def receiver(self) -> ParticleReceiver:
        """Particle receiver component with smart dependency tracking."""
        return component(self._select_receiver())

    def _select_receiver(self) -> ParticleReceiver:
        """Factory method to select the appropriate receiver type."""
        type_ = self.rcv_type.upper()
        
        # Common parameters for all receiver types
        common_params = {
            'rcv_nom_power': constraint(self.rcv_power),
            'heat_flux_avg': constraint(self.flux_avg),
            'temp_ini': constraint(self.stg_temp_cold),
            'temp_out': constraint(self.stg_temp_hot),
            'thickness_parts': constraint(self.stg_thickness),
        }
        
        if type_ == 'HPR_2D':
            return spr.HPR2D(**common_params)
        elif type_ == 'HPR_0D':
            return spr.HPR0D(**common_params)
        elif type_ == "TPR_2D":
            return spr.TPR2D(**common_params)
        elif type_ == "TPR_0D":
            raise ValueError(f"Receiver type {type_} is not implemented yet.")
        else:
            raise ValueError(f"Receiver type '{type_}' not recognized. Use 'HPR_2D' or 'HPR_0D'.")

    def run_thermal_subsystem(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the thermal subsystem simulation using smart components.
        
        Returns:
            tuple: (R2 ray dataset, SF solar field dataframe)
        """
        # Access components (will be created/cached as needed)
        HSF = self.HSF
        HB = self.HB
        TOD = self.TOD
        Gbn = self.Gbn
        A_h1 = self.Ah1

        # Load ray dataset
        R0, SF = HSF.load_dataset(save_plk=True)

        # Hyperboloid mirror ray tracing
        R1 = HB.mcrt_direct(R0, refl_error=True)
        R1['hel_in'] = True
        HB.rmin = Var(R1['rb'].quantile(0.0001), "m")
        HB.rmax = Var(R1['rb'].quantile(0.9981), "m")
        R1['hit_hb'] = (R1['rb'] > HB.rmin.gv("m")) & (R1['rb'] < HB.rmax.gv("m"))
        
        # Shadow calculations
        SF = HB.shadow_simple(
            lat=self.lat.v, lng=self.lng.v, type_shdw="simple", SF=SF
        )
        
        # TOD ray tracing
        R2 = TOD.mcrt_solver(R1, refl_error=False)
        
        # Optical efficiency calculations
        SF = bdr.optical_efficiencies(R2, SF, irradiance=Gbn, area_hel=A_h1)
        
        # Receiver simulation
        receiver = self.receiver
        if isinstance(receiver, spr.HPR2D):
            receiver.run_model(TOD, SF, R2)
        elif isinstance(receiver, spr.HPR0D):
            receiver.run_model(SF)
        else:
            raise ValueError(f"Receiver type '{self.rcv_type}' not recognized or not implemented.")

        # Extract results
        T_p = receiver.temps_parts
        N_hel = int(receiver.n_hels.gv("-"))
        Qstg = receiver.heat_stored.gv("MWh")
        M_p = receiver.mass_stg.gv("kg/s")
        eta_rcv = receiver.eta_rcv.gv("-")

        # Heliostat selection
        Gbn = self.Gbn.gv("W/m2")
        A_h1 = self.Ah1.gv("m2")
        Q_acc = SF['Q_h1'].cumsum()
        hlst = Q_acc.iloc[:N_hel].index
        SF['hel_in'] = SF.index.isin(hlst)
        R2['hel_in'] = R2['hel'].isin(hlst)
        
        # Update outputs
        self.sf_n_hels = Var(N_hel, "-")
        self.sf_power = Var(SF[SF["hel_in"]]['Q_h1'].sum(), "MW")
        self.rcv_power = Var(self.sf_power.v * eta_rcv, "MW")
        self.sf_area = Var(N_hel * A_h1, "m2")
        self.rcv_massflowrate = Var(M_p, "kg/s")
        
        # Update HB geometry
        HB.rmin = Var(R2[R2["hel_in"]]['rb'].quantile(0.0001), "m")
        HB.rmax = Var(R2[R2["hel_in"]]['rb'].quantile(0.9981), "m")
        R2['hit_hb'] = (R2['rb'] > HB.rmin.v) & (R2['rb'] < HB.rmax.v)
        
        HB.update_geometry(R2)
        HB.height_range()
        HB.calculate_mass(R2, SF, Gbn, A_h1)
        
        # Cost calculations
        self.costs_out = spr.plant_costing_calculations(self, SF)
        
        return R2, SF

    def eta_optical_hourly(
            self,
            df: pd.DataFrame,
            file_alt_azi: str = os.path.join(DIR_DATA, 'annual_perf', '1-Grid_AltAzi_vF.csv'),
            lbl: str = 'eta_SF'
        ) -> np.ndarray:
        """Calculate hourly optical efficiency from solar position data."""
        latitude = self.lat.gv("deg")
        
        # Load efficiency grid
        df_grid = pd.read_csv(file_alt_azi, header=0, index_col=0)
        df_grid.sort_values(by=['lat', 'alt', 'azi'], axis=0, inplace=True)
        lat = df_grid['lat'].unique()
        alt = df_grid['alt'].unique()
        azi = df_grid['azi'].unique()
        eta = np.array(df_grid[lbl]).reshape(len(lat), len(alt), len(azi))
        
        # Create interpolator
        f_int = RegularGridInterpolator(
            (lat, alt, azi),
            eta,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Apply to dataframe
        Npoints = len(df)
        lats = latitude * np.ones(Npoints)
        azis = df["azimuth"].to_numpy()
        azis = np.where(azis > 180, 360 - azis, azis)
        eles = df["elevation"].to_numpy()
        eta_SF = f_int(np.array([lats, eles, azis]).transpose())
        
        return eta_SF
    
    def eta_receiver_hourly(
            self,
            df: pd.DataFrame,
            func_receiver: Callable | None = None,
    ) -> pd.DataFrame:
        """Calculate hourly receiver efficiency."""
        if func_receiver is None:
            raise ValueError("Receiver efficiency function not provided.")
        
        sf_area = self.sf_area.gv("m2")
        temp_parts_avg = (self.stg_temp_cold.gv("K") + self.stg_temp_hot.gv("K")) / 2
        rcv_area = self.TOD.receiver_area.gv("m2")  # Use smart component
        Npoints = len(df)

        temp_parts = temp_parts_avg * np.ones(Npoints)
        temp_ambsK = df["temp_amb"].to_numpy()
        rcv_flux_avg = df["DNI"] * df["sf_eta"] * (sf_area / rcv_area) * 1e-6
        
        return func_receiver(np.array([temp_parts, temp_ambsK, rcv_flux_avg]).transpose())

    def eta_pb_hourly(
            self,
            df: pd.DataFrame,
            func_pb: Callable | None = None,
    ) -> pd.DataFrame:
        """Calculate hourly power block efficiency."""
        if func_pb is None:
            raise ValueError("Power block efficiency function not provided.")
            
        temp_pb_max = self.pb_temp_max.gv("K")
        Npoints = len(df)
        TambsC = df["temp_amb"].to_numpy() - 273.15
        T_maxC = (temp_pb_max - 273.15) * np.ones(Npoints)
        
        return func_pb(np.array([T_maxC, TambsC]).transpose())

    def run_simulation(self, verbose: bool = False) -> dict[str, Var|Array|float]:
        """
        Run complete plant simulation including thermal and annual performance.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary of simulation results
        """
        # Run thermal subsystem
        R2, SF = self.run_thermal_subsystem()
        
        # Annual performance parameters
        year_i = 2019
        year_f = 2019
        rcv_power = self.rcv_power
        Ntower = self.Ntower
        stg_cap = self.stg_cap
        solar_multiple = self.solar_multiple
        pb_eta_des = self.pb_eta_des

        latitude = self.lat.gv("deg")
        longitude = self.lng.gv("deg")

        # Load weather and price data
        dT = 0.5
        df_weather = load_weather_data(latitude, longitude, year_i, year_f, dT)
        df_sp = load_spotprice_data(self.state, year_i, year_f, dT)
        df = df_weather.merge(df_sp, how="inner", left_index=True, right_index=True)
        
        # Calculate power block parameters
        self.pb_power_th = Ntower * rcv_power / solar_multiple
        self.pb_power_el = pb_eta_des * self.pb_power_th
        self.storage_heat = self.pb_power_th * stg_cap

        # Run annual performance analysis
        self.out = annual_performance(self, df)
        return self.out


# ============================================================================
# UTILITY FUNCTIONS (unchanged from original)
# ============================================================================

def eta_optical(
        file_alt_azi: str,
        lbl: str = 'eta_SF'
    ) -> RegularGridInterpolator:
    """Create optical efficiency interpolator from file."""
    df = pd.read_csv(file_alt_azi, header=0, index_col=0)
    df.sort_values(by=['lat', 'alt', 'azi'], axis=0, inplace=True)
    lat = df['lat'].unique()
    alt = df['alt'].unique()
    azi = df['azi'].unique()
    eta = np.array(df[lbl]).reshape(len(lat), len(alt), len(azi))
    return RegularGridInterpolator(
        (lat, alt, azi),
        eta,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )


def eta_power_block(
        file_PB: str = os.path.join(DIR_DATA, "sCO2_cycle", "sCO2_OD_lookuptable.csv")
        ) -> RegularGridInterpolator:
    """Create power block efficiency interpolator."""
    df_sCO2 = pd.read_csv(file_PB, index_col=0)
    df_sCO2 = df_sCO2[df_sCO2["part_load_od"] == 1.0]
    
    lbl_x = 'T_htf_hot_des'
    lbl_y = 'T_amb_od'
    lbl_f = 'eta_thermal_net_less_cooling_od'
    df_sCO2.sort_values(by=[lbl_x, lbl_y], axis=0, inplace=True)
    X, Y = df_sCO2[lbl_x].unique(), df_sCO2[lbl_y].unique()
    F = np.array(df_sCO2[lbl_f]).reshape(len(X), len(Y))
    
    return RegularGridInterpolator(
        (X, Y),
        F,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )


def eta_HPR_0D() -> RegularGridInterpolator:
    """Create HPR receiver efficiency interpolator."""
    F_c = 5.
    air = Air()
    # Particle characteristics
    ab_p = 0.91    
    em_p = 0.75
    rho_b = 1810
    
    Tps = np.arange(600, 2001, 100, dtype='int64')
    Tambs = np.arange(5, 56, 5, dtype='int64') + 273.15
    qis = np.arange(0.25, 4.1, 0.5)
    eta_ths = np.zeros((len(Tps), len(Tambs), len(qis)))
    
    for (i, j, k) in [
        (i, j, k)
        for i in range(len(Tps))
        for j in range(len(Tambs))
        for k in range(len(qis))
    ]:
        Tp = Tps[i]
        T_amb = Tambs[j]
        qi = qis[k]
        
        Tamb = T_amb
        Tsky = Tamb - 15
        hcov = F_c * h_horizontal_surface_upper_hot(Tp, Tamb, 5.0, fluid=air)
        hrad = em_p * 5.67e-8 * (Tp**4 - Tsky**4) / (Tp - Tamb)
        hrc = hcov + hrad
        qloss = hrc * (Tp - Tamb)
        eta_th = (qi * 1e6 * ab_p - qloss) / (qi * 1e6)
        eta_ths[i, j, k] = eta_th
        
    return RegularGridInterpolator(
        (Tps, Tambs, qis),
        eta_ths,
        bounds_error=False,
        fill_value=np.nan
    )


def eta_TPR_0D(CSTo: dict) -> RegularGridInterpolator:
    """Create TPR receiver efficiency interpolator."""
    CST = CSTo.copy()
    dT = CST['T_pH'] - CST['T_pC']
    Tps = np.arange(600, 2001, 100, dtype='int64')
    Tambs = np.arange(5, 56, 5, dtype='int64') + 273.15
    Qavgs = np.arange(0.25, 4.1, 0.5)
    eta_ths = np.zeros((len(Tps), len(Tambs), len(Qavgs)))
    
    for (i, j, k) in [
        (i, j, k)
        for i in range(len(Tps))
        for j in range(len(Tambs))
        for k in range(len(Qavgs))
    ]:
        Tp = Tps[i]
        T_amb = Tambs[j]
        Qavg = Qavgs[k]
        CST['T_pC'] = Tp - dT/2
        CST['T_pH'] = Tp + dT/2
        CST['T_amb'] = T_amb
        CST['Qavg'] = Qavg
        eta_rcv = spr.initial_eta_rcv(CST)[0]
        eta_ths[i, j, k] = eta_rcv

    return RegularGridInterpolator(
        (Tps, Tambs, Qavgs),
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
    """Load and process weather data."""
    tz = 'Australia/Brisbane'
    if file_weather is None:
        file_weather = os.path.join(
            DIR_DATA, "weather", 'MERRA2_Processed_All.nc'
        )
    
    lat_v = lat
    lon_v = lng
    data_weather = xr.open_dataset(file_weather, engine="netcdf4")
    lats = np.array(data_weather.lat)
    lons = np.array(data_weather.lon)
    lon_a = lons[(abs(lons - lon_v)).argmin()]
    lat_a = lats[(abs(lats - lat_v)).argmin()]
    df_weather = data_weather.sel(lon=lon_a, lat=lat_a).to_dataframe()

    del data_weather  # Clean memory
    
    df_weather['DNI_dirint'] = df_weather['DNI_dirint'].fillna(0)
    df_weather.index = pd.to_datetime(df_weather.index).tz_localize('UTC')
    df_weather = df_weather.resample(f'{dT:.1f}h').interpolate()
    df_weather['DNI'] = df_weather['DNI_dirint']

    # Calculate solar position
    tz = 'Australia/Brisbane'
    df_weather.index = pd.to_datetime(df_weather.index).tz_convert(tz)
    sol_pos = Location(lat_v, lon_v, tz=tz).get_solarposition(df_weather.index)
    df_weather['azimuth'] = sol_pos['azimuth'].copy()
    df_weather['elevation'] = sol_pos['elevation'].copy()    

    year = pd.to_datetime(df_weather.index).year
    df_weather = df_weather[(year >= year_i) & (year <= year_f)]

    # Finish weather data
    df_weather = df_weather[['DNI', 'T2M', 'WS', 'azimuth', 'elevation']].copy()
    df_weather.rename(columns={'T2M': 'temp_amb'}, inplace=True)
    
    return df_weather


def load_spotprice_data(
        state: str = "NSW",
        year_i: int = 2019,
        year_f: int = 2019,
        dT: float = 0.5,
        file_data: str | None = None
) -> pd.DataFrame:
    """Load electricity spot price data."""
    if file_data is None:
        dir_spotprice = os.path.join(DIR_DATA, 'NEM_spotprice')
        file_data = os.path.join(dir_spotprice, 'NEM_TRADINGPRICE_2010-2020.PLK')
        
    df_SP = pd.read_pickle(file_data)
    df_sp_state = (
        df_SP[
        (df_SP.index.year >= year_i) & (df_SP.index.year <= year_f)
        ][['SP_' + state]]
    )
    df_sp_state.rename(
        columns={'Demand_' + state: 'Demand', 'SP_' + state: 'SP'},
        inplace=True
    )
    return df_sp_state


@dataclass
class PrimaryHeatExchanger:
    """Primary heat exchanger for particle-sCO2 heat transfer."""
    fluid = CO2()
    parts = Carbo()
    
    pb_power_th: Var = Var(20., "MW")
    temp_sco2_in: Var = Var(550 + 273.15, "K")
    temp_sco2_out: Var = Var(650 + 273.15, "K")
    temp_parts_in: Var = Var(950, "K")
    temp_parts_out: Var = Var(1200, "K")
    U: Var = Var(240., "W/m2-K")
    press_sco2: Var = Var(25e6, "Pa")
    n_pass: Var = Var(4, "-")
    n_tubes: Var = Var(400, "1/m2")
    d_tubes: Var = Var(0.03, "m")

    def __post_init__(self):
        # Outputs
        self.heat_exchanged: Var = Var(None, "MW")
        self.epsilon: Var = Var(None, "-")
        self.lmtd: Var = Var(None, "K")
        self.surface_area: Var = Var(None, "m2")
        self.volume: Var = Var(None, "m3")
        self.parts_mfr: Var = Var(None, "kg/s")
        self.sco2_mfr: Var = Var(None, "kg/s")

    @property
    def cost(self) -> Var:
        """Calculate heat exchanger cost."""
        self.run_model()
        T_ps = np.linspace(self.temp_parts_in.gv("K"), self.temp_parts_out.gv("K"), 10)
        f_HXTs = [
            1 if (T_pi < 550 + 273.) else float(1 + 5.4e-5 * (T_pi - 273.15 - 550)**2)
            for T_pi in T_ps
        ]
        return Var(
            float(np.mean(f_HXTs) * 1000 * self.surface_area.gv("m2") / self.pb_power_th.gv("MW")),
            "MUSD/MW"
        )

    def run_model(self) -> None:
        """Run heat exchanger model calculations."""
        # Tube and shell HX using log-mean temperature difference
        temp_sco2_in = self.temp_sco2_in
        temp_sco2_out = self.temp_sco2_out
        temp_parts_in = self.temp_parts_in
        temp_parts_out = self.temp_parts_out
        pb_power_th = self.pb_power_th
        U_HX = self.U
        press_sco2 = self.press_sco2
        Npass = int(self.n_pass.gv("-"))
        n_tubes = int(self.n_tubes.gv("1/m2"))
        d_tubes = self.d_tubes.gv("m")

        dT_sco2 = temp_sco2_out - temp_sco2_in
        temp_sco2_avg = (temp_sco2_out + temp_sco2_in) / 2
        cp_sco2 = self.fluid.cp(temp_sco2_avg, press_sco2)
        sco2_mfr = pb_power_th / (cp_sco2 * dT_sco2) / Npass
        sco2_C = sco2_mfr * cp_sco2
        heat_exchanged = sco2_C * dT_sco2
        
        dtemp_parts = temp_parts_in - temp_parts_out
        temp_parts_avg = (temp_parts_in + temp_parts_out) / 2
        parts_cp = self.parts.cp(temp_parts_avg)
        parts_C = heat_exchanged / dtemp_parts
        parts_mfr = parts_C / parts_cp
        
        lmtd = Var(
            float(
                ((temp_parts_out - temp_sco2_in) - (temp_parts_in - temp_sco2_out)).v
               / np.log(((temp_parts_out - temp_sco2_in) / (temp_parts_in - temp_sco2_out)).v)
            ),
            "K"
        )
        C_min = min(parts_C, sco2_C)
        C_max = max(parts_C, sco2_C)
        heat_max = C_max * (temp_parts_in - temp_sco2_in)
        epsilon = heat_exchanged / heat_max
        
        surface_area = heat_exchanged / (U_HX * lmtd)
        dAdV = Var(1, "1/m") * (np.pi * d_tubes) * n_tubes
        
        volume = surface_area / dAdV
        
        self.heat_exchanged = heat_exchanged.su("MW")
        self.epsilon = epsilon.su("-")
        self.lmtd = lmtd.su("K")
        self.surface_area = surface_area.su("m2")
        self.volume = volume.su("m3")
        self.parts_mfr = parts_mfr.su("kg/s")
        self.sco2_mfr = sco2_mfr.su("kg/s")


def dispatch_model_simple(
        df: pd.DataFrame,
        plant: PlantCSPBeamDownParticle,
    ) -> pd.DataFrame:
    """Simplified dispatch model for CSP plant with storage."""
    stg_cap_heat = plant.storage_heat.gv("MWh")
    pb_power_el = plant.pb_power_el.gv("MW")
    pb_power_th = plant.pb_power_th.gv("MW")
    stg_cap = plant.stg_cap.gv("hr")
    sf_area = plant.sf_area.gv("m2")
    Ntower = plant.Ntower
    DNI_min = plant.DNI_min.gv("W/m2")

    dT = 0.5  # [hr]
    
    # Calculate collected energy and potential dispatched energy
    df["rcv_op_fraction"] = np.where(df["DNI"] > DNI_min, 1, 0)
    df["sf_heat_to_rcv"] = Ntower * df["rcv_op_fraction"] * df["sf_eta"] * df["DNI"] * sf_area * CF("W", "MW").v * dT
    df['rcv_heat'] = df["sf_heat_to_rcv"] * df["rcv_eta"]
    df["rcv_heat"] = (
        Ntower * df["DNI"] * sf_area * CF("W", "MW").v * dT
         * df["sf_eta"] * df["rcv_eta"] * df["rcv_op_fraction"]
    )
    df['rcv_heat'] = df["rcv_heat"].fillna(0)

    # Power block requirements
    df["pb_heat_req"] = pb_power_th * dT
    df["energy_gen_pot"] = df['pb_eta'] * df["pb_heat_req"]
    df["revenue_gen_pot"] = df["energy_gen_pot"] * df['SP']
    
    # Day indexing (sunrise to sunrise)
    df["new_day"] = (df["elevation"] * df["elevation"].shift(1) < 0) & (df["elevation"] > 0)
    df.loc[df.index[0], "new_day"] = True
    df["day_index"] = df["new_day"].cumsum()
    df["day_index"] = df["day_index"].ffill()
    
    # Total heat delivered by receiver per day
    df["rcv_heat_collected"] = df.groupby(df["day_index"])['rcv_heat'].transform('sum')
    df["heat_in_excess"] = df["rcv_heat_collected"] - stg_cap_heat
    
    # Dispatch ranking and logic (simplified version)
    df["rank_dispatch_while_rcv_op"] = (
        df[df["rcv_heat"] > df["pb_heat_req"]]
        .groupby(df["day_index"])["revenue_gen_pot"]
        .rank('first', ascending=False)
    )
    
    # Simplified dispatch logic
    df["dispatch_while_rcv_op"] = np.where(
        (df["rank_dispatch_while_rcv_op"].notnull()) & (df["SP"] > 0), 1, 0
    )
    
    # Calculate final energy and revenue
    df["dispatch_all"] = df["dispatch_while_rcv_op"]
    df["heat_stored_in"] = df["rcv_op_fraction"] * df["rcv_heat"]
    df["heat_stored_out"] = df["dispatch_all"] * df["pb_heat_req"]
    df['heat_stored_net'] = df["heat_stored_in"] - df["heat_stored_out"]
    df['heat_stored_cum_total'] = df["heat_stored_net"].cumsum()
    df['heat_stored_cum_day'] = df.groupby(df["day_index"])['heat_stored_net'].cumsum()
    df['energy_gen_final'] = df["dispatch_all"] * df["energy_gen_pot"]
    df['revenue_final'] = df["energy_gen_final"] * df["SP"]
    
    return df


def calc_financial_metrics(
        revenue_tot: float,
        cost_capital: float,
        discount_rate: float,
        n_years: int,
        cost_om: float
    ) -> tuple[float, float, float]:
    """Calculate financial metrics including ROI, NPV, and payback period."""
    tpf = (1. / discount_rate) * (1. - 1. / (1. + discount_rate)**n_years)
    roi2 = tpf * revenue_tot / cost_capital
    npv = revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)
    roi = (revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)) / cost_capital
    
    def fNPV(i):
        tpf = (1. / discount_rate) * (1. - 1. / (1. + discount_rate)**i)
        return revenue_tot * tpf - cost_capital * (1. + cost_om * tpf)
    
    sol = spo.fsolve(fNPV, n_years, full_output=True)
    payback_period = sol[0][0]
    if sol[2] != 1:
        print(sol)

    return npv, roi, payback_period


def annual_performance(
        plant: PlantCSPBeamDownParticle,
        df: pd.DataFrame,
        ) -> dict[str, Var|Array|float]:
    """Calculate annual performance metrics for the CSP plant."""
    dT = 0.5
    rcv_power = plant.rcv_power.gv("MW")
    pb_power_el = plant.pb_power_el.gv("MW")
    Ntower = plant.Ntower
    pb_temp_max = plant.pb_temp_max.gv("K")

    # Calculate efficiencies on hourly basis
    df["date"] = pd.to_datetime(df.index).date
    Ndays = len(df["date"].unique())
    years = Ndays / 365
    
    df["sf_eta"] = plant.eta_optical_hourly(df)
    df["rcv_eta"] = plant.eta_receiver_hourly(df, eta_HPR_0D())
    df["pb_eta"] = plant.eta_pb_hourly(df, eta_power_block())
    
    df_out = dispatch_model_simple(df, plant)

    # Calculate annual metrics
    stg_heat_stored = Var(df_out["rcv_heat"].sum() / years, "MWh/yr")
    pb_energy_dispatched = Var(df_out["energy_gen_final"].sum() / years, "MWh/yr")
    revenue_tot = Var(df_out["revenue_final"].sum() / 1e6 / years, "MUSD/yr")
    revenue_daily = Var(df_out["revenue_final"].sum() / 1000 / Ndays, "kUSD/day")
    sf_cf = Var(df_out["rcv_heat"].sum() / (Ntower * rcv_power * dT * len(df_out)), "-")
    pb_cf = Var(df_out["energy_gen_final"].sum() / (pb_power_el * dT * len(df_out)), "-")
    sf_time_operation = Var(df_out["rcv_op_fraction"].sum() * dT / years, "-")
    pb_time_operation = Var(df_out["dispatch_all"].sum() * dT / years, "-")
    stg_min_stored = Var(df_out["heat_stored_cum_day"].min(), "MWh")
    stg_max_stored = Var(df_out["heat_stored_cum_day"].max(), "MWh")
    
    # Cost calculations
    USDtoAUD = 1.4
    costs_in = plant.cost_input
    costs_in['CF_sf'] = sf_cf.gv("-")
    costs_in['CF_pb'] = pb_cf.gv("-")
    
    pb_red_cost_factor = float(costs_in['F_pb'])
    file_pb = os.path.join(DIR_DATA, 'sCO2_cycle', 'sCO2_lookuptable_costs.csv')
    df_pb = pd.read_csv(file_pb, index_col=0)
    lbl_x = 'P_el'
    lbl_y = 'T_htf_hot_des'
    df_pb.sort_values(by=[lbl_x, lbl_y], axis=0, inplace=True)
    X = df_pb[lbl_x]
    Y = df_pb[lbl_y]
    XY = list(zip(X, Y))

    if costs_in['C_pb_est'] == 'Neises':
        lbl_f = 'cycle_spec_cost'
    elif costs_in['C_pb_est'] == 'Sandia':
        lbl_f = 'cycle_spec_cost_Sandia'
    else:
        raise ValueError('Invalid value for C_pb_est')
        
    f_pb = LinearNDInterpolator(XY, df_pb[lbl_f])
    pb_cost = pb_red_cost_factor * f_pb([pb_power_el, pb_temp_max - 273.15])[0]
    
    # Get sCO2 temperatures
    lbl_f = 'T_co2_PHX_in'
    f_sCO2_in = LinearNDInterpolator(XY, df_pb[lbl_f])
    temp_sco2_in = f_sCO2_in([pb_power_el, pb_temp_max - 273.15])[0]
    
    lbl_f = 'T_turb_in'
    f_sCO2_out = LinearNDInterpolator(XY, df_pb[lbl_f])
    temp_sco2_out = f_sCO2_out([pb_power_el, pb_temp_max - 273.15])[0]
    
    # Calculate PHX cost
    phx = PrimaryHeatExchanger(
        pb_power_th=plant.pb_power_th,
        temp_sco2_in=Var(temp_sco2_in + 273.15, "K"),
        temp_sco2_out=Var(temp_sco2_out + 273.15, "K"),
        temp_parts_in=plant.stg_temp_hot,
        temp_parts_out=plant.stg_temp_cold,
        U=plant.hx_u,
    )
    
    costs_in['C_pb'] = pb_cost * 1e3
    costs_in['C_PHX'] = phx.cost.gv("USD/MW")

    cost_capital = plant.costs_out["Elec"]
    discount_rate = float(costs_in['DR'])
    n_years = int(costs_in['Ny'])
    cost_om = float(costs_in['C_OM'])
    
    lcoh = spr.get_lcoh(
        plant.costs_out["Heat"], cost_om, rcv_power, sf_cf.v, discount_rate, n_years,
    ).su("USD/MWh")
    lcoe = spr.get_lcoe(
        plant.costs_out["Elec"], cost_om, pb_power_el, pb_cf.v, discount_rate, n_years,
    ).su("USD/MWh")
    npv, roi, payback_period = calc_financial_metrics(
        revenue_tot.gv("MUSD/yr"),
        cost_capital,
        discount_rate,
        n_years,
        cost_om
    )
    
    results = {
        "stg_min_stored": stg_min_stored,
        "stg_max_stored": stg_max_stored,
        "stg_heat_stored": stg_heat_stored,
        'pb_energy_dispatched': pb_energy_dispatched,
        'sf_cf': sf_cf,
        'pb_cf': pb_cf,
        'revenue_tot': revenue_tot,
        'revenue_daily': revenue_daily,
        'lcoh': lcoh,
        'lcoe': lcoe,
        'pb_cost': Var(pb_cost, "MUSD"),
        'cost_capital': Var(cost_capital, "MUSD"),
        'roi': Var(roi, "-"),
        'payback_period': Var(payback_period, "yr"),
        'npv': Var(npv, "MUSD"),
        'sf_time_operation': sf_time_operation,
        'pb_time_operation': pb_time_operation,
        'date_simulation': time.time(),
    }
    
    return results