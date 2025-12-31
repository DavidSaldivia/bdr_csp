from __future__ import annotations
from dataclasses import dataclass, field
from os.path import isfile
from typing import Callable, Union, Protocol, runtime_checkable
import json
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
from antupy.utils.props import Air, CO2, Carbo
from antupy.utils.htc import h_horizontal_surface_upper_hot

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

# ============================================================================
# PROTOCOLS FOR DATA LOADERS
# ============================================================================

@runtime_checkable
class WeatherLoader(Protocol):
    """Protocol for weather data loaders.
    
    Must return DataFrame with columns: DNI, temp_amb, WS, azimuth, elevation
    Index must be timezone-aware datetime.
    """
    lat: Var
    lng: Var
    year_i: int
    year_f: int
    dT: float
    
    def load_data(self) -> pd.DataFrame:
        """Load and return weather data as DataFrame."""
        ...

@runtime_checkable
class MarketLoader(Protocol):
    """Protocol for electricity market data loaders.
    
    Must return DataFrame with column: SP (spot price in local currency)
    Index must be timezone-aware datetime.
    """
    year_i: int
    year_f: int
    dT: float
    
    def load_data(self) -> pd.DataFrame:
        """Load and return market data as DataFrame."""
        ...

@dataclass
class ThermalSubsystemCSPPlant():
    """Thermal subsystem of CSP Beam Down Plant."""
    
    # Core plant parameters
    zf: Var = Var(50, "m")                        # Tower focal height

    # Solar field characteristics
    A_h1 = Var(2.92**2, "m2")

    # Receiver and storage characteristics
    material: Carbo = Carbo()
    zrc: Var = Var(10, "m")                       # Receiver center height
    rcv_type: str = 'TPR_2D'                     # Receiver model type
    rcv_flux_max: Var = Var(3.0, "MW/m2")       # Maximum flux on receiver
    stg_temp_cold: Var = Var(950, "K")          # Cold tank temperature
    stg_temp_hot: Var = Var(1200, "K")          # Hot tank temperature
    stg_thickness: Var = Var(0.05, "m")         # Particle layer thickness
    stg_h_to_d: Var = Var(0.5, "-")             # Storage tank height/diameter ratio
    
    lat: Var = Var(-23., "deg")                  # Latitude
    lng: Var = Var(115.9, "deg")                 # Longitude


    def __post_init__(self):
        # Output parameters

        self.pb_power_th = Var(None, "MW")   # Power block thermal input
        self.pb_power_el = Var(None, "MW")   # Power block electrical output
        self.storage_heat = Var(None, "MWh") # Storage capacity

        self.rcv_massflowrate = Var(None, "kg/s")  # Receiver mass flow rate

    @property
    def fzv(self) -> Var:
        return (Var(0.20 + 0.031*np.exp(0.774*self.zf.gv("m")), "m2"))
    
    @property
    def rcv_nom_power(self) -> Var:
        return (Var(0.57*self.zf.gv("m") - 9.7, "MW"))
    
    @property
    def flux_avg(self) -> Var:
        return (Var(-2.266 - 0.054*np.exp(-1.553*self.zf.gv("m")), "MW/m2"))

    @property
    def rcv_area(self) -> Var:
        return (Var(6.244 * np.exp(0.018*self.zf.gv("m")), "m2"))
    
    @property
    def hb_surface_area(self) -> Var:
        return (Var(226.901 * np.exp(0.033*self.zf.gv("m")), "m2"))
    
    @property
    def tod_surface_area(self) -> Var:
        return (Var(16.634 * np.exp(0.025*self.zf.gv("m")), "m2"))
    
    @property
    def land_surface(self) -> Var:
        return (Var(6.504 * np.exp(0.020*self.zf.gv("m")), "ha"))

    @property
    def n_hels(self) -> Var:
        return (Var(124.53 * self.zf.gv("m") - 1788.5, "-"))
    
    @property
    def rcv_eta(self) -> Var:
        return (Var(-0.223 - 0.056*np.exp(0.846*self.zf.gv("m")),"-"))
    
    @property
    def sf_eta(self) -> Var:
        return (Var(-0.175 - 0.038*np.exp(0.651*self.zf.gv("m")),"-"))
    
    @property
    def sf_area(self) -> Var:
        return (self.n_hels * self.A_h1)
    
    @property
    def sf_power(self) -> Var:
        return (self.rcv_nom_power / self.rcv_eta)
    
    @property
    def lcoh(self) -> Var:
        return (Var(23.52-0.058*np.exp(21.691*self.zf.gv("m")),"USD/MWh"))


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
        rcv_area = self.rcv_area.gv("m2")  # Use smart component
        Npoints = len(df)

        temp_parts = temp_parts_avg * np.ones(Npoints)
        temp_ambsK = df["temp_amb"].to_numpy()
        rcv_flux_avg = df["DNI"] * df["sf_eta"] * (sf_area / rcv_area) * 1e-6
        
        return func_receiver(np.array([temp_parts, temp_ambsK, rcv_flux_avg]).transpose())


@dataclass
class CO2PowerBlock():
    """Supercritical CO2 power block model."""
    
    # Core plant parameters
    zf: Var = Var(50, "m")                        # Tower focal height
    rcv_nom_power: Var = Var(19., "MW")          # Receiver thermal power
    stg_cap: Var = Var(8., "hr")                # Storage capacity
    solar_multiple: Var = Var(2.0, "-")         # Solar multiple
    n_tower: Var = Var(4, "-")                   # Number of towers

    temp_sco2_in: Var = Var(550 + 273.15, "K")      # sCO2 inlet temperature
    temp_sco2_out: Var = Var(650 + 273.15, "K")     # sCO2 outlet temperature
    press_sco2: Var = Var(25e6, "Pa")               # sCO2 operating pressure
    
    hx_u = Var(240., "W/m2-K")                  # Heat exchanger U-value
    pb_temp_max: Var = Var(875 + 273.15, "K")   # Max power block temperature
    pb_eta_des: Var = Var(0.50, "-")            # Power block design efficiency
    stg_eta_des: Var = Var(1.00, "-")           # Storage efficiency

    def __post_init__(self):
        # Outputs
        self.sco2_massflowrate: Var = Var(None, "kg/s")  # sCO2 mass flow rate

    @property
    def pb_power_th(self) -> Var:
        return (self.n_tower * self.rcv_nom_power / self.solar_multiple).su("MW")
    
    @property
    def pb_power_el(self) -> Var:
        return (self.pb_eta_des * self.pb_power_th).su("MW")
    
    @property
    def storage_heat(self) -> Var:
        return (self.pb_power_th * self.stg_cap).su("MWh")


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
        cost_per_area = Var(float(np.mean(f_HXTs) * 1000), "USD/m2")
        return (cost_per_area * self.surface_area / self.pb_power_th).su("MUSD/MW")

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

@dataclass
class WeatherMerra2():
    lat: Var = Var(-23., "deg")
    lng: Var = Var(115.9, "deg")
    year_i: int = 2019
    year_f: int = 2019
    dT: float = 0.5
    file_weather: str | None = None

    def load_data(self) -> pd.DataFrame:
        """Load and process weather data."""
        tz = 'Australia/Brisbane'
        if self.file_weather is None:
            self.file_weather = os.path.join(
                DIR_DATA, "weather", 'MERRA2_Processed_All.nc'
            )
        
        lat_v = self.lat.gv("deg")
        lon_v = self.lng.gv("deg")
        data_weather = xr.open_dataset(self.file_weather, engine="netcdf4")
        lats = np.array(data_weather.lat)
        lons = np.array(data_weather.lon)
        lon_a = lons[(abs(lons - lon_v)).argmin()]
        lat_a = lats[(abs(lats - lat_v)).argmin()]
        df_weather = data_weather.sel(lon=lon_a, lat=lat_a).to_dataframe()

        del data_weather  # Clean memory
        
        df_weather['DNI_dirint'] = df_weather['DNI_dirint'].fillna(0)
        df_weather.index = pd.to_datetime(df_weather.index).tz_localize('UTC')
        df_weather = df_weather.resample(f'{self.dT:.1f}h').interpolate()
        df_weather['DNI'] = df_weather['DNI_dirint']

        # Calculate solar position
        tz = 'Australia/Brisbane'
        df_weather.index = pd.to_datetime(df_weather.index).tz_convert(tz)
        sol_pos = Location(lat_v, lon_v, tz=tz).get_solarposition(df_weather.index)
        df_weather['azimuth'] = sol_pos['azimuth'].copy()
        df_weather['elevation'] = sol_pos['elevation'].copy()    

        year = pd.to_datetime(df_weather.index).year
        df_weather = df_weather[(year >= self.year_i) & (year <= self.year_f)]

        # Finish weather data
        df_weather = df_weather[['DNI', 'T2M', 'WS', 'azimuth', 'elevation']].copy()
        df_weather.rename(columns={'T2M': 'temp_amb'}, inplace=True)
        return df_weather

@dataclass
class MarketAU():
    state: str = "NSW"
    year_i: int = 2019
    year_f: int = 2019
    dT: float = 0.5
    file_data: str | None = None

    def load_data(
            self,
    ) -> pd.DataFrame:
        """Load electricity spot price data."""
        if self.file_data is None:
            dir_spotprice = os.path.join(DIR_DATA, 'energy_market', "nem")
            self.file_data = os.path.join(dir_spotprice, 'NEM_TRADINGPRICE_2010-2020.PLK')
            
        df_SP = pd.read_pickle(self.file_data)
        df_sp_state = (
            df_SP[
            (df_SP.index.year >= self.year_i) & (df_SP.index.year <= self.year_f)
            ][['SP_' + self.state]]
        )
        df_sp_state.rename(
            columns={'Demand_' + self.state: 'Demand', 'SP_' + self.state: 'spot_price'},
            inplace=True
        )
        return df_sp_state


# ============================================================================
# CHILEAN DATA LOADERS
# ============================================================================

@dataclass
class WeatherCL:
    """Chilean weather data loader using PSDA station data."""
    lat: Var = Var(-23.66, "deg")                # PSA/PSDA latitude
    lng: Var = Var(-70.40, "deg")                # PSA/PSDA longitude
    year_i: int = 2024
    year_f: int = 2024
    dT: float = 0.5
    file_weather: str | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load Chilean weather data with solar position calculation."""
        # Load raw weather data
        DIR_WEATHER = os.path.join(DIR_DATA, "weather", "psda_2024")
        
        vars_dict = {
            "temp_amb": {"file": "Tamb_2024.csv", "col_name": "AirTC"},
            "DNI": {"file": "DNI_2024.csv", "col_name": "DNI"}
        }
        
        df_weather = pd.DataFrame()
        for var_name, var_info in vars_dict.items():
            file_path = os.path.join(DIR_WEATHER, var_info["file"])
            if not isfile(file_path):
                raise FileNotFoundError(f"Weather data file not found: {file_path}")
            
            df_var = pd.read_csv(file_path, sep=";", index_col=0, header=0, skiprows=[1,2])
            df_var.rename(columns={f"{var_info['col_name']}_Avg": var_name}, inplace=True)
            df_var.drop(
                columns=[f"{var_info['col_name']}_{text}" for text in ["Min", "Max", "Std"]], 
                inplace=True
            )
            df_weather = pd.concat([df_weather, df_var], axis=1)
        
        # Add WS column (wind speed) - if not available, fill with nominal value
        if 'WS' not in df_weather.columns:
            df_weather['WS'] = 5.0  # nominal wind speed in m/s
        
        # Timezone handling - PSDA data is in Chilean time
        tz = 'America/Santiago'
        df_weather.index = pd.to_datetime(df_weather.index)
        
        # Localize to Chilean timezone if naive
        if df_weather.index.tz is None:
            try:
                df_weather.index = df_weather.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                try:
                    df_weather.index = df_weather.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
                except Exception:
                    df_weather.index = df_weather.index.tz_localize(tz)
        
        # Remove duplicates by averaging
        df_weather = df_weather.groupby(level=0).mean(numeric_only=True)
        
        # Resample to desired frequency
        df_weather = df_weather.resample(f'{self.dT:.1f}h').interpolate(method='time')
        
        # Calculate solar position
        lat_v = self.lat.gv("deg")
        lon_v = self.lng.gv("deg")
        sol_pos = Location(lat_v, lon_v, tz=tz).get_solarposition(df_weather.index)
        df_weather['azimuth'] = sol_pos['azimuth'].copy()
        df_weather['elevation'] = sol_pos['elevation'].copy()
        
        # Filter by year range
        year = pd.to_datetime(df_weather.index).year
        df_weather = df_weather[(year >= self.year_i) & (year <= self.year_f)]
        
        # Ensure temp_amb is in Kelvin
        df_weather['temp_amb'] = df_weather['temp_amb'] + 273.15
        
        return df_weather[['DNI', 'temp_amb', 'WS', 'azimuth', 'elevation']].copy()


@dataclass
class MarketCL:
    """Chilean electricity market data loader (SEN - Sistema Eléctrico Nacional)."""
    location: str = "crucero"  # Default barra
    year_i: int = 2024
    year_f: int = 2024
    dT: float = 0.5
    file_data: str | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load Chilean market spot price data."""
        if self.file_data is None:
            # Try JSON format first (datos-de-costos-marginales-en-linea.tsv)
            json_path = os.path.join(DIR_DATA, "energy_market", "sen", "datos-costos-marginales.tsv")
            
            # Fallback to CSV format
            csv_path = os.path.join(DIR_DATA, "energy_market", "sen", f"{self.year_i}_{self.location}.csv")
            
            if isfile(json_path):
                self.file_data = json_path
                return self._load_json_format()
            elif isfile(csv_path):
                self.file_data = csv_path
                return self._load_csv_format()
            else:
                raise FileNotFoundError(
                    f"Market data not found. Tried:\n  {json_path}\n  {csv_path}"
                )
        
        # Determine format from file extension
        if self.file_data.endswith('.tsv') or self.file_data.endswith('.json'):
            return self._load_json_format()
        else:
            return self._load_csv_format()
    
    def _load_json_format(self) -> pd.DataFrame:
        """Load market data from JSON format."""
        with open(self.file_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data, columns=["fecha", "barra", "cmg"])
        df["barra"] = df["barra"].str.lower()
        df["date"] = pd.to_datetime(df["fecha"])
        
        # Filter by location and year
        df_filtered = df[
            (df["barra"] == self.location.lower()) & 
            (df["date"].dt.year >= self.year_i) & 
            (df["date"].dt.year <= self.year_f)
        ].copy()
        
        df_filtered = df_filtered.rename(columns={"cmg": "spot_price"})
        df_filtered = df_filtered[["date", "spot_price"]].set_index("date")
        
        # Localize to Chilean timezone
        tz = 'America/Santiago'
        if df_filtered.index.tz is None:
            try:
                df_filtered.index = df_filtered.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                try:
                    df_filtered.index = df_filtered.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
                except Exception:
                    df_filtered.index = df_filtered.index.tz_localize(tz)
        
        # Resample to match dT
        df_filtered = df_filtered.resample(f'{self.dT:.1f}h').interpolate(method='time')
        
        return df_filtered
    
    def _load_csv_format(self) -> pd.DataFrame:
        """Load market data from CSV format."""
        df_market = pd.read_csv(str(self.file_data), sep=",", index_col=3, header=0)
        df_market.drop(["fecha", "hora"], inplace=True, axis=1, errors='ignore')
        df_market.index = pd.to_datetime(df_market.index)
        
        # Localize to Chilean timezone
        tz = 'America/Santiago'
        if df_market.index.tz is None:
            try:
                df_market.index = df_market.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                try:
                    df_market.index = df_market.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
                except Exception:
                    df_market.index = df_market.index.tz_localize(tz)
        
        # Filter by year
        df_market = df_market[
            (df_market.index.year >= self.year_i) & 
            (df_market.index.year <= self.year_f)
        ]
        
        return df_market


@dataclass
class CSPDesignPoint():
    # Environmental conditions
    Gbn: Var = Var(950, "W/m2")                  # Design-point DNI
    temp_amb: Var = Var(300., "K")               # Ambient temperature
    day: int = 80                                 # Design-point day
    omega: Var = Var(0.0, "rad")                 # Design-point hour angle


@dataclass
class ModularCSPPlant(Plant):
    """Enhanced CSP Beam Down Receiver Plant with smart component management."""
    # Core plant parameters
    zf: Var = Var(50, "m")                        # Tower focal height
    stg_cap: Var = Var(8., "hr")                # Storage capacity
    solar_multiple: Var = Var(2.0, "-")         # Solar multiple
    n_tower: Var = Var(4, "-")                   # Number of towers

    # Economic parameters
    cost_input: dict[str, float|str] = field(default_factory=spr.get_plant_costs)

    design_point: CSPDesignPoint = field(default_factory=CSPDesignPoint)
    weather: WeatherLoader = field(default_factory=WeatherMerra2)
    market: MarketLoader = field(default_factory=MarketAU)

    DNI_min: Var = Var(400., "W/m2")             # Minimum DNI for operation

    # Output dictionary
    out: dict[str, Var|Array|float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived parameters and output variables."""
        self.costs_out: dict[str, float] = {}     # Cost calculation outputs

        self.thermal_subsystem = ThermalSubsystemCSPPlant(
            zf=self.zf,
        )
        self.power_subsystem = CO2PowerBlock(
            zf=self.zf,
            stg_cap=self.stg_cap,
            solar_multiple=self.solar_multiple,
            n_tower=self.n_tower,
        )

    def run_costing_calculations(self) -> dict[str, Var]:
        cp = self.thermal_subsystem.material.cp()
        M_p = (
            self.thermal_subsystem.rcv_nom_power / (
                cp * (self.thermal_subsystem.stg_temp_hot - self.thermal_subsystem.stg_temp_cold)
            )
        )
        costing_params = {
        "zrc": self.thermal_subsystem.zrc,
        "A_h1": self.thermal_subsystem.A_h1,
        "T_stg": self.stg_cap,
        "SM": self.solar_multiple,
        "pb_eta": self.power_subsystem.pb_eta_des,
        "T_pH": self.thermal_subsystem.stg_temp_hot,
        "T_pC": self.thermal_subsystem.stg_temp_cold,
        "HtD_stg": self.thermal_subsystem.stg_h_to_d,
        "Ntower": self.n_tower,
        "material": self.thermal_subsystem.material,
        "costs_in": self.cost_input,
        "Arcv": self.thermal_subsystem.rcv_area,
        "P_rcv": self.thermal_subsystem.rcv_nom_power,
        "S_HB" : self.thermal_subsystem.hb_surface_area,
        "zmax" : self.zf*1.1,
        "S_TOD": self.thermal_subsystem.tod_surface_area,
        "M_p": M_p,
        "M_HB_fin": self.thermal_subsystem.hb_surface_area*Var(15, "kg/m2"),
        "M_HB_t": self.thermal_subsystem.hb_surface_area*Var(30, "kg/m2"),
        "S_land" : self.thermal_subsystem.land_surface,
        "S_hel" : self.thermal_subsystem.sf_area,
    }
        return spr.plant_costing_calculations(costing_params=costing_params)
    
    def annual_performance(
        self,
        df: pd.DataFrame,
        ) -> dict[str, Var|Array|float|pd.DataFrame]:
        """Calculate annual performance metrics for the CSP plant."""
        dT = 0.5
        rcv_nom_power = self.thermal_subsystem.rcv_nom_power.gv("MW")
        pb_power_el = self.power_subsystem.pb_power_el.gv("MW")
        n_tower = self.n_tower.gv("-")

        # Calculate efficiencies on hourly basis
        df["date"] = pd.to_datetime(df.index).date
        Ndays = len(df["date"].unique())
        years = Ndays / 365
        
        df["sf_eta"] = self.thermal_subsystem.eta_optical_hourly(df)
        df["rcv_eta"] = self.thermal_subsystem.eta_receiver_hourly(df, eta_HPR_0D())
        df["pb_eta"] = self.power_subsystem.eta_pb_hourly(df, eta_power_block())
        
        df_out = dispatch_model(df, self)

        # Calculate annual metrics
        stg_heat_stored = Var(
            df_out["rcv_heat"].sum()/years, "MWh/yr"
        )                                           # annual energy stored per tower
        pb_energy_dispatched = Var(
            df_out["energy_gen_final"].sum()/years, "MWh/yr"
        )                                           # annual energy dispatched per power block
        revenue_tot = Var(
            df_out["revenue_final"].sum()/years, "USD/yr"
        )    # total annual revenue per power block
        revenue_daily = Var(
            df_out["revenue_final"].sum()/Ndays, "USD/day"
        )   #daily revenue
        sf_cf = Var(
            df_out["rcv_heat"].sum() / (n_tower * rcv_nom_power * dT * len(df_out)), "-"
        )
        pb_cf = Var(
            df_out["energy_gen_final"].sum() / (pb_power_el * dT * len(df_out)), "-"
        )
        sf_time_operation = Var(
            df_out["rcv_op_fraction"].sum()*dT / years, "-"
        )                #hrs
        pb_time_operation = Var(
            df_out["dispatch_all"].sum()*dT/years, "-"
        )        #hrs
        stg_min_stored = Var(
            df_out["heat_stored_cum_day"].min(), "MWh"
        )  # minimum energy stored in the storage system
        stg_max_stored = Var(
            df_out["heat_stored_cum_day"].max(), "MWh"
        )  # maximum energy stored in the storage system
        return {
            "stg_heat_stored": stg_heat_stored,
            'pb_energy_dispatched':pb_energy_dispatched,
            'sf_cf':sf_cf,
            'pb_cf':pb_cf,
            'revenue_tot':revenue_tot,
            'revenue_daily':revenue_daily,
            'sf_time_operation': sf_time_operation,
            'pb_time_operation': pb_time_operation,
            'stg_min_stored': stg_min_stored,
            'stg_max_stored': stg_max_stored,
        }

    def financial_calculations(
            self,
            out_performance: dict[str, Var|Array|float|pd.DataFrame],
        ) -> dict[str, Var|float]:
        # Cost calculations
        sf_cf: Var = out_performance["sf_cf"]
        pb_cf: Var = out_performance["pb_cf"]
        revenue_tot: Var = out_performance["revenue_tot"]
        pb_power_el = self.power_subsystem.pb_power_el.gv("MW")
        pb_temp_max = self.power_subsystem.pb_temp_max.gv("K")

        costs_in = self.cost_input
        costs_in['CF_sf'] = sf_cf.gv("-")
        costs_in['CF_pb'] = pb_cf.gv("-")
        
        discount_rate = float(costs_in['DR'])
        n_years = int(costs_in['Ny'])
        cost_om = float(costs_in['C_OM'])

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
            pb_power_th=self.power_subsystem.pb_power_th,
            temp_sco2_in=Var(temp_sco2_in + 273.15, "K"),
            temp_sco2_out=Var(temp_sco2_out + 273.15, "K"),
            temp_parts_in=self.thermal_subsystem.stg_temp_hot,
            temp_parts_out=self.thermal_subsystem.stg_temp_cold,
            U=self.power_subsystem.hx_u,
        )
        costs_in['C_pb'] = pb_cost * 1e3
        costs_in['C_PHX'] = phx.cost.gv("USD/MW")
        self.cost_input = costs_in
        costs_out = self.run_costing_calculations()
        cost_capital = costs_out["Elec"]
        lcoh = costs_out["LCOH"]
        lcoe = costs_out["LCOE"]

        npv, roi, payback_period = calc_financial_metrics(
            revenue_tot.gv("MUSD/yr"),
            cost_capital.gv("MUSD"),
            discount_rate,
            n_years,
            cost_om
        )
        results = {
            'lcoh': lcoh,
            'lcoe': lcoe,
            'pb_cost': Var(pb_cost, "MUSD"),
            'cost_capital': cost_capital,
            'roi': Var(roi, "-"),
            'payback_period': Var(payback_period, "yr"),
            'npv': Var(npv, "MUSD"),
            'date_simulation': time.time(),
        }
        return results

    def run_simulation(self, verbose: bool = False) -> dict[str, Var|Array|float]:
        """
        Run complete plant simulation including thermal and annual performance.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Dictionary of simulation results
        """

        # Load weather and price data
        df_weather = self.weather.load_data()
        df_sp = self.market.load_data()
        df = df_weather.merge(df_sp, how="inner", left_index=True, right_index=True)

        # Run annual performance analysis
        out_performance = self.annual_performance(df)
        out_financial = self.financial_calculations(out_performance)

        self.out = {**out_performance, **out_financial}
        return self.out



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



def dispatch_model(
        df: pd.DataFrame,
        plant: ModularCSPPlant,
    ) -> pd.DataFrame:


    stg_cap_heat = plant.power_subsystem.storage_heat.gv("MWh")
    pb_power_th = plant.power_subsystem.pb_power_th.gv("MW")
    sf_area = plant.thermal_subsystem.sf_area.gv("m2")
    n_tower = plant.n_tower.gv("-")
    DNI_min = plant.DNI_min.gv("W/m2")
    dT = plant.market.dT

    #Calculations for Collected Energy and potential Dispatched energy
    # Is the Solar Field operating?
    df["rcv_op_fraction"]  = np.where( df["DNI"]>DNI_min, 1, 0)
    df["sf_heat_to_rcv"]   = (
        n_tower * df["rcv_op_fraction"] 
        * df["sf_eta"] * df["DNI"] * sf_area
        * CF("W", "MW").v * dT
    )   #[MWh]
    df['rcv_heat']  = df["sf_heat_to_rcv"] * df["rcv_eta"]                       #[MWh]
    df['rcv_heat'] = df["rcv_heat"].fillna(0)

    #It is assumed the same amount of energy is extracted from storage,
    # so particle-sCO2 HX always try to work at design capacity.
    # The energy dispatched is lower then.
    df["pb_heat_req"] = pb_power_th * dT                        #[MWh_th]
    df["energy_gen_pot"] = df['pb_eta'] * df["pb_heat_req"]     #[MWh_e]
    df["revenue_gen_pot"] = df["energy_gen_pot"] * df["spot_price"]     #[USD]
    
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
            (df["pb_heat_req_cum_while_rcv_op"] < df["heat_in_excess"]) & (df["spot_price"]>0) ,
            1,
            0
        )
    )
    df["stg_stored_while_rcv_op"] = (
        df["rcv_op_fraction"] * df["rcv_heat"] - df["pb_heat_req"] * df["dispatch_while_rcv_op"]
    )
    df["stg_stored_cum_while_rcv_op"] =(
        df["stg_stored_while_rcv_op"]
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
                & (df["spot_price"]>0)
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
        df[(df["dispatch_both"]<=0) & (df["spot_price"]>0)]
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
    df['revenue_final'] = df["energy_gen_final"] * df["spot_price"]                 # [USD]
    return df


def dispatch_model_simple(
        df: pd.DataFrame,
        plant: ModularCSPPlant,
    ) -> pd.DataFrame:
    """Simplified dispatch model for CSP plant with storage."""
    stg_cap_heat = plant.storage_heat.gv("MWh")
    pb_power_el = plant.pb_power_el.gv("MW")
    pb_power_th = plant.pb_power_th.gv("MW")
    stg_cap = plant.stg_cap.gv("hr")
    sf_area = plant.thermal_subsystem.sf_area.gv("m2")
    n_tower = plant.n_tower.gv("-")
    DNI_min = plant.DNI_min.gv("W/m2")

    dT = 0.5  # [hr]
    
    # Calculate collected energy and potential dispatched energy
    df["rcv_op_fraction"] = np.where(df["DNI"] > DNI_min, 1, 0)
    df["sf_heat_to_rcv"] = n_tower * df["rcv_op_fraction"] * df["sf_eta"] * df["DNI"] * sf_area * CF("W", "MW").v * dT
    df['rcv_heat'] = df["sf_heat_to_rcv"] * df["rcv_eta"]
    df["rcv_heat"] = (
        n_tower * df["DNI"] * sf_area * CF("W", "MW").v * dT
         * df["sf_eta"] * df["rcv_eta"] * df["rcv_op_fraction"]
    )
    df['rcv_heat'] = df["rcv_heat"].fillna(0)

    # Power block requirements
    df["pb_heat_req"] = pb_power_th * dT
    df["energy_gen_pot"] = df['pb_eta'] * df["pb_heat_req"]
    df["revenue_gen_pot"] = df["energy_gen_pot"] * df["spot_price"]
    
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
        (df["rank_dispatch_while_rcv_op"].notnull()) & (df["spot_price"] > 0), 1, 0
    )
    
    # Calculate final energy and revenue
    df["dispatch_all"] = df["dispatch_while_rcv_op"]
    df["heat_stored_in"] = df["rcv_op_fraction"] * df["rcv_heat"]
    df["heat_stored_out"] = df["dispatch_all"] * df["pb_heat_req"]
    df['heat_stored_net'] = df["heat_stored_in"] - df["heat_stored_out"]
    df['heat_stored_cum_total'] = df["heat_stored_net"].cumsum()
    df['heat_stored_cum_day'] = df.groupby(df["day_index"])['heat_stored_net'].cumsum()
    df['energy_gen_final'] = df["dispatch_all"] * df["energy_gen_pot"]
    df['revenue_final'] = df["energy_gen_final"] * df["spot_price"]
    
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