from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, Any, TypeAlias

import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy.interpolate as spi

from antupy import Var, Array, Frame, SimulationOutput, component, constraint, derived
from antupy.utils.props import Air, Carbo
from antupy.utils import htc

from bdr_csp import bdr as bdr
import bdr_csp.htc

from bdr_csp.dir import DIRECTORY

from bdr_csp.bdr import (
    SolarField,
    HyperboloidMirror,
    TertiaryOpticalDevice
)

if TYPE_CHECKING:
    from bdr_csp.pb import ModularCSPPlant

DIR_DATA = DIRECTORY.DIR_DATA

air = Air()
COLS_INPUT = [
    'location',
    'temp_stg',
    'solar_multiple',
    'receiver_power'
]

COLS_OUTPUT = [
    'pb_power_th',
    'receiver_temp_output',
    'sf_n_hels',
    'receiver_rad_flux_max', 
    'receiver_rad_flux_avg',
    'receiver_eta',
    'stg_heat',
    'stg_mass',
    'stg_res_time',
    'stg_vel_flow',
    'date_simulation'
]


@dataclass
class CSPBeamDownParticlePlant:
    """CSP Beam-Down Particle-Receiver Plant."""

    zf: Var = Var(50, "m")
    fzv: Var = Var(0.818161, "-")
    rcv_nom_power: Var = Var(19.,"MW")
    flux_avg: Var = Var(1.25,"MW/m2")
    flux_max: Var = Var(3.0, "MW/m2")

    stg_capacity: Var = Var(8.0, "hr")
    solar_multiple: Var = Var(2.0, "-")
    Ntower: Var = Var(1, "-")

    # Design conditions
    Gbn: Var = Var(950, "W/m2")
    DNI_min: Var = Var(400, "W/m2")
    day: int = 80
    omega: Var = Var(0.0, "rad")
    lat: Var = Var(-23., "deg")
    lng: Var = Var(115.9, "deg")
    temp_amb: Var = Var(300., "K")
    type_shadow: str = "simple"
    file_weather = os.path.join(
        DIR_DATA, 'weather', "Alice_Springs_Real2000_Created20130430.csv"
    )
    
    # Efficiency targets
    rcv_eta_des : Var = Var(0.83, "-")
    pb_eta_des: Var = Var(0.50, "-")
    eta_stg: Var = Var(0.95, "-")
    
    # BDR and Tower characteristics
    xrc: Var = Var(0., "m")
    yrc: Var = Var(0., "m")
    zrc: Var = Var(10., "m")
    
    # TOD and receiver characteristics
    geometry: str = 'PB'
    array: str = 'A'
    Cg: Var = Var(2, "-")
    rcv_type: str = "HPR0D"  # Options: "HPR0D", "HPR2D", "TPR2D"
    thickness_part: Var = Var(0.05, "m")

    #Storage
    stg_temp_cold: Var = Var(950., "K")
    stg_temp_hot: Var = Var(1200., "K")
    stg_h_to_d: Var = Var(0.5, "-")

    #Economic
    costs_in: dict = field(default_factory=dict)
    
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.zv = self.fzv * self.zf
        self.receiver_area: Var = (
            self.rcv_nom_power / self.rcv_eta_des / self.flux_avg
        ).su("m2")

    # Create BDR components
    @property
    def HSF(self) -> SolarField:
        def _file_SF(zf:Var)-> str:
            return os.path.join(
                DIR_DATA,
                'mcrt_datasets_final',
                'Dataset_zf_{:.0f}'.format(self.zf.gv("m"))
            )
        return component(
            SolarField(
                zf=self.zf,
                A_h1=Var(2.92**2, "m2"),
                N_pan=Var(1,"-"),
                file_SF=derived(_file_SF, self.zf),
                )
            )
    
    @property
    def HB(self) -> HyperboloidMirror:
        return component(
            HyperboloidMirror(
                zf=constraint(self.zf),
                fzv=constraint(self.fzv),
                xrc=constraint(self.xrc),
                yrc=constraint(self.yrc),
                zrc=constraint(self.zrc),
            )
        )
    
    @property
    def TOD(self) -> TertiaryOpticalDevice:
        return component(
            TertiaryOpticalDevice(
                geometry=constraint(self.geometry),
                array=constraint(self.array),
                Cg=constraint(self.Cg),
                receiver_area=constraint(self.receiver_area),
                xrc=constraint(self.xrc),
                yrc=constraint(self.yrc),
                zrc=constraint(self.zrc)
            )
        )
    
    @property
    def receiver(self) -> ParticleReceiver:
        if self.rcv_type == "HPR0D":
            receiver = HPR0D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part
            )
        elif self.rcv_type == "HPR2D":
            receiver = HPR2D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part,
                tod=self.TOD,
            )
        elif self.rcv_type == "TPR0D":
            receiver = TPR0D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part,
                tod = self.TOD
            )
        elif self.rcv_type == "TPR2D":
            receiver = TPR2D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part,
                tod = self.TOD
            )
        else:
            raise ValueError(f"Invalid receiver type: {self.rcv_type}.")
        return component(receiver)
    
    def run_optic_simulation(
                self,
                testing: bool = False,
                verbose: bool = True
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        hsf = self.HSF
        hb = self.HB
        tod = self.TOD

        if verbose:
            print("Getting the RayDataset")
        R0, SF = hsf.load_dataset(save_plk=True)

        if testing:
            R0 = R0.sample(n=1000, random_state=1).reset_index(drop=True)

        if verbose:
            print("Getting interceptions with HB")
        R1, SF = hb.run_model(
            R0, SF, lat=self.lat, lng=self.lng, refl_error=True
        )

        if verbose:
            print("Getting interceptions with TOD")
        R2 = tod.mcrt_solver(R1, refl_error=True)

        ### Optical Efficiencies
        SF = bdr.optical_efficiencies(
            R2, SF,
            irradiance=self.Gbn,
            area_hel=hsf.A_h1,
            reflectivity=hsf.eta_rfl
        )
        return R2, SF
    
    def run_costing_calculations(self,
        result_output: dict[str, float],
        SF: pd.DataFrame
    ) -> dict[str, float]:
        costing_params = {
        "zrc": self.zrc,
        "A_h1": self.HSF.A_h1,
        "T_stg": self.stg_capacity,
        "SM": self.solar_multiple,
        "pb_eta": self.pb_eta_des,
        "T_pH": self.stg_temp_hot,
        "T_pC": self.stg_temp_cold,
        "HtD_stg": self.stg_h_to_d,
        "Gbn": self.Gbn,
        "DNI_min": self.DNI_min,
        "Ntower": self.Ntower,
        "material": self.receiver.material,
        "file_weather": self.file_weather,
        "costs_in": self.costs_in,
        "S_HB" : result_output["hb_surface_area"],
        "zmax" : result_output["hb_zmax"],
        "S_TOD": result_output["tod_surface_area"],
        "Arcv": result_output["receiver_area"],
        "M_p": result_output["mass_stg"],
        "P_rcv": result_output["rcv_power_sim"],
        "M_HB_fin": result_output["hb_mass_fin"],
        "M_HB_t": result_output["hb_mass_total"],
        "S_land" : get_land_area(SF, self.HSF.A_h1)[0],
        "S_hel" : get_land_area(SF, self.HSF.A_h1)[1],
    }
        return plant_costing_calculations(costing_params=costing_params)

    def run_simulation(self, verbose: bool = True, testing: bool = False) -> SimulationOutput:
        
        receiver = self.receiver

        if verbose:
            print("Running optical simulation...")
        R2, SF = self.run_optic_simulation(testing=testing, verbose=verbose)

        if verbose:
            print("Running receiver simulation...")
        if isinstance(receiver, (HPR0D, TPR0D)):
            rcvr_output = receiver.run_model(SF)
        elif isinstance(receiver, (HPR2D, TPR2D)):
            rcvr_output = receiver.run_model(SF,R2)
        else:
            raise ValueError(f"Receiver type '{self.rcv_type}' not recognized.")

        if isinstance(rcvr_output["n_hels"], Var):
            N_hel = rcvr_output["n_hels"]
        else:
            raise ValueError("Receiver output for n_hels is not valid.")
        if isinstance(rcvr_output["eta_rcv"], Var):
            eta_rcv = rcvr_output["eta_rcv"]
        else:
            raise ValueError("Receiver output for eta_rcv is not valid.")
        if isinstance(rcvr_output["temps_parts"], Array):
            temp_parts = rcvr_output["temps_parts"]
            temp_parts_avg = Var(temp_parts.v.mean(), "K")
            temp_parts_max = Var(temp_parts.v.max(), "K")
        else:
            raise ValueError("Receiver output for temps_parts is not valid.")

        #Plant Parameters
        rcv_nom_power = receiver.rcv_nom_power
        eta_pb = self.pb_eta_des
        solar_multiple = self.solar_multiple
        A_h1 = self.HSF.A_h1
        
        # Heliostat selection
        Q_acc    = SF['Q_h1'].cumsum()
        hlst     = Q_acc.iloc[:int(N_hel.v)].index
        SF['hel_in'] = SF.index.isin(hlst)
        R2['hel_in'] = R2['hel'].isin(hlst)
        sf_power_sim  = Var(SF[SF["hel_in"]]['Q_h1'].sum(), "MW")
        rcv_power_sim = sf_power_sim * eta_rcv
        pb_power_sim  = rcv_power_sim * eta_pb
        
        # Calculating HB surface
        hb = self.HB
        tod = self.TOD
        hb.rmin = Var(R2[R2["hel_in"]]['rb'].quantile(0.0001), "m")
        hb.rmax = Var(R2[R2["hel_in"]]['rb'].quantile(0.9981), "m")
        R2['hit_hb'] = (R2['rb']>hb.rmin.v)&(R2['rb']<hb.rmax.v)
        hb.update_geometry(R2)
        hb.height_range()
        M_HB_fin, M_HB_mirr, M_HB_str, M_HB_tot = bdr.HB_mass_cooling(hb, R2, SF)

        hb.mass_fin = Var(M_HB_fin, "ton")
        hb.mass_mirror = Var(M_HB_mirr, "ton")
        hb.mass_structure = Var(M_HB_str, "ton")
        hb.mass_total = Var(M_HB_tot, "ton")
        
        Etas = SF[SF['hel_in']].mean()
        hlst = SF[SF["hel_in"]].index
        eta_SF = Var(Etas['Eta_SF'], "-")
        eta_BDR = Var(Etas['Eta_BDR'], "-")
        eta_TOD = Var(Etas['Eta_tdi']*Etas['Eta_tdr'], "-")
        eta_StH = eta_SF * eta_rcv

        self.sf_n_hels = N_hel                    # Number of heliostats
        self.sf_area = N_hel*A_h1                     # Solar Field area
        self.sf_power = sf_power_sim                     # Solar Field power (output)
        self.pb_power_th = rcv_power_sim        # Power Block thermal power (input)
        self.pb_power_el = pb_power_sim        # Power Block electrical power (output)
        self.storage_heat = rcvr_output["heat_stored"]      # Storage heat (output)
        self.rcv_massflowrate = rcvr_output["mass_stg"]  # Receiver mass flow rate (to be calculated)

        # Outputs
        results_output = {
            "R2": R2,
            "SF": SF,
            "temp_parts": temp_parts,
            "temp_part_max": temp_parts_max,
            "temp_part_avg": temp_parts_avg,
            "eta_rcv": eta_rcv,
            "eta_sf": eta_SF,
            "eta_bdr": eta_BDR,
            "eta_tod": eta_TOD,
            "eta_StH": eta_StH,
            "rad_flux_max": rcvr_output["rad_flux_max"],
            "rad_flux_avg": rcvr_output["rad_flux_avg"],
            "time_res": rcvr_output["time_res"],
            "vel_parts": rcvr_output["vel_parts"],
            "mass_stg": rcvr_output["mass_stg"],
            "heat_stored": rcvr_output["heat_stored"],
            "n_hels": N_hel,
            "pb_power_th": rcv_nom_power/solar_multiple,
            "pb_power_el": (rcv_nom_power/solar_multiple)*eta_pb,
            "sf_power": rcv_nom_power/eta_rcv,
            "sf_power_sim": sf_power_sim.su("MW"),
            "rcv_power_sim": rcv_power_sim.su("MW"),
            "pb_power_sim": pb_power_sim.su("MW"),
            "hb_rmin": hb.rmin,
            "hb_rmax": hb.rmax,
            "hb_zmin": hb.zmin,
            "hb_zmax": hb.zmax,
            "hb_surface_area": hb.surface_area,
            "tod_height": tod.height,
            "tod_surface_area": tod.surface_area,
            "receiver_area": tod.receiver_area,
            "sf_surface_area": N_hel*A_h1,
            "total_surface_area": hb.surface_area + tod.surface_area + N_hel*A_h1,
            "hb_zmax": hb.zmax,
            "hb_mass_fin": hb.mass_fin,
            "hb_mass_total": hb.mass_total,
        }
        
        if verbose:
            print("Running costing calculations...")
        costs_out = self.run_costing_calculations(result_output=results_output, SF=SF)
        results_output["costs_out"] = costs_out
        results_output["lcoh"] = costs_out['LCOH']
        results_output["lcoe"] = costs_out['LCOE']
        results_output["land_prod"] = costs_out['land_prod']
        results_output["stg_vol"] = costs_out['V_stg']
        results_output["stg_height"] = costs_out['H_stg']

        return results_output

class ReceiverOutput(TypedDict):
    temps_parts: Array
    temps_diff: Var
    n_hels: Var
    rad_flux_max: Var
    rad_flux_avg: Var
    power_sf_i: Var
    heat_stored: Var
    rcv_power_in: Var
    eta_rcv: Var
    mass_stg: Var
    time_res: Var
    vel_parts: Var
    iteration: int | None
    solve_t_res: bool
    full: dict[str, Frame|pd.DataFrame] | None


@dataclass
class HPR0D():
    rcv_nom_power: Var = Var(10., "MW")      # Initial target for Receiver nominal power
    heat_flux_max: Var = Var(3.0, "MW/m2")     # Maximum radiation flux on receiver
    heat_flux_avg: Var = Var(0.5, "MW/m2")     # Average radiation flux on receiver (initial guess)
    temp_ini: Var = Var(950, "K")          # Particle temperature in cold tank
    temp_out: Var = Var(1200, "K")         # Particle temperature in hot tank
    material: Carbo = Carbo()                        # [-] Thermal Storage Material
    thickness_parts: Var = Var(0.05, "m")  # Thickness of material on conveyor belt
    factor_htc: Var = Var(2.57, "-")       # factor to magnified the HTC

    def __post_init__(self):
        self.rcv_power_in: Var = Var(None, "MW")
        self.temps_parts: Array = Array(None, "K")
        self.n_hels: Var = Var(None, "-")
        self.rad_flux_max: Var = Var(None, "MW/m2")
        self.rad_flux_avg: Var = Var(None, "MW/m2")
        self.heat_stored: Var = Var(None, "MWh")
        self.eta_rcv: Var = Var(None, "-")
        self.mass_stg: Var = Var(None, "kg/s")
        self.time_res: Var = Var(None, "s")
        self.vel_parts: Var = Var(None, "m/s")

    def run_model(self, SF) -> ReceiverOutput:
    
        Prcv = self.rcv_nom_power
        Qavg = self.heat_flux_avg
        Tin = self.temp_ini
        Tout = self.temp_out
        tz = self.thickness_parts
        Fc = self.factor_htc
        material = self.material
        
        air= Air()
        Tp = 0.5*(Tin+Tout)
        eta_rcv = _HTM_0D_blackbox(Tp.gv("K"), Qavg.gv("MW/m2"), Fc=Fc.gv("-"), air=air)[0]
        
        rho_b = material.rho()
        cp = material.cp(Tp)
        t_res = rho_b * cp * tz * (Tout - Tin ) / (Qavg*eta_rcv)
        m_p = Prcv / (cp*(Tout-Tin))
        P_bdr = Prcv / eta_rcv

        Arcv  = P_bdr / Qavg
        Ltot  = Var(Arcv.gv("m2")**0.5, "m")
        vel_p = Ltot / t_res
        Q_acc    = SF['Q_h1'].cumsum()

        # Output
        self.rcv_power_in = P_bdr.su("MW")
        self.temps_parts = Array([Tp], "K")
        self.n_hels =  Var( len( Q_acc[ Q_acc < P_bdr.gv("MW") ] ) + 1, "-" )
        self.rad_flux_max = Var( np.nan, "MW/m2" )
        self.rad_flux_avg = P_bdr/Arcv.su("MW/m2")
        self.heat_stored = Var( np.nan, "MWh" )
        self.eta_rcv = Var( eta_rcv, "-" )
        self.mass_stg = m_p.su("kg/s")
        self.time_res = t_res.su("s" )
        self.vel_parts = vel_p.su("m/s" )

        rcvr_output: ReceiverOutput = {
            "temps_parts": self.temps_parts,
            "temps_diff": Var(Tout.v - Tin.v, "K"),
            "rad_flux_max": self.heat_flux_max,
            "rad_flux_avg": self.heat_flux_avg,
            "power_sf_i": Var(np.nan, "MW"),
            "rcv_power_in": self.rcv_power_in,
            "heat_stored": self.heat_stored,
            "eta_rcv": self.eta_rcv,
            "n_hels" : self.n_hels,
            "mass_stg": self.mass_stg,
            "time_res": self.time_res,
            "vel_parts": self.vel_parts,
            "iteration": None,
            "solve_t_res": False,
            "full": None,
        }
        return rcvr_output


@dataclass
class HPR2D():
    rcv_nom_power: Var = Var(10., "MW")      # Initial target for Receiver nominal power
    heat_flux_max: Var = Var(3.0, "MW/m2")     # Maximum radiation flux on receiver
    heat_flux_avg: Var = Var(0.5, "MW/m2")     # Average radiation flux on receiver (initial guess)
    temp_ini: Var = Var(950, "K")          # Particle temperature in cold tank
    temp_out: Var = Var(1200, "K")         # Particle temperature in hot tank
    material: Carbo = Carbo()        # [-] Thermal Storage Material
    thickness_parts: Var = Var(0.05, "m")  # Thickness of material on conveyor belt
    factor_htc: Var = Var(2.57, "-")       # factor to magnified the HTC
    tod: TertiaryOpticalDevice | None = None

    x_fin: Var = Var(None, "m")
    x_ini: Var = Var(None, "m")
    y_bot: Var = Var(None, "m")
    y_top: Var = Var(None, "m")

    def __post_init__(self):
        # Output
        self.rcr_power_in: Var = Var(None, "MW")
        self.temps_parts = Array([], "degC")
        self.n_hels: Var = Var(None, "-")
        self.rad_flux_max: Var = Var(None, "MW/m2")
        self.rad_flux_avg: Var = Var(None, "MW/m2")
        self.heat_stored: Var = Var(None, "MWh")
        self.eta_rcv: Var = Var(None, "-")
        self.mass_stg: Var = Var(None, "kg/s")
        self.time_res: Var = Var(None, "s")
        self.vel_parts: Var = Var(None, "m/s")
        self.time_sim: Var = Var(None, "m/s")



    def get_dimensions(self, tod_index: int = 0) -> None:
        if self.tod is None:
            raise ValueError("No TOD provided")
        xO,yO = self.tod.perimeter_points(self.tod.radius_out.v, tod_index=tod_index)
        self.x_fin = xO.min()
        self.x_ini = xO.max()
        self.y_bot = yO.min()
        self.y_top = yO.max()
        self.area = self.tod.receiver_area
        return None


    def run_model(
            self,
            SF: pd.DataFrame,
            R2: pd.DataFrame,
            polygon_i: int = 1,
            full_output: bool = True
        ) -> ReceiverOutput:
        
        if self.tod is None:
            raise ValueError("No TOD provided")

        def func_temp_out_avg(t_res_g,*args):
            rcvr_input = args[0]
            vel_p  = rcvr_input["x_avg"]/t_res_g[0]
            T_p, _, _, _ = _HTM_2D_moving_particles(rcvr_input, vel_p)
            return T_p.gv("K").mean()-temp_out
        
        temp_ini = self.temp_ini
        temp_out = self.temp_out
        thickness_parts = self.thickness_parts
        material = self.material
        power_rcv = self.rcv_nom_power
        heat_flux_avg = self.heat_flux_avg
        receiver_area = self.tod.receiver_area

        #Parameters for receiver
        xO,yO = self.tod.perimeter_points(self.tod.radius_out.v, tod_index=polygon_i-1)
        lims = [Var(xO.min(),"m"), Var(xO.max(),"m"), Var(yO.min(),"m"), Var(yO.max(),"m")]
        x_fin = lims[0]
        x_ini = lims[1]
        y_bot = lims[2]
        y_top = lims[3]
        area_rcv_1 = self.tod.receiver_area/self.tod.n_tods
        x_avg = area_rcv_1/(y_top-y_bot)
        func_eta = _get_func_eta()

        rcvr_input: dict[str, Var|spi.RegularGridInterpolator|None|Carbo] = {
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
            "func_eta": func_eta,
            "func_heat_flux_rcv1": None,
        }

        #Initial guess for number of heliostats
        temps_parts_avg = ( temp_ini + temp_out ) / 2
        eta_rcv = Var(func_eta([ temps_parts_avg.gv("K"), heat_flux_avg.gv("MW/m2") ])[0],"-")
        Q_acc = SF['Q_h1'].cumsum()

        power_sf_in = (power_rcv / eta_rcv)
        N_hel = len( Q_acc[ Q_acc < power_sf_in.gv("MW") ] ) + 1

        R2a = R2[(R2["hit_rcv"])&(R2["Npolygon"]==polygon_i)].copy()  #Rays into 1 receiver
        R2_all = R2[(R2["hit_rcv"])].copy()                           #Total Rays into receivers
        
        rho_b = material.rho()
        cp = material.cp(temps_parts_avg)
        
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
            P_SF_it = Var(SF_it["Q_pen"].sum(),"MW")         #Value only valid in iteration it
            eta_SF_it = Var(Etas_it['Eta_SF'], "-")       #Value only valid in iteration it
            
            #Getting the rays dataset and the total energy it should represent
            rays = R2a[R2a["hel"].isin(hlst)].copy()
            r_i = len(rays)/len(R2_all[R2_all["hel"].isin(hlst)])  #Fraction of rays that goes into one TOD
            P_SF_i = r_i * P_SF_it
            
            func_heat_flux_rcv1,heat_flux_rcv1,_,_ = _get_func_heat_flux_rcv1(
                rays['xr'], rays['yr'], lims, P_SF_i
            )
            rcvr_input["func_heat_flux_rcv1"] = func_heat_flux_rcv1

            #initial estimates for residence time and receiver efficiency
            if it==1:
                heat_flux_avg = P_SF_it / receiver_area
                eta_rcv_g = Var(func_eta([temps_parts_avg.gv("K"),heat_flux_avg.gv("MW/m2")])[0], "-")
            else:      
                eta_rcv_g = eta_rcv
            
            if solve_t_res:
                t_res_g = (
                    rho_b * cp * thickness_parts * (temp_out - temp_ini )
                    / (eta_rcv_g * heat_flux_avg )
                )
                vel_p  = x_avg/t_res_g                #Belt velocity magnitud
                inputs = (rcvr_input)
                sol  = spo.fsolve(func_temp_out_avg, t_res_g, args=inputs, full_output=True)
                t_res = Var(sol[0][0], "s")
                n_evals = sol[1]['nfev']
            else:
                t_res = (
                    rho_b * cp * thickness_parts * (temp_out - temp_ini ) 
                    / (eta_rcv_g * heat_flux_avg * 1e6 )
                )
                
            # Final values for temperature, heat_stored_1 and mass_stg_1
            vel_p  = x_avg/t_res                    #Belt velocity magnitud
            temps_parts, heat_stored_1, mass_stg_1, Rcvr_full = _HTM_2D_moving_particles(
                rcvr_input, vel_p, full_output=full_output
            )
            
            #Performance parameters
            eta_rcv = heat_stored_1*1e6/P_SF_i
            heat_stored = heat_stored_1 / r_i
            mass_stg = mass_stg_1 / r_i
            heat_flux_avg = P_SF_it / receiver_area
            heat_flux_max = heat_flux_rcv1.max()/1000.
            temp_parts_avg = Var(temps_parts.gv("K").mean(), "K")


            P_SF  = power_rcv / eta_rcv
            Q_acc = SF['Q_h1'].cumsum()
            N_new = len( Q_acc[ Q_acc < P_SF ] ) + 1
            
            data.append([receiver_area, N_hel, heat_flux_max, heat_flux_avg,
                        P_SF_i, heat_stored, P_SF_it, eta_rcv,
                        eta_SF_it, mass_stg, t_res, temp_parts_avg])
            print('\t'.join('{:.3f}'.format(x) for x in data[-1]))
            
            conv = (
                abs((heat_stored-power_rcv).gv("MW"))<0.1
                and (abs((temp_parts_avg-temp_out).gv("K"))<1.)
            )
            
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

        self.rcr_power_in: Var = P_SF_it.su("MW")
        self.temps_parts = temps_parts.su("K")
        self.n_hels: Var = Var(N_hel, "-")
        self.rad_flux_max: Var = Var(heat_flux_max, "MW/m2")
        self.rad_flux_avg: Var = heat_flux_avg.su("MW/m2")
        self.heat_stored: Var = heat_stored.su("MWh")
        self.eta_rcv: Var = eta_rcv.su("-")
        self.mass_stg: Var = mass_stg.su("kg/s")
        self.time_res: Var = t_res.su("m/s")
        self.vel_p: Var = vel_p.su("m/s")

        rcvr_output: ReceiverOutput = {
            "temps_parts": temps_parts,
            "temps_diff": Var(temps_parts.gv("K").max()-temps_parts.gv("K").min(), "K"),
            "n_hels" : self.n_hels,
            "rad_flux_max": self.rad_flux_max,
            "rad_flux_avg": self.heat_flux_avg,
            "power_sf_i": P_SF_i,
            "heat_stored": heat_stored,
            "rcv_power_in": P_SF_it,
            "eta_rcv": eta_rcv,
            "mass_stg": mass_stg,
            "time_res": t_res,
            "vel_parts": vel_p,
            "iteration": it,
            "solve_t_res": solve_t_res,
            "full": Rcvr_full
        }
        return rcvr_output


@dataclass
class TPR0D():
    """Tilted Particle Receiver 2D model implementation."""
    
    rcv_nom_power: Var = Var(10., "MW")      # Initial target for Receiver nominal power
    heat_flux_max: Var = Var(3.0, "MW/m2")  # Maximum radiation flux on receiver
    heat_flux_avg: Var = Var(0.5, "MW/m2")  # Average radiation flux on receiver (initial guess)
    temp_ini: Var = Var(950, "K")           # Particle temperature in cold tank
    temp_out: Var = Var(1200, "K")          # Particle temperature in hot tank
    temp_amb: Var = Var(300., "K")          # Ambient temperature
    material: Carbo = Carbo()                # Thermal Storage Material
    thickness_parts: Var = Var(0.06, "m")   # Thickness of material on conveyor belt
    factor_htc: Var = Var(2.57, "-")        # Factor to magnify the HTC
    model_htc: str = 'NellisKlein'
    
    # TPR specific parameters
    wall_temp: Var = Var(800., "K")         # Wall temperature
    tilt_angle: Var = Var(-27., "deg")      # Tilt angle (negative for tilted down)
    absorptivity: Var = Var(0.91, "-")     # Particle absorptivity
    emissivity_part: Var = Var(0.85, "-")  # Particle emissivity
    emissivity_wall: Var = Var(0.20, "-")  # Wall emissivity
    conductivity: Var = Var(0.833, "W/m2-K") # Conduction coefficient
    view_factor_avg: Var = Var(0.4, "-")   # Average view factor

    tod: TertiaryOpticalDevice | None = None
    
    
    def __post_init__(self):

        # Receiver dimensions (calculated from TOD)
        if self.tod is not None:
            self.x_ini: Var = Var(self.tod.perimeter_points(self.tod.radius_out.v)[0].max(),"m")
            self.x_fin: Var = Var(self.tod.perimeter_points(self.tod.radius_out.v)[0].min(),"m")
            self.y_bot: Var = Var(self.tod.perimeter_points(self.tod.radius_out.v)[1].min(),"m")
            self.y_top: Var = Var(self.tod.perimeter_points(self.tod.radius_out.v)[1].max(),"m")
        else:
            self.x_ini: Var = Var(None, "m")
            self.x_fin: Var = Var(None, "m")
            self.y_bot: Var = Var(None, "m")
            self.y_top: Var = Var(None, "m")
        self.x_avg: Var = Var(None, "m")

        # Output variables
        self.rcv_power_in: Var = Var(None, "MW")
        self.temps_parts = np.array([])
        self.n_hels: Var = Var(None, "-")
        self.rad_flux_max: Var = Var(None, "MW/m2")
        self.rad_flux_avg: Var = Var(None, "MW/m2")
        self.heat_stored: Var = Var(None, "MWh")
        self.eta_rcv: Var = Var(None, "-")
        self.mass_stg: Var = Var(None, "kg/s")
        self.time_res: Var = Var(None, "s")
        self.vel_parts: Var = Var(None, "m/s")
        
        # Internal parameters
        self._air = Air()
    
    def run_model(
        self,
        SF: pd.DataFrame | None = None,
        corr: bool = True,
    ) -> ReceiverOutput:

        Fv    = self.view_factor_avg
        T_ini = self.temp_ini
        T_out = self.temp_out
        tz = self.thickness_parts
        Q_avg = self.heat_flux_avg
        Prcv = self.rcv_nom_power
        if self.tod is not None:
            Arcv = self.tod.receiver_area / self.tod.n_tods
        else:
            raise ValueError("receiver.tod is not provided")
        self.get_dimensions(tod_index=0)
        y_top = self.y_top
        y_bot = self.y_bot
        Tp_avg = (T_ini+T_out)/2.
        material = self.material

        eta_rcv = _eta_th_tilted(self, Tp_avg, Q_avg, Fv=Fv)[0]

        if isinstance(material, Carbo):
            rho_b = material.rho()
            cp = material.cp(Tp_avg)
        else:
            raise ValueError(f"Material {material} not implemented")
        t_res = rho_b * cp * tz * (T_out - T_ini ) / (Q_avg*eta_rcv)
        M_p = Prcv / (cp*(T_out - T_ini))
        P_bdr = Prcv / eta_rcv
        Ltot  = Arcv/(y_top-y_bot)
        vel_p = Ltot / t_res

        #Correcting 0D model with linear fit
        if corr:
            m = 0.6895
            n = 0.2466
        else:
            m,n=1.0,0.0
        eta_rcv_corr = Var(eta_rcv.v*m+n, "-")

        if SF is None:
            n_hels = Var(np.nan, "-")
        else:
            Q_acc = SF['Q_h1'].cumsum()
            n_hels = Var(len( Q_acc[ Q_acc < P_bdr ] ) + 1, "-")
        
        rcvr_output: ReceiverOutput = {
            "temps_parts": Array([Tp_avg.v], "K"),
            "temps_diff": T_out-T_ini,
            "n_hels" : n_hels,
            "rad_flux_max": Var(np.nan, "-"),
            "rad_flux_avg": P_bdr/Arcv,
            "power_sf_i": Var(np.nan, "MW"),
            "heat_stored": Var(np.nan, "MWh"),
            "rcv_power_in": P_bdr,
            "eta_rcv": eta_rcv_corr,
            "mass_stg": M_p,
            "time_res": t_res,
            "vel_parts": vel_p,
            "iteration": None,
            "solve_t_res": False,
            "full": None
        }
        return rcvr_output

    def get_dimensions(self, tod_index: int = 0) -> None:
        """Get receiver dimensions from TOD geometry."""
        if self.tod is None:
            raise ValueError("No TOD provided")
        xO, yO = self.tod.perimeter_points(self.tod.radius_out.v, tod_index=tod_index)
        self.x_fin = Var(xO.min(), "m")
        self.x_ini = Var(xO.max(), "m")
        self.y_bot = Var(yO.min(), "m")
        self.y_top = Var(yO.max(), "m")
        self.area = self.tod.receiver_area
        return None

@dataclass
class TPR2D():
    """Tilted Particle Receiver 2D model implementation."""
    
    rcv_nom_power: Var = Var(10., "MW")      # Initial target for Receiver nominal power
    heat_flux_max: Var = Var(3.0, "MW/m2")  # Maximum radiation flux on receiver
    heat_flux_avg: Var = Var(0.5, "MW/m2")  # Average radiation flux on receiver (initial guess)
    temp_ini: Var = Var(950, "K")           # Particle temperature in cold tank
    temp_out: Var = Var(1200, "K")          # Particle temperature in hot tank
    temp_amb: Var = Var(300., "K")          # Ambient temperature
    material: Carbo = Carbo()                # Thermal Storage Material
    thickness_parts: Var = Var(0.06, "m")   # Thickness of material on conveyor belt
    factor_htc: Var = Var(2.57, "-")        # Factor to magnify the HTC
    model_htc: str = 'NellisKlein'
    
    # TPR specific parameters
    wall_temp: Var = Var(800., "K")         # Wall temperature
    tilt_angle: Var = Var(-27., "deg")      # Tilt angle (negative for tilted down)
    absorptivity: Var = Var(0.91, "-")     # Particle absorptivity
    emissivity_part: Var = Var(0.85, "-")  # Particle emissivity
    emissivity_wall: Var = Var(0.20, "-")  # Wall emissivity
    conductivity: Var = Var(0.833, "W/m2-K") # Conduction coefficient
    view_factor_avg: Var = Var(0.4, "-")   # Average view factor

    tod: TertiaryOpticalDevice | None = None
    
    
    def __post_init__(self):

        # Receiver dimensions (calculated from TOD)
        self.x_ini: Var = Var(None, "m")
        self.x_fin: Var = Var(None, "m")
        self.y_bot: Var = Var(None, "m")
        self.y_top: Var = Var(None, "m")
        self.x_avg: Var = Var(None, "m")

        # Output variables
        self.rcv_power_in: Var = Var(None, "MW")
        self.temps_parts = np.array([])
        self.n_hels: Var = Var(None, "-")
        self.rad_flux_max: Var = Var(None, "MW/m2")
        self.rad_flux_avg: Var = Var(None, "MW/m2")
        self.heat_stored: Var = Var(None, "MWh")
        self.eta_rcv: Var = Var(None, "-")
        self.mass_stg: Var = Var(None, "kg/s")
        self.time_res: Var = Var(None, "s")
        self.vel_parts: Var = Var(None, "m/s")
        
        # Internal parameters
        self._air = Air()
    
    
    def run_model(
        self,
        SF: pd.DataFrame,
        R2: pd.DataFrame,
        Gbn: Var = Var(950., "W/m2"),
        A_h1: Var = Var(2.92**2, "m2"),
        polygon_i: int = 1,
        full_output: bool = True
    ) -> ReceiverOutput:
        """
        Run the complete TPR 2D model simulation.
        
        Args:
            TOD: Tertiary Optical Device
            SF: Solar field DataFrame
            R2: Ray tracing results DataFrame
            CST: Configuration dictionary
            polygon_i: Polygon index for receiver
            full_output: Whether to return detailed results
            
        Returns:
            Dictionary with simulation results
        """
        temp_ini = self.temp_ini
        temp_out = self.temp_out
        thickness_parts = self.thickness_parts
        power_rcv = self.rcv_nom_power
        heat_flux_avg = self.heat_flux_avg
        
        # Get receiver dimensions
        if self.x_ini is None:
            self.get_dimensions(polygon_i-1)
        
        # Process tilted rays
        R3, xp, yp, zp = self._process_tilted_rays(R2, polygon_i)
        R2_all = R2[R2["hit_rcv"]].copy()
        
        # Initial guess for number of heliostats
        T_av = (temp_ini + temp_out) / 2
        eta_rcv = Var(
            _HTM_0D_blackbox(
                T_av.gv("K"), heat_flux_avg.gv("MW/m2"), Fc=self.factor_htc.gv("-"), air=self._air
            )[0],
            "-"
        )
        Q_acc = SF['Q_h1'].cumsum()
        N_hel = Var(len(Q_acc[Q_acc < (power_rcv / eta_rcv).gv("MW")]) + 1, "-")
        
        # Material properties
        rho_b = self.material.rho()

        # Iteration parameters
        it_max = 20
        it = 1
        loop = 0
        data = []
        N_hels: list[Var] = []
        solve_t_res = False
        Loop_break = False
        T_w = Var(None, "K")
        cp = Var(None, "J/kg-K")
        Q_av = Var(None, "MW/m2")
        
        # Main iteration loop
        while True:
            # Calculate heat flux function
            f_Qrc1, Q_max, lims = self._calculate_heat_flux_function(
                R3, R2, SF, N_hel, polygon_i, Gbn, A_h1
            )
            
            # Update receiver dimensions from corrected limits
            x_fin, x_ini, y_bot, y_top = lims
            if self.tod is None:
                raise ValueError("No TOD given")
            A_rc1 = self.tod.receiver_area / self.tod.n_tods
            xavg = A_rc1 / (y_top - y_bot)
            
            # Get view factor function
            f_ViewFactor = self._get_view_factor_function(polygon_i, lims, xp, yp, zp)
            
            #TODO This needs to be updated
            Fv = self.view_factor_avg  # Use average for now
            
            # Calculate wall temperature
            if it == 1:
                T_pavg = (temp_ini + temp_out) / 2
                beta = self.tilt_angle
                args = (T_pavg, x_ini, x_fin, y_top, y_bot, beta, Fv,
                       self.emissivity_part, self.emissivity_wall,
                       A_rc1, self.conductivity)
                try:
                    res = spo.fsolve(_getting_T_w, self.wall_temp.gv("K"), args=args)[0]
                    T_w = Var(res, "K")
                except:
                    raise RuntimeError(f"Wall temperature not found {self.wall_temp}")
            
            # Update receiver conditions
            self.x_ini = x_ini
            self.x_fin = x_fin
            self.y_bot = y_bot
            self.y_top = y_top
            self.wall_temp = T_w
            self.xavg = xavg
            
            # Calculate residence time
            hlst = SF.iloc[:int(N_hel.v)].index
            SF_it = SF.loc[hlst]
            P_SF_it = Var(SF_it["Q_pen"].sum(), "MW")
            receiver_area = self.tod.receiver_area
            
            if it == 1:
                Q_av = P_SF_it / receiver_area
                cp = self.material.cp(T_av)
                eta_rcv_g = _eta_th_tilted(self, T_av, heat_flux_avg, Fv=Fv)[0]
            else:
                eta_rcv_g = eta_rcv
            
            if solve_t_res:
                t_res_g = rho_b * cp * thickness_parts * (temp_out - temp_ini) / (eta_rcv_g * Q_av)
                args = (self, f_Qrc1, _eta_th_tilted, f_ViewFactor)
                try:
                    sol = spo.fsolve(_tpr2d_func_temp_out_avg, t_res_g.gv("s"), args=args, xtol=1e-2, full_output=True)
                    t_res = Var(sol[0][0], "s")
                except:
                    t_res = t_res_g
            else:
                t_res_g = rho_b * cp * thickness_parts * (temp_out - temp_ini) / (eta_rcv_g * Q_av)
                t_res = t_res_g
            
            # Final simulation
            vel_p = xavg / t_res
            t_sim = (x_ini - x_fin) / vel_p
            
            self.vel_parts = vel_p
            self.time_sim = t_sim
            
            T_p, Qstg1, Mstg1, Rcvr_full = _HTM_2D_tilted_surface(
                self, vel_p, t_sim, f_Qrc1, _eta_th_tilted, f_ViewFactor, full_output=full_output
            )
            
            # Calculate performance parameters
            rays = R3[R3["hel"].isin(hlst)].copy()
            r_i = len(rays) / len(R2_all[R2_all["hel"].isin(hlst)]) if len(R2_all[R2_all["hel"].isin(hlst)]) > 0 else 1.0
            P_SF_i = r_i * P_SF_it
            
            eta_rcv = Qstg1 / P_SF_i if P_SF_i > Var(0,"MW") else Var(0, "MW")
            Qstg = Qstg1 / r_i
            Mstg = Mstg1 / r_i
            Q_av = P_SF_it / receiver_area
            
            # Check convergence
            P_SF = power_rcv / eta_rcv if eta_rcv.v > 0 else np.inf
            Q_acc = SF['Q_h1'].cumsum()
            N_new = Var(len(Q_acc[Q_acc < P_SF]) + 1, "-")
            
            conv = (
                abs((Qstg - power_rcv).gv("MW")) < 0.1
                and abs(T_p.gv("K").mean() - temp_out.gv("K")) < 1.0
            )
            
            if (N_new == N_hel) and conv:
                break
            
            if Loop_break:
                break
            
            if loop == 5:
                N_new = Var((N_hel + N_new).v // 2, "-")
                Loop_break = True
            
            if N_new in N_hels:
                solve_t_res = True
                N_new = Var((N_hel + N_new).v // 2, "-")
                if N_new in N_hels:
                    N_new = N_new + Var(1,"-")
                loop += 1
            
            if (it == it_max) and conv and not solve_t_res:
                it_max = 40
                solve_t_res = True
            
            if it == it_max and solve_t_res:
                break
            
            N_hel = N_new
            N_hels.append(N_new)
            it += 1
        
        # Store results
        self.rcv_power_in = P_SF_it.su("MW")
        self.temps_parts = T_p.su("K")
        self.n_hels = N_hel.su("-")
        self.rad_flux_max = Q_max.su("MW/m2")
        self.rad_flux_avg = Q_av.su("MW/m2")
        self.heat_stored = Qstg.su("MW")
        self.eta_rcv = eta_rcv.su("-")
        self.mass_stg = Mstg.su("kg/s")
        self.time_res = t_res.su("s")
        self.vel_parts = vel_p.su("m/s")
        
        # Return results
        rcvr_output: ReceiverOutput = {
            "temps_parts": T_p,
            "temps_diff": Var(T_p.gv("K").max()-T_p.gv("K").min(), "K"),
            "n_hels": self.n_hels,
            "rad_flux_max": self.rad_flux_max,
            "rad_flux_avg": self.rad_flux_avg,
            "power_sf_i": P_SF_i,
            "heat_stored": Qstg,
            "rcv_power_in": P_SF_it,
            "eta_rcv": eta_rcv,
            "mass_stg": Mstg,
            "time_res": t_res,
            "vel_parts": vel_p,
            "iteration": it,
            "solve_t_res": solve_t_res,
            "full": Rcvr_full
        }
        
        return rcvr_output

    def get_dimensions(self, tod_index: int = 0) -> None:
        """Get receiver dimensions from TOD geometry."""
        if self.tod is None:
            raise ValueError("No TOD provided")
        xO, yO = self.tod.perimeter_points(self.tod.radius_out.v, tod_index=tod_index)
        self.x_fin = Var(xO.min(), "m")
        self.x_ini = Var(xO.max(), "m")
        self.y_bot = Var(yO.min(), "m")
        self.y_top = Var(yO.max(), "m")
        self.area = self.tod.receiver_area
        return None
    
    def _get_view_factor_function(
            self,
            polygon_i: int,
            lims: Array,
            xp: float,
            yp: float,
            zp: float
    ) -> spi.RegularGridInterpolator:
        """Create view factor interpolation function."""
        if self.tod is None:
            raise ValueError("No TOD provided")
        beta = self.tilt_angle.gv("rad")
        X1, Y1, Z1, F12 = _get_view_factor(self.tod, polygon_i, lims, xp, yp, zp, beta)
        
        # Extract coordinate arrays from the meshgrid
        X2 = X1[0, :]  # x-coordinates (1D array)
        Y2 = Y1[:, 0]  # y-coordinates (1D array)
        
        # Create RegularGridInterpolator
        return spi.RegularGridInterpolator(
            (Y2, X2),  # Note: order is (y, x) for RegularGridInterpolator
            F12,       # 2D array of values
            bounds_error=False,
            fill_value=0.0,  # Use 0.0 for view factors outside bounds
            method='linear'
        )
    
    def _process_tilted_rays(self, R2: pd.DataFrame, polygon_i: int) -> tuple[pd.DataFrame, float, float, float]:
        """Process rays for tilted surface intersection."""
        R3 = R2[(R2["Npolygon"] == polygon_i) & R2["hit_rcv"]].copy()
        
        # Point of pivoting belonging to the plane
        xp = R3["xr"].max() + 0.1
        yp = 0.0  # Valid for 1st hexagon
        zp = R3["zr"].mean()
        
        # Calculate intersection with tilted plane
        beta = self.tilt_angle.gv("rad")
        kt = (xp - R3["xr"]) / (R3["uzr"] / np.tan(beta) + R3["uxr"])
        R3['xt'] = R3["xr"] + kt * R3["uxr"]
        R3['yt'] = R3["yr"] + kt * R3["uyr"]
        R3['zt'] = R3["zr"] + kt * R3["uzr"]
        
        return R3, xp, yp, zp
    
    def _calculate_heat_flux_function(
            self,
            R3: pd.DataFrame,
            R2: pd.DataFrame,
            SF: pd.DataFrame,
            N_hel: Var,
            polygon_i: int,
            Gbn: Var = Var(950., "W/m2"),
            A_h1: Var = Var(2.92*2.92, "m2"),
        ) -> tuple[spi.RectBivariateSpline, Var, Array]:
        """Calculate heat flux distribution and create interpolation function."""
        if self.tod is None:
            raise ValueError("No TOD provided")
        hlst = SF.iloc[:N_hel.v].index
        SF_it = SF.loc[hlst]
        Etas_it = SF_it.mean()
        eta_SF_it = Var(Etas_it['Eta_SF'], "-")
        
        # Get rays for this iteration
        R3i = R3[R3["hel"].isin(hlst)].copy()
        R2_all = R2[R2["hit_rcv"]].copy()
        r_i = len(R3i) / len(R2_all[R2_all["hel"].isin(hlst)])
        
        # Setup grid for heat flux calculation
        Nx, Ny = 100, 50
        xA, yA = self.tod.perimeter_points(self.tod.radius_ap.v, tod_index=polygon_i-1)

        xmin = Var(max(xA.min(), R3i["xt"].quantile(0.01)), "m")
        xmax = Var(min(xA.max(), R3i["xt"].quantile(0.99)), "m")
        ymin = Var(max(yA.min(), R3i["yt"].quantile(0.01)), "m")
        ymax = Var(min(yA.max(), R3i["yt"].quantile(0.99)), "m")

        dx = (xmax - xmin) / Nx
        dy = (ymax - ymin) / Ny
        dA = dx * dy
        
        # Calculate number of rays hitting receiver
        Nrays = len(R3i["xt"][(R3i["xt"] > xmin) & (R3i["xt"] < xmax) &
                          (R3i["yt"] > ymin) & (R3i["yt"] < ymax)])
        
        # Total power calculation
        P_SF = eta_SF_it * (Gbn * A_h1 * N_hel)
        P_SF1 = P_SF * r_i
        Fbin = P_SF1 / ( dA * Nrays)
        
        # Create heat flux distribution
        Q_rc1, X_rc1, Y_rc1 = np.histogram2d(
            R3i["xt"], R3i["yt"],
            bins=[Nx, Ny],
            range=[[xmin.gv("m"), xmax.gv("m")], [ymin.gv("m"), ymax.gv("m")]],
            density=False
        )
        Q_rc1 = Fbin.gv("MW/m2") * Q_rc1
        Q_max = Var(Q_rc1.max(), "MW/m2")
        
        # Reduce receiver size based on minimum heat flux
        Q_avg_min = Var(250, "kW/m2")  #Minimal average radiation allowed per axis
        
        # Filter in y-direction
        valid_y = Q_rc1.mean(axis=0) > Q_avg_min.gv("MW/m2")
        if np.any(valid_y):
            ymin_corr = Y_rc1[:-1][valid_y][0]
            ymax_corr = Y_rc1[:-1][valid_y][-1]
        else:
            ymin_corr, ymax_corr = ymin.gv("m"), ymax.gv("m")
        
        # Filter in x-direction
        valid_x = Q_rc1.mean(axis=1) > Q_avg_min
        if np.any(valid_x):
            xmin_corr = X_rc1[:-1][valid_x][0]
            xmax_corr = X_rc1[:-1][valid_x][-1]
        else:
            xmin_corr, xmax_corr = xmin.gv("m"), xmax.gv("m")
        
        # Create corrected arrays
        X_av = np.array([(X_rc1[i] + X_rc1[i+1]) / 2 for i in range(len(X_rc1)-1)])
        Y_av = np.array([(Y_rc1[i] + Y_rc1[i+1]) / 2 for i in range(len(Y_rc1)-1)])
        
        x_mask = (X_av > xmin_corr) & (X_av < xmax_corr)
        y_mask = (Y_av > ymin_corr) & (Y_av < ymax_corr)
        
        Q_rcf = Q_rc1[np.ix_(x_mask, y_mask)]
        X_rcf = X_av[x_mask]
        Y_rcf = Y_av[y_mask]
        
        # Apply reflectivity correction
        one = Var(1.0, "-")
        em_w = self.emissivity_wall
        Q_rcf_accum = Var(Q_rcf.sum(), "MW/m2")
        eta_tilt = Q_rcf_accum * dA / P_SF1
        eta_tilt_rfl = eta_tilt + (one - eta_tilt) * (one - em_w)
        F_corr = eta_tilt_rfl / eta_tilt if eta_tilt.v > 0 else one
        Q_rcf = F_corr.gv("-") * Q_rcf
        
        # Create interpolation function
        f_Qrc1 = spi.RectBivariateSpline(X_rcf, Y_rcf, Q_rcf)

        lims = Array([xmin_corr, xmax_corr, ymin_corr, ymax_corr], "m")
        return f_Qrc1, Q_max, lims

ParticleReceiver: TypeAlias = HPR0D | HPR2D | TPR0D | TPR2D

def _HTM_0D_blackbox(
        Tp: float,
        qi: float,
        Tamb: float = 300.,
        Fc: float = 2.57,
        air: Air = air,
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
    ab_p = 0.91
    em_p = 0.85
    Tp = float(Tp)
    Tsky = Tamb - 15.
    # air.TP = (Tp+Tamb)/2., P_atm
    if HTC in ['NellisKlein', "Holman"]:
        hconv = Fc* htc.h_horizontal_surface_upper_hot(Tp, Tamb, 0.01, fluid=air, correlation=HTC)
    elif HTC == 'Experiment':
        hconv = Fc* bdr_csp.htc.h_conv_Experiment(Tp, Tamb, 0.1, fluid=air)
    else:
        raise ValueError(f"HTC {HTC} not implemented")
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
        eta_rcv = 0.
    return (eta_rcv,hrad,hconv)


def _tpr2d_func_temp_out_avg(t_res_g, *args):
    """Objective function for residence time calculation."""
    rcvr, f_Qrc1, f_eta, f_ViewFactor = args
    vel_p = rcvr['xavg'] / t_res_g[0]
    t_sim = (rcvr['x_ini'] - rcvr['x_fin']) / rcvr['vel_p']
    T_p, _, _, _ = _HTM_2D_tilted_surface(rcvr, vel_p, t_sim, f_Qrc1, f_eta, f_ViewFactor)
    return T_p.gv("K").mean() - rcvr.temp_out


def _HTM_2D_moving_particles(
        rcvr_input: dict,
        vel_p: Var,
        full_output: bool = False
    ) -> tuple[Array, Var, Var, dict[str,pd.DataFrame|Frame]]:

    x_ini = rcvr_input["x_ini"]
    x_fin = rcvr_input["x_fin"]
    y_bot = rcvr_input["y_bot"]
    y_top = rcvr_input["y_top"]
    thickness_parts = rcvr_input["thickness_parts"]
    temp_ini = rcvr_input["temp_ini"]
    material = rcvr_input["material"]
    f_Qrc1 = rcvr_input["func_heat_flux_rcv1"]
    f_eta = rcvr_input["func_eta"]
    
    if not isinstance(material, Carbo):
        raise TypeError("Material must be of type Carbo")

    #Belt vector direction and vector direction
    B_xi = x_ini
    B_yi = (y_bot + y_top)/2.
    B_ux  = -1.             #Should be unitary vector (B_ux**2+B_uy**2 = 1)
    B_uy  = 0.              #Should be unitary vector
    
    B_Vx  = vel_p.gv("m/s")*B_ux      #Belt X axis velocity
    B_Vy  = vel_p.gv("m/s")*B_uy      #Belt Y axis velocity
    
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
    P = {x:[] for x in ["t", "x", "y", "Tp", "Q_in", "eta", "Tnew", "rcv_in", "dT", "Q_gain"]}
    units = ["s", "m", "m", "K", "W", "-", "K", "-","K","W"]
    
    while t <= t_sim+dt:
        
        Q_in = f_Qrc1(np.array([B_x,B_y]).transpose())/1000
        
        eta = np.array([f_eta([T_B[j],Q_in[j]])[0] for j in range(len(T_B))])
        eta = np.nan_to_num(eta, nan=0.0)

        Q_in = np.where(Q_in>0.01,Q_in,0.0)
        rcv_in = np.where(Q_in>0.01,True,False)
        
        Q_abs = Q_in * eta
        Q_gain = Q_abs * rcv_in
        rho_b = material.rho()

        if np.isnan(T_B).any():
            print(T_B)
            sys.exit()
        cp_b  = np.array([148*temp**0.3093 for temp in T_B])
        
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
    
    P_out = {}
    if full_output:
        for x in P.keys():
            P_out[x] = pd.DataFrame(P[x])

    temp_out_av = T_B.mean()
    rho_b = material.rho().gv("kg/m3")
    cp_b = np.array([(148*((temp+temp_ini)/2)**0.3093) for temp in T_B]).mean()
    M_stg1 = rho_b * (B_Lz*B_Ly*vel_p)
    Qstg1 = M_stg1 * cp_b * (temp_out_av - temp_ini) /1e6

    temp_parts_out = Array(T_B, "K")
    heat_stg_1 = Var(Qstg1, "W")
    mass_stg_1 = Var(M_stg1, "kg")
    
    return (temp_parts_out, heat_stg_1, mass_stg_1, P_out)


def _HTM_2D_tilted_surface(
        rcvr: TPR2D,
        vel_p: Var,
        t_sim: Var,
        f_Qrc1: spi.RectBivariateSpline,
        f_eta: Callable[[TPR2D|TPR0D, Var, Var, Var], list[Var]],
        f_ViewFactor:spi.RegularGridInterpolator,
        full_output=False
    ) -> tuple[Array, Var, Var, dict[str,pd.DataFrame]]:
    
    x_ini =  rcvr.x_ini
    y_bot = rcvr.y_bot
    y_top = rcvr.y_top
    tz = rcvr.thickness_parts
    T_ini = rcvr.temp_ini

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
    dt     = Var(1., "s")             #Temporal discretisation [s]
    
    #Initial positions for Belt elements
    B_y = np.array([ (B_yi-B_Ly/2 + (j+0.5)*B_dy).gv("m") for j in range(B_Ny)])
    B_x = np.array([ B_xi.gv("m") for j in range(B_Ny)])
    T_B = np.array([ T_ini.gv("K") for j in range(B_Ny)])
    
    t=dt
    P = {'t':[],'x':[],'y':[],'Tp':[],'Q_in':[], 'eta':[], 'Tnew':[], 'rcv_in':[], 'dT':[], 'Q_abs':[], 'Fv':[]}
    units = ["s", "m", "m", "K", "MW", "-", "-", "K", "MW", "-"]

    while t <= t_sim+dt:
        
        Q_in = f_Qrc1.ev(B_x,B_y)
        
        # eta = np.array([f_eta(T_B[j],Q_in[j])[0] for j in range(len(T_B))])
        
        # Fv  = np.array([f_ViewFactor(B_x[j],B_y[j])[0] for j in range(len(T_B))])
        points = np.column_stack([B_y, B_x])
        Fv = f_ViewFactor(points)
        eta = np.array([f_eta(rcvr,T_B[j],Q_in[j],Fv[j])[0].v for j in range(len(T_B))])
        Q_in = np.where(Q_in>0.01,Q_in,0.0)
        rcv_in = np.where(Q_in>0.01,True,False)
        
        Q_abs = Array(Q_in * eta, "MW/m2")
        rho_b = Array(1810*np.ones(len(T_B)), "kg/m3")
        cp_b  = Array(148*T_B**0.3093, "J/kg-K")
        T_B1 =  (Q_abs/(cp_b*rho_b)).gv("K-m/s")/(B_Lz/dt).gv("m/s") + T_B
        dT  = T_B1-T_B
        
        if full_output:
            P['t'].append(t.gv("s")*np.ones(len(B_x)))
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
        
        B_x = B_x + (dt*B_Vx).gv("m")
        B_y = B_y + (dt*B_Vy).gv("m")
        T_B = T_B1
        t += dt
    
    P_out = {}
    if full_output:
        for x in P.keys():
            P_out[x] = pd.DataFrame(P[x])
    out_detailed = P_out
    
    T_out_av = Var(T_B.mean(), "K")
    rho_b = Var(1810., "kg/m3")
    cp_b  = Var((148*((T_B+T_ini.gv("K"))/2)**0.3093).mean(), "J/kg-K")
    
    temp_parts = Array(T_B, "K")
    mass_stg_1 = rho_b * (B_Lz*B_Ly*vel_p)
    heat_stg_1 = mass_stg_1 * cp_b * (T_out_av - T_ini)
    
    return (temp_parts, heat_stg_1, mass_stg_1, out_detailed)

def _HTM_3D_moving_particles(CST,TOD,inputs,f_Qrc1,f_eta,full_output=True):
    
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
    else:
        P = {}
    Py_3D = np.array([])

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
        else:
            raise ValueError(f"Material {TSM} not implemented")
        
        
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
    
    P_out = {}
    if full_output:
        for x in P.keys():
            P_out[x] = np.array(P[x])
    
    T_out_av = T_B.mean()
    # cp_b  = (365*((T_B+T_in)/2 + 273.15)**0.18).mean()
    cp_b  = (148*((T_B+T_in)/2)**0.3093).mean()
    
    if TSM=='CARBO':
        rho_p = 1810.
    else:
        raise ValueError(f"Material {TSM} not implemented")
    M_stg1 = rho_p * (Lz*Ly*P_vel)
    Qstg1 = M_stg1 * cp_b * (T_out_av - T_in) /1e6
    
    print(T_in,T_p.mean(),T_p.max())
    return T_p, Qstg1, M_stg1, Py_3D


def _get_func_eta():
    Fc = 2.57
    air  = Air()
    Tps = np.arange(700.,2001.,100.)
    qis = np.arange(0.10,4.01,0.1)
    eta_ths = np.zeros((len(Tps),len(qis)))
    for (i,j) in [(i,j) for i in range(len(Tps)) for j in range(len(qis))]:
        eta_ths[i,j] = _HTM_0D_blackbox(Tps[i], qis[j], Fc=Fc, air=air)[0]
    return spi.RegularGridInterpolator(
        (Tps,qis),
        eta_ths,
        bounds_error = False,
        fill_value = None   #type: ignore
        )


def _get_func_heat_flux_rcv1(xr: pd.Series,yr: pd.Series,lims: list[Var],P_SF1: Var):
    
    xmin,xmax,ymin,ymax = lims
    
    Nx = 50; Ny = 50
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Nx
    dA=dx*dy
    Nrays = len(xr)
    Fbin    = P_SF1 /(dA*Nrays)
    Q_rc1,X_rc1, Y_rc1 = np.histogram2d(
        xr,
        yr,
        bins=[Nx,Ny],
        range=[[xmin.gv("m"), xmax.gv("m")], [ymin.gv("m"), ymax.gv("m")]],
        density=False
    )
    Q_rc1 = Fbin.gv("MW/m2") * Q_rc1
    
    # f_Qrc1 = spi.RectBivariateSpline(X_rc1[:-1],Y_rc1[:-1],Q_rc1)     #Function to interpolate
    f_Qrc1 = spi.RegularGridInterpolator(
        [X_rc1[:-1],Y_rc1[:-1]],
        Q_rc1,
        bounds_error = False,
        fill_value = np.nan
        )
    return f_Qrc1, Q_rc1, X_rc1, Y_rc1


#---------------------
def get_plant_costs() -> dict[str, float | str]:
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

#---------------------

def get_land_area(SF: pd.DataFrame, A_h1: Var, N_sec=8) -> tuple[Var, Var]:
    SF_in = SF[SF["hel_in"]].copy()
    S_hel = len(SF_in) * A_h1
    x_c = SF_in["xi"].mean()
    y_c = SF_in["yi"].mean()
    SF_in['ri_c']  = ((SF_in["xi"] - x_c)**2 + (SF_in["yi"] - y_c)**2)**0.5
    SF_in['theta'] = np.arctan2(SF_in["yi"],SF_in["xi"])
    SF_in['bin'] = SF_in['theta']//(2*np.pi/N_sec) + N_sec/2
    points = pd.DataFrame([],columns=['xi','yi','ri_c'])
    for n in range(N_sec):
        SF_n = SF_in[SF_in["bin"]==n]
        set_n = SF_n.loc[SF_n["ri_c"].nlargest(10).index][['xi','yi','ri_c']]
        points = set_n if n==0 else pd.concat([points,set_n])
    S_land = Var(np.pi * (points["ri_c"].mean())**2, "m2")
    return S_land, S_hel

def get_cost_hb(
        C_hel: Var,
        M_HB_fin: Var,
        M_HB_t: Var,
        F_struc_hel: Var,
        zhel: Var,
        zmax: Var
    ) -> Var:
    F_HB_fin    = M_HB_fin / (M_HB_t - M_HB_fin)
    F_mirr_HB   = Var(0.34, "-")               # Double than heliostat's
    F_struc_HB  = F_struc_hel * np.log((zmax.gv("m")-3.0)/0.003) / np.log((zhel.gv("m")-3.0)/0.003)
    F_cool_HB   = F_struc_HB * F_HB_fin
    F_oth_HB    = Var(0.20, "-")               # Double than heliostat's
    return C_hel * (F_mirr_HB + F_struc_HB + F_cool_HB + F_oth_HB)
    
def get_mass_tod(S_TOD):
    M_TOD_mirr  = (S_TOD * Var(15.1, "kg/m2"))        # Mass of TOD mirror
    F_TOD_pipe  = 2.0                       # Extra factor due to fins (must be corrected)
    return M_TOD_mirr*(1 + F_TOD_pipe)

def get_cost_tod(
        C_hel: Var,
        F_struc_hel: Var,
        zhel: Var,
        S_TOD: Var,
        zrc: Var
    ) -> Var:
    F_mirr_TOD  = Var(0.68, "-")              # Fourfold than heliostat's
    F_struc_TOD = F_struc_hel * np.log((zrc.gv("m")-3.0)/0.003) / np.log((zhel.gv("m")-3.0)/0.003)
    F_cool_TOD  = Var(8000 + 252.9*S_TOD.gv("m2")**0.91,"USD") / (C_hel*S_TOD)  # Normalized Hall's correlation
    F_oth_TOD   = Var(0.20, "-")               # Double than heliostats
    return C_hel * (F_mirr_TOD + F_struc_TOD + F_cool_TOD + F_oth_TOD)

def get_tower_cost_antenna(
        zmax: Var,
        zrc: Var,
        M_HB_t: Var,
        M_TOD_t: Var,
        F_tow: Var = Var(1.0, "-")):
    def _cost_per_tower(M_mirror: Var, z: Var) -> Var:
        return Var(( (123.21 + 362.6*M_mirror.gv("ton")) * np.exp(0.0224*z.gv("m")) ) * 4, "USD")
    M_HB_1   = M_HB_t / 4.          #Weight per column
    M_TOD_1  = M_TOD_t / 4.
    C_tow_HB = _cost_per_tower(M_HB_1, zmax)
    C_tow_TOD = _cost_per_tower(M_TOD_1, zrc)
    return F_tow * (C_tow_HB + C_tow_TOD)

def get_storage_cost_dims(
        material: Carbo,
        T_pH: Var,
        T_pC: Var,
        Q_stg: Var,
        C_parts: Var,
        HtD_stg: Var,
        F_stg: Var = Var(1.0, "-")
    ) -> tuple[Var, Var, Var]:
    dT_stg  = T_pH - T_pC                   # K
    Tp_avg  = (T_pH + T_pC)/2.              # K
    cp_stg  = material.cp(Tp_avg)
    rho_stg = material.rho()
    
    M_stg = (Q_stg / (cp_stg * dT_stg)).su("kg")            #[kg]
    V_stg = (M_stg / rho_stg).su("m3")                                     #[m3]
    C_stg_p  = (1.1 * C_parts * M_stg).su("USD")                      #[USD]
    
    C_stg_H = Var(1000.*(1 + 0.3*(1 + (T_pH.gv("K") - 273.15 - 600)/400.)), "USD/m2")
    C_stg_C = Var(1000.*(1 + 0.3*(1 + (T_pC.gv("K") - 273.15 - 600)/400.)), "USD/m2")
    D_stg   = Var((4.*V_stg/np.pi/HtD_stg).gv("m3")**(1./3.), "m")
    H_stg   = HtD_stg*D_stg
    A_tank  = np.pi * ( Var(D_stg.gv("m")**2/2., "m2") + D_stg*H_stg)
    C_stg_t = A_tank * (C_stg_H + C_stg_C)            #[MM USD]

    return F_stg*C_stg_p + C_stg_t, H_stg, V_stg

def get_rcv_cost(
        F_rcv: Var,
        C_tpr: Var,
        Arcv: Var,
        M_p: Var,
        H_stg: Var,
        SM: Var) -> Var:
    C_rcv_tpr = C_tpr * Arcv
    hlift_CH = H_stg + Var(2.0, "m")
    hlift_HC = H_stg + Var(5.0, "m")
    C_lift = Var(58.38, "USD-s/kg-m") * (hlift_CH + hlift_HC / SM) * M_p
    return F_rcv * ( C_rcv_tpr + C_lift )

def get_lcoh(
        C_heat: Var,
        C_OM: Var,
        P_rcv: Var,
        CF_sf: Var,
        DR: Var,
        Ny: Var) -> Var:
        one = Var(1.0, "-")
        TPF  = (one/DR)*(one - one/(one+DR).v**Ny.v)
        P_yr = CF_sf * Var(8760,"hr") * P_rcv
        lcoh = (C_heat * (one + C_OM*TPF) / (P_yr * TPF)).su("USD/MWh")
        return lcoh if P_yr.v>0 else Var(np.inf, "USD/MWh")


def get_lcoe(
        C_elec: Var,
        C_OM: Var,
        P_el: Var,
        CF_pb: Var,
        DR: Var,
        Ny: Var) -> Var:
    one = Var(1.0, "-")
    TPF  = (one/DR)*(one - one/(one+DR).v**Ny.v)
    E_yr = CF_pb * Var(8760,"hr") * P_el
    lcoe = (C_elec * (one + C_OM*TPF) / (E_yr * TPF)).su("USD/MWh")
    return lcoe if E_yr.v>0 else Var(np.inf, "USD/MWh")


def plant_costing_calculations(
        costing_params: dict[str, Var],
) -> dict[str,Var]:         # ex BDR_cost

    zrc = costing_params['zrc']
    A_h1 = costing_params['A_h1']
    T_stg = costing_params['T_stg']
    SM = costing_params['SM']
    pb_eta = costing_params['pb_eta']
    T_pH = costing_params['T_pH']
    T_pC = costing_params['T_pC']
    HtD_stg = costing_params['HtD_stg']
    Ntower = costing_params['Ntower']
    material: Carbo = costing_params['material'] #type: ignore
    costs_in: dict[str, float] = costing_params['costs_in'] #type: ignore
    S_HB = costing_params['S_HB']
    zmax = costing_params['zmax']
    S_TOD = costing_params['S_TOD']
    Arcv = costing_params['Arcv']
    M_p = costing_params['M_p']
    P_rcv = costing_params['P_rcv']
    M_HB_fin = costing_params['M_HB_fin']
    M_HB_t = costing_params['M_HB_t']
    S_land = costing_params['S_land']
    S_hel = costing_params['S_hel']
    
    #Solar field related costs
    R_ftl = Var(costs_in['R_ftl'], "-")         # Field to land ratio, SolarPilot
    C_land = Var(costs_in['C_land'], "USD/m2")       # USD per m2 of land. SAM/SolarPilot (10000 USD/acre)
    C_site = Var(costs_in['C_site'], "USD/m2")       # USD per m2 of heliostat. SAM/SolarPilot
    C_hel = Var(costs_in['C_hel'], "USD/m2")         # USD per m2 of heliostat. Projected by Pfal et al (2017)
    C_tpr = Var(costs_in['C_tpr'], "USD/m2")         # [USD/m2]
    C_tow = Var(costs_in['C_tow'], "USD")         # USD fixed. SAM/SolarPilot, assumed tower 25% cheaper
    e_tow = Var(costs_in['e_tow'], "-")         # Scaling factor for tower
    C_parts = Var(costs_in['C_parts'], "USD/kg")     # USD/kg (Gonzalez-Portillo et al. 2022). CARBO HSP
    tow_type = costs_in['tow_type']
    C_OM = Var(costs_in['C_OM'], "-")           # % of capital costs.
    DR = Var(costs_in['DR'], "-")               # Discount rate
    Ny = Var(costs_in['Ny'], "yr")               # Horizon project
    C_pb = Var(costs_in['C_pb'], "USD/MW")           #USD/MWe
    C_PHX = Var(costs_in['C_PHX'], "USD/MW") if "C_PHX" in costs_in else Var(0.0, "USD/MW")      #USD/MWe
    C_xtra = Var(costs_in['C_xtra'], "-")       # Engineering and Contingency

    #Factors to apply on costs. Usually they are 1.0.
    F_HB   = Var(costs_in['F_HB'], "-")
    F_TOD  = Var(costs_in['F_TOD'], "-")
    F_rcv  = Var(costs_in['F_rcv'], "-")
    F_tow  = Var(costs_in['F_tow'], "-")
    F_stg  = Var(costs_in['F_stg'], "-")

    # P_rcv = plant.rcv_nom_power  # [MW] Nominal power stored by receiver
    P_pb  = P_rcv / SM              #[MWth]  Thermal energy delivered to power block
    Q_stg = P_pb * T_stg            #[MWh]  Energy storage required to meet T_stg
    
    Co: dict[str, Var] = {}      #Everything in MM USD, unless explicitely indicated
    
    # CAPACITY FACTOR
    # CF_sf = Capacity Factor (Solar field)
    # CF_pb = Capacity Factor (Power block)
    
    CF_sf  = Var(costs_in['CF_sf'], "-")
    CF_pb  = Var(costs_in['CF_pb'], "-") if 'CF_pb' in costs_in else CF_sf*SM
    
    Co['CF_sf'] = CF_sf
    Co['CF_pb'] = CF_pb
    
    # LAND CALCULATIONS
    Co['land'] = ( C_land*S_land*R_ftl + C_site*S_hel )
    Co['land_prod'] = P_rcv/S_land
    
    # MIRROR MASSES AND COSTS
    F_struc_hel = Var(0.17, "-")
    zhel        = Var(4.0, "m")

    C_HB = get_cost_hb(C_hel, M_HB_fin, M_HB_t, F_struc_hel, zhel, zmax)
    C_TOD = get_cost_tod(C_hel, F_struc_hel, zhel,S_TOD, zrc)
    M_TOD_t = get_mass_tod(S_TOD)
    Co['hel'] = C_hel * S_hel
    Co['HB']  = F_HB * C_HB * S_HB
    Co['TOD'] = F_TOD * C_TOD * S_TOD
    
    # TOWER COST
    if tow_type == 'Conv':
        Co['tow'] = C_tow * np.exp(e_tow.v*zmax.v)
    elif 'Antenna':    
        Co['tow'] = get_tower_cost_antenna(zmax, zrc, M_HB_t, M_TOD_t, F_tow)
    
    # STORAGE COST
    C_stg, H_stg, V_stg = get_storage_cost_dims(material, T_pH, T_pC, Q_stg, C_parts, HtD_stg, F_stg)
    Co['stg'] = C_stg      #[MM USD]
    Co['H_stg'] = H_stg
    Co['V_stg'] = V_stg

    # RECEIVER COST    
    Co['rcv'] = get_rcv_cost(F_rcv, C_tpr, Arcv, M_p, H_stg, SM)
    
    # TOTAL HEAT COSTS
    Co['Heat'] = (Co['land'] + Co['hel'] + Co['HB'] + Co['TOD'] + Co['rcv'] + Co['tow'] + Co['stg']) * C_xtra
    Co['SCH'] = Co['Heat'] / P_rcv if P_rcv.v>0 else Var(None, "USD/W")      #(USD/Wp) Specific cost of Heat (from receiver)
    
    #Levelised cost of heat (sun-to-storage)
    Co['LCOH'] =  get_lcoh(Co['Heat'], C_OM, P_rcv, CF_sf, DR, Ny)

    # Levelised cost of electricity (sun-to-electricity)
    Pel        = Ntower * pb_eta * P_pb
    Co['PB']   = C_pb * Pel
    Co['PHX']  = C_PHX * P_pb
    Co['Elec'] = Ntower*Co['Heat'] +  (Co['PHX']+Co['PB']) * C_xtra
    Co['LCOE'] =  get_lcoe(Co['Elec'], C_OM, Pel, CF_pb, DR, Ny)

    return Co


def _rcvr_HPR_0D(CST, Fc=2.57, air=air):
    
    Prcv, Qavg, Tin, Tout, tz  = [CST[x] for x in ['P_rcv','Qavg','T_pC','T_pH', 'tz']]
    
    Tp = 0.5*(Tin+Tout)
    eta_rcv = _HTM_0D_blackbox(Tp, Qavg,Fc=Fc,air=air)[0]
    
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


def _rcvr_HPR_0D_simple(CST, SF: pd.DataFrame, Fc: float=2.57, air=Air()):
    
    Prcv, Qavg, Tin, Tout, tz  = [CST[x] for x in ['P_rcv','Qavg','T_pC','T_pH', 'tz']]
    
    Tp = 0.5*(Tin+Tout)
    eta_rcv = _HTM_0D_blackbox(Tp, Qavg,Fc=Fc,air=air)[0]
    
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


def _rcvr_HPR_2D_simple(
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
        T_p, _, _, _ = _HTM_2D_moving_particles(rcvr_input, vel_p)
        return T_p.gv("K").mean()-temp_out
    
    temp_ini, temp_out, thickness_parts, material, power_rcv, heat_flux_avg = [
        CST[x] for x in ['T_pC', 'T_pH', 'tz', 'TSM', 'P_rcv', 'Qavg']
    ]

    #Parameters for receiver
    xO,yO = TOD.perimeter_points(TOD.radius_out.v, tod_index=polygon_i-1)
    lims = xO.min(), xO.max(), yO.min(), yO.max()
    x_fin = lims[0]
    x_ini = lims[1]
    y_bot = lims[2]
    y_top = lims[3]
    area_rcv_1 = TOD.receiver_area.v/TOD.n_tods
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
        "func_eta": _get_func_eta(),
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
    else:
        raise ValueError(f"Material {material} not implemented")
    
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
        
        func_heat_flux_rcv1,heat_flux_rcv1,_,_ = _get_func_heat_flux_rcv1(
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
            temps_parts, heat_stored_1, mass_stg_1, Rcvr_full = _HTM_2D_moving_particles(
                rcvr_input, vel_p, full_output=True
            )
        else:
            temps_parts, heat_stored_1, mass_stg_1 = _HTM_2D_moving_particles(
                rcvr_input, vel_p,
            )
            Rcvr_full = None
        
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
def _eta_th_tilted(
        rcvr: TPR0D|TPR2D,
        Tp: Var|float,
        qi: Var|float,
        Fv: Var|float = Var(1.0, "-")
        ) -> list[Var]:
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
    if isinstance(Tp, float):
        T_part = Var(Tp, "K")
    elif isinstance(Tp, Var):
        T_part = Tp
    else:
        raise ValueError("Tp must be float or Var")
    if isinstance(qi, float):
        q_in = Var(qi, "MW/m2")
    elif isinstance(qi, Var):
        q_in = qi
    else:
        raise ValueError("qi must be float or Var")
    if isinstance(Fv, float):
        F_t_rcv = Var(Fv, "-")
    elif isinstance(Fv, Var):
        F_t_rcv = Fv
    else:
        raise ValueError("Fv must be float or Var")

    Tw = rcvr.wall_temp
    Fc = rcvr.factor_htc
    air = rcvr._air
    HTC: str = rcvr.model_htc

    T_amb = rcvr.temp_amb
    ab_p = rcvr.absorptivity
    em_p = rcvr.emissivity_part
    em_w = rcvr.emissivity_wall
    h_cond = rcvr.conductivity

    xmin = rcvr.x_fin
    xmax = rcvr.x_ini
    ymin = rcvr.y_bot
    ymax = rcvr.y_top
    beta = rcvr.tilt_angle


    if rcvr.tod is not None:
        Arc1 = rcvr.tod.receiver_area / rcvr.tod.n_tods
    else:
        raise ValueError("rcvr.tod is not provided")

    sigma = Var(5.674e-8, "W/m2-K4")
    one = Var(1.0, "-")
    
    Tsky = T_amb - Var(15.,"K")
    if HTC in ['NellisKlein', "Holman"]:
        hconv_f = htc.h_horizontal_surface_upper_hot(T_part.gv("K"), T_amb.gv("K"), 0.01, fluid=air, correlation=HTC)
    elif HTC=='Experiment':
        hconv_f = bdr_csp.htc.h_conv_Experiment(T_part.gv("K"), T_amb.gv("K"), 0.1, fluid=air)
    else:
        raise ValueError(f"HTC {HTC} not implemented")
    
    hconv = Fc * Var(hconv_f, "W/m2-K")
    dTp4 = Var(T_part.gv("K")**4-Tsky.gv("K")**4, "K4")

    hrad = F_t_rcv*em_p*sigma*dTp4/(T_part-T_amb)
    
    F_tw = one-F_t_rcv
    X_tpr = (xmax-xmin)
    Y_tpr = (ymax-ymin)
    Z_tpr = X_tpr * float(abs(np.tan(beta.gv("rad"))))
    A_w = Z_tpr*Y_tpr + (Z_tpr*X_tpr/2)*2 + (X_tpr*Y_tpr - Arc1)
    A_t = Y_tpr*X_tpr/float(np.cos(beta.gv("rad")))

    C_tw = (one-em_p)/(em_p*A_t) + one/(F_tw*A_t) + (one-em_w)/(em_w*A_w)
    
    h_t_wall = (sigma/(C_tw*A_t))*dTp4/(T_part-Tw)
    hwall = one/(one/h_t_wall + one/h_cond)
    
    hrc = hconv + hrad + hwall
    qloss = hrc * (T_part - T_amb)
    
    if q_in.v<=0.:
        eta_th = Var(0., "-")
    else:
        eta_th = (q_in*ab_p - qloss)/q_in

    if eta_th.v<0:
        eta_th = Var(0., "-")

    return [eta_th, hrad, hconv, hwall]


def _get_view_factor(
        TOD: TertiaryOpticalDevice,
        polygon_i: int,
        lims1: Array,
        xp: float,
        yp: float,
        zp: float,
        beta: float) -> list[np.ndarray]:
    
    Array = TOD.array
    x0 = TOD.x0
    y0 = TOD.y0
    V_TOD = TOD.n_sides
    H_TOD = TOD.height.gv("m")
    rO = TOD.radius_out.gv("m")
    zrc = TOD.zrc.gv("m")

    xmin1,xmax1,ymin1,ymax1 = [x.gv("m") for x in lims1]
    n1,n2 = (-np.sin(beta),0.,np.cos(beta)), (0.,0.,-1)
    x0i,y0i = x0[polygon_i-1], y0[polygon_i-1]
    xO, yO = TOD.perimeter_points(rO, tod_index=(polygon_i-1))
    xmin2,xmax2,ymin2,ymax2 =xO.min(),xO.max(),yO.min(),yO.max()
    
    Nx2 = 100
    Ny2 = 100
    dx2 = (xmax2-xmin2)/Nx2
    dy2 = (ymax2-ymin2)/Nx2
    dA2=dx2*dy2
    Px = np.array([xmin2+dx2/2 + i*dx2 for i in range(Nx2)])
    Py = np.array([ymin2+dy2/2 + j*dy2 for j in range(Ny2)])
    X2, Y2 = np.meshgrid(Px, Py)
    Z2    = (zrc-H_TOD) * np.ones(X2.shape)
    rcvr_in = bdr.CPC_enter(X2, Y2, rO, Array, V_TOD, x0i, y0i)
    
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
        
    return [X1,Y1,Z1,F12]


def _getting_T_w(T_w:float,*args: Var):
    T_pavg,xmax,xmin,ymax,ymin,beta,F_t_ap,em_p,em_w,A_ap,h_w_amb = args
    T_wall = Var(T_w, "K")
    beta_rad = beta.gv("rad")
    one = Var(1.0, "-")
    Tamb = Var(300., "K")
    Tsky = Tamb - Var(15., "K")
    sigma = Var(5.674e-8, "W/m2-K4")
    
    X_tpr = (xmax-xmin)
    Y_tpr = (ymax-ymin)
    Z_tpr = Var(np.abs(X_tpr.gv("m") * np.tan(beta_rad)), "m")
    
    A_w_aux = Var(max((X_tpr*Y_tpr - A_ap).gv("m2"),0.), "m2")
    A_w = Z_tpr*Y_tpr + (Z_tpr*X_tpr/2)*2 + A_w_aux
    A_t = Y_tpr*X_tpr/float(np.cos(beta_rad))
    
    F_t_w   = one - F_t_ap
    F_w_ap = (one - F_t_ap * (A_t/A_ap)) * (A_ap/A_w)

    C_tw = (one-em_p)/(em_p*A_t) + one/(F_t_w*A_t) + (one-em_w)/(em_w*A_w)

    dTw4 = Var(T_pavg.gv("K")**4 - T_wall.gv("K")**4, "K4")
    dTsky4 = Var(T_wall.gv("K")**4-Tsky.gv("K")**4, "K4")
    q_w_net = (
        (sigma/C_tw) * dTw4
        - F_w_ap*em_w*sigma*A_w*dTsky4
        - h_w_amb*A_w*(T_wall-Tamb)
    )
    # print(F_w_ap,F_t_w, A_t,A_ap,F_t_ap)
    return q_w_net.gv("W")

def _rcvr_TPR_0D_corr(
        receiver: TPR0D,
        SF: pd.DataFrame | None = None,
        corr: bool = True,
    ) -> dict:

    Fv    = receiver.view_factor_avg
    T_ini = receiver.temp_ini
    T_out = receiver.temp_out
    tz = receiver.thickness_parts
    Q_avg = receiver.heat_flux_avg
    Prcv = receiver.rcv_nom_power
    if receiver.tod is not None:
        Arcv = receiver.tod.receiver_area / receiver.tod.n_tods
    else:
        raise ValueError("receiver.tod is not provided")
    receiver.get_dimensions(tod_index=0)
    y_top = receiver.y_top
    y_bot = receiver.y_bot
    Tp_avg = (T_ini+T_out)/2.
    material = receiver.material

    eta_rcv = _eta_th_tilted(receiver, Tp_avg, Q_avg, Fv=Fv)[0]
    
    if isinstance(material, Carbo):
        rho_b = material.rho()
        cp = material.cp(Tp_avg)
    else:
        raise ValueError(f"Material {material} not implemented")
    t_res = rho_b * cp * tz * (T_out - T_ini ) / (Q_avg*eta_rcv)
    M_p = Prcv / (cp*(T_out - T_ini))
    P_bdr = Prcv / eta_rcv
    Ltot  = Arcv/(y_top-y_bot)
    vel_p = Ltot / t_res

    #Correcting 0D model with linear fit
    if corr:
        m = 0.6895
        n = 0.2466
    else:
        m,n=1.0,0.0
    eta_rcv_corr = Var(eta_rcv.v*m+n, "-")

    if SF is None:
        n_hels = Var(np.nan, "-")
    else:
        Q_acc = SF['Q_h1'].cumsum()
        n_hels = Var(len( Q_acc[ Q_acc < P_bdr ] ) + 1, "-")

    rcvr_output = {
        "temps_parts" : Array([Tp_avg.v], "K"),
        "n_hels" : n_hels,
        "rad_flux_max" : Var(np.nan, "-"),            #No calculated
        "rad_flux_avg" : P_bdr/Arcv,
        "heat_stored": Var(np.nan, "-"),            #No calculated
        "eta_rcv": eta_rcv_corr,
        "mass_stg": M_p,
        "time_res": t_res,
        "vel_parts": vel_p,
    }
    return rcvr_output


def bdr_cost(SF,CST):
    
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
    
    #Solar field related costs
    Ci      = CST['costs_in']
    R_ftl  = Ci['R_ftl']                   # Field to land ratio, SolarPilot
    C_land = Ci['C_land']                  # USD per m2 of land. SAM/SolarPilot (10000 USD/acre)
    C_site = Ci['C_site']                 # USD per m2 of heliostat. SAM/SolarPilot
    C_hel  = Ci['C_hel']                  # USD per m2 of heliostat. Projected by Pfal et al (2017)
    C_tpr  = Ci['C_tpr']                  # [USD/m2]
    C_tow = Ci['C_tow']             # USD fixed. SAM/SolarPilot, assumed tower 25% cheaper
    e_tow = Ci['e_tow']                 # Scaling factor for tower
    C_parts = Ci['C_parts']
    tow_type = Ci['tow_type']
    C_OM = Ci['C_OM']                    # % of capital costs.
    DR   = Ci['DR']                    # Discount rate
    Ny   = Ci['Ny']                      # Horizon project
    C_pb    = Ci['C_pb']                                #USD/MWe
    C_PHX   = Ci['C_PHX'] if 'C_PHX' in Ci else 0.      #USD/MWe
    C_xtra = Ci['C_xtra']                   # Engineering and Contingency
    
    #Factors to apply on costs. Usually they are 1.0.
    F_HB   = Ci['F_HB']
    F_TOD  = Ci['F_TOD']
    F_rcv  = Ci['F_rcv']
    F_tow  = Ci['F_tow']
    F_stg  = Ci['F_stg']
    
    P_rcv = CST['P_rcv_sim'] if 'P_rcv_sim' in CST else CST['P_rcv'] #[MW] Nominal power stored by receiver
    P_pb  = P_rcv / SM              #[MWth]  Thermal energy delivered to power block
    Q_stg = P_pb * T_stg            #[MWh]  Energy storage required to meet T_stg
    
    Co = {}      #Everything in MM USD, unless explicitely indicated
    
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
    else:       #If nothing is given, assumed default values
        CF_sf = Ci['CF_sf']
        CF_pb = CF_sf*SM
    
    Co['CF_sf'] = CF_sf
    Co['CF_pb'] = CF_pb
    
    # LAND CALCULATIONS
    S_land, S_hel = get_land_area(SF, A_h1)
    Co['land'] = ( C_land*S_land*R_ftl + C_site*S_hel )  / 1e6
    Co['land_prod'] = P_rcv/(S_land/1e4)            #MW/ha
    
    # MIRROR MASSES AND COSTS
    M_HB_fin  = CST['M_HB_fin']          # Fins components of HB weight
    M_HB_t    = CST['M_HB_tot']          # Total weight of HB
    F_struc_hel = 0.17
    zhel        = 4.0
    
    C_HB = get_cost_hb(C_hel, M_HB_fin, M_HB_t, F_struc_hel, zhel, zmax)
    C_TOD = get_cost_tod(C_hel, F_struc_hel, zhel,S_TOD, zrc)
    M_TOD_t = get_mass_tod(S_TOD)
    Co['hel'] = C_hel * S_hel /1e6
    Co['HB']  = F_HB * C_HB * S_HB /1e6
    Co['TOD'] = F_TOD * C_TOD * S_TOD /1e6
    
    # TOWER COST
    if tow_type == 'Conv':
        Co['tow'] = C_tow * np.exp(e_tow*zmax)/1e6
    elif 'Antenna':    
        Co['tow'] = get_tower_cost_antenna(zmax, zrc, M_HB_t, M_TOD_t, F_tow)  # USD
    
    # 
    # STORAGE COST
    C_stg, H_stg, V_stg = get_storage_cost_dims(T_pH, T_pC, Q_stg, C_parts, HtD_stg, F_stg)
    Co['stg'] = C_stg      #[MM USD]
    Co['H_stg'] = H_stg
    Co['V_stg'] = V_stg

    # RECEIVER COST    
    C_tpr = 37400  #[USD/m2]
    Co['rcv'] = get_rcv_cost(F_rcv, C_tpr, Arcv, M_p, H_stg, SM)  # USD
    
    # TOTAL HEAT COSTS
    Co['Heat'] = (Co['land'] + Co['hel'] + Co['HB'] + Co['TOD'] + Co['rcv'] + Co['tow'] + Co['stg']) * C_xtra
    Co['SCH'] = Co['Heat'] / P_rcv if P_rcv>0 else np.inf      #(USD/Wp) Specific cost of Heat (from receiver)
    
    #Levelised cost of heat (sun-to-storage)
    TPF  = (1./DR)*(1. - 1./(1.+DR)**Ny)
    P_yr = CF_sf * 8760 * P_rcv  /1e6            #[TWh_th/yr]
    Co['LCOH'] =  Co['Heat'] * (1. + C_OM*TPF) / (P_yr * TPF) if P_yr>0 else np.inf #USD/MWh delivered from receiver
    
    # Levelised cost of electricity (sun-to-electricity)
    Ntower     = CST['Ntower'] if 'Ntower' in CST else 1
    Pel        = Ntower * eta_pb * P_pb           #[MWe]  Nominal power of power block
    Co['PB']   = C_pb * Pel / 1e6
    Co['PHX']  = C_PHX * P_pb / 1e6
    Co['Elec'] = Ntower*Co['Heat'] +  (Co['PHX']+Co['PB']) * C_xtra
    E_yr       = CF_pb * 8760 * Pel   /1e6            #[TWh_e/yr]
    Co['LCOE'] =  Co['Elec'] * (1. + C_OM*TPF) / (E_yr * TPF)  if E_yr>0 else np.inf
    return Co




def initial_eta_rcv(
        plant: ModularCSPPlant|CSPBeamDownParticlePlant
    ) -> tuple[Var, Var, Var]:

    Tp_avg = (plant.stg_temp_hot + plant.stg_temp_cold) / 2

    eta_rcv1 = _HTM_0D_blackbox(Tp_avg.gv("K"), plant.flux_avg.gv("MW/m2"))[0]
    Arcv = (plant.rcv_nom_power / eta_rcv1) / plant.flux_avg
    TOD = bdr.TertiaryOpticalDevice(
            geometry=plant.geometry,
            array=plant.array,
            Cg=plant.Cg,
            receiver_area=Arcv,
            xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc,
        )
    results = _rcvr_TPR_0D_corr(plant.receiver, corr=True)
    eta_rcv = results["eta_rcv"]

    return eta_rcv, Arcv, TOD.radius_out


def run_coupled_simulation(
        plant: ModularCSPPlant | CSPBeamDownParticlePlant,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    hsf = plant.HSF
    hb = plant.HB
    tod = plant.TOD
    receiver = plant.receiver

    #Getting the RayDataset
    R0, SF = hsf.load_dataset(save_plk=True)

    # ##########################
    # # DELETE AFTER! This is only for testing purposes
    # R0 = R0.sample(n=1000, random_state=1).reset_index(drop=True)
    # ############################

    #Getting interceptions with HB
    R1, SF = hb.run_model(R0, SF, lat=plant.lat, lng=plant.lng, refl_error=False)

    #Interceptions with TOD
    R2 = tod.mcrt_solver(R1, refl_error=False)

    ### Optical Efficiencies
    SF = bdr.optical_efficiencies(
        R2, SF,
        irradiance=plant.Gbn,
        area_hel=hsf.A_h1,
        reflectivity=hsf.eta_rfl
    )

    ### Running receiver simulation and getting the results
    if isinstance(receiver, (HPR0D, TPR0D)):
        rcvr_output = receiver.run_model(SF)
    elif isinstance(receiver, (HPR2D, TPR2D)):
        rcvr_output = receiver.run_model(SF,R2)
    else:
        raise ValueError(f"Receiver type '{plant.rcv_type}' not recognized.")

    N_hel = rcvr_output["n_hels"]
    eta_rcv = rcvr_output["eta_rcv"]

    #Plant Parameters
    Prcv = receiver.rcv_nom_power
    eta_pb = plant.pb_eta_des
    SM = plant.solar_multiple
    A_h1 = plant.HSF.A_h1
    # Heliostat selection
    Q_acc    = SF['Q_h1'].cumsum()
    hlst     = Q_acc.iloc[:N_hel.v].index
    SF['hel_in'] = SF.index.isin(hlst)
    R2['hel_in'] = R2['hel'].isin(hlst)
    sf_power_sim  = Var(SF[SF["hel_in"]]['Q_h1'].sum(), "MW")
    rcv_power_sim = sf_power_sim * eta_rcv
    pb_power_sim  = rcv_power_sim * eta_pb
    
    # Calculating HB surface
    hb.rmin = Var(R2[R2["hel_in"]]['rb'].quantile(0.0001), "m")
    hb.rmax = Var(R2[R2["hel_in"]]['rb'].quantile(0.9981), "m")
    R2['hit_hb'] = (R2['rb']>hb.rmin.v)&(R2['rb']<hb.rmax.v)
    hb.update_geometry(R2)
    hb.height_range()
    M_HB_fin, M_HB_mirr, M_HB_str, M_HB_tot = bdr.HB_mass_cooling(hb, R2, SF)

    hb.mass_fin = Var(M_HB_fin, "ton")
    hb.mass_mirror = Var(M_HB_mirr, "ton")
    hb.mass_structure = Var(M_HB_str, "ton")
    hb.mass_total = Var(M_HB_tot, "ton")

    plant.sf_n_hels = N_hel                    # Number of heliostats
    plant.sf_area = N_hel*A_h1                     # Solar Field area
    plant.sf_power = sf_power_sim                     # Solar Field power (output)
    plant.pb_power_th = rcv_power_sim        # Power Block thermal power (input)
    plant.pb_power_el = pb_power_sim        # Power Block electrical power (output)
    plant.storage_heat = rcvr_output["heat_stored"]      # Storage heat (output)
    plant.rcv_massflowrate = rcvr_output["mass_stg"]  # Receiver mass flow rate (to be calculated)

    # Outputs
    results_output = {
        "temp_parts": rcvr_output["temps_parts"],
        "eta_rcv": rcvr_output["eta_rcv"],
        "rad_flux_max": rcvr_output["rad_flux_max"].v,
        "rad_flux_avg": rcvr_output["rad_flux_avg"].v,
        "time_res": rcvr_output["time_res"],
        "vel_parts": rcvr_output["vel_p"],
        "mass_stg": rcvr_output["mass_stg"],
        "heat_stored": rcvr_output["heat_stored"],
        "n_hels": N_hel,
        "pb_power_th": Prcv/SM,
        "pb_power_el": (Prcv/SM)*eta_pb,
        "sf_power": Prcv/eta_rcv,
        "sf_power_sim": sf_power_sim.su("MW"),
        "rcv_power_sim": rcv_power_sim.su("MW"),
        "pb_power_sim": pb_power_sim.su("MW"),
        "rmin": hb.rmin,
        "rmax": hb.rmax,
        "zmin": hb.zmin,
        "zmax": hb.zmax,
        "HB_surface_area": hb.surface_area,
        "TOD_surface_area": tod.surface_area,
        "sf_surface_area": N_hel*A_h1,
        "total_surface_area": hb.surface_area + tod.surface_area + N_hel*A_h1,
        "HB_zmax": hb.zmax,
        "M_HB_fin": hb.mass_fin,
        "M_HB_total": hb.mass_total,
        "TOD_receiver_area": tod.receiver_area,
    }

    costs_out = plant_costing_calculations(plant, results_output=results_output, SF=SF)
    results_output["costs_out"] = costs_out

    return R2, SF, results_output

