# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:00:57 2020

@author: z5158936
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import bdr_csp.bdr as bdr
from antupy import Var
from bdr_csp.bdr import (
    HyperboloidMirror,
    SolarField,
    TertiaryOpticalDevice,
)

DIR_DATA = "C:/Users/david/OneDrive/academia-pega/2024_25-serc_postdoc/data"
DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

@dataclass
class CSPBDRPlant:
    """Enhanced CSP Beam Down Receiver Plant configuration with all parameters."""
    
    # Environment conditions
    Gbn: Var = Var(950, "W/m2")                    # Design-point DNI [W/m2]
    day: int = 80                                   # Design-point day [-]
    omega: Var = Var(0.0, "rad")                   # Design-point hour angle [rad]
    lat: Var = Var(-23., "deg")                    # Latitude [°]
    lng: Var = Var(115.9, "deg")                   # Longitude [°]
    temp_amb: Var = Var(300., "K")                 # Ambient Temperature [K]
    type_weather: str = 'TMY'                       # Weather source (for CF)
    file_weather: str | None = None                 # Weather source file path
    
    # Receiver and Power Block
    power_el: Var = Var(10.0, "MW")                # Target for Net Electrical Power
    eta_pb: Var = Var(0.50, "-")                   # Power Block efficiency target
    eta_stg: Var = Var(0.95, "-")                  # Storage efficiency target
    eta_rcv: Var = Var(0.75, "-")                  # Receiver efficiency target
    
    # Characteristics of Solar Field
    eta_sfr: Var = Var(0.97*0.95*0.95, "-")        # Solar field reflectivity
    eta_rfl: Var = Var(0.95, "-")                  # Includes mirror refl, soiling and refl. surf. ratio
    A_h1: Var = Var(2.97**2, "m2")                 # Area of one heliostat
    N_pan: int = 1                                  # Number of panels per heliostat
    err_x: Var = Var(0.001, "rad")                 # Reflected error mirror in X direction
    err_y: Var = Var(0.001, "rad")                 # Reflected error mirror in Y direction
    type_shadow: str = "simple"                     # Type of shadow modelling
    
    # Characteristics of BDR and Tower
    zf: Var = Var(50., "m")                        # Focal point height
    fzv: Var = Var(0.83, "-")                      # Position of HB vertex (fraction of zf)
    eta_hbi: Var = Var(0.95, "-")                  # Desired hbi efficiency
    
    # TOD characteristics
    geometry: str = 'PB'                            # Type of TOD
    array: str = 'A'                                # Array of polygonal TODs
    xrc: Var = Var(0., "m")                        # Second focal point (TOD & receiver)
    yrc: Var = Var(0., "m")                        # Second focal point (TOD & receiver)
    fzc: Var = Var(0.20, "-")                      # Second focal point (Height of TOD Aperture, fraction of zf)
    
    # Flux characteristics
    Q_av: Var = Var(0.5, "MW/m2")                  # Desired average radiation flux on receiver
    Q_mx: Var = Var(2.0, "MW/m2")                  # Maximum radiation flux on receiver
    
    def __post_init__(self):
        """Calculate derived parameters."""
        # Calculate zrc if not explicitly provided
        if not hasattr(self, '_zrc_set'):
            self.zrc = (self.fzc * self.zf).su("m")
        
        # Calculate zv (vertex height)
        self.zv = (self.fzv * self.zf).su("m")
        
        # Calculate required solar field power
        self.P_SF = (self.power_el / (self.eta_pb * self.eta_stg * self.eta_rcv)).su("MW")
    
    @property
    def power_th(self) -> Var:
        """Calculate thermal power requirement."""
        return (self.power_el / (self.eta_pb * self.eta_stg)).su("MW")
    
    def set_zrc(self, zrc: Var) -> None:
        """Explicitly set zrc value."""
        self.zrc = zrc
        self.fzc = (self.zrc / self.zf).su("-")
        self._zrc_set = True
    
    def set_zv(self, zv: Var) -> None:
        """Explicitly set vertex height."""
        self.zv = zv
        self.fzv = (zv / self.zf).su("-")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with legacy functions."""
        return {
            'Gbn': self.Gbn.gv("W/m2"),
            'day': self.day,
            'omega': self.omega.gv("rad"),
            'lat': self.lat.gv("deg"),
            'lng': self.lng.gv("deg"),
            'T_amb': self.temp_amb.gv("K"),
            'P_el': self.power_el.gv("MW"),
            'eta_pb': self.eta_pb.gv("-"),
            'eta_sg': self.eta_stg.gv("-"),
            'eta_rcv': self.eta_rcv.gv("-"),
            'eta_sfr': self.eta_sfr.gv("-"),
            'eta_rfl': self.eta_rfl.gv("-"),
            'A_h1': self.A_h1,
            'N_pan': self.N_pan,
            'err_x': self.err_x.gv("rad"),
            'err_y': self.err_y.gv("rad"),
            'zf': self.zf.gv("m"),
            'fzv': self.fzv.gv("-"),
            'eta_hbi': self.eta_hbi.gv("-"),
            'Type': self.geometry,
            'Array': self.array,
            'xrc': self.xrc.gv("m"),
            'yrc': self.yrc.gv("m"),
            'zrc': self.zrc.gv("m"),
            'fzc': self.fzc.gv("-"),
            'Q_av': self.Q_av.gv("MW/m2"),
            'Q_mx': self.Q_mx.gv("MW/m2"),
            'P_SF': self.P_SF.gv("MW"),
            'zv': self.zv.gv("m"),
            'type_shdw': self.type_shadow,
        }


def plot_hb_rad_map(
        case: str,
        R2: pd.DataFrame,
        Etas: pd.DataFrame,
        plant: CSPBDRPlant,
        HB: HyperboloidMirror,
        fldr_rslt: str
    ) -> None:
    """Plot hyperboloid mirror radiation map."""
    
    A_h1 = plant.A_h1.gv("m2")
    hb_rmin = HB.rmin.gv("m")
    hb_rmax = HB.rmax.gv("m")

    f_s = 18
    out2 = R2[(R2['hel_in']) & (R2['hit_hb'])]
    xmin = out2['xb'].min()
    xmax = out2['xb'].max()
    ymin = out2['yb'].min()
    ymax = out2['yb'].max()
    Nx = 100
    Ny = 100
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Nx
    dA = dx*dy
    N_hel = len(R2[R2["hel_in"]]["hel"].unique())
    Fbin = (
        plant.eta_rfl.gv("-") * Etas['Eta_cos'] * Etas['Eta_blk']
        * (plant.Gbn.gv("W/m2") * A_h1 * N_hel)
        / (1e3 * dA * len(out2))
    ) 
    Q_HB, X, Y = np.histogram2d(
        out2['xb'], out2['yb'],
        bins=[Nx, Ny],
        range=[[xmin, xmax], [ymin, ymax]],
        density=False
    )
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, aspect='equal')
    vmin = 0
    vmax = (np.ceil(Fbin*Q_HB.max()/10)*10)
    surf = ax.pcolormesh(
        X, Y, Fbin*Q_HB.transpose(),
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlabel('E-W axis (m)', fontsize=f_s)
    ax.set_ylabel('N-S axis (m)', fontsize=f_s)
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    fig.text(0.77, 0.62, r'$Q_{HB}(kW/m^2)$', fontsize=f_s)
    ax.tick_params(axis='both', which='major', labelsize=f_s)
    
    fig.text(0.77, 0.35, 'Main Parameters', fontsize=f_s-3)
    fig.text(0.77, 0.33, f'$z_{{f\\;}}={plant.zf.gv("m"):.0f} m$', fontsize=f_s-3)
    fig.text(0.77, 0.31, f'$f_{{zv}}={plant.fzv.gv("-"):.2f}$', fontsize=f_s-3)
    fig.text(0.77, 0.29, f'$z_{{rc}}={plant.zrc.gv("m"):.0f} m$', fontsize=f_s-3)
    fig.text(0.77, 0.27, f'$\\eta_{{hbi}}={plant.eta_hbi.gv("-"):.2f}$', fontsize=f_s-3)
    
    r1 = patches.Circle(
        (0., 0.0), hb_rmin,
        zorder=10, color='black', fill=None
    )
    r2 = patches.Circle(
        (0., 0.0), hb_rmax,
        zorder=10, edgecolor='black', fill=None
    )
    ax.add_artist(r1)
    ax.add_artist(r2)
    ax.grid(zorder=20)
    fig.savefig(f'{fldr_rslt}/{case}_QHB_upper.png', bbox_inches='tight')
    plt.close(fig)
    print(Q_HB.sum()*Fbin*dA)


def plot_receiver_rad_map(
        case: str,
        R2: pd.DataFrame,
        Etas: pd.DataFrame,
        plant: CSPBDRPlant,
        TOD: TertiaryOpticalDevice,
        fldr_rslt: str
    ) -> None:
    """Plot receiver radiation map."""
    
    n_tods = TOD.n_tods
    A_h1 = plant.A_h1.gv("m2")
    radius_ap = TOD.radius_ap.gv("m")
    radius_out = TOD.radius_out.gv("m")

    N_hel = len(R2[R2["hel_in"]]["hel"].unique())
    total_rad = Etas['Eta_SF'] * (plant.Gbn.gv("W/m2") * A_h1 * N_hel)
    Q_TOD, X, Y = TOD.radiation_flux(R2, total_rad)
    Q_max = Q_TOD.max()

    print(Q_max, total_rad)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, aspect='equal')
    f_s = 16

    for N in range(n_tods):
        xA, yA = TOD.perimeter_points(radius_ap, tod_index=N)
        xO, yO = TOD.perimeter_points(radius_out, tod_index=N)
        ax.plot(xA, yA, c='k')
        ax.plot(xO, yO, c='k')
    vmin = 0
    vmax = 2000
    surf = ax.pcolormesh(X, Y, Q_TOD, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xlabel('E-W axis (m)', fontsize=f_s)
    ax.set_ylabel('N-S axis (m)', fontsize=f_s)
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s-2)
    fig.text(0.77, 0.65, r'$Q_{{rcv}}(kW/m^2)$', fontsize=f_s)
    fig.savefig(f'{fldr_rslt}/{case}_radmap_out.png', bbox_inches='tight')
    plt.grid()
    plt.close(fig)


def plot_solar_field_etas(
        case: str,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        Etas: pd.DataFrame,
        fldr_rslt: str
    ) -> None:
    """Plot solar field efficiencies."""
    
    R2f = pd.merge(R2, SF[(SF['hel_in'])]['Eta_SF'], how='inner', on=['hel'])
    N_hel = len(R2[R2["hel_in"]]["hel"].unique())

    f_s = 18
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    vmin = 0.0
    vmax = 1.0
    surf = ax1.scatter(
        R2f['xi'], R2f['yi'],
        s=0.5, c=R2f['Eta_SF'],
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax 
    )
    cb = fig.colorbar(surf, shrink=0.25, aspect=4)
    cb.ax.tick_params(labelsize=f_s)
    cb.ax.locator_params(nbins=4)
    
    fig.text(
        0.76, 0.70,
        r'$\overline{\eta_{{SF}}}$' + '={:.3f}'.format(Etas['Eta_SF']),
        fontsize=f_s
    )
    fig.text(
        0.76, 0.65,
        r'$N_{{hel}}$' + '={:d}'.format(N_hel),
        fontsize=f_s
    )
    ax1.set_xlabel('E-W axis (m)', fontsize=f_s)
    ax1.set_ylabel('N-S axis (m)', fontsize=f_s)
    ax1.tick_params(axis='both', which='major', labelsize=f_s)
    ax1.grid()
    fig.savefig(f'{fldr_rslt}/{case}_SF.png', bbox_inches='tight')
    plt.close(fig)


def get_radius_out(*args) -> Var:
    """Calculate required outer radius for given receiver area."""
    def area_required(rO, *args):
        A_rcv_rq, geometry, array, xrc, yrc, zrc, Cg = args
        A_rcv = TertiaryOpticalDevice(
            geometry=geometry, array=array, Cg=Cg, radius_out=Var(rO, "m"),
            xrc=xrc, yrc=yrc, zrc=zrc,
        ).receiver_area.gv("m2")
        return A_rcv - A_rcv_rq
    
    return Var(fsolve(area_required, 1.0, args=args)[0], "m")


def run_parametric(
        tower_heights: List[float],
        fzvs: List[float],
        Cgs: List[float],
        arrays: List[str],
        Pels: List[float],
        geometries: List[str],
        write_results: bool = False,
        plot: bool = True,
    ) -> pd.DataFrame:
    """
    Run parametric study using CSPBDRPlant dataclass instead of CST dictionary.
    
    Args:
        tower_heights: List of focal heights to study
        fzvs: List of vertex height fractions to study
        Cgs: List of concentration ratios to study
        arrays: List of array configurations to study
        Pels: List of electrical powers to study
        geometries: List of TOD geometries to study
        write_results: Whether to write results to files
        plot: Whether to generate plots
    
    Returns:
        DataFrame with parametric study results
    """
    
    fldr_dat = os.path.join(DIR_DATA, 'mcrt_datasets_final')
    file_SF0 = os.path.join(fldr_dat, 'Dataset_zf_{:.0f}')
    
    fldr_rslt = os.path.join(DIR_PROJECT, "testing")
    os.makedirs(fldr_rslt, exist_ok=True)
    
    file_rslt = os.path.join(fldr_rslt, 'testing.txt')
    file_rslt2 = os.path.join(fldr_rslt, 'testing.csv')

    data = []
    COLS_OUTPUT = [
        'Pel', 'zf', 'fzv', 'Cg', 'geometry', 'array',
        'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 'eta_tdi', 'eta_tdr',
        'eta_TOD', 'eta_BDR', 'eta_SF', 'Pel_real', 'N_hel', 'S_hel',
        'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max', 'status',
    ]
    
    txt_header = '\t'.join(COLS_OUTPUT) + '\n'
    
    if write_results:
        with open(file_rslt, 'w') as f:
            f.write(txt_header)

    for (zf, fzv, Cg, array, Pel, geometry) in [
        (zf, fzv, Cg, array, Pel, geometry) 
        for zf in tower_heights 
        for fzv in fzvs 
        for Cg in Cgs 
        for array in arrays 
        for Pel in Pels 
        for geometry in geometries
    ]:
        
        # Create plant configuration
        case = f'zf_{zf:d}_fzv_{fzv:.3f}_Cg_{Cg:.1f}_{geometry}-{array}_Pel_{Pel:.1f}'
        print(case)
        
        # Create CSPBDRPlant instance with current parameters
        plant = CSPBDRPlant(
            zf=Var(zf, "m"),
            fzv=Var(fzv, "-"),
            power_el=Var(Pel, "MW"),
            N_pan=1,
            A_h1=Var(2.92**2, "m2"),
            type_shadow='point',
            geometry=geometry,
            array=array,
        )
        
        # Files for initial data set and HB intersections dataset
        file_SF = file_SF0.format(zf)
        
        Cg_v = Var(Cg, "-")
        
        # Calculate required receiver area
        A_rcv_rq = plant.P_SF.gv("MW") / plant.Q_av.gv("MW/m2")
        rO = get_radius_out(A_rcv_rq, geometry, array, plant.xrc, plant.yrc, plant.zrc, Cg_v)

        # Create BDR components
        HSF = SolarField(zf=plant.zf, A_h1=plant.A_h1, N_pan=plant.N_pan, file_SF=file_SF)
        HB = HyperboloidMirror(
            zf=plant.zf, fzv=plant.fzv, 
            xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc, 
            eta_hbi=plant.eta_hbi
        )
        TOD = TertiaryOpticalDevice(
            geometry=geometry, array=array, radius_out=rO, Cg=Cg_v,
            xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc,
        )

        # Convert plant to dict for compatibility with existing functions
        plant_dict = plant.to_dict()
        
        # Call the heliostat selection function
        R2, Etas, SF, status = bdr.heliostat_selection(plant, HSF, HB, TOD)
        
        # Post-calculations
        N_hel = len(R2[R2["hel_in"]]["hel"].unique())
        S_hel = N_hel * HSF.A_h1.gv("m2")
        Pth_real = SF[SF['hel_in']]['Q_h1'].sum()
        Pel_real = Pth_real * (plant.eta_pb.gv("-") * plant.eta_stg.gv("-") * plant.eta_rcv.gv("-"))
        Q_max = TOD.radiation_flux(R2, Pth_real*1e6)[0].max()
        
        # Store results
        results_row = [
            Pel, zf, fzv, Cg, geometry, array,
            Etas['Eta_cos'], Etas['Eta_blk'], Etas['Eta_att'], 
            Etas['Eta_hbi'], Etas['Eta_tdi'], Etas['Eta_tdr'], 
            Etas['Eta_TOD'], Etas['Eta_BDR'], Etas['Eta_SF'],
            Pel_real, N_hel, S_hel,
            HB.surface_area.gv("m2"), 
            TOD.surface_area.gv("m2"),
            TOD.height.gv("m"),
            TOD.radius_out.gv("m"),
            Q_max,
            status
        ]
        
        data.append(results_row)
        
        # Format text output
        text_r = (
            '\t'.join(f'{x:.3f}' for x in [Pel, zf, fzv, Cg])
            + f'\t{geometry}\t{array}\t'
            + '\t'.join(f'{Etas[x]:.4f}' for x in [
                'Eta_cos', 'Eta_blk', 'Eta_att', 
                'Eta_hbi', 'Eta_tdi', 'Eta_tdr', 
                'Eta_TOD', 'Eta_BDR', 'Eta_SF'
            ]) + '\t'
            + '\t'.join(f'{x:.2f}' for x in [
                Pel_real, N_hel, S_hel,
                HB.surface_area.gv("m2"),
                TOD.surface_area.gv("m2"),
                TOD.height.gv("m"),
                TOD.radius_out.gv("m"),
                Q_max
            ]) + f'\t{status}\n'
        )
        
        print(text_r[:-2])
        
        if write_results:
            with open(file_rslt, 'a') as f:
                f.write(text_r)
            
            df = pd.DataFrame(data, columns=COLS_OUTPUT)
            df.to_csv(file_rslt2, index=False)

        # Generate plots if requested
        if plot:
            plot_hb_rad_map(case, R2, Etas, plant, HB, fldr_rslt)
            plot_receiver_rad_map(case, R2, Etas, plant, TOD, fldr_rslt)
            plot_solar_field_etas(case, R2, SF, Etas, fldr_rslt)

    return pd.DataFrame(data, columns=COLS_OUTPUT)


def main():
    """Main execution function."""
    
    results_df = run_parametric(
        tower_heights=[50],
        fzvs=[0.83],
        Cgs=[2.0],
        arrays=['A'],
        geometries=['PB', "CPC"],
        Pels=[10],
        write_results=True,
        plot=True
    )
    
    print(f"\nParametric study completed.")
    print(f"Total configurations analyzed: {len(results_df)}")
    print(f"Results saved to: {os.path.join(DIR_PROJECT, 'testing')}")


if __name__ == "__main__":
    main()