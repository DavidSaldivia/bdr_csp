# -*- coding: utf-8 -*-
"""
Refactored BDR Selection Analysis Tool
Created on Thu Oct 15 20:00:57 2020
@author: z5158936
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from enum import Enum

import bdr_csp.bdr as bdr
from antupy import Var
from bdr_csp.bdr import (
    CSPBDRPlant,
    HyperboloidMirror,
    SolarField,
    TertiaryOpticalDevice,
)

# Use the same data directory structure as example_parametric.py
DIR_DATA = "C:/Users/david/OneDrive/academia-pega/2024_25-serc_postdoc/data"
DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

class OperationMode(Enum):
    PAPER = 1
    THESIS = 2
    SINGLE_POINT = 3
    VARYING_POWER = 4

@dataclass
class BDRStudyConfig:
    """Configuration for BDR parametric studies."""
    
    # Operation mode
    op_mode: OperationMode = OperationMode.SINGLE_POINT
    
    # Output configuration
    write_results: bool = False
    plot_results: bool = True
    save_detailed_plots: bool = True
    
    # Study parameters (will be overridden based on op_mode)
    focal_heights: List[float] = field(default_factory=lambda: [50])
    vertex_fractions: List[float] = field(default_factory=lambda: [0.83])
    concentration_ratios: List[float] = field(default_factory=lambda: [2.0])
    array_configs: List[str] = field(default_factory=lambda: ['A'])
    geometries: List[str] = field(default_factory=lambda: ['PB'])
    electrical_powers: List[float] = field(default_factory=lambda: [10])
    
    # Solar field parameters
    n_panels: int = 1
    heliostat_area: float = 2.92**2
    latitude: float = -23.0
    shadow_type: str = 'simple'
    
    # Data directories
    data_folder: str = ""
    results_folder: str = ""
    
    def __post_init__(self):
        """Configure parameters based on operation mode."""
        if self.op_mode == OperationMode.PAPER:
            self._configure_paper_mode()
        elif self.op_mode == OperationMode.THESIS:
            self._configure_thesis_mode()
        elif self.op_mode == OperationMode.SINGLE_POINT:
            self._configure_single_point_mode()
        elif self.op_mode == OperationMode.VARYING_POWER:
            self._configure_varying_power_mode()
        
        # Set data and results folders using new structure
        if not self.data_folder:
            # Use the same data directory structure as example_parametric.py
            self.data_folder = os.path.join(DIR_DATA, 'mcrt_datasets_final')
        
        if not self.results_folder:
            folder_names = {
                OperationMode.PAPER: "results_paper",
                OperationMode.THESIS: "results_thesis", 
                OperationMode.SINGLE_POINT: "results_single_point",
                OperationMode.VARYING_POWER: "results_varying_p_el"
            }
            self.results_folder = os.path.join(DIR_PROJECT, folder_names[self.op_mode])
    
    def _configure_paper_mode(self):
        """Configure parameters for paper analysis."""
        self.focal_heights = [50]
        self.vertex_fractions = [float(x) for x in np.arange(0.750, 0.910, 0.02)]
        self.concentration_ratios = [1.5, 2.0, 3.0, 4.0]
        self.array_configs = ['A', 'B', 'C', 'D', 'E', 'F']
        self.geometries = ['PB']
        self.electrical_powers = [10.0]
        self.n_panels = 16
        self.heliostat_area = 7.07 * 7.07
    
    def _configure_thesis_mode(self):
        """Configure parameters for thesis analysis."""
        self.focal_heights = [50]
        self.vertex_fractions = [float(x) for x in np.arange(0.770, 0.910, 0.02)]
        self.concentration_ratios = [1.5, 2.0, 3.0, 4.0]
        self.array_configs = ['A', 'B', 'C', 'D', 'E', 'F']
        self.geometries = ['PB']
        self.electrical_powers = [10.0]
        self.shadow_type = 'simple'
    
    def _configure_single_point_mode(self):
        """Configure parameters for single point analysis."""
        self.focal_heights = [50]
        self.vertex_fractions = [0.82]
        self.concentration_ratios = [2.0]
        self.array_configs = ['A']
        self.geometries = ['PB']
        self.electrical_powers = [10]
        self.shadow_type = 'simple'
    
    def _configure_varying_power_mode(self):
        """Configure parameters for varying power analysis."""
        self.focal_heights = [75, 100]
        self.vertex_fractions = [float(x) for x in np.arange(0.71, 0.74, 0.02)]
        self.concentration_ratios = [2.0]
        self.array_configs = ['A']
        self.geometries = ['PB']
        self.electrical_powers = [float(x) for x in np.arange(3, 6, 2)]
        self.shadow_type = 'simple'

class BDRAnalyzer:
    """Main analyzer for BDR parametric studies."""
    
    def __init__(self, config: BDRStudyConfig):
        self.config = config
        self._ensure_directories()
        self._setup_output_files()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.config.results_folder, exist_ok=True)
        if not os.path.exists(self.config.data_folder):
            print(f"Warning: Data folder does not exist: {self.config.data_folder}")
    
    def _setup_output_files(self) -> None:
        """Setup output file paths and headers."""
        mode_suffix = self.config.op_mode.name.lower()
        
        if self.config.op_mode == OperationMode.PAPER:
            self.output_file = os.path.join(self.config.results_folder, f'1-TOD_{self.config.geometries[0]}.txt')
        elif self.config.op_mode == OperationMode.THESIS:
            self.output_file = os.path.join(self.config.results_folder, f'1-{self.config.geometries[0]}_Selection_final.txt')
        elif self.config.op_mode == OperationMode.VARYING_POWER:
            self.output_file = os.path.join(self.config.results_folder, '1-Pvar.txt')
        else:
            self.output_file = os.path.join(self.config.results_folder, f'1-{mode_suffix}.txt')
        
        self.csv_file = self.output_file.replace('.txt', '.csv')
        
        # Define output columns
        self.output_columns = [
            'Pel', 'zf', 'fzv', 'Cg', 'Type', 'Array', 'eta_hbi',
            'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 'eta_tdi', 'eta_tdr',
            'eta_TOD', 'eta_BDR', 'eta_SF', 'Pel_real', 'N_hel', 'S_hel',
            'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max', 'status'
        ]
        
        # Write header if needed
        if self.config.write_results:
            header = '\t'.join(self.output_columns) + '\n'
            with open(self.output_file, 'w') as f:
                f.write(header)
    
    def get_radius_out(self, A_rcv_rq: float, array: str, plant: CSPBDRPlant, Cg: float) -> float:
        """Calculate required outer radius for given receiver area."""
        def area_required(rO):
            TOD = TertiaryOpticalDevice(
                geometry=plant.geometry, array=array, Cg=Var(Cg, "-"), radius_out=Var(rO, "m"),
                xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc,
            )
            return TOD.receiver_area.gv("m2") - A_rcv_rq
        
        return fsolve(area_required, 1.0)[0]
    
    def run_parametric_study(self) -> pd.DataFrame:
        """Run the complete parametric study."""
        print(f"Starting BDR {self.config.op_mode.name} Analysis...")
        print(f"Data folder: {self.config.data_folder}")
        print(f"Results folder: {self.config.results_folder}")
        print("=" * 80)
        
        results_data = []
        file_SF0 = os.path.join(self.config.data_folder, 'Dataset_zf_{:.0f}')
        
        # Generate all parameter combinations
        param_combinations = [
            (zf, fzv, Cg, array, Pel, geometry)
            for zf in self.config.focal_heights
            for fzv in self.config.vertex_fractions  
            for Cg in self.config.concentration_ratios
            for array in self.config.array_configs
            for Pel in self.config.electrical_powers
            for geometry in self.config.geometries
        ]
        
        for zf, fzv, Cg, array, Pel, geometry in param_combinations:
            case_name = f'zf_{zf:d}_fzv_{fzv:.3f}_Cg_{Cg:.1f}_{geometry}-{array}_Pel_{Pel:.1f}'
            print(f"Processing: {case_name}")
            
            try:
                # Create plant configuration
                plant = CSPBDRPlant(
                    zf=Var(zf, "m"),
                    fzv=Var(fzv, "-"),
                    power_el=Var(Pel, "MW"),
                    N_pan=self.config.n_panels,
                    A_h1=Var(self.config.heliostat_area, "m2"),
                    type_shadow=self.config.shadow_type,
                    geometry=geometry,
                    array=array,
                    lat=Var(self.config.latitude, "deg")
                )
                
                # Setup file paths
                file_SF = file_SF0.format(zf)
                
                # Calculate required receiver area and outer radius
                A_rcv_rq = plant.P_SF.gv("MW") / plant.Q_av.gv("MW/m2")
                rO = self.get_radius_out(A_rcv_rq, array, plant, Cg)
                
                # Create BDR components using new dataclasses
                HSF = SolarField(
                    zf=plant.zf, A_h1=plant.A_h1, N_pan=plant.N_pan, file_SF=file_SF
                )
                HB = HyperboloidMirror(
                    zf=plant.zf, fzv=plant.fzv, 
                    xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc, 
                    eta_hbi=plant.eta_hbi
                )
                TOD = TertiaryOpticalDevice(
                    geometry=geometry, array=array, radius_out=Var(rO, "m"), Cg=Var(Cg, "-"),
                    xrc=plant.xrc, yrc=plant.yrc, zrc=plant.zrc,
                )
                
                # Run heliostat selection using new function
                R2, Etas, SF, status = bdr.heliostat_selection(plant, HSF, HB, TOD)
                
                # Calculate results
                results = self._calculate_results(plant, R2, Etas, SF, HB, TOD, status)
                results.update({
                    'Pel': Pel, 'zf': zf, 'fzv': fzv, 'Cg': Cg,
                    'Type': geometry, 'Array': array
                })
                
                results_data.append(results)
                
                # Write results if requested
                if self.config.write_results:
                    self._write_result_line(results)
                
                # Generate plots if requested
                if self.config.plot_results:
                    self._generate_plots(case_name, R2, Etas, SF, plant, HB, TOD)
                
                # Print summary
                self._print_result_summary(results)
                
                # Clean up memory
                del R2, SF
                
            except Exception as e:
                print(f"Error processing {case_name}: {str(e)}")
                continue
        
        # Create and save results DataFrame
        df_results = pd.DataFrame(results_data)
        if len(df_results) > 0:
            df_results.to_csv(self.csv_file, index=False)
        
        print(f"\nParametric study completed.")
        print(f"Total configurations analyzed: {len(df_results)}")
        print(f"Results saved to: {self.config.results_folder}")
        
        return df_results
    
    def _calculate_results(
        self, 
        plant: CSPBDRPlant, 
        R2: pd.DataFrame, 
        Etas: pd.Series, 
        SF: pd.DataFrame,
        HB: HyperboloidMirror,
        TOD: TertiaryOpticalDevice,
        status: str
    ) -> Dict[str, float]:
        """Calculate all result metrics."""
        N_hel = len(R2[R2["hel_in"]]["hel"].unique())
        S_hel = N_hel * plant.A_h1.gv("m2")
        Pth_real = SF[SF['hel_in']]['Q_h1'].sum()
        Pel_real = Pth_real * (plant.eta_pb.gv("-") * plant.eta_stg.gv("-") * plant.eta_rcv.gv("-"))
        
        # Calculate radiation flux
        N_hel_total = len(R2[R2["hel_in"]]["hel"].unique())
        total_rad = Etas['Eta_SF'] * (plant.Gbn.gv("W/m2") * plant.A_h1.gv("m2") * N_hel_total)
        Q_rcv, _, _ = TOD.radiation_flux(R2, total_rad*1e6)
        Q_max = Q_rcv.max()
        
        return {
            'eta_cos': Etas['Eta_cos'],
            'eta_blk': Etas['Eta_blk'], 
            'eta_att': Etas['Eta_att'],
            'eta_hbi': Etas['Eta_hbi'],
            'eta_tdi': Etas['Eta_tdi'],
            'eta_tdr': Etas['Eta_tdr'],
            'eta_TOD': Etas['Eta_TOD'],
            'eta_BDR': Etas['Eta_BDR'],
            'eta_SF': Etas['Eta_SF'],
            'Pel_real': Pel_real,
            'N_hel': N_hel,
            'S_hel': S_hel,
            'S_HB': HB.surface_area.gv("m2"),
            'S_TOD': TOD.surface_area.gv("m2"),
            'H_TOD': TOD.height.gv("m"),
            'rO': TOD.radius_out.gv("m"),
            'Q_max': Q_max,
            'status': status        #type: ignore
        }
    
    def _write_result_line(self, results: Dict[str, Any]) -> None:
        """Write a single result line to the output file."""
        # Format the result line as in original code
        text_r = (
            '\t'.join(f'{results[x]:.3f}' for x in ['Pel', 'zf', 'fzv', 'Cg'])
            + f'\t{results["Type"]}\t{results["Array"]}\t'
            + '\t'.join(f'{results[x]:.4f}' for x in [
                'eta_hbi', 'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 
                'eta_tdi', 'eta_tdr', 'eta_TOD', 'eta_BDR', 'eta_SF'
            ]) + '\t'
            + '\t'.join(f'{results[x]:.2f}' for x in [
                'Pel_real', 'N_hel', 'S_hel', 'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max'
            ]) + f'\t{results["status"]}\n'
        )
        
        with open(self.output_file, 'a') as f:
            f.write(text_r)
    
    def _print_result_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the current result."""
        print(f"  Status: {results['status']}")
        print(f"  N_hel: {results['N_hel']:.0f}")
        print(f"  eta_SF: {results['eta_SF']:.3f}")
        print(f"  Q_max: {results['Q_max']:.1f} kW/m²")
    
    def _generate_plots(
        self,
        case_name: str,
        R2: pd.DataFrame,
        Etas: pd.Series,
        SF: pd.DataFrame,
        plant: CSPBDRPlant,
        HB: HyperboloidMirror,
        TOD: TertiaryOpticalDevice
    ) -> None:
        """Generate all visualization plots."""
        if self.config.save_detailed_plots:
            self._plot_hb_radiation_map(case_name, R2, Etas, plant, HB)
            self._plot_receiver_radiation_map(case_name, R2, Etas, plant, TOD)
            self._plot_solar_field_efficiencies(case_name, R2, SF, Etas)
    
    def _plot_hb_radiation_map(
        self, 
        case_name: str, 
        R2: pd.DataFrame, 
        Etas: pd.Series, 
        plant: CSPBDRPlant,
        HB: HyperboloidMirror
    ) -> None:
        """Plot hyperboloid radiation map."""
        f_s = 18
        out2 = R2[(R2['hel_in']) & (R2['hit_hb'])]
        
        if len(out2) == 0:
            return
        
        xmin, xmax = out2['xb'].min(), out2['xb'].max()
        ymin, ymax = out2['yb'].min(), out2['yb'].max()
        Nx, Ny = 100, 100
        dx, dy = (xmax-xmin)/Nx, (ymax-ymin)/Ny
        dA = dx*dy
        
        N_hel = len(R2[R2["hel_in"]]["hel"].unique())
        Fbin = (
            plant.eta_rfl.gv("-") * Etas['Eta_cos'] * Etas['Eta_blk']
            * (plant.Gbn.gv("W/m2") * plant.A_h1.gv("m2") * N_hel)
            / (1e3 * dA * len(out2))
        )
        
        Q_HB, X, Y = np.histogram2d(
            out2['xb'], out2['yb'],
            bins=[Nx, Ny],
            range=[[xmin, xmax], [ymin, ymax]],
            density=False
        )
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, aspect='equal')
        X, Y = np.meshgrid(X, Y)
        
        vmin = 0
        vmax = np.ceil(Fbin*Q_HB.max()/10)*10
        surf = ax.pcolormesh(X, Y, Fbin*Q_HB.transpose(), cmap="YlOrRd", vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('E-W axis (m)', fontsize=f_s)
        ax.set_ylabel('N-S axis (m)', fontsize=f_s)
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s)
        fig.text(0.77, 0.62, r'$Q_{HB}(kW/m^2)$', fontsize=f_s)
        ax.tick_params(axis='both', which='major', labelsize=f_s)
        
        # Add parameter text
        fig.text(0.77, 0.35, 'Main Parameters', fontsize=f_s-3)
        fig.text(0.77, 0.33, f'$z_{{f}}={plant.zf.gv("m"):.0f} m$', fontsize=f_s-3)
        fig.text(0.77, 0.31, f'$f_{{zv}}={plant.fzv.gv("-"):.2f}$', fontsize=f_s-3)
        fig.text(0.77, 0.29, f'$z_{{rc}}={plant.zrc.gv("m"):.0f} m$', fontsize=f_s-3)
        fig.text(0.77, 0.27, f'$\\eta_{{hbi}}={plant.eta_hbi.gv("-"):.2f}$', fontsize=f_s-3)
        
        # Add HB radius circles
        rmin, rmax = HB.rmin.gv("m"), HB.rmax.gv("m")
        r1 = patches.Circle((0., 0.), rmin, zorder=10, color='black', fill=None)
        r2 = patches.Circle((0., 0.), rmax, zorder=10, edgecolor='black', fill=None)
        ax.add_artist(r1)
        ax.add_artist(r2)
        ax.grid(zorder=20)
        
        fig.savefig(f'{self.config.results_folder}/{case_name}_QHB_upper.png', bbox_inches='tight')
        plt.close(fig)
    
    def _plot_receiver_radiation_map(
        self,
        case_name: str,
        R2: pd.DataFrame,
        Etas: pd.Series,
        plant: CSPBDRPlant,
        TOD: TertiaryOpticalDevice
    ) -> None:
        """Plot receiver radiation map."""
        xmin, xmax, ymin, ymax = TOD.limits()
        Nx, Ny = 100, 100
        dx, dy = (xmax-xmin)/Nx, (ymax-ymin)/Ny
        dA = dx*dy
        
        out = R2[(R2['hel_in']) & (R2['hit_rcv'])]
        if len(out) == 0:
            return
        
        Nrays = len(out)
        N_hel = len(R2[R2["hel_in"]]["hel"].unique())
        Fbin = Etas['Eta_SF'] * (plant.Gbn.gv("W/m2") * plant.A_h1.gv("m2") * N_hel) / (1e3*dA*Nrays)
        
        Q_TOD, X, Y = np.histogram2d(
            out['xr'], out['yr'],
            bins=[Nx, Ny],
            range=[[xmin, xmax], [ymin, ymax]],
            density=False
        )
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, aspect='equal')
        f_s = 16
        
        # Plot TOD perimeters
        for N in range(TOD.n_tods):
            xA, yA = TOD.perimeter_points(TOD.radius_ap.gv("m"), tod_index=N)
            xO, yO = TOD.perimeter_points(TOD.radius_out.gv("m"), tod_index=N)
            ax.plot(xA, yA, c='k')
            ax.plot(xO, yO, c='k')
        
        X, Y = np.meshgrid(X, Y)
        vmin, vmax = 0, 2000
        surf = ax.pcolormesh(X, Y, Fbin*Q_TOD.transpose(), cmap="YlOrRd", vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('E-W axis (m)', fontsize=f_s)
        ax.set_ylabel('N-S axis (m)', fontsize=f_s)
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s-2)
        fig.text(0.77, 0.65, r'$Q_{rcv}(kW/m^2)$', fontsize=f_s)
        
        fig.savefig(f'{self.config.results_folder}/{case_name}_radmap_out.png', bbox_inches='tight')
        plt.grid()
        plt.close(fig)
    
    def _plot_solar_field_efficiencies(
        self,
        case_name: str,
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        Etas: pd.Series
    ) -> None:
        """Plot solar field efficiencies."""
        R2f = pd.merge(R2, SF[(SF['hel_in'])]['Eta_SF'], how='inner', on=['hel'])
        N_hel = len(R2[R2["hel_in"]]["hel"].unique())
        
        f_s = 18
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        
        vmin, vmax = 0.0, 1.0
        surf = ax1.scatter(
            R2f['xi'], R2f['yi'], 
            s=0.5, c=R2f['Eta_SF'], 
            cmap="YlOrRd", vmin=vmin, vmax=vmax
        )
        
        cb = fig.colorbar(surf, shrink=0.25, aspect=4)
        cb.ax.tick_params(labelsize=f_s)
        cb.ax.locator_params(nbins=4)
        
        fig.text(0.76, 0.70, r'$\overline{\eta_{SF}}$' + f'={Etas["Eta_SF"]:.3f}', fontsize=f_s)
        fig.text(0.76, 0.65, r'$N_{hel}$' + f'={N_hel:d}', fontsize=f_s)
        
        ax1.set_xlabel('E-W axis (m)', fontsize=f_s)
        ax1.set_ylabel('N-S axis (m)', fontsize=f_s)
        ax1.tick_params(axis='both', which='major', labelsize=f_s)
        ax1.grid()
        
        fig.savefig(f'{self.config.results_folder}/{case_name}_SF.png', bbox_inches='tight')
        plt.close(fig)

def run_paper_analysis(write_results: bool = False, plot_results: bool = True) -> pd.DataFrame:
    """Run paper analysis."""
    config = BDRStudyConfig(
        op_mode=OperationMode.PAPER,
        write_results=write_results,
        plot_results=plot_results
    )
    analyzer = BDRAnalyzer(config)
    return analyzer.run_parametric_study()

def run_thesis_analysis(write_results: bool = True, plot_results: bool = True) -> pd.DataFrame:
    """Run thesis analysis."""
    config = BDRStudyConfig(
        op_mode=OperationMode.THESIS,
        write_results=write_results,
        plot_results=plot_results
    )
    analyzer = BDRAnalyzer(config)
    return analyzer.run_parametric_study()

def run_single_point_analysis(write_results: bool = False, plot_results: bool = True) -> pd.DataFrame:
    """Run single point analysis."""
    config = BDRStudyConfig(
        op_mode=OperationMode.SINGLE_POINT,
        write_results=write_results,
        plot_results=plot_results
    )
    analyzer = BDRAnalyzer(config)
    return analyzer.run_parametric_study()

def run_varying_power_analysis(write_results: bool = True, plot_results: bool = True) -> pd.DataFrame:
    """Run varying power analysis."""
    config = BDRStudyConfig(
        op_mode=OperationMode.VARYING_POWER,
        write_results=write_results,
        plot_results=plot_results
    )
    analyzer = BDRAnalyzer(config)
    return analyzer.run_parametric_study()

def main():
    """Main execution function - choose which analyses to run."""
    
    print("BDR Parametric Analysis Suite")
    print("=" * 50)
    
    # Choose which analyses to run by uncommenting the desired lines
    
    # 1. Single point analysis (quick test)
    # print("\n1. Running single point analysis...")
    # results_single = run_single_point_analysis(write_results=False, plot_results=True)
    
    # 2. Paper analysis (comprehensive, for publication)
    # print("\n2. Running paper analysis...")
    # results_paper = run_paper_analysis(write_results=False, plot_results=True)
    
    # 3. Thesis analysis (CPC focus)
    print("\n3. Running thesis analysis...")
    results_thesis = run_thesis_analysis(write_results=True, plot_results=True)
    
    # 4. Varying power analysis
    # print("\n4. Running varying power analysis...")
    # results_power = run_varying_power_analysis(write_results=True, plot_results=True)
    
    print("\nAll requested analyses completed!")

if __name__ == "__main__":
    main()