# -*- coding: utf-8 -*-
"""
Refactored HPR (Horizontal Particle Receiver) 2D Analysis Tool
Created on Mon Oct 17 19:33:51 2022
@author: z5158936
"""

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum

from antupy import Var
from bdr_csp.pb import PlantCSPBeamDownParticle
from bdr_csp.bdr import SolarField, HyperboloidMirror, TertiaryOpticalDevice
from bdr_csp import spr as SPR

class HPRStudyMode(Enum):
    STANDARD = 1
    THICKNESS_STUDY = 2
    COMPARISON_0D = 3
    PARAMETRIC_FULL = 4

@dataclass
class HPRStudyConfig:
    """Configuration for HPR (Horizontal Particle Receiver) parametric studies."""
    
    # Study mode
    study_mode: HPRStudyMode = HPRStudyMode.STANDARD
    
    # Output configuration
    save_detailed_results: bool = False
    output_folder: str = "results_hpr_2d/"
    base_case_folder: str = "cases/"
    
    # Study parameters
    focal_heights: List[float] = field(default_factory=lambda: [50])
    avg_heat_fluxes: List[float] = field(default_factory=lambda: [1.0])
    receiver_powers: List[float] = field(default_factory=lambda: [20])
    thickness_values: List[float] = field(default_factory=lambda: [0.05])
    
    # Base parameters
    geometry: str = "PB"
    array: str = "A"
    concentration_ratio: float = 2.0
    heliostat_area: float = 2.92**2
    n_panels: int = 1
    temp_part_cold: float = 950
    temp_part_hot: float = 1200
    xrc: float = 0.0
    yrc: float = 0.0
    zrc: float = 10.0
    
    # Analysis options
    comparison_0d: bool = False
    thickness_study: bool = False
    
    def __post_init__(self):
        """Configure parameters based on study mode."""
        if self.study_mode == HPRStudyMode.STANDARD:
            self._configure_standard_mode()
        elif self.study_mode == HPRStudyMode.THICKNESS_STUDY:
            self._configure_thickness_mode()
        elif self.study_mode == HPRStudyMode.COMPARISON_0D:
            self._configure_comparison_mode()
        elif self.study_mode == HPRStudyMode.PARAMETRIC_FULL:
            self._configure_parametric_mode()
    
    def _configure_standard_mode(self):
        """Configure for standard single-case analysis."""
        self.focal_heights = [50]
        self.avg_heat_fluxes = [1.0]
        self.receiver_powers = [20]
        self.thickness_values = [0.05]
    
    def _configure_thickness_mode(self):
        """Configure for thickness influence study."""
        self.focal_heights = [50]
        self.avg_heat_fluxes = [1.0]
        self.receiver_powers = [20]
        self.thickness_values = [float(tz) for tz in np.arange(0.01, 0.11, 0.01)]
        self.thickness_study = True
    
    def _configure_comparison_mode(self):
        """Configure for 0D/2D comparison study."""
        self.focal_heights = [50]
        self.avg_heat_fluxes = [1.0, 0.75, 0.50, 1.25, 1.50]
        self.receiver_powers = list(np.append(np.arange(20, 41, 5), np.arange(15, 4, -5)))
        self.thickness_values = [0.05]
        self.comparison_0d = True
    
    def _configure_parametric_mode(self):
        """Configure for full parametric study."""
        self.focal_heights = [40, 50, 60]
        self.avg_heat_fluxes = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.receiver_powers = [float(x) for x in np.arange(10, 31, 5)]
        self.thickness_values = [0.03, 0.05, 0.07]

class HPRAnalyzer:
    """Main analyzer for HPR (Horizontal Particle Receiver) systems."""
    
    def __init__(self, config: HPRStudyConfig):
        self.config = config
        self._ensure_directories()
        self._setup_base_parameters()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.config.output_folder, exist_ok=True)
        os.makedirs(self.config.base_case_folder, exist_ok=True)
    
    def _setup_base_parameters(self) -> None:
        """Setup base parameters for HPR analysis."""
        self.base_params = {
            'geometry': self.config.geometry,
            'array': self.config.array,
            'Cg': self.config.concentration_ratio,
            'Ah1': self.config.heliostat_area,
            'Npan': self.config.n_panels,
            'temp_part_cold': self.config.temp_part_cold,
            'temp_part_hot': self.config.temp_part_hot,
            'xrc': self.config.xrc,
            'yrc': self.config.yrc,
            'zrc': self.config.zrc
        }
    
    def run_single_case_analysis(
        self,
        zf: float,
        Qavg: float,
        Prcv: float,
        tz: float = 0.05,
        polygon_i: int = 1,
        full_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run analysis for a single HPR case.
        
        Args:
            zf: Focal height [m]
            Qavg: Average heat flux [MW/m2]
            Prcv: Receiver power [MW]
            tz: Particle thickness [m]
            polygon_i: Polygon index for receiver
            full_output: Whether to return detailed results
        
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        print(f"Analyzing case: zf={zf:.0f}m, Q_avg={Qavg:.2f}MW/m², P_rcv={Prcv:.1f}MW")
        
        # Create plant configuration
        plant = self._create_plant_configuration(zf, Qavg, Prcv, tz)
        
        # Run coupled simulation
        try:
            R2, SF, simulation_results = SPR.run_coupled_simulation(
                plant, plant.HSF, plant.HB, plant.TOD
            )
            
            # Extract detailed results
            results = self._extract_detailed_results(
                simulation_results, R2, SF, plant, polygon_i
            )
            
            # Add case parameters
            results.update({
                'case_name': f'zf{zf:.0f}_Q{Qavg:.2f}_P{Prcv:.1f}_tz{tz:.3f}',
                'focal_height': zf,
                'avg_heat_flux': Qavg,
                'receiver_power': Prcv,
                'thickness': tz,
                'simulation_time': time.time() - start_time,
                'success': True
            })
            
            # Generate detailed plots if requested
            if self.config.save_detailed_results and full_output:
                self._generate_detailed_plots(
                    results['case_name'], 
                    simulation_results.get('full', {}), 
                    plant.TOD,
                    polygon_i
                )
            
            return results
            
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return {
                'case_name': f'zf{zf:.0f}_Q{Qavg:.2f}_P{Prcv:.1f}_tz{tz:.3f}',
                'focal_height': zf,
                'avg_heat_flux': Qavg,
                'receiver_power': Prcv,
                'thickness': tz,
                'simulation_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _create_plant_configuration(
        self,
        zf: float,
        Qavg: float,
        Prcv: float,
        tz: float
    ) -> PlantCSPBeamDownParticle:
        """Create plant configuration with specified parameters."""
        # Calculate initial receiver area estimate
        Tp_avg = (self.config.temp_part_cold + self.config.temp_part_hot) / 2
        eta_rcv = SPR.HTM_0D_blackbox(Tp_avg, Qavg)[0]
        Arcv = (Prcv / eta_rcv) / Qavg
        
        plant = PlantCSPBeamDownParticle(
            zf=Var(zf, "m"),
            fzv=Var(0.82, '-'),  # Initial guess
            Ah1=Var(self.config.heliostat_area, 'm2'),
            xrc=Var(self.config.xrc, 'm'),
            yrc=Var(self.config.yrc, 'm'),
            zrc=Var(self.config.zrc, 'm'),
            Cg=Var(self.config.concentration_ratio, '-'),
            rcv_area=Var(Arcv, "m2"),
            rcv_type="HPR_2D"
        )
        
        # Update receiver thickness
        plant.receiver.thickness_parts = Var(tz, "m")
        plant.receiver.temp_ini = Var(self.config.temp_part_cold, "K")
        plant.receiver.temp_out = Var(self.config.temp_part_hot, "K")
        plant.receiver.heat_flux_avg = Var(Qavg, "MW/m2")
        plant.receiver.rcv_nom_power = Var(Prcv, "MW")
        
        return plant
    
    def _extract_detailed_results(
        self,
        simulation_results: Dict[str, Any],
        R2: pd.DataFrame,
        SF: pd.DataFrame,
        plant: PlantCSPBeamDownParticle,
        polygon_i: int
    ) -> Dict[str, Any]:
        """Extract detailed results from simulation."""
        # Get receiver results
        temps_parts = simulation_results.get("temp_parts", np.array([]))
        n_hels = simulation_results.get("n_hels", 0)
        rad_flux_max = simulation_results.get("rad_flux_max", 0.0)
        rad_flux_avg = simulation_results.get("rad_flux_avg", 0.0)
        heat_stored = simulation_results.get("heat_stored", 0.0)
        eta_rcv = simulation_results.get("eta_rcv", 0.0)
        mass_stg = simulation_results.get("mass_stg", 0.0)
        time_res = simulation_results.get("time_res", 0.0)
        vel_parts = simulation_results.get("vel_parts", 0.0)
        iteration = simulation_results.get("iteration", 0)
        solve_t_res = simulation_results.get("solve_t_res", False)
        
        # Calculate solar field metrics
        if n_hels > 0 and len(SF) > n_hels:
            eta_SF = SF.iloc[:n_hels]['Eta_SF'].mean()
            eta_hel = SF.iloc[:n_hels]['Eta_hel'].mean()
            eta_BDR = SF.iloc[:n_hels]['Eta_BDR'].mean()
        else:
            eta_SF = eta_hel = eta_BDR = 0.0
        
        # Calculate thermal efficiency
        eta_thermal = eta_SF * eta_rcv
        
        # Calculate receiver area
        receiver_area = plant.TOD.receiver_area.gv("m2")
        
        return {
            'n_heliostats': n_hels,
            'receiver_area': receiver_area,
            'max_heat_flux': rad_flux_max,
            'avg_heat_flux': rad_flux_avg,
            'heat_stored': heat_stored,
            'residence_time': time_res,
            'particle_velocity': vel_parts,
            'heliostat_efficiency': eta_hel,
            'bdr_efficiency': eta_BDR,
            'solar_field_efficiency': eta_SF,
            'receiver_efficiency': eta_rcv,
            'thermal_efficiency': eta_thermal,
            'mass_flow_rate': mass_stg,
            'temp_parts_avg': temps_parts.mean() if len(temps_parts) > 0 else 0.0,
            'temp_parts_max': temps_parts.max() if len(temps_parts) > 0 else 0.0,
            'temp_parts_min': temps_parts.min() if len(temps_parts) > 0 else 0.0,
            'temp_parts_std': temps_parts.std() if len(temps_parts) > 0 else 0.0,
            'iterations': iteration,
            'solve_time_res': solve_t_res,
            'simulation_status': 'converged' if solve_t_res else 'standard'
        }
    
    def run_parametric_study(self) -> pd.DataFrame:
        """Run comprehensive parametric study for HPR systems."""
        results_data = []
        study_start_time = time.time()
        
        print(f"Starting HPR {self.config.study_mode.name} Analysis...")
        print("=" * 80)
        
        # Generate parameter combinations based on study mode
        if self.config.study_mode == HPRStudyMode.THICKNESS_STUDY:
            param_combinations = self._get_thickness_combinations()
        elif self.config.study_mode == HPRStudyMode.COMPARISON_0D:
            param_combinations = self._get_comparison_combinations()
        else:
            param_combinations = self._get_standard_combinations()
        
        for zf, Qavg, Prcv, tz in param_combinations:
            try:
                # Run single case analysis
                results = self.run_single_case_analysis(
                    zf, Qavg, Prcv, tz, full_output=True
                )
                
                if results['success']:
                    results_data.append(results)
                    self._print_result_summary(results)
                else:
                    print(f"Failed: {results['case_name']}")
                
            except Exception as e:
                print(f"Error processing case zf={zf}, Q={Qavg}, P={Prcv}, tz={tz}: {str(e)}")
                continue
        
        # Create and save results DataFrame
        df_results = pd.DataFrame(results_data)
        
        if len(df_results) > 0:
            output_file = os.path.join(
                self.config.output_folder,
                f"HPR_{self.config.study_mode.name.lower()}_results.csv"
            )
            df_results.to_csv(output_file, index=False)
            
            # Generate summary plots
            self._generate_summary_plots(df_results)
        
        print(f"\nParametric study completed in {time.time() - study_start_time:.2f} seconds")
        print(f"Total configurations analyzed: {len(df_results)}")
        print(f"Results saved to: {self.config.output_folder}")
        
        return df_results
    
    def _get_thickness_combinations(self) -> List[Tuple[float, float, float, float]]:
        """Get parameter combinations for thickness study."""
        return [
            (self.config.focal_heights[0], self.config.avg_heat_fluxes[0], 
             self.config.receiver_powers[0], tz)
            for tz in self.config.thickness_values
        ]
    
    def _get_comparison_combinations(self) -> List[Tuple[float, float, float, float]]:
        """Get parameter combinations for 0D comparison study."""
        combinations = []
        for Qavg in self.config.avg_heat_fluxes:
            for Prcv in self.config.receiver_powers:
                combinations.append((
                    self.config.focal_heights[0], Qavg, Prcv, self.config.thickness_values[0]
                ))
        return combinations
    
    def _get_standard_combinations(self) -> List[Tuple[float, float, float, float]]:
        """Get parameter combinations for standard study."""
        combinations = []
        for zf in self.config.focal_heights:
            for Qavg in self.config.avg_heat_fluxes:
                for Prcv in self.config.receiver_powers:
                    for tz in self.config.thickness_values:
                        combinations.append((zf, Qavg, Prcv, tz))
        return combinations
    
    def _print_result_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the current result."""
        print(f"  Case: {results['case_name']}")
        print(f"  N_hel: {results['n_heliostats']:.0f}")
        print(f"  eta_SF: {results['solar_field_efficiency']:.3f}")
        print(f"  eta_rcv: {results['receiver_efficiency']:.3f}")
        print(f"  Q_max: {results['max_heat_flux']:.1f} MW/m²")
        print(f"  Time: {results['simulation_time']:.1f}s")
    
    def run_thickness_influence_study(self) -> pd.DataFrame:
        """Study the influence of particle thickness on receiver performance."""
        print("Running thickness influence study...")
        
        # Set configuration for thickness study
        original_mode = self.config.study_mode
        self.config.study_mode = HPRStudyMode.THICKNESS_STUDY
        self.config._configure_thickness_mode()
        
        # Run parametric study
        results = self.run_parametric_study()
        
        # Restore original mode
        self.config.study_mode = original_mode
        
        # Generate thickness-specific plots
        if len(results) > 0:
            self._plot_thickness_influence(results)
        
        return results
    
    def run_0d_comparison_study(self) -> pd.DataFrame:
        """Compare HPR_2D results with HPR_0D model."""
        print("Running 0D/2D comparison study...")
        
        comparison_data = []
        
        for Qavg in [1.0]:  # Focus on one heat flux for comparison
            for Prcv in [15, 20, 25, 30]:
                
                # HPR_2D case
                results_2d = self.run_single_case_analysis(50, Qavg, Prcv, 0.05)
                
                # HPR_0D case (using simplified model)
                plant_0d = self._create_plant_configuration(50, Qavg, Prcv, 0.05)
                plant_0d.rcv_type = "HPR_0D"
                
                # Run 0D simulation
                results_0d = SPR.rcvr_HPR_0D_simple(
                    plant_0d.to_dict(), plant_0d.HSF.load_dataset()[1]
                )
                
                # Store comparison
                comparison_data.append({
                    'avg_heat_flux': Qavg,
                    'receiver_power': Prcv,
                    'eta_2d': results_2d['receiver_efficiency'],
                    'eta_0d': results_0d['eta_rcv'],
                    'efficiency_ratio': results_2d['receiver_efficiency'] / results_0d['eta_rcv'] if results_0d['eta_rcv'] > 0 else 0,
                    'flux_max_2d': results_2d['max_heat_flux'],
                    'temp_max_2d': results_2d['temp_parts_max'],
                    'time_res_2d': results_2d['residence_time'],
                    'time_res_0d': results_0d['time_res'],
                    'vel_2d': results_2d['particle_velocity'],
                    'vel_0d': results_0d['vel_p']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison results
        output_file = os.path.join(
            self.config.output_folder,
            "HPR_0D_2D_comparison.csv"
        )
        df_comparison.to_csv(output_file, index=False)
        
        # Generate comparison plots
        self._plot_0d_2d_comparison(df_comparison)
        
        return df_comparison
    
    def _generate_detailed_plots(
        self,
        case_name: str,
        rcvr_full: Dict[str, Any],
        TOD: TertiaryOpticalDevice,
        polygon_i: int = 1,
        labels: List[str] = ['eta', 'Tp', 'Q_in']
    ) -> None:
        """Generate detailed visualization plots for receiver analysis."""
        if not rcvr_full:
            return
        
        for label in labels:
            try:
                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(111, aspect='equal')
                
                X = rcvr_full.get('x', np.array([]))
                Y = rcvr_full.get('y', np.array([]))
                Z = rcvr_full.get(label, np.array([]))
                
                if len(X) == 0 or len(Y) == 0 or len(Z) == 0:
                    plt.close(fig)
                    continue
                
                f_s = 16
                
                # Set color map and limits based on label
                if label == 'eta':
                    vmin, vmax = 0.7, 0.9
                    cmap = "YlOrRd"
                    cb_label = r'$\eta_{rcv}$ [-]'
                elif label == 'Tp':
                    vmin = (np.floor(Z.min() / 100) * 100)
                    vmax = (np.ceil(Z.max() / 100) * 100)
                    cmap = "YlOrRd"
                    cb_label = r'$T_p$ [K]'
                elif label == 'Q_in':
                    vmin = 0.0
                    vmax = np.ceil(Z.max())
                    cmap = "YlOrRd"
                    cb_label = r'$Q_{in}$ [MW/m²]'
                else:
                    vmin, vmax = Z.min(), Z.max()
                    cmap = "YlOrRd"
                    cb_label = label
                
                surf = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xlabel('E-W axis (m)', fontsize=f_s)
                ax.set_ylabel('N-S axis (m)', fontsize=f_s)
                cb = fig.colorbar(surf, shrink=0.5, aspect=4)
                cb.ax.tick_params(labelsize=f_s-2)
                cb.set_label(cb_label, fontsize=f_s)
                
                # Add TOD boundaries
                xO, yO = TOD.perimeter_points(TOD.radius_out.gv("m"), tod_index=polygon_i-1)
                lims = xO.min(), xO.max(), yO.min(), yO.max()
                
                x0f = TOD.x0[polygon_i-1]
                y0f = TOD.y0[polygon_i-1]
                radius = TOD.radius_out.gv("m") / np.cos(np.pi / TOD.n_sides)
                ax.add_artist(patches.RegularPolygon(
                    (x0f, y0f),
                    TOD.n_sides,
                    radius=radius,
                    orientation=np.pi / TOD.n_sides,
                    zorder=10, color='black', fill=None
                ))
                
                radius = TOD.radius_ap.gv("m") / np.cos(np.pi / TOD.n_sides)
                ax.add_artist(patches.RegularPolygon(
                    (x0f, y0f),
                    TOD.n_sides,
                    radius=radius,
                    orientation=np.pi / TOD.n_sides,
                    zorder=10, color='black', fill=None
                ))
                
                ax.set_xlim(lims[0], lims[1])
                ax.set_ylim(lims[2], lims[3])
                
                output_path = os.path.join(
                    self.config.output_folder,
                    f'{case_name}_{label}.png'
                )
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error generating plot for {label}: {str(e)}")
                # if 'fig' in locals():
                #     plt.close(fig)
    
    def _generate_summary_plots(self, df_results: pd.DataFrame) -> None:
        """Generate summary plots based on study mode."""
        if self.config.study_mode == HPRStudyMode.THICKNESS_STUDY:
            self._plot_thickness_influence(df_results)
        elif self.config.study_mode == HPRStudyMode.COMPARISON_0D:
            self._plot_efficiency_comparison(df_results)
        else:
            self._plot_performance_trends(df_results)
    
    def _plot_thickness_influence(self, df_results: pd.DataFrame) -> None:
        """Plot thickness influence on receiver performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        thickness = df_results['thickness']
        
        # Efficiency vs thickness
        ax1.plot(thickness, df_results['receiver_efficiency'], 'o-', label='Receiver')
        ax1.plot(thickness, df_results['thermal_efficiency'], 's-', label='Thermal')
        ax1.set_xlabel('Thickness (m)')
        ax1.set_ylabel('Efficiency (-)')
        ax1.set_title('Efficiency vs Thickness')
        ax1.legend()
        ax1.grid(True)
        
        # Residence time and velocity vs thickness
        ax2.plot(thickness, df_results['residence_time'], 'o-', color='blue', label='Residence time')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(thickness, df_results['particle_velocity'], 's-', color='red', label='Velocity')
        ax2.set_xlabel('Thickness (m)')
        ax2.set_ylabel('Residence time (s)', color='blue')
        ax2_twin.set_ylabel('Velocity (m/s)', color='red')
        ax2.set_title('Flow Characteristics vs Thickness')
        ax2.grid(True)
        
        # Temperature statistics vs thickness
        ax3.plot(thickness, df_results['temp_parts_max'], 'o-', label='Max')
        ax3.plot(thickness, df_results['temp_parts_avg'], 's-', label='Avg')
        ax3.plot(thickness, df_results['temp_parts_min'], '^-', label='Min')
        ax3.set_xlabel('Thickness (m)')
        ax3.set_ylabel('Temperature (K)')
        ax3.set_title('Temperature Distribution vs Thickness')
        ax3.legend()
        ax3.grid(True)
        
        # Temperature standard deviation vs thickness
        ax4.plot(thickness, df_results['temp_parts_std'], 'o-', color='purple')
        ax4.set_xlabel('Thickness (m)')
        ax4.set_ylabel('Temperature Std Dev (K)')
        ax4.set_title('Temperature Uniformity vs Thickness')
        ax4.grid(True)
        
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.output_folder,
            "thickness_influence_analysis.png"
        )
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _plot_efficiency_comparison(self, df_results: pd.DataFrame) -> None:
        """Plot efficiency comparison for different configurations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Efficiency vs Power
        for Qavg in df_results['avg_heat_flux'].unique():
            subset = df_results[df_results['avg_heat_flux'] == Qavg]
            ax1.plot(
                subset['receiver_power'],
                subset['thermal_efficiency'],
                marker='o',
                label=f'Q_avg = {Qavg:.2f} MW/m²'
            )
        
        ax1.set_xlabel('Receiver Power (MW)')
        ax1.set_ylabel('Thermal Efficiency (-)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Thermal Efficiency vs Receiver Power')
        
        # Efficiency vs Heat Flux
        for Prcv in sorted(df_results['receiver_power'].unique()):
            subset = df_results[df_results['receiver_power'] == Prcv]
            if len(subset) > 1:
                ax2.plot(
                    subset['avg_heat_flux'],
                    subset['receiver_efficiency'],
                    marker='s',
                    label=f'P_rcv = {Prcv:.0f} MW'
                )
        
        ax2.set_xlabel('Average Heat Flux (MW/m²)')
        ax2.set_ylabel('Receiver Efficiency (-)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Receiver Efficiency vs Heat Flux')
        
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.output_folder,
            "efficiency_comparison.png"
        )
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _plot_performance_trends(self, df_results: pd.DataFrame) -> None:
        """Plot general performance trends."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance vs receiver area
        ax1.scatter(df_results['receiver_area'], df_results['thermal_efficiency'], 
                   c=df_results['receiver_power'], cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Receiver Area (m²)')
        ax1.set_ylabel('Thermal Efficiency (-)')
        ax1.set_title('Efficiency vs Receiver Area')
        cb1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cb1.set_label('Receiver Power (MW)')
        
        # Heat flux distribution
        ax2.scatter(df_results['avg_heat_flux'], df_results['max_heat_flux'],
                   c=df_results['receiver_efficiency'], cmap='plasma', alpha=0.7)
        ax2.set_xlabel('Average Heat Flux (MW/m²)')
        ax2.set_ylabel('Maximum Heat Flux (MW/m²)')
        ax2.set_title('Heat Flux Distribution')
        cb2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cb2.set_label('Receiver Efficiency (-)')
        
        # Flow characteristics
        ax3.scatter(df_results['residence_time'], df_results['particle_velocity'],
                   c=df_results['receiver_efficiency'], cmap='coolwarm', alpha=0.7)
        ax3.set_xlabel('Residence Time (s)')
        ax3.set_ylabel('Particle Velocity (m/s)')
        ax3.set_title('Flow Characteristics')
        cb3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cb3.set_label('Receiver Efficiency (-)')
        
        # Number of heliostats vs power
        ax4.scatter(df_results['receiver_power'], df_results['n_heliostats'],
                   c=df_results['thermal_efficiency'], cmap='Spectral', alpha=0.7)
        ax4.set_xlabel('Receiver Power (MW)')
        ax4.set_ylabel('Number of Heliostats (-)')
        ax4.set_title('Field Size vs Power')
        cb4 = plt.colorbar(ax4.collections[0], ax=ax4)
        cb4.set_label('Thermal Efficiency (-)')
        
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.output_folder,
            "performance_trends.png"
        )
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def _plot_0d_2d_comparison(self, df_comparison: pd.DataFrame) -> None:
        """Plot comparison between 0D and 2D models."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Efficiency comparison
        ax1.scatter(df_comparison['eta_0d'], df_comparison['eta_2d'], alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
        ax1.set_xlabel('0D Model Efficiency (-)')
        ax1.set_ylabel('2D Model Efficiency (-)')
        ax1.set_title('Efficiency Comparison: 0D vs 2D')
        ax1.legend()
        ax1.grid(True)
        
        # Efficiency ratio vs power
        ax2.plot(df_comparison['receiver_power'], df_comparison['efficiency_ratio'], 'o-')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Receiver Power (MW)')
        ax2.set_ylabel('Efficiency Ratio (2D/0D)')
        ax2.set_title('Efficiency Ratio vs Power')
        ax2.grid(True)
        
        # Residence time comparison
        ax3.scatter(df_comparison['time_res_0d'], df_comparison['time_res_2d'], alpha=0.7)
        ax3.plot([0, df_comparison['time_res_0d'].max()], 
                [0, df_comparison['time_res_0d'].max()], 'r--', label='Perfect agreement')
        ax3.set_xlabel('0D Residence Time (s)')
        ax3.set_ylabel('2D Residence Time (s)')
        ax3.set_title('Residence Time Comparison')
        ax3.legend()
        ax3.grid(True)
        
        # Velocity comparison
        ax4.scatter(df_comparison['vel_0d'], df_comparison['vel_2d'], alpha=0.7)
        ax4.plot([0, df_comparison['vel_0d'].max()], 
                [0, df_comparison['vel_0d'].max()], 'r--', label='Perfect agreement')
        ax4.set_xlabel('0D Velocity (m/s)')
        ax4.set_ylabel('2D Velocity (m/s)')
        ax4.set_title('Velocity Comparison')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.output_folder,
            "0d_2d_comparison.png"
        )
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

def run_standard_analysis(
    save_detailed: bool = False,
    output_folder: str = "results_hpr_2d/"
) -> pd.DataFrame:
    """Run standard HPR analysis."""
    config = HPRStudyConfig(
        study_mode=HPRStudyMode.STANDARD,
        save_detailed_results=save_detailed,
        output_folder=output_folder
    )
    analyzer = HPRAnalyzer(config)
    return analyzer.run_parametric_study()

def run_thickness_study(
    save_detailed: bool = True,
    output_folder: str = "results_hpr_2d/"
) -> pd.DataFrame:
    """Run thickness influence study."""
    config = HPRStudyConfig(
        study_mode=HPRStudyMode.THICKNESS_STUDY,
        save_detailed_results=save_detailed,
        output_folder=output_folder
    )
    analyzer = HPRAnalyzer(config)
    return analyzer.run_thickness_influence_study()

def run_comparison_study(
    save_detailed: bool = False,
    output_folder: str = "results_hpr_2d/"
) -> pd.DataFrame:
    """Run 0D/2D comparison study."""
    config = HPRStudyConfig(
        study_mode=HPRStudyMode.COMPARISON_0D,
        save_detailed_results=save_detailed,
        output_folder=output_folder
    )
    analyzer = HPRAnalyzer(config)
    return analyzer.run_0d_comparison_study()

def run_parametric_analysis(
    save_detailed: bool = False,
    output_folder: str = "results_hpr_2d/"
) -> pd.DataFrame:
    """Run full parametric analysis."""
    config = HPRStudyConfig(
        study_mode=HPRStudyMode.PARAMETRIC_FULL,
        save_detailed_results=save_detailed,
        output_folder=output_folder
    )
    analyzer = HPRAnalyzer(config)
    return analyzer.run_parametric_study()

def main():
    """Main execution function for HPR analysis."""
    print("HPR Analysis Suite")
    print("=" * 50)
    
    # Choose which analyses to run by uncommenting the desired lines
    
    # 1. Standard single-case analysis
    # print("\n1. Running standard analysis...")
    # results_standard = run_standard_analysis(save_detailed=True)
    
    # 2. Thickness influence study
    # print("\n2. Running thickness influence study...")
    # results_thickness = run_thickness_study(save_detailed=True)
    
    # 3. 0D/2D comparison study
    # print("\n3. Running 0D/2D comparison study...")
    # results_comparison = run_comparison_study(save_detailed=False)
    
    # 4. Full parametric analysis
    print("\n4. Running full parametric analysis...")
    results_parametric = run_parametric_analysis(save_detailed=False)
    
    print("\nAll requested HPR analyses completed!")

if __name__ == "__main__":
    main()