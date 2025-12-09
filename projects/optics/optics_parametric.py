import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, Dict, List
from scipy.optimize import fsolve

from antupy import Array, Var
from antupy.analyser import Parametric

import bdr_csp.bdr as bdr
from bdr_csp.bdr import SolarField, HyperboloidMirror, TertiaryOpticalDevice
from bdr_csp.dir import DIRECTORY

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = DIRECTORY.DIR_DATA

# Input parameters that can be varied in parametric studies
COLS_INPUT = [
    'zf', 'fzv', 'Cg', 'geometry', 'array', 'power_el', 
    'eta_pb', 'eta_stg', 'eta_rcv', 'Q_av'
]

# Output parameters from BDR optical system simulation
COLS_OUTPUT = [
    'eta_cos', 'eta_blk', 'eta_att', 'eta_hbi', 'eta_tdi', 'eta_tdr',
    'eta_TOD', 'eta_BDR', 'eta_SF', 'Pel_real', 'N_hel', 'S_hel',
    'S_HB', 'S_TOD', 'H_TOD', 'rO', 'Q_max', 'status'
]

@dataclass
class CSPBDRPlant:
    """Enhanced CSP Beam Down Receiver Plant configuration for optical analysis."""
    
    # Environment conditions
    Gbn: Var = Var(950, "W/m2")
    day: int = 80
    omega: Var = Var(0.0, "rad")
    lat: Var = Var(-23., "deg")
    lng: Var = Var(115.9, "deg")
    temp_amb: Var = Var(300., "K")
    
    # Power and efficiency targets
    power_el: Var = Var(10.0, "MW")
    eta_pb: Var = Var(0.50, "-")
    eta_stg: Var = Var(0.95, "-")
    eta_rcv: Var = Var(0.75, "-")
    
    # Solar field characteristics
    eta_sfr: Var = Var(0.97*0.95*0.95, "-")
    eta_rfl: Var = Var(0.95, "-")
    A_h1: Var = Var(2.92**2, "m2")
    N_pan: int = 1
    err_x: Var = Var(0.001, "rad")
    err_y: Var = Var(0.001, "rad")
    type_shadow: str = "simple"
    
    # BDR and Tower characteristics
    zf: Var = Var(50., "m")
    fzv: Var = Var(0.83, "-")
    eta_hbi: Var = Var(0.95, "-")
    
    # TOD characteristics
    geometry: str = 'PB'
    array: str = 'A'
    xrc: Var = Var(0., "m")
    yrc: Var = Var(0., "m")
    fzc: Var = Var(0.20, "-")
    Cg: Var = Var(2.0, "-")
    
    # Flux characteristics
    Q_av: Var = Var(0.5, "MW/m2")
    Q_mx: Var = Var(2.0, "MW/m2")
    
    def __post_init__(self):
        """Calculate derived parameters and initialize output dictionary."""
        # Calculate zrc from fzc and zf
        self.zrc = (self.fzc * self.zf).su("m")
        
        # Calculate zv (vertex height)
        self.zv = (self.fzv * self.zf).su("m")

        # Calculate required solar field power
        self.P_SF = (self.power_el / (self.eta_pb * self.eta_stg * self.eta_rcv)).su("MW")
        
        # Initialize output dictionary for parametric analysis
        self.out = {}

    
    def _get_radius_out(self, A_rcv_rq: float) -> float:
        """Calculate required outer radius for given receiver area."""
        def area_required(rO):
            TOD = TertiaryOpticalDevice(
                geometry=self.geometry, array=self.array, Cg=self.Cg, radius_out=Var(rO, "m"),
                xrc=self.xrc, yrc=self.yrc, zrc=self.zrc,
            )
            return TOD.receiver_area.gv("m2") - A_rcv_rq
        
        return fsolve(area_required, 1.0)[0]
    
    
    def run_simulation(self, verbose: bool = True) -> None:
        """
        Run BDR optical system simulation.
        
        This method contains the core simulation logic extracted from the original
        run_parametric function, but adapted to work with antupy.Parametric framework.
        Results are stored in self.out dictionary.
        """
        try:
            if verbose:
                print(f"Running BDR simulation: zf={self.zf.gv('m'):.0f}m, "
                      f"fzv={self.fzv.gv('-'):.3f}, Cg={self.Cg.gv('-'):.1f}, "
                      f"{self.geometry}-{self.array}")
            
            # Setup file paths for solar field data
            file_SF = os.path.join(
                DIR_DATA, 
                'mcrt_datasets_final', 
                f'Dataset_zf_{self.zf.gv("m"):.0f}'
            )
            
            # Calculate required receiver area and outer radius
            rO = Var( self._get_radius_out((self.P_SF / self.Q_av).gv("m2")), "m")
            
            # Create BDR components
            HSF = SolarField(
                zf=self.zf, A_h1=self.A_h1, N_pan=self.N_pan, file_SF=file_SF
            )
            HB = HyperboloidMirror(
                zf=self.zf, fzv=self.fzv, 
                xrc=self.xrc, yrc=self.yrc, zrc=self.zrc, 
                eta_hbi=self.eta_hbi
            )
            TOD = TertiaryOpticalDevice(
                geometry=self.geometry, array=self.array, radius_out=rO, 
                Cg=self.Cg, xrc=self.xrc, yrc=self.yrc, zrc=self.zrc,
            )
            
            # Run heliostat selection using BDR module
            R2, Etas, SF, status = bdr.heliostat_selection(self, HSF, HB, TOD)
            
            # Calculate results
            N_hel = Var(len(R2[R2["hel_in"]]["hel"].unique()), "-")
            S_hel = N_hel * self.A_h1
            Pth_real = Var(SF[SF['hel_in']]['Q_h1'].sum(), "MW")
            Pel_real = Pth_real * (self.eta_pb * self.eta_stg * self.eta_rcv)
            
            # Calculate radiation flux
            N_hel_total = len(R2[R2["hel_in"]]["hel"].unique())
            total_rad = float(Etas['Eta_SF']) * (self.Gbn * self.A_h1 * N_hel_total)
            Q_rcv, _, _ = TOD.radiation_flux(R2, total_rad.gv("MW"))
            Q_max = Q_rcv.max()
            
            # Store results in self.out dictionary
            self.out = {
                'eta_cos': Var(Etas['Eta_cos'], "-"),
                'eta_blk': Var(Etas['Eta_blk'], "-"), 
                'eta_att': Var(Etas['Eta_att'], "-"),
                'eta_hbi': Var(Etas['Eta_hbi'], "-"),
                'eta_tdi': Var(Etas['Eta_tdi'], "-"),
                'eta_tdr': Var(Etas['Eta_tdr'], "-"),
                'eta_TOD': Var(Etas['Eta_TOD'], "-"),
                'eta_BDR': Var(Etas['Eta_BDR'], "-"),
                'eta_SF': Var(Etas['Eta_SF'], "-"),
                'Pel_real': Pel_real.su("MW"),
                'N_hel': N_hel.su("-"),
                'S_hel': S_hel.su("m2"),
                'S_HB': HB.surface_area.su("m2"),
                'S_TOD': TOD.surface_area.su("m2"),
                'H_TOD': TOD.height.su("m"),
                'rO': TOD.radius_out.su("m"),
                'Q_max': Q_max.su("kW/m2"),
                'status': status
            }
            
            if verbose:
                print(f"  Success: N_hel={N_hel:.0f}, eta_SF={Etas['Eta_SF']:.3f}, "
                      f"Q_max={Q_max:.1f} kW/m²")
                
        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            if verbose:
                print(f"  {error_msg}")
            
            # Store error information
            self.out = {
                'status': 'FAILED',
                'error': error_msg,
                # Initialize other outputs with NaN or zero
                'eta_SF': Var(0.0, "-"),
                'N_hel': Var(0, "-"),
                'Q_max': Var(0.0, "kW/m2"),
                'Pel_real': Var(0.0, "MW"),
            }

def test_optics_parametric():
    """Test function for BDR optical parametric analysis using antupy.Parametric framework."""
    
    # Create base case plant configuration
    base_case = CSPBDRPlant(
        zf=Var(50., "m"),
        fzv=Var(0.83, "-"),
        power_el=Var(10., "MW"),
        Cg=Var(2.0, "-"),
        geometry='PB',
        array='A',
        Q_av=Var(0.5, "MW/m2"),
        eta_pb=Var(0.50, "-"),
        eta_stg=Var(0.95, "-"),
        eta_rcv=Var(0.75, "-"),
    )
    
    # Define parameter ranges for parametric study
    params_in = {
        "zf": Array([40, 50, 60], "m"),
        "fzv": Array(np.arange(0.80, 0.87, 0.02), "-"),
        "Cg": Array([1.5, 2.0, 2.5], "-"),
        "geometry": ['PB', 'CPC'],
        "array": ['A', 'B'],
        "power_el": Array([8, 10, 12], "MW"),
    }
    
    # Create parametric study
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        params_out=COLS_OUTPUT,  # Specify output parameters
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "optics_parametric"),
        path_results=os.path.join(DIR_PROJECT, "optics_parametric", "results.csv"),
        include_gitignore=True,
        verbose=True
    )
    
    # Run parametric analysis
    results_frame = study.run_analysis()
    
    print(f"\nParametric study completed!")
    print(f"Total configurations analyzed: {len(results_frame)}")
    print(f"Results saved to: {study.path_results}")
    
    return results_frame

def test_single_case():
    """Test function for single BDR case to verify functionality."""
    
    print("Testing single BDR case...")
    
    # Create test plant
    plant = CSPBDRPlant(
        zf=Var(50., "m"),
        fzv=Var(0.83, "-"),
        power_el=Var(10., "MW"),
        Cg=Var(2.0, "-"),
        geometry='PB',
        array='A',
    )
    
    # Run simulation
    plant.run_simulation(verbose=True)
    
    # Check results
    if plant.out.get('status') != 'FAILED':
        print(f"✓ Single case test passed!")
        print(f"  eta_SF: {plant.out['eta_SF'].gv('-'):.3f}")
        print(f"  N_hel: {plant.out['N_hel'].gv('-'):.0f}")
        print(f"  Q_max: {plant.out['Q_max'].gv('kW/m2'):.1f} kW/m²")
        return True
    else:
        print(f"✗ Single case test failed: {plant.out.get('error', 'Unknown error')}")
        return False

def main():
    """Main execution function."""
    print("BDR Optical Parametric Analysis")
    print("=" * 50)
    
    # Test single case first
    if test_single_case():
        print("\nRunning parametric study...")
        results = test_optics_parametric()
    else:
        print("Single case test failed. Skipping parametric study.")

if __name__ == "__main__":
    main()