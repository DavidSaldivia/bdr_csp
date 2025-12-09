import os
import time
from typing import Any
from dataclasses import dataclass, field

import numpy as np
from antupy import Var, Array, Frame
from antupy import Plant, SimulationOutput, Parametric, component, constraint, derived

import bdr_csp.bdr as bdr
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


def run_test_parametric_optics() -> Frame:
    base_case = bdr.CSPBDRPlant(
        zf=Var(50., "m"),
        fzv=Var(0.83, "-"),
        power_el=Var(10., "MW"),
        Cg=Var(2.0, "-"),
        flux_avg=Var(0.5, "MW/m2"),
        eta_pb=Var(0.50, "-"),
        eta_stg=Var(0.95, "-"),
        eta_rcv=Var(0.75, "-"),
    )
    params_in = {
        "fzv": Array(np.arange(0.80, 0.87, 0.02), "-"),
        "Cg": Array([1.5, 2.0, 3.0, 4.0], "-"),
    }
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "param_optics_test"),
        path_results=os.path.join(DIR_PROJECT, "param_optics_test", "results.csv"),
        include_gitignore=True,
        verbose=True  # Keep verbose to see component caching in action
    )
    # Run parametric analysis and time it
    start_time = time.time()
    results = study.run_analysis()
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Execution time: {exec_time:.2f} seconds")
    return results



def run_parametric_optics():
    """Test function for BDR optical parametric analysis using antupy.Parametric framework."""
    
    # Create base case plant configuration
    base_case = bdr.CSPBDRPlant(
        zf=Var(50., "m"),
        fzv=Var(0.83, "-"),
        power_el=Var(10., "MW"),
        Cg=Var(2.0, "-"),
        geometry='PB',
        array='A',
        flux_avg=Var(0.5, "MW/m2"),
        eta_pb=Var(0.50, "-"),
        eta_stg=Var(0.95, "-"),
        eta_rcv=Var(0.75, "-"),
    )
    params_in = {
        "fzv": Array(np.arange(0.78, 0.95, 0.02), "-"),
        "Cg": Array([1.5, 2.0, 3.0, 4.0], "-"),
    }
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "param_optics"),
        path_results=os.path.join(DIR_PROJECT, "param_optics", "results.csv"),
        include_gitignore=True,
        verbose=True  # Keep verbose to see component caching in action
    )
    
    # Run parametric analysis and time it
    start_time = time.time()
    results_frame = study.run_analysis()
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Execution time: {exec_time:.2f} seconds")
    print(results_frame)
    return None


def main():
    """Main execution function for testing smart components."""
    
    if False:
        print("Testing real BDR simulation performance...")
        run_parametric_optics()

    if True:
        results = run_test_parametric_optics()
        print("Test parametric optics results:")
        print(results)

    pass

if __name__ == "__main__":
    main()