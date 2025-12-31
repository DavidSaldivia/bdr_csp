import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from antupy import Array, Var
from antupy.analyser import Parametric

import bdr_csp.pb_2 as pb

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

COLS_INPUT = pb.COLS_INPUT
COLS_OUTPUT = pb.COLS_OUTPUT

def testing_plant_design_params():
    plant = pb.ModularCSPPlant(
        zf=Var(50., "m"),
        n_tower=Var(4, "-"),
        solar_multiple=Var(2.0, "-"),
        stg_cap=Var(8.0, "hr"),
    )

    pass


def test_parametric_dispatch():
    base_case = pb.ModularCSPPlant(
        zf = Var(50., "m"),
        n_tower = Var(4, "-"),
        solar_multiple=Var(2.0, "-"),
        stg_cap=Var(8.0, "hr"),
    )
    params_in = {
        "solar_multiple": Array(np.arange(1.0, 3.1, 0.10),"-"),
        "stg_cap": Array(np.arange(4,17,2),"hr"),
    }
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "parametric_dispatch"),
        path_results=os.path.join(DIR_PROJECT, "parametric_dispatch", "results.csv"),
        include_gitignore=True,
    )
    df_out = study.run_analysis()
    pass


def parametric_dispatch_cl():
    base_case = pb.ModularCSPPlant(
        zf=Var(50., "m"),
        n_tower=Var(4, "-"),
        solar_multiple=Var(2.0, "-"),
        stg_cap=Var(8.0, "hr"),
        weather=pb.WeatherCL(),
        market=pb.MarketCL(location="crucero", year_i=2024, year_f=2024, dT=0.5),
    )
    params_in = {
        "solar_multiple": Array(np.arange(1.0, 3.1, 0.10), "-"),
        "stg_cap": Array(np.arange(4, 17, 2), "hr"),
    }
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "parametric_dispatch_cl"),
        path_results=os.path.join(DIR_PROJECT, "parametric_dispatch_cl", "results.csv"),
        include_gitignore=True,
    )
    return study.run_analysis()


def parametric_dispatch_au():
    base_case = pb.ModularCSPPlant(
        zf=Var(50., "m"),
        n_tower=Var(4, "-"),
        solar_multiple=Var(2.0, "-"),
        stg_cap=Var(8.0, "hr"),
        weather=pb.WeatherMerra2(year_i=2019, year_f=2019, dT=0.5),
        market=pb.MarketAU(state="NSW", year_i=2019, year_f=2019, dT=0.5),
    )
    params_in = {
        "solar_multiple": Array(np.arange(1.0, 3.1, 0.10), "-"),
        "stg_cap": Array(np.arange(4, 17, 2), "hr"),
    }
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output=os.path.join(DIR_PROJECT, "parametric_dispatch_au"),
        path_results=os.path.join(DIR_PROJECT, "parametric_dispatch_au", "results.csv"),
        include_gitignore=True,
    )
    return study.run_analysis()

def main():
    if False:
        test_parametric_dispatch()
    if False:
        parametric_dispatch_cl()
    if True:
        parametric_dispatch_au()
    pass

if __name__ == "__main__":
    main()