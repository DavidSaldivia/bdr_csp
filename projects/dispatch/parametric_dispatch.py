import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from antupy import Array, Var
from antupy.analyser import Parametric

import bdr_csp.pb as pb

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

COLS_INPUT = pb.COLS_INPUT
COLS_OUTPUT = pb.COLS_OUTPUT


def test_parametric_dispatch():
    base_case = pb.ModularCSPPlant(
        zf = Var(50., "m"),
        fzv = Var(0.818161, "-"),
        rcv_power = Var(19.,"MW"),
        flux_avg = Var(1.25,"MW/m2"),
        Ntower = 4,
        rcv_type="HPR_0D",
        solar_multiple=Var(2.0, "-"),
    )
    params_in = {
        "solar_multiple": Array(np.arange(1.0, 3.1, 0.25),"-"),
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

def main():
    test_parametric_dispatch()
    pass

if __name__ == "__main__":
    main()