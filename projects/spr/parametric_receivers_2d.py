from dataclasses import dataclass

import numpy as np
import pandas as pd

from antupy import Var, Array, Plant, SimulationOutput, Parametric

import bdr_csp.bdr as bdr
import bdr_csp.spr as spr

ParticleReceiver = spr.HPR0D | spr.HPR2D | spr.TPR2D
        

def test_parametric() -> pd.DataFrame:

    params_in = {
        "rcv_nom_power": Array([5.,10.,15.], "MW"),
        "rcv_type": ["TPR2D", "HPR0D", "HPR2D"],
    }
    base_case = spr.CSPBeamDownParticlePlant()
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        params_out = ["eta_rcv", "n_hels", "rcv_power_sim", "sf_power_sim", "M_HB_total" ],
    )
    df_out = study.run_analysis()
    return df_out
        

def test_parametric_comparison() -> pd.DataFrame:

    params_in = {
        "rcv_nom_power": Array(np.arange(5,41,5), "MW"),
        "flux_avg": Array(np.arange(0.5,1.51,0.25), "MW/m2"),
        "rcv_type": ["HPR2D", "TPR2D"],
    }
    base_case = BaseCase()
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
    )
    df_out = study.run_analysis()
    return df_out


def main():
    df_out = test_parametric_comparison()
    print(df_out)
    pass

if __name__ == "__main__":
    main()
    pass
