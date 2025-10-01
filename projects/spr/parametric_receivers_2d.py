from dataclasses import dataclass

import numpy as np
import pandas as pd

from antupy import Var, Array, Simulation, Plant

from antupy.analyser.par import Parametric

import bdr_csp.bdr as bdr
import bdr_csp.spr as spr

Receiver = spr.HPR0D | spr.HPR2D | spr.TPR2D

@dataclass
class BaseCase(Simulation):

    rcv_nom_power: Var = Var(19.0, "MW")
    zf : Var = Var(50.0, "m")
    fzv: Var = Var(0.818161, "-")
    flux_avg: Var = Var(1.25, "MW/m2")
    rcv_type: str = "HPR0D"  # Options: "HPR0D", "HPR2D", "TPR2D"
    thickness_part: Var = Var(0.05, "m")

    def __post_init__(self):
        self.plant_bdr = spr.CSPBeamDownPlant(
            rcv_power=self.rcv_nom_power,
            zf=self.zf,
            fzv=self.fzv,
            flux_avg=self.flux_avg,
        )
        self.receiver = self._select_receiver()

    def _select_receiver(self) -> Receiver:
        if self.rcv_type == "HPR0D":
            return spr.HPR0D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part
            )
        elif self.rcv_type == "HPR2D":
            return spr.HPR2D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part
            )
        elif self.rcv_type == "TPR2D":
            return spr.TPR2D(
                rcv_nom_power=self.rcv_nom_power,
                heat_flux_avg=self.flux_avg,
                thickness_parts=self.thickness_part
            )
        else:
            raise ValueError("Invalid receiver type.")


    def run_simulation(self, verbose: bool = True) -> None:
        if verbose:
            print("Running simulation...")

        HSF = self.plant_bdr.HSF
        HB = self.plant_bdr.HB
        TOD = self.plant_bdr.TOD
        BDR = self.plant_bdr
        lat = self.plant_bdr.lat.v
        lng = self.plant_bdr.lng.v
        
        #Getting the RayDataset
        R0, SF = HSF.load_dataset(save_plk=True)

        #Getting interceptions with HB
        R1 = self.plant_bdr.HB.mcrt_direct(R0, refl_error=True)
        R1['hel_in'] = True
        HB.rmin = Var( R1['rb'].quantile(0.0001), "m")
        HB.rmax = Var( R1['rb'].quantile(0.9981), "m")
        R1['hit_hb'] = (R1['rb']>HB.rmin.v) & (R1['rb']<HB.rmax.v)

        SF = HB.shadow_simple( lat=lat, lng=lng, type_shdw="simple", SF=SF)

        #Interceptions with TOD
        R2 = TOD.mcrt_solver(R1, refl_error=False)
        
        ### Optical Efficiencies
        SF = bdr.optical_efficiencies(R2, SF, irradiance=BDR.Gbn, area_hel=BDR.A_h1, reflectivity=BDR.eta_rfl)
        
        ### Running receiver simulation and getting the results
        if isinstance(self.receiver, spr.HPR0D):
            rcvr_output = self.receiver.run_model(SF)
        elif isinstance(self.receiver, spr.HPR2D):
            rcvr_output = self.receiver.run_model(TOD,SF,R2)
        elif isinstance(self.receiver, spr.TPR2D):
            rcvr_output = self.receiver.run_model(TOD,SF,R2)
        else:
            raise ValueError(f"Receiver type '{self.rcv_type}' not recognized.")

        N_hel = rcvr_output["n_hels"] if isinstance(rcvr_output["n_hels"], Var) else Var(rcvr_output["n_hels"], "-")
        eta_rcv = rcvr_output["eta_rcv"] if isinstance(rcvr_output["eta_rcv"], Var) else Var(rcvr_output["eta_rcv"], "-")

        #Plant Parameters
        Prcv = self.plant_bdr.rcv_power
        A_h1 = self.plant_bdr.A_h1

        # Heliostat selection
        Q_acc    = SF['Q_h1'].cumsum()
        hlst     = Q_acc.iloc[:N_hel.v].index
        SF['hel_in'] = SF.index.isin(hlst)
        R2['hel_in'] = R2['hel'].isin(hlst)
        sf_power_sim  = SF[SF["hel_in"]]['Q_h1'].sum()
        rcv_power_sim = sf_power_sim * eta_rcv
        
        SF_avg = SF[SF["hel_in"]].mean()
        eta_bdr = Var(SF_avg['Eta_BDR'], "-")
        eta_sf = Var(SF_avg['Eta_SF'], "-")

        # Calculating HB surface
        HB.rmin = Var(R2[R2["hel_in"]]['rb'].quantile(0.0001), "m")
        HB.rmax = Var(R2[R2["hel_in"]]['rb'].quantile(0.9981), "m")
        R2['hit_hb'] = (R2['rb']>HB.rmin.v)&(R2['rb']<HB.rmax.v)
        HB.update_geometry(R2)
        HB.height_range()
        M_HB_fin, M_HB_mirr, M_HB_str, M_HB_tot = bdr.HB_mass_cooling(HB, R2, SF)
        self.plant_bdr.HB.mass_fin = Var(M_HB_fin, "ton")
        self.plant_bdr.HB.mass_mirror = Var(M_HB_mirr, "ton")
        self.plant_bdr.HB.mass_structure = Var(M_HB_str, "ton")
        self.plant_bdr.HB.mass_total = Var(M_HB_tot, "ton")

        # Outputs
        results_output = {
            "temp_parts": rcvr_output["temps_parts"],
            "temps_diff": rcvr_output["temps_diff"],
            "eta_bdr": eta_bdr,
            "eta_sf": eta_sf,
            "eta_rcv": rcvr_output["eta_rcv"],
            "rad_flux_max": rcvr_output["rad_flux_max"],
            "rad_flux_avg": rcvr_output["rad_flux_avg"],
            "time_res": rcvr_output["time_res"],
            "vel_parts": rcvr_output["vel_parts"],
            "mass_stg": rcvr_output["mass_stg"],
            "heat_stored": rcvr_output["heat_stored"],
            "n_hels": N_hel,
            "iteration": rcvr_output["iteration"],
            "solve_t_res": rcvr_output["solve_t_res"],
            "full": rcvr_output["full"],
            "sf_power": Prcv/eta_rcv,
            "sf_power_sim": sf_power_sim,
            "rcv_power_sim": rcv_power_sim,
            "rmin": HB.rmin,
            "rmax": HB.rmax,
            "zmin": HB.zmin,
            "zmax": HB.zmax,
            "HB_surface_area": HB.surface_area,
            "TOD_surface_area": TOD.surface_area,
            "sf_surface_area": N_hel*A_h1,
            "total_surface_area": HB.surface_area + TOD.surface_area + N_hel*A_h1,
            "M_HB_fin": Var(M_HB_fin, "ton"),
            "M_HB_total": Var(M_HB_tot, "ton"),
        }

        self.out = results_output
        if verbose:
            print("Simulation completed.")
        return None
        

def test_parametric() -> pd.DataFrame:

    params_in = {
        "rcv_nom_power": Array([5.,10.,15.], "MW"),
        "rcv_type": ["TPR2D", "HPR0D", "HPR2D"],
    }
    base_case = BaseCase()
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
        params_out = ["eta_rcv", "n_hels", "rcv_power_sim", "sf_power_sim", "temps_diff" ],
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
