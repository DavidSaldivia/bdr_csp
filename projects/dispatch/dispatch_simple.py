import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from antupy import Var

import bdr_csp.pb as pb

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

COLS_INPUT = pb.COLS_INPUT
COLS_OUTPUT = pb.COLS_OUTPUT

def get_data_location(plant: pb.ModularCSPPlant, location: int) -> None:
    if location == 1:
        plant.lat = Var(-20.7,"deg")
        plant.lng = Var(139.5, "deg")
        plant.state = "QLD"
        
    elif location == 2:
        plant.lat = Var(-27.0,"deg")
        plant.lng = Var(135.5, "deg")
        plant.state = "SA"
    
    elif location == 3:
        plant.lat = Var(-34.5,"deg")
        plant.lng = Var(141.5, "deg")
        plant.state = "VIC"
        
    elif location == 4:
        plant.lat = Var(-31.0,"deg")
        plant.lng = Var(142.0, "deg")
        plant.state = "NSW"
    
    elif location == 5:
        plant.lat = Var(-25.0,"deg")
        plant.lng = Var(119.0, "deg")
        plant.state = "SA"
    
    else:
        raise ValueError("Location not recognized")


def dispatch_single_case(
        plant: pb.ModularCSPPlant,
        year_i: int = 2019,
        year_f: int = 2019,
        ) -> dict:

    rcv_power = plant.rcv_power
    Ntower = plant.Ntower
    stg_cap = plant.stg_cap
    solar_multiple = plant.solar_multiple
    pb_eta_des = plant.pb_eta_des

    latitude = plant.lat.gv("deg")
    longitude = plant.lng.gv("deg")
    
    dT = 0.5            #Time in hours

    df_weather = pb.load_weather_data(latitude, longitude, year_i, year_f, dT)
    df_sp = pb.load_spotprice_data(plant.state, year_i, year_f, dT)
    df = df_weather.merge(df_sp, how="inner", left_index=True, right_index=True)
    
    dT = pd.to_datetime(df.index).freq
    print(dT)

    data = []
    print('\t'.join('{:}'.format(x) for x in COLS_INPUT + COLS_OUTPUT))
    
    #Design power block efficiency and capacity
    plant.pb_power_th = Ntower * rcv_power / solar_multiple  #[MW] Design value for Power from receiver to power block
    plant.pb_power_el = pb_eta_des * plant.pb_power_th       #[MW] Design value for Power Block generation
    plant.storage_heat = plant.pb_power_th * stg_cap        #[MWh] Capacity of storage per tower
    
    # Annual performance
    out = pb.annual_performance(plant, df)

    date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    data.append([Ntower, stg_cap.v, solar_multiple.v, rcv_power.v, 
                 plant.pb_power_el.v] + [out[x].v for x in COLS_OUTPUT] + [date_sim])
    print('\t'.join('{:8.3f}'.format(x) for x in data[-1][:-1]))
    
    return out

def main():

    plant = pb.ModularCSPPlant(
        zf = Var(50., "m"),
        fzv = Var(0.818161, "-"),
        rcv_power = Var(19.,"MW"),
        flux_avg = Var(1.25,"MW/m2"),
        Ntower = 4,
        rcv_type="HPR_0D",
        solar_multiple=Var(2.0, "-"),
    )
    R2, SF = plant.run_thermal_subsystem()
    results = dispatch_single_case(plant, year_i=2016, year_f=2020)
    print(results)

    return


if __name__ == "__main__":
    main()


