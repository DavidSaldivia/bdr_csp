from dataclasses import dataclass, field
import os
import time

import pandas as pd

from antupy.units import Variable

import bdr_csp.PowerCycle as ppc

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

COLS_INPUT = ppc.COLS_INPUT
COLS_OUTPUT = ppc.COLS_OUTPUT


def get_data_location(plant: ppc.PlantCSPBeamDownParticle, location: int) -> None:
    if location == 1:
        plant.lat = Variable(-20.7,"deg")
        plant.lng = Variable(139.5, "deg")
        plant.state = "QLD"
        
    elif location == 2:
        plant.lat = Variable(-27.0,"deg")
        plant.lng = Variable(135.5, "deg")
        plant.state = "SA"
    
    elif location == 3:
        plant.lat = Variable(-34.5,"deg")
        plant.lng = Variable(141.5, "deg")
        plant.state = "VIC"
        
    elif location == 4:
        plant.lat = Variable(-31.0,"deg")
        plant.lng = Variable(142.0, "deg")
        plant.state = "NSW"
    
    elif location == 5:
        plant.lat = Variable(-25.0,"deg")
        plant.lng = Variable(119.0, "deg")
        plant.state = "SA"

def load_base_case():
    pass


def dispatch_single_case(
        plant: ppc.PlantCSPBeamDownParticle,
        year_i: float = 2019,
        year_f: float = 2019,
        ) -> None:

    receiver_power = plant.receiver_power.get_value("MW")
    Ntower = plant.Ntower
    stg_cap = plant.stg_cap.get_value("hr")
    solar_multiple = plant.solar_multiple.get_value("-")
    pb_eta_des = plant.pb_eta_des.get_value("-")

    
    dT = 0.5            #Time in hours
    print('Charging new dataset')

    df_weather = ppc.load_weather_data(plant.lat, plant.lng, year_i, year_f, dT)
    df_sp = ppc.load_spotprice_data(plant.state, year_i, year_f, dT)
    df = df_weather.merge(df_sp, how="inner", left_index=True, right_index=True)
    
    dT = pd.to_datetime(df_weather.index).freq
    print(dT)

    data = []
    print('\t'.join('{:}'.format(x) for x in COLS_INPUT + COLS_OUTPUT))
    
    #Design power block efficiency and capacity
    pb_power_th = receiver_power / solar_multiple        #[MW] Design value for Power from receiver to power block
    pb_power_el = Ntower * pb_eta_des * pb_power_th      #[MW] Design value for Power Block generation
    storage_heat  = pb_power_th * stg_cap            #[MWh] Capacity of storage per tower

    plant.pb_power_th  = Variable(pb_power_th,"MW")
    plant.pb_power_el  = Variable(pb_power_el,"MW")
    plant.storage_heat = Variable(storage_heat,"MWh")
    
    # Annual performance
    out = ppc.annual_performance(plant, df)
    date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    data.append([Ntower, stg_cap, solar_multiple, receiver_power, 
                 pb_power_el] + [out[x] for x in COLS_OUTPUT] + [date_sim])
    print('\t'.join('{:8.3f}'.format(x) for x in data[-1][:-1]))
    
    return pd.DataFrame(data,columns=COLS_INPUT + COLS_OUTPUT)


def main():
    
    plant = ppc.PlantCSPBeamDownParticle(
        zf = Variable(50., "m"),
        fzv = Variable(0.818161, "-"),
        receiver_power = Variable(19.,"MW"),
        flux_avg = Variable(1.25,"MW/m2"),
    )

    results = dispatch_single_case(plant)

    return


if __name__ == "__main__":
    main()


