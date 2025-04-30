from dataclasses import dataclass, field
import os

import pandas as pd

from antupy.units import Variable

import bdr_csp.SolidParticleReceiver as SPR
import bdr_csp.PerformancePowerCycle as PPC

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))

COLS_INPUT = ['loc', 'Ntower', 'T_stg', 'SM', 'Prcv']
COLS_OUTPUT = ['P_el', 'Q_stg', 'CF_sf', 
               'CF_pb', 'Rev_tot', 'LCOH', 'LCOE', 
               'C_pb', 'C_C', 'RoI', 'PbP', 
               'NPV','Stg_min','date_sim']

def get_data_location(plant: PPC.CSPPlant, location: int) -> None:
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
        plant: PPC.CSPPlant,
        year_i: float = 2019,
        year_f: float = 2019,
        ) -> None:

    receiver_power = plant.receiver_power.get_value("MW")
    Ntower = plant.Ntower
    storage_cap = plant.storage_cap.get_value("hr")
    solar_multiple = plant.solar_multiple.get_value("-")
    eta_pb_des = plant.eta_pb_des.get_value("-")

    
    dT = 0.5            #Time in hours
    print('Charging new dataset')


    df_weather = PPC.load_weather_data(plant.lat, plant.lng, year_i, year_f, dT)
    df_sp = PPC.load_spotprice_data(plant.state, year_i, year_f, dT)
    df = df_weather.merge(df_sp, how="inner", left_index=True, right_index=True)
    
    dT = pd.to_datetime(df_weather.index).freq
    print(dT)

    data = []
    print('\t'.join('{:}'.format(x) for x in COLS_INPUT + COLS_OUTPUT))
    
    #Design power block efficiency and capacity
    pb_power_th = receiver_power / solar_multiple        #[MW] Design value for Power from receiver to power block
    pb_power_el = Ntower * eta_pb_des * pb_power_th      #[MW] Design value for Power Block generation
    storage_heat  = pb_power_th * storage_cap            #[MWh] Capacity of storage per tower


    CSTo['P_pb']  = pb_power_th
    CSTo['P_el']  = pb_power_el
    CSTo['Q_stg'] = storage_heat
    
    # Annual performance
    args = (f_eta_opt, f_eta_TPR, f_eta_pb, dT, DNI_min)
    out = PPC.annual_performance(plant, df, CSTo, SF, args)
    CF_sf,CF_pb,Rev_tot,Rev_day,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV,Stg_min = [out[x] for x in ['CF_sf', 'CF_pb', 'Rev_tot', 'Rev_day', 'LCOH', 'LCOE','C_pb','C_C','RoI','PbP','NPV','Stg_min']]
    
    date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    data.append([Ntower,T_stg,SM,Prcv,P_el,Q_stg,CF_sf,CF_pb,Rev_tot,LCOH,LCOE,C_pb,C_C,RoI,PbP,NPV,Stg_min,date_sim])
    print('\t'.join('{:8.3f}'.format(x) for x in data[-1][:-1]))
    
    res = pd.DataFrame(data,columns=COLS_INPUT + COLS_OUTPUT)


def main():

    zf = Variable(50., "m")
    fzv: Variable  = Variable(0.818161, "-")
    receiver_power: Variable = Variable(19.,"MW")
    flux_avg: Variable = Variable(1.25,"MW/m2")
    
    plant = PPC.CSPPlant(zf, fzv, receiver_power, flux_avg)

    dispatch_single_case(plant)

    return


if __name__ == "__main__":
    main()


