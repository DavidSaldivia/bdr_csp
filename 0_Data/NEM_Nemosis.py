# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:17:19 2022

@author: z5158936
"""
#%% SPOT PRICES: USING NEMOSIS
import pandas as pd
import numpy as np
import scipy.optimize as spo
from scipy.integrate import quad
import scipy.interpolate as spi
import scipy.linalg as sla
import cantera as ct
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from os.path import isfile
import os
import pickle
import time
import sys

from nemosis import dynamic_data_compiler
from nemosis import defaults
print(defaults.dynamic_tables)

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
mainDir = os.path.dirname(fileDir)

# start_time = '2010/01/01 00:00:00'
# end_time = '2021/01/01 00:00:00'
# # table = 'DISPATCHPRICE'
# table = 'TRADINGREGIONSUM'
# raw_data_cache = os.path.dirname(os.path.realpath(__file__))+'/SpotPrice_NEM'

# # price_data = dynamic_data_compiler(start_time, end_time, table, raw_data_cache)

# price_data2 = dynamic_data_compiler(start_time, end_time, table, raw_data_cache)

# print(price_data2.groupby('REGIONID').mean())
# sys.exit()
#%% SPOT PRICES: USING NEMOSIS
from nemosis import dynamic_data_compiler
from nemosis import defaults
print(defaults.dynamic_tables)

start_time = '2022/01/01 00:00:00'
end_time = '2022/01/31 00:00:00'
table = 'DISPATCHPRICE'
# table = 'TRADINGREGIONSUM'
# table = 'BIDDAYOFFER_D'

raw_data_cache = os.path.dirname(os.path.realpath(__file__))+'/NEM_SpotPrice'

# price_data = dynamic_data_compiler(start_time, end_time, table, raw_data_cache)

# price_data2 = dynamic_data_compiler(start_time, end_time, table, raw_data_cache)
data_bids = dynamic_data_compiler(start_time, end_time, table, raw_data_cache)

# print(data_bids.groupby('REGIONID').mean())

#%% REDUCING SIZE OF FILES

# fldr_data = os.path.dirname(os.path.realpath(__file__))+'/NEM_Bidding'

# list_files = os.listdir(fldr_data)
# list_files.sort()
# list_files =  [x for x in list_files if ('BIDPEROFFER_D') in x]

# for file in list_files:
    
#     df = pd.read_csv(os.path.join(fldr_data,file),header=1)
#     df = df[df.BIDTYPE=='ENERGY']
#     df.drop(['I','BID','BIDPEROFFER_D','2','BIDTYPE','ENABLEMENTMIN','ENABLEMENTMAX','LOWBREAKPOINT','HIGHBREAKPOINT','MR_CAPACITY','LASTCHANGED','OFFERDATE'],axis=1,inplace=True,errors='ignore')
#     df.to_csv(os.path.join(fldr_data,file))
    

#%% PPREPROCESSING SPOT PRICES
file_SP = os.path.join(mainDir,'0_Data','NEM_SpotPrice','NEM_TRADINGPRICE_2010-2020.CSV')
df = pd.read_csv(file_SP)
df.rename(columns={'NSW1':'SP_NSW','QLD1':'SP_QLD','SA1':'SP_SA','TAS1':'SP_TAS','VIC1':'SP_VIC'}, inplace=True)


tz = 'Australia/Brisbane'
df['datetime'] = pd.to_datetime(df['time'])
df['datetime'] = df.datetime.dt.tz_localize(tz)
df.set_index('datetime',inplace=True)

df['year'] = df.index.year
df['month'] = df.index.month

file_SP = os.path.join(mainDir,'0_Data','NEM_SpotPrice','NEM_TRADINGPRICE_2010-2020.PLK')
df.to_pickle(file_SP)

#%% PLOTTING THE SPOT PRICES

file_SP = os.path.join(mainDir,'0_Data','NEM_SpotPrice','NEM_TRADINGPRICE_2010-2020.PLK')
df = pd.read_pickle(file_SP)

# df = df[(df.index.year>=2012)&(df.index.year<=2020)]
dff = df.groupby(df.index.to_period('Q')).mean()

fig, ax = plt.subplots(figsize=(9,6))
fs = 16
ms=['o','v','s','d','*','H','P']

states = {1:'QLD',2:'SA',3:'VIC',4:'NSW'}
colors = {'WA':'red', 'SA':'orange', 'NT':'orange', 'QLD':'maroon', 'VIC':'darkblue', 'NSW':'dodgerblue', 'TAS':'green'}



sts=['NSW','QLD','SA','VIC']
i=0
for i in range(len(sts)):
    st = sts[i]
    ax.plot(dff.year+dff.month/12,dff['SP_'+st],lw=2,marker=ms[i],c=colors[st],label=st)
    i+=1
ax.set_xlabel('Time (-)',fontsize=fs)
ax.set_ylabel('Quarterly Average Spot Price [AUD/MWh]',fontsize=fs)
ax.set_xlim(2012,2021)
ax.tick_params(axis='both', which='major', labelsize=fs-2)
ax.tick_params(axis='x', which='major', rotation=45,labelsize=fs-2)
ax.legend(loc=0,fontsize=fs)

ax2 = ax.twinx()
mn, mx = ax.get_ylim()
ax2.set_ylim(mn/1.4, mx/1.4)
ax2.set_ylabel('Quarterly Average Spot Price [USD/MWh]',fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)

ax.grid()
plt.show()

fig.savefig(os.path.join(mainDir,'0_Data','NEM_SpotPrice','Prices.png'), bbox_inches='tight')


#################################################
for year in [2016,2017,2018,2019,2020]:
    dff = df[(df.index.year==year)]
    dff = dff.groupby(dff.index.hour).mean()
    
    fig, ax = plt.subplots(figsize=(9,6))
    fs = 16
    ms=['o','v','s','d','*','H','P']
    
    sts=['NSW','QLD','SA','VIC']
    i=0
    for i in range(len(sts)):
        st = sts[i]
        ax.plot(dff.index,dff['SP_'+st],lw=3,marker=ms[i],markersize=10,c=colors[st],label=st)
        i+=1
    ax.set_xlabel('Time (-)',fontsize=fs)
    ax.set_ylabel('Hourly Average Spot Price [AUD/MWh]',fontsize=fs)
    ax.set_xlim(-1,24)
    ax.set_ylim(0,220)
    ax.set_xticks(np.arange(0,25,4))
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.tick_params(axis='x', which='major',labelsize=fs-2)
    ax.legend(loc=0,fontsize=fs)
    ax.grid()
    
    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn/1.4, mx/1.4)
    ax2.set_ylabel('Hourly Average Spot Price [USD/MWh]',fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)
    
    fig.savefig(os.path.join(mainDir,'0_Data','NEM_SpotPrice','Prices_{:0d}.png'.format(year)), bbox_inches='tight')
    plt.show()

dff = df[(df.index.year>=2016)&(df.index.year<=2020)]
print(dff.describe())