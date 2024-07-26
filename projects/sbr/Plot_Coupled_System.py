# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:47:45 2020

@author: z5158936
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


file = 'Coupled_Plot'

data = pd.read_excel(file+'.xlsx',header=0)

X1a = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.25]['SED']
Y1a = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.25]['E_d']
m1a = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.25]['m_f']
H1a = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.25]['HDR']

X2a = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.25]['SED']
Y2a = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.25]['E_d']
m2a = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.25]['m_f']
H2a = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.25]['HDR']

X3a = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.25]['SED']
Y3a = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.25]['E_d']
m3a = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.25]['m_f']
H2a = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.25]['HDR']

X1b = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.5]['SED']
Y1b = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.5]['E_d']
m1b = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.5]['m_f']
H1b = data.loc[data['Tavmx']==1000].loc[data['HDR']==0.5]['HDR']

X2b = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.5]['SED']
Y2b = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.5]['E_d']
m2b = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.5]['m_f']
H2b = data.loc[data['Tavmx']==1050].loc[data['HDR']==0.5]['HDR']

X3b = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.5]['SED']
Y3b = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.5]['E_d']
m3b = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.5]['m_f']
H2b = data.loc[data['Tavmx']==1100].loc[data['HDR']==0.5]['HDR']

f1, ax = plt.subplots(1,1,figsize=(14,12))
f_s = 16

plt.scatter( X1a , Y1a , color='navy'    , marker='o', s=50,  label=r'$T_{avmx}=1000 K$')
plt.scatter( X1b , Y1b , color='skyblue' , marker='o', s=50,  label=r'$T_{avmx}=1000 K$')
plt.scatter( X2a , Y2a , color='darkred'  , marker='^', s=50,  label=r'$T_{avmx}=1050 K$')
plt.scatter( X2b , Y2b , color='lightcoral'  , marker='^', s=50,  label=r'$T_{avmx}=1050 K$')
plt.scatter( X3a , Y3a , color='darkgreen'   , marker='s', s=50,  label=r'$T_{avmx}=1100 K$')
plt.scatter( X3b , Y3b , color='yellowgreen'   , marker='s', s=50,  label=r'$T_{avmx}=1100 K$')


for i in X1a.keys():
    if i==4:
        plt.annotate(r'$m_f$='+str(m1a[i])+'kg/s', (X1a[i]-4, Y1a[i]-1.0), fontsize=12)
    else:
        plt.annotate(r'$m_f$='+str(m1a[i])+'kg/s', (X1a[i]-8, Y1a[i]+0.5), fontsize=12)

for i in X1b.keys():
    if i==1:
        plt.annotate(r'$m_f$='+str(m1b[i])+'kg/s', (X1b[i]-10, Y1b[i]-1.1), fontsize=12)
    elif i==2:
        plt.annotate(r'$m_f$='+str(m1b[i])+'kg/s', (X1b[i]-8, Y1b[i]+0.5), fontsize=12)
    else:
        plt.annotate(r'$m_f$='+str(m1b[i])+'kg/s', (X1b[i]-5, Y1b[i]+0.5), fontsize=12)
        
for i in X2a.keys():
    if i==13:
        plt.annotate(r'$m_f$='+str(m2a[i])+'kg/s', (X2a[i]-1., Y2a[i]+0.4), fontsize=12)
    elif i==14:
        plt.annotate(r'$m_f$='+str(m2a[i])+'kg/s', (X2a[i]-2., Y2a[i]-1.1), fontsize=12)
    else:
        plt.annotate(r'$m_f$='+str(m2a[i])+'kg/s', (X2a[i]+1, Y2a[i]-1), fontsize=12)
        
for i in X2b.keys():
    plt.annotate(r'$m_f$='+str(m2b[i])+'kg/s', (X2b[i]+1, Y2b[i]-1), fontsize=12)

for i in X3a.keys():
    if i==25:
        plt.annotate(r'$m_f$='+str(m3a[i])+'kg/s', (X3a[i]-2.5, Y3a[i]-1.2), fontsize=12)
    else:
        plt.annotate(r'$m_f$='+str(m3a[i])+'kg/s', (X3a[i]-8, Y3a[i]+0.7), fontsize=12)
for i in X3b.keys():
    plt.annotate(r'$m_f$='+str(m3b[i])+'kg/s', (X3b[i]-2.5, Y3b[i]-1.2), fontsize=12)

plt.annotate(r'$D_{st}=4m$', (17,7),  fontsize=14, color='blue')
plt.annotate(r'$D_{st}=6m$', (15,17), fontsize=14, color='blue')
plt.annotate(r'$D_{st}=8m$', (15,30), fontsize=14, color='blue')
plt.annotate(r'$D_{st}=10m$',(17,46), fontsize=14, color='blue')

ax.add_patch(Ellipse((83,6),  110, 6,  facecolor='none', edgecolor='blue'))
ax.add_patch(Ellipse((80,15), 120, 7, facecolor='none', edgecolor='blue'))
ax.add_patch(Ellipse((60,28),  80, 5,  facecolor='none', edgecolor='blue'))
ax.add_patch(Ellipse((80,44), 110, 7,  facecolor='none', edgecolor='blue'))

plt.legend(loc=8, bbox_to_anchor=(0.55, -0.2),ncol=3, fontsize=f_s)
plt.text(0.13, 0.025, r'$HDR=0.25(-) \longrightarrow$', fontsize=f_s, transform=plt.gcf().transFigure)
plt.text(0.13, -0.007, r'$HDR=0.50(-) \longrightarrow$', fontsize=f_s, transform=plt.gcf().transFigure)
#plt.annotate(,(-2,-10), color='black',fontsize=f_s)
plt.xlabel(r'Stored Energy Density $(kWh_t/m^3)$', fontsize=f_s)
plt.ylabel(r'Daily Delivered Energy (normalised to 8h charging) $(MWh_e)$', fontsize=f_s)
plt.xticks(fontsize=f_s); plt.yticks(fontsize=f_s)
plt.ylim(0,50)
plt.xlim(0,140)
plt.grid(ls='--'); plt.show()
f1.savefig(file+".png", bbox_inches='tight')
f1.savefig(file+".pdf", bbox_inches='tight')