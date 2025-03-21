import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

from bdr_csp import bdr as BDR
from bdr_csp import SolidParticleReceiver as SPR

zf = 50.

file_tpr = 'Results_TPR/0_Results_zf_50.csv'
file_hpr = 'Results_HPR_2D/0_Results_zf_50.csv'

df_tpr = pd.read_csv(file_tpr,index_col=0)
df_hpr = pd.read_csv(file_hpr,index_col=0)

Qis = df_hpr.Qavg.unique()
df_tpr = df_tpr[df_tpr.Qavg.isin(Qis)]


fig = plt.figure(figsize=(14,8), constrained_layout=True)
ax1 = fig.add_subplot(121,aspect='equal')
fs=18

ms=['o','s','v','d','*','H','P']
i=0
# cmap = cm.Pastel2
# cmap = cm.get_cmap(cm.viridis, 8)
cmap = cm.viridis
for qi in Qis:
    df_tpra = df_tpr[df_tpr.Qavg==qi]
    df_hpra = df_hpr[df_hpr.Qavg==qi]
    surf = ax1.scatter(df_hpra.Eta_rcv, df_tpra.Eta_rcv, c=df_tpra.Prcv, s=100,marker=ms[i], label=r'$Q_{{avg}}={:.2f}$'.format(qi),cmap=cmap)
    i+=1
    
ax1.plot([0,1],[0,1],lw=3,c='grey',ls='--')
ax1.set_xlim(0.7,0.9)
ax1.set_ylim(0.7,0.9)

cb = fig.colorbar(surf, shrink=0.5, aspect=4)
cb.ax.tick_params(labelsize=fs-2)
fig.text(0.45,0.72,r'$P_{rcv}[MW_{th}]$',fontsize=fs)

ax1.set_xlabel('HPR Efficiency (-)', fontsize=fs)
ax1.set_ylabel('TPR Efficiency (-)', fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs-2)
ax1.legend(loc=4,fontsize=fs-6)
ax1.grid()
# fig.savefig('Results_TPR/Comparison_HPR_TPR_eta.png', bbox_inches='tight')

################################################
# fig = plt.figure(figsize=(9,6))
ax2 = fig.add_subplot(122,aspect='equal')
fs=18

ms=['o','s','v','d','*','H','P']
i=0
cmap = cm.YlOrRd
for qi in Qis:
    df_tpra = df_tpr[df_tpr.Qavg==qi]
    df_hpra = df_hpr[df_hpr.Qavg==qi]
    tpr_Tdiff = df_tpra.Tp_max - df_tpra.Tp_min
    hpr_Tdiff = df_hpra.Tp_max - df_hpra.Tp_min
    ax2.scatter(hpr_Tdiff, tpr_Tdiff, c=df_tpra.Prcv, s=100,marker=ms[i], label=r'$Q_{{avg}}={:.2f}$'.format(qi))
    # surf = ax2.scatter(df_hpra.Tp_min, df_tpra.Tp_min, c=df_tpra.Prcv, s=80,marker=ms[i], label=r'$Q_{{avg}}={:.2f}$'.format(qi))
    i+=1
# fig.text(0.75,0.75,r'$P_{rcv}[MW_{th}]$',fontsize=fs)


ax2.plot([0,350],[0,350],lw=3,c='grey',ls='--')
ax2.set_xlim(100,350)
ax2.set_ylim(100,350)
ax2.set_xlabel('HPR Temperature Difference (K)', fontsize=fs)
ax2.set_ylabel('TPR Temperature Difference (K)', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs-2)
ax2.legend(loc=2, fontsize=fs-6)
ax2.grid()
fig.savefig('Results_TPR/Comparison_HPR_TPR.png', bbox_inches='tight')

plt.show()

####################################################
[results, T_p_TPR, Rcvr, Rcvr_TPR] = pickle.load(open('Results_TPR_basecase.plk','rb'))

lims = Rcvr['x_ini'], Rcvr['x_fin'],Rcvr['y_bot'], Rcvr['y_top'] 
yrcv_min, yrcv_max = lims[2],lims[3]
Ytpr = np.linspace(yrcv_min,yrcv_max,len(T_p_TPR))
plt.scatter(Ytpr,T_p_TPR)

[T_p_HPR, results, Rcvr_HPR] = pickle.load(open('Results_HPR_basecase.plk','rb'))
lims = Rcvr_HPR['x'].max(), Rcvr_HPR['x'].min(),Rcvr_HPR['y'].min(), Rcvr_HPR['y'].max()
yrcv_min,yrcv_max = lims[2],lims[3]
Yhpr = np.linspace(yrcv_min,yrcv_max,len(T_p_HPR))
plt.scatter(Yhpr,T_p_HPR)
plt.show()


plt.show()