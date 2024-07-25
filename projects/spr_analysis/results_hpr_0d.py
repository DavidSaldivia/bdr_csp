# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:13:56 2022

@author: z5158936
"""
import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import os
from bdr_csp import SolidParticleReceiver as SPR

def efficiency_plots(
        fldr_rslt: str,
        plot_file: str,
    ) -> None:
    
    air  = ct.Solution('air.yaml')
    Tps = np.arange(700.,2001.,100.)
    qis = [0.25,0.50,1.0,2.0,4.0,0.72]
    data = []
    Fc = 2.57
    for (Tp,qi) in [(Tp,qi) for Tp in Tps for qi in qis]:
        (eta_th,h_rad,h_conv) = SPR.HTM_0D_blackbox(Tp, qi, Fc=Fc, air=air)
        data.append([Tp,qi,eta_th,h_rad,h_conv])
        print(data[-1])
    df = pd.DataFrame(data, columns=['Tp','qi','eta','h_rad','h_conv'])
    
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    markers=['o','v','s','d','*','H','P']
    i=0
    for qi in qis:
        dfq = df[df["qi"]==qi]
        if i<5:
            ax1.plot(
                dfq["Tp"], dfq["eta"],
                lw=2.0, marker=markers[i], markersize=10,
                label=r'{:.2f}$[MW/m^2]$'.format(qi)
            )
        else:
            ax1.plot(
                dfq["Tp"], dfq["eta"],
                lw=2.0, ls='--', c='gray', marker=markers[i], markersize=10,
                label=r'{:.2f}$[MW/m^2](exp)$'.format(qi)
            )
        i+=1
    ax1.set_xlim(min(Tps),max(Tps))
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r'Average Particle temperature $(K)$',fontsize=fs)
    ax1.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=1,bbox_to_anchor=(1.48, 1.01),fontsize=fs-2)
    ax1.grid()
    fig.savefig(os.path.join(fldr_rslt,plot_file), bbox_inches='tight')
    plt.show()
    return None


def radiative_fraction_plot(
        fldr_rslt: str,
        plot_file: str,
    ) -> None:

    air  = ct.Solution('air.yaml')
    Tps = np.arange(700.,2001.,100.)
    Fcs = [1.0, 2.0, 2.57, 5.0, 10.]
    markers=['o','s','v','d','*','H','P']
    cs = ['C0','C1','C2','C3','C4','C5']
    qi  = 1.0
    data = []
    for (Tp,Fc) in [(Tp,Fc) for Tp in Tps for Fc in Fcs]:
        
        eta_th,h_rad,h_conv = SPR.HTM_0D_blackbox(Tp, qi, Fc=Fc, air=air)
        data.append([Tp,Fc,eta_th,h_rad,h_conv])
        print(data[-1])
    df = pd.DataFrame(data,columns=['Tp','Fc','eta','h_rad','h_conv'])


    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    fs = 18
    markers=['o','s','v','d','*','H','P']
    i=0
    for Fc in Fcs:
        dfq = df[df.Fc==Fc]
        rad_frac = dfq["h_rad"] / (dfq["h_rad"] + dfq["h_conv"])
        ax1.plot(
            dfq["Tp"], rad_frac,
            c=cs[i], lw=2.0, marker=markers[i], markersize=10,
            label=r'$F_C={:.2f}$'.format(Fc)
        )
        i+=1
    ax1.set_xlim(min(Tps),max(Tps))
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r'Particle temperature $(K)$',fontsize=fs)
    ax1.set_ylabel(r'Fraction of radiative losses $(-)$',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.grid()

    ax2 = fig.add_subplot(122)
    fs = 18
    markers=['o','s','v','d','*','H','P']
    i=0
    for Fc in Fcs:
        dfq = df[df.Fc==Fc]
        rad_frac = dfq["h_rad"] / (dfq["h_rad"]+dfq["h_conv"])
        ax2.plot(dfq.Tp,dfq.eta,c=cs[i],lw=2.0,marker=markers[i],markersize=10,label=r'$F_C={:.1f}$'.format(Fc))
        i+=1
    ax2.set_xlim(min(Tps),max(Tps))
    ax2.set_ylim(0,1)
    ax2.set_xlabel('Particle temperature $(K)$',fontsize=fs)
    ax2.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=4,fontsize=fs-2)
    ax2.grid()
    fig.savefig(os.path.join(fldr_rslt,plot_file), bbox_inches='tight')
    plt.show()
    return None


def main():
    fldr_rslt = os.path.join(os.path.dirname(__file__), 'results_HPR_0D')
    if not os.path.isdir(fldr_rslt):
        os.mkdir(fldr_rslt)
    efficiency_plots(fldr_rslt = fldr_rslt, plot_file = 'efficiency_chart.png')
    radiative_fraction_plot(fldr_rslt = fldr_rslt, plot_file = 'FC_vs_efficiency.png')


if __name__ == "__main__":
    main()
