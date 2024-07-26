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
from bdr_csp.models.spr import BlackboxModel

DIR_PROJECT = os.path.dirname(__file__)
DIR_RESULTS = os.path.join(DIR_PROJECT, 'results_HPR_0D')


def parametric_temps_fluxs(
        verbose: bool = False,
) -> pd.DataFrame:
    
    temps = np.arange(700.,2001.,100.)
    fluxes = [0.25,0.50,1.0,2.0,4.0,0.72]
    temp_amb = 300.
    data = []
    for (temp,flux) in [(temp,flux) for temp in temps for flux in fluxes]:
        receiver = BlackboxModel()
        output = receiver.run_0D_model(temp=temp, flux=flux, temp_amb=temp_amb)
        data.append([temp, flux, output["eta_rcv"], output["h_rad"], output["h_conv"]])
        if verbose:
            print(data[-1])
    df = pd.DataFrame(data, columns=['temp','flux','eta','h_rad','h_conv'])
    return df


def parametric_temps_fluxs_plot(
        results: pd.DataFrame,
        figname: str | None = None,
        showfig: bool = True,
        ) -> None:
    fig, ax1 = plt.subplots(figsize=(9,6))
    fs = 18
    markers = ['o','v','s','d','*','H','P']
    temps = results["temp"].unique()
    fluxes = results["flux"].unique()
    for (i,flux) in enumerate(fluxes):
        df_flux = results[results["flux"]==flux]
        if i<5:
            ax1.plot(
                df_flux["temp"], df_flux["eta"],
                lw=2.0, marker=markers[i], markersize=10,
                label=r'{:.2f}$[MW/m^2]$'.format(flux)
            )
        else:
            ax1.plot(
                df_flux["temp"], df_flux["eta"],
                lw=2.0, ls='--', c='gray', marker=markers[i], markersize=10,
                label=r'{:.2f}$[MW/m^2](exp)$'.format(flux)
            )
    ax1.set_xlim(min(temps),max(temps))
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r'Average Particle temperature $(K)$',fontsize=fs)
    ax1.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=1,bbox_to_anchor=(1.48, 1.01),fontsize=fs-2)
    ax1.grid()
    if figname is not None:
        fig.savefig(os.path.join(DIR_RESULTS,figname), bbox_inches='tight')
    if showfig:
        plt.show()
    return None


def parametric_radiative_fraction(verbose: bool = True) -> pd.DataFrame:
    temps = np.arange(700.,2001.,100.)
    Fcs = [1.0, 2.0, 2.57, 5.0, 10.]
    flux  = 1.0
    temp_amb = 300.
    data = []
    for (temp,Fc) in [(temp,Fc) for temp in temps for Fc in Fcs]:
        
        receiver = BlackboxModel(Fc=Fc)
        output = receiver.run_0D_model(temp=temp, flux=flux, temp_amb=temp_amb)
        data.append([temp, Fc, output["eta_rcv"], output["h_rad"], output["h_conv"]])
        print(data[-1])
    return pd.DataFrame(data,columns=['temp','Fc','eta','h_rad','h_conv'])


def parametric_radiative_fraction_plot(
        results: pd.DataFrame,
        figname: str | None = None,
        showfig: bool = True,
        ) -> None:

    markers=['o','s','v','d','*','H','P']
    cs = ['C0','C1','C2','C3','C4','C5']

    Fcs = results["Fc"].unique()
    temps = results["temp"].unique()
    
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    fs = 18

    for (i,Fc) in enumerate(Fcs):
        df_Fc = results[results["Fc"]==Fc]
        rad_frac = df_Fc["h_rad"] / (df_Fc["h_rad"] + df_Fc["h_conv"])
        ax1.plot(
            df_Fc["temp"], rad_frac,
            c=cs[i], lw=2.0, marker=markers[i], markersize=10,
            label=r'$F_C={:.2f}$'.format(Fc)
        )
    ax1.set_xlim(min(temps),max(temps))
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r'Particle temperature $(K)$',fontsize=fs)
    ax1.set_ylabel(r'Fraction of radiative losses $(-)$',fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.grid()

    ax2 = fig.add_subplot(122)
    fs = 18
    markers=['o','s','v','d','*','H','P']
    for (i,Fc) in enumerate(Fcs):
        df_Fc = results[results["Fc"]==Fc]
        ax2.plot(
            df_Fc["temp"], df_Fc["eta"],
            c=cs[i], lw=2.0, marker=markers[i], markersize=10, 
            label=r'$F_C={:.1f}$'.format(Fc)
        )
    ax2.set_xlim(min(temps),max(temps))
    ax2.set_ylim(0,1)
    ax2.set_xlabel('Particle temperature $(K)$',fontsize=fs)
    ax2.set_ylabel(r'Receiver efficiency $(-)$',fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs-2)
    ax1.legend(loc=4,fontsize=fs-2)
    ax2.grid()

    if figname is not None:
        fig.savefig(os.path.join(DIR_PROJECT,figname), bbox_inches='tight')
    if showfig:
        plt.show()
    return None


def main():
    if not os.path.isdir(DIR_RESULTS):
        os.mkdir(DIR_RESULTS)
    verbose = True

    results = parametric_temps_fluxs(verbose=verbose)
    parametric_temps_fluxs_plot(results, "efficiency_chart.png")

    results = parametric_radiative_fraction(verbose=verbose)
    parametric_radiative_fraction_plot(results, "rad_frac_eta.png")

    # radiative_fraction_plot(fldr_rslt = fldr_rslt, plot_file = 'FC_vs_efficiency.png')


if __name__ == "__main__":
    main()

