"""
Plot results from parallel Bayesian optimization (TPR0D_quick_fzv_prcv_parallel.csv)
Uses same plotting functions as plots_bd_tpr_optim.py but with different output filenames.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.interpolate as spi

pd.set_option('display.max_columns', None)

def f_minPoly(X, *args):
    Xs, Ys = args
    try:
        f_inter = spi.interp1d(Xs, Ys, kind='quadratic', fill_value='extrapolate') # type: ignore
    except Exception as e:
        f_inter = spi.interp1d(Xs, Ys, kind='linear', fill_value='extrapolate') # type: ignore
    
    return f_inter(X)

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))
DIR_RESULTS = os.path.join(DIR_PROJECT, 'final_results')
fs = 18


cLCOH = 'mediumblue'
cPrcv = 'orangered'

def _get_min_df_by_height(df2: pd.DataFrame) -> pd.DataFrame:
    mins = []
    zfs = df2["zf"].unique()
    for zf in zfs:
        df3 = df2[(df2["zf"]==zf)].copy()
        df3.drop_duplicates(subset=['rcv_nom_power','zf','flux_avg'],inplace=True)
        bounds = (df3["rcv_nom_power"].min(),df3["rcv_nom_power"].max())
        args = (df3['rcv_nom_power'],df3["lcoh"])
        res = spo.minimize_scalar(f_minPoly, bounds=bounds, args=args, method='bounded')
        Prcv_min = res.x
        LCOH_min = f_minPoly(Prcv_min,*args)
        Nhel_min = spi.interp1d(
            df3['rcv_nom_power'], df3["n_hels"], kind='cubic',fill_value='extrapolate'
        )(Prcv_min)
        etaSF_min = spi.interp1d(
            df3['rcv_nom_power'], df3["eta_sf"], kind='cubic',fill_value='extrapolate'
        )(Prcv_min)
        fzv_min  = spi.interp1d(
            df3['rcv_nom_power'], df3["fzv"], kind='cubic',fill_value='extrapolate'
        )(Prcv_min)
        mins.append([zf,Prcv_min,LCOH_min,Nhel_min,etaSF_min,fzv_min])
    
    mins = pd.DataFrame(mins,columns=('zf','rcv_nom_power','lcoh','n_hels','eta_sf','fzv'))
    mins.sort_values(by='zf',inplace=True)
    return mins


def plot_min_LCOH_for_diff_Qavg(
    file_results
) -> None:

    Q_avs = np.arange(0.50,1.51,0.5)
    ms = ['o','s','d']
    i=0
    df_i = pd.read_csv(file_results)
    df_i = df_i.round({'flux_avg':2})
    fs = 18

    for Q_av in Q_avs:
        df = df_i[(df_i["flux_avg"]==Q_av)].copy()
        df.sort_values(['zf','rcv_nom_power'],inplace=True)
        
        mins = _get_min_df_by_height(df)

        # TOWER HEIGHT VS RECEIVER POWER
        
        def func1(x, *params):
            A,b = params
            return A*np.exp(b*x)

        X = mins["zf"]
        Y = mins['rcv_nom_power']
        p0 = (1., 0.05)
        coefs, covariance = spo.curve_fit( func1, X, Y, maxfev=10000, p0=p0)
        Yc = func1(X,*coefs)
        r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-float(np.mean(Y)))**2))
        Xc = np.linspace(20,75,100)
        Yc = func1(Xc,*coefs)
        A,b = coefs
        
        def func3(x, *params):
            A,b,c = params
            return c+A*np.exp(-x*b)
        
        X3 = mins["zf"]
        Y3 = mins['lcoh']
        p0 = (10., 0.05, 20.)
        coefs_LCOH, covariance = spo.curve_fit( func3, X3, Y3, maxfev=10000, p0=p0)
        Yc3 = func3(X3,*coefs_LCOH)
        r2 = 1 - (np.sum((Y3 - Yc3)**2) / np.sum((Y3-float(np.mean(Y3)))**2))
        A3,b3,c3 = coefs_LCOH
        print(A3,b3,c3,r2)
        
        figb, ax1b = plt.subplots(figsize=(9,6))
        ax2b = ax1b.twinx()
        fs=18
        ax1b.scatter(mins["zf"], mins['lcoh'], c=cLCOH, marker=ms[i], s=200, label='lcoh')
        ax2b.scatter(mins["zf"], mins['rcv_nom_power'], c=cPrcv, marker=ms[i], s=200, label=r'$P_{rcv}$')
        
        ax2b.plot(Xc,Yc,c=cPrcv,lw=2,ls=':')
        ax1b.plot(X3,Yc3,lw=3, c=cLCOH, ls=':')
        
        ax2b.annotate(r'$P_{{rcv}}={:.1f}e^{{{:.3f}z_f}}$'.format(A,b),(Xc[-1]-18,Yc[-1]),c=cPrcv,fontsize=fs)
        
        ax1b.annotate(r'${:.1f}+{:.1f}e^{{-{:.2f}z_f}}$'.format(c3,A3,b3),(X3.iloc[-1]-20,Yc3.iloc[-1]+1),c=cLCOH,fontsize=fs-2)
        
        ax1b.scatter([],[],lw=3,c=cPrcv,marker='s',s=200,label=r'$P_{rcv}$')
        
        ax1b.legend(loc=2,fontsize=fs)
        ax1b.set_ylim(20,35)
        ax1b.grid()
        # ax1b.set_title('LCOH and optimal receiver power for different tower heights with $Q_{{avg}}={:.2f}$'.format(Q_av),fontsize=fs)
        ax1b.set_xlabel(r'Tower height $(m)$',fontsize=fs)
        ax1b.set_ylabel(r'LCOH $(USD/MW_t)$',fontsize=fs)
        ax2b.set_ylabel('Receiver Power $(MW_t)$',fontsize=fs)
        ax2b.spines['left'].set_color('mediumblue')
        ax2b.spines['right'].set_color('C1')
        ax1b.tick_params(axis='y', colors=cLCOH,size=10)
        ax2b.tick_params(axis='y', colors=cPrcv,size=10)
        ax1b.yaxis.label.set_color(cLCOH)
        ax2b.yaxis.label.set_color(cPrcv)
        ax1b.tick_params(axis='both', which='major', labelsize=fs-2)
        ax2b.tick_params(axis='both', which='major', labelsize=fs-2)
        ax2b.set_yticks(np.linspace(ax2b.get_yticks()[0], ax2b.get_yticks()[-1], len(ax1b.get_yticks())))
        ax2b.set_yticks(np.linspace(0, 35, len(ax1b.get_yticks())))
        # plt.show()
        
        i+=1
        figb.savefig(
            os.path.join(DIR_RESULTS,f'zf_Prcv_min_Qavg_{Q_av:.2f}.png'),
            bbox_inches='tight'
        )
    return None
    


def plot_influence_rad_flux(
    file_results: str
) -> pd.DataFrame:
    """
    Generate plots for influence of radiation flux on LCOH.
    Similar to plots_bd_tpr_optim.py but for Bayesian parallel results.
    """
    
    df_i = pd.read_csv(file_results)
    df_i = df_i.round({'flux_avg': 1})
    zfs = np.arange(25, 96, 5)
    mins = []
    fs = 18
    
    for zf in zfs:
        pd.set_option('display.max_columns', None)
        df = df_i[(df_i["zf"] == zf)].copy()
        
        if len(df) == 0:
            print(f"⚠️  No data found for zf={zf}m, skipping...")
            continue
            
        df.sort_values(['flux_avg', 'rcv_nom_power'], inplace=True)
        
        idx_min = df["lcoh"].idxmin()
        Prcv_min = df.loc[idx_min]['rcv_nom_power']
        LCOH_min = df.loc[idx_min]['lcoh']
        Nhel_min = df.loc[idx_min]['n_hels']
        etaSF_min = df.loc[idx_min]['eta_sf']
        fzv_min = df.loc[idx_min]['fzv']
        Q_av = df.loc[idx_min]['rad_flux_avg']
        Arcv = df.loc[idx_min]['receiver_area']
        eta_rcv = df.loc[idx_min]['eta_rcv']
        S_HB = df.loc[idx_min]['hb_surface_area']
        S_TOD = df.loc[idx_min]['tod_surface_area']
        S_land = Prcv_min / df.loc[idx_min]['land_prod']
        mins.append([idx_min, zf, Prcv_min, LCOH_min, Nhel_min, etaSF_min, fzv_min, Q_av, Arcv, eta_rcv, S_TOD, S_HB, S_land])
        
        fig, ax1 = plt.subplots(figsize=(9, 6))
        Qavgs = np.arange(0.5, 2.1, 0.1)
        for Q_av in Qavgs:
            df2 = df[(df["flux_avg"] == np.round(Q_av,1))]
            df2 = df2.drop_duplicates(subset=['rcv_nom_power', 'zf', 'fzv', 'flux_avg'])
            if len(df2) > 0:
                ax1.plot(df2["heat_stored"], df2["lcoh"], lw=3, label=r'${:.2f} MW/m^2$'.format(Q_av))
        
        ax1.scatter([Prcv_min], [LCOH_min], lw=3, c='red', marker='*', s=200, label='Design')
        y1, y2 = 20, 30
        ax1.plot([Prcv_min, Prcv_min], [y1, y2], lw=2, c='red', ls=':')
        ax1.tick_params(axis='both', which='major', labelsize=fs-2)
        ax1.set_xlim(0, 40)
        ax1.set_ylim(y1, y2)
        ax1.set_ylabel(r'LCOH $(USD/MW_t)$', fontsize=fs)
        ax1.set_xlabel('Receiver Power $(MW_t)$', fontsize=fs)
        ax1.legend(loc=1, bbox_to_anchor=(1.35, 1.00), fontsize=fs-2)
        ax1.grid()
        
        fig.savefig(
            os.path.join(DIR_RESULTS, f'LCOH_Prcv_Qavg_zf_{zf:.0f}m.png'),
            bbox_inches='tight'
        )
        plt.close(fig)
        print(f"✅ Saved plot for zf={zf}m")

    mins = pd.DataFrame(
        mins,
        columns=(
            'idx_min', 'zf', 'rcv_nom_power', 'lcoh', 
            'n_hels', 'eta_sf', 'fzv', 'flux_avg', 
            'receiver_area', 'eta_rcv', 'tod_surface_area', 'hb_surface_area', 
            'S_land'
        )
    )
    return mins


def final_optimization_plot(
        mins: pd.DataFrame,
        dir_results: str = DIR_RESULTS,
) -> None:
    """
    Generate final optimization plots.
    Similar to plots_bd_tpr_optim.py but saves with PARALLEL_ prefix.
    """

    def func1(x, *params):
        a, b = params
        return a*x + b

    def func2(x, *params):
        A, b = params
        return A * np.exp(b * x)

    def func3(x, *params):
        A, b, c = params
        return c + A * np.exp(x * b)
    
    def func6(x, *params):
        a, b, c = params
        return a*x**2+b*x+c
    
    
    VARY_CONFIG = {
        "rcv_nom_power": {
            "label_y": r"Optimal Receiver Power ($MW_{th}$)",
            "func_fit": func1,
            "p0": (1., 0.05),
            "color": 'orangered',
            "marker": "o",
            "eq_text": r'$P_{{rcv}}={:.3f}z_f+{:.1f}$',
            },
        "lcoh": {
            "label_y": r"Minimum LCOH ($USD/MW_{th}$)",
            "func_fit": func3,
            "p0": (10., -0.05, 20.),
            "color": 'mediumblue',
            "marker": "s",
            "eq_text": r'$LCOE={:.2f}+{:.3f}e^{{{:.3f}z_f}}$',
            },
        "flux_avg": {
            "label_y": r"Average Radiation flux ($MW_{th}/m^2$)",
            "func_fit": func3,
            "p0": (0.5, -0.05, 0.2),
            "color": 'darkgreen',
            "marker": "o",
            "eq_text": r'$Q_{{avg}}={:.3f}+{:.3f}e^{{-{:.3f}z_f}}$',
        },
        "fzv": {
            "label_y" : r'Optimal $f_{zv} (-)$',
            "func_fit": func3,
            "p0": (0.5, -0.05, 0.90),
            "color": 'darkviolet',
            "marker": "o",
            "eq_text": r'$f_{{zv}} = {:.3f}+{:.3f}e^{{{:.3f}z_f}}$',
        },
        "n_hels": {
            "label_y": r"Optimal Number of Heliostats $N_{hel} (-)$",
            "func_fit": func1,
            "p0": (1000., -0.05),
            "color": 'saddlebrown',
            "marker": "o",
            "eq_text": r'$N_{{hel}} = {:.3f}z_f+{:.1f}$',
        },
        "eta_sf": {
            "label_y": r"Optimal Solar Field Efficiency $\eta_{SF} (-)$",
            "func_fit": func3,
            "p0": (0.5, -0.05, 0.4),
            "color": 'teal',
            "marker": "o",
            "eq_text": r'$\eta_{{SF}} = {:.3f}{:.3f}e^{{{:.3f}z_f}}$',
        },
        "eta_rcv": {
            "label_y": r"Optimal Receiver Efficiency $\eta_{rcv} (-)$",
            "func_fit": func3,
            "p0": (0.5, -0.05, 0.7),
            "color": 'olive',
            "marker": "o",
            "eq_text": r'$\eta_{{rcv}} = {:.3f}{:.3f}e^{{{:.3f}z_f}}$',
        },
        "tod_surface_area": {
            "label_y": r"Optimal TOD Surface Area ($m^2$)",
            "func_fit": func2,
            "p0": (200., 0.2),
            "color": 'coral',
            "marker": "o",
            "eq_text": r'${:.3f}e^{{{:.3f} z_f}}$',
        },
        "hb_surface_area": {
            "label_y": r"Optimal HB Surface Area ($m^2$)",
            "func_fit": func2,
            "p0": (200., 0.2),
            "color": 'slateblue',
            "marker": "o",
            "eq_text": r'${:.3f}e^{{{:.3f} z_f}}$',
        },
        "receiver_area": {
            "label_y": r"Optimal Receiver Area ($m^2$)",
            "func_fit": func2,
            "p0": (200., 0.2),
            "color": 'darkorange',
            "marker": "o",
            "eq_text": r'${:.3f}e^{{{:.3f} z_f}}$',
        },
        "S_land": {
            "label_y": r"Land Surface ($ha$)",
            "func_fit": func2,
            "p0": (200., 0.2),
            "color": 'darkorange',
            "marker": "o",
            "eq_text": r'${:.3f}e^{{{:.3f} z_f}}$',
        }
    }


    VARX = "zf"
    label_x = r"Tower Height ($m$)"

    for VARY in VARY_CONFIG.keys():
        X = mins[VARX]
        Y = mins[VARY]
        label_y = VARY_CONFIG[VARY]["label_y"]
        func_fit = VARY_CONFIG[VARY]["func_fit"]
        p0 = VARY_CONFIG[VARY]["p0"]
        color = VARY_CONFIG[VARY]["color"]
        marker = VARY_CONFIG[VARY]["marker"]
        eq_text = VARY_CONFIG[VARY]["eq_text"]


        coefs, covariance = spo.curve_fit(func_fit, X, Y, maxfev=10000, p0=p0)
        Yc = func_fit(X, *coefs)
        r2 = 1 - (np.sum((Y - Yc)**2) / np.sum((Y-float(np.mean(Y)))**2))
        print(f"{VARY} fit: Coefs={coefs}, R²={r2:.4f}")

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.scatter(X, Y, lw=3, c=color, marker=marker, s=150, label=eq_text.format(*coefs))
        ax1.plot(X, Yc, lw=3, c=color, ls=':', label=f'R²={r2:.4f}')

        ax1.set_xlim(18, 97)

        ax1.set_xlabel(label_x, fontsize=fs)
        ax1.set_ylabel(label_y, fontsize=fs)

        ax1.spines['left'].set_color(color)
        ax1.tick_params(axis='y', colors=color, size=10)
        ax1.yaxis.label.set_color(color)
        ax1.tick_params(axis='both', which='major', labelsize=fs-2)
        ax1.legend(loc=0, fontsize=fs-2)
        ax1.grid()
        
        fig.savefig(os.path.join(dir_results, f'FINAL_optimal_{VARY}.png'), bbox_inches='tight')
        plt.close(fig)


    
    return None


def main():

    
    file_results = os.path.join(DIR_RESULTS, "results_quick_optim_final.csv")
    
    verbose = True

    if not os.path.exists(file_results):
        print(f"❌ Error: File not found: {file_results}")
        return
    
    if verbose:
        print(f"- Loading results from: {file_results}")
        print()
    
        print("- Generating plots of min LCOH for different Q_avg.")
    plot_min_LCOH_for_diff_Qavg(file_results)

    if verbose:
        print("- Generating radiation flux influence plots.")
    mins = plot_influence_rad_flux(file_results)
    print()
    
    if verbose:
        print("Generating final optimization plots")
    final_optimization_plot(mins)
    print()
    
    if verbose:
        print("=" * 60)
        print("✅ All plots generated successfully!")
    return None


if __name__ == "__main__":
    main()
