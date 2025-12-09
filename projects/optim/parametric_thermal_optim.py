import os
import sys
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import threading

import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import pickle

from antupy import Var, Array, Frame

from bdr_csp import spr
from bdr_csp.dir import DIRECTORY

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))
DIR_DATASET = os.path.join(DIRECTORY.DIR_DATA, 'mcrt_datasets_final')


def f_Optim1D(X,*args):
    stime_i = time.time()
    plant, label = args

    if not isinstance(plant, spr.CSPBeamDownParticlePlant):
        raise ValueError('Plant must be a CSPBeamDownParticlePlant instance')

    #Updating the variables before running the simulation
    if label == 'fzv':
        plant.fzv = Var(X/100. if X>1. else X, "-")
    # if label == 'Arcv':
    #     plant.TOD.receiver_area = Var(X, "m2")
    #     plant.TOD.update_params()
    # if label == 'Prcv':
    #     plant.rcv_nom_power = Var(X, "MW")
    #     eta_rcv_i, Arcv, _ = spr.initial_eta_rcv(plant)
    #     plant.rcv_eta_des = eta_rcv_i
    #     plant.TOD.receiver_area = Arcv
    #     plant.TOD.update_params()
    # if label == 'zf':
    #     zf = round(X)
    #     if zf < 20:
    #         zf=20
    #     if zf > 100:
    #         zf=100
    #     plant.zf = Var(zf, "m")
    
    #Running Optical-Thermal simulation
    results = plant.run_simulation(verbose=False)
    #Objective function
    fobj = results["costs_out"]["LCOH"]

    print(f"{plant.rcv_nom_power}\t{plant.zf}\t{plant.fzv}\t{plant.flux_avg}\t{fobj:.4f}\tTime: {time.time()-stime_i:.2f}s")
    return fobj.gv("USD/MWh")

# def plot_detailed_hsf(
#         SF: pd.DataFrame,
#         hlst: list[int],
# ):
#     f_s=18
#     fig = plt.figure(figsize=(12,8))
#     ax1 = fig.add_subplot(111)
#     SF2 = SF[SF['hel_in']].loc[hlst]
#     N_hel = len(hlst)
#     vmin = SF2['Eta_SF'].min()
#     vmax = SF2['Eta_SF'].max()
#     surf = ax1.scatter(SF2['xi'],SF2['yi'], s=5, c=SF2['Eta_SF'], cmap=cm.YlOrRd, vmin=vmin, vmax=vmax )
#     cb = fig.colorbar(surf, shrink=0.25, aspect=4)
#     cb.ax.tick_params(labelsize=f_s)
#     cb.ax.locator_params(nbins=4)
    
#     fig.text(0.76,0.70,r'$\overline{\eta_{{SF}}}$'+'={:.3f}'.format(Etas['Eta_SF']), fontsize=f_s)
#     fig.text(0.76,0.65,r'$N_{{hel}}$'+'={:d}'.format(N_hel),fontsize=f_s)
#     # plt.title(title+' (av. eff. {:.2f} %)'.format(Etas_SF[eta_type]*100))
#     ax1.set_xlabel('E-W axis (m)',fontsize=f_s);ax1.set_ylabel('N-S axis (m)',fontsize=f_s);
#     ax1.add_artist(patches.Wedge((0, 0), rmax, 0, 360, width=rmax-rmin,color='C0'))
#     for tick in ax1.xaxis.get_major_ticks():
#         tick.label.set_fontsize(f_s)
#     for tick in ax1.yaxis.get_major_ticks():
#         tick.label.set_fontsize(f_s)
#     ax1.grid()
#     fig.savefig(fldr_rslt+case+'_SF.png', bbox_inches='tight')
#     plt.show()
#     plt.close(fig)


# def plot_detailed_hb():
#     # HYPERBOLOID MIRROR
#     f_s = 18
#     out  = R2[(R2['hel_in'])&(R2['hit_hb'])]
#     xmin = out['xb'].min(); xmax = out['xb'].max()
#     ymin = out['yb'].min(); ymax = out['yb'].max()
#     Nx = 100; Ny = 100
#     dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
#     Fbin = CSTo['eta_rfl']*Etas['Eta_cos']*Etas['Eta_blk']*(CSTo['Gbn']*CSTo['A_h1']*N_hel)/(1e3*dA*len(out))
#     Q_HB,X,Y = np.histogram2d(out['xb'],out['yb'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]],density=False)
#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(111, aspect='equal')
#     X, Y = np.meshgrid(X, Y)
    
#     vmin = 0
#     vmax = (np.ceil(Fbin*Q_HB.max()/10)*10)
#     surf = ax.pcolormesh(X, Y, Fbin*Q_HB.transpose(),cmap=cm.YlOrRd,vmin=vmin,vmax=vmax)
#     ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
#     cb = fig.colorbar(surf, shrink=0.25, aspect=4)
#     cb.ax.tick_params(labelsize=f_s)
#     fig.text(0.77,0.62,r'$Q_{HB}(kW/m^2)$',fontsize=f_s)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(f_s)
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(f_s)
    
#     # from matplotlib import rc
#     # rc('text', usetex=True)
#     fig.text(0.77,0.35,'Main Parameters',fontsize=f_s-3)
#     fig.text(0.77,0.33,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
#     fig.text(0.77,0.31,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
#     fig.text(0.77,0.29,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
#     fig.text(0.77,0.27,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
    
#     ax.add_artist(patches.Circle((0.,0.0), rmin, zorder=10, color='black', fill=None))
#     ax.add_artist(patches.Circle((0.,0.0), rmax, zorder=10, edgecolor='black', fill=None))
#     ax.grid(zorder=20)
#     fig.savefig(fldr_rslt+case+'_QHB_upper.png', bbox_inches='tight')
#     plt.show()
#     plt.close()


# def plot_detailed_rcvr():
#     Nx = 100; Ny = 100
#     rO,Cg,zrc,Type,Array = [CSTo[x] for x in ['rO_TOD', 'Cg_TOD', 'zrc', 'Type', 'Array']]
#     TOD = BDR.TOD_Params({'Type':Type, 'Array':Array,'rO':rO,'Cg':Cg},0.,0.,zrc)
#     N_TOD,V_TOD,rO,rA,x0,y0 = [TOD[x] for x in ['N','V','rO','rA','x0','y0']]
#     out   = R2[(R2['hel_in'])&(R2['hit_rcv'])].copy()
#     xmin = out['xr'].min(); xmax = out['xr'].max()
#     ymin = out['yr'].min(); ymax = out['yr'].max()
#     if Array!='N':
#         xmin=min(x0)-rA/np.cos(np.pi/V_TOD) 
#         xmax=max(x0)+rA/np.cos(np.pi/V_TOD)
#         ymin=min(y0)-rA
#         ymax=max(y0)+rA
    
#     dx = (xmax-xmin)/Nx; dy = (ymax-ymin)/Nx; dA=dx*dy
#     Nrays = len(out)
#     Fbin  = Etas['Eta_SF'] * (CSTo['Gbn']*CSTo['A_h1']*N_hel)/(1e3*dA*Nrays)
#     Q_BDR,X,Y = np.histogram2d(out['xr'],out['yr'],bins=[Nx,Ny],range=[[xmin, xmax], [ymin, ymax]], density=False)
#     Q_BDR = Fbin * Q_BDR
#     Q_max = Q_BDR.max()
    
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, aspect='equal')
#     X, Y = np.meshgrid(X, Y)
#     f_s = 16
#     vmin = 0
#     vmax = (np.ceil(Q_BDR.max()/100)*100)
#     surf = ax.pcolormesh(X, Y, Q_BDR.transpose(), cmap=cm.YlOrRd, vmin=vmin, vmax=vmax)
#     ax.set_xlabel('E-W axis (m)',fontsize=f_s);ax.set_ylabel('N-S axis (m)',fontsize=f_s);
#     cb = fig.colorbar(surf, shrink=0.5, aspect=4)
#     cb.ax.tick_params(labelsize=f_s-2)
    
#     if Array=='F' or Array=='N':
#         ax.add_artist(patches.Circle((x0[0],y0[0]), rO, zorder=10, color='black', fill=None))
#     else:
#         for i in range(N_TOD):
#             radius = rO/np.cos(np.pi/V_TOD)
#             ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
#             radius = rA/np.cos(np.pi/V_TOD)
#             ax.add_artist(patches.RegularPolygon((x0[i],y0[i]), V_TOD,radius, np.pi/V_TOD, zorder=10, color='black', fill=None))
        
#     fig.text(0.77,0.27,'Main Parameters',fontsize=f_s-3)
#     fig.text(0.77,0.25,r'$z_{{f\;}}={:.0f} m$'.format(zf),fontsize=f_s-3)
#     fig.text(0.77,0.23,r'$f_{{zv}}={:.2f} m$'.format(fzv),fontsize=f_s-3)
#     fig.text(0.77,0.21,r'$z_{{rc}}={:.1f} m$'.format(zrc),fontsize=f_s-3)
#     fig.text(0.77,0.19,r'$r_{{hb}}={:.1f} m$'.format(rmax),fontsize=f_s-3)
#     ax.set_title('b) Radiation flux in receiver aperture',fontsize=f_s)
    
#     fig.text(0.77,0.70,r'$Q_{{rcv}}(kW/m^2)$',fontsize=f_s)
#     fig.savefig(fldr_rslt+case+'_radmap_out.png', bbox_inches='tight')
#     plt.show()
#     plt.close()


def simulate_zf_flux_combination(args):
    """
    Simulate one combination of zf and flux_avg with multiple rcv_nom_powers.
    Designed to be parallelizable.
    
    Parameters
    ----------
    args : tuple
        (zf, flux_avg, rcv_nom_powers, tol, DIR_RESULTS, worker_id, result_queue)
        
    Returns
    -------
    results : list
        List of dictionaries with simulation results
    """
    zf, flux_avg, rcv_nom_powers, tol, DIR_RESULTS, worker_id, result_queue = args
    
    results = []
    
    try:
        print(f"Worker {worker_id}: Creating plant for zf={zf} and flux_avg={flux_avg}.")
        plant = spr.CSPBeamDownParticlePlant(
            zf=zf,
            flux_avg=flux_avg,
            rcv_nom_power=Var(zf.gv("m")*0.35, "MW"),      #Initial guess for Receiver power
            costs_in=spr.get_plant_costs(),
            rcv_type='TPR0D'
        )
        
        print(f"Worker {worker_id}: Obtaining initial optimized value for fzv")

        # Read existing results (cache) once per worker if available
        file_df = os.path.join(DIR_RESULTS, 'TPR0D_quick_fzv_prcv_parallel.csv')
        df_existing = None
        if os.path.isfile(file_df):
            try:
                df_existing = pd.read_csv(file_df)
            except Exception as _e:
                print(f"Worker {worker_id}: Warning reading existing CSV '{file_df}': {_e}")
                df_existing = None

        # Optional partial-cache optimization for fzv: use cached fzv if any row exists for this (zf, flux_avg)
        cached_fzv = None
        if df_existing is not None and {'zf', 'flux_avg', 'fzv'}.issubset(df_existing.columns):
            try:
                zf_arr = df_existing['zf'].to_numpy()
                flux_arr = df_existing['flux_avg'].to_numpy()
                zf_target = np.round(zf.gv("m"), 1)
                flux_target = np.round(flux_avg.gv("MW/m2"), 1)
                mask_fzv = (
                    (np.round(zf_arr, 1) == zf_target)
                    & (np.round(flux_arr, 1) == flux_target)
                )
                matches_fzv = df_existing.loc[mask_fzv]
                if not matches_fzv.empty:
                    cached_fzv = float(matches_fzv.iloc[-1]['fzv'])
            except Exception as _e:
                print(f"Worker {worker_id}: Cached fzv lookup failed, will optimize. Details: {_e}")

        if cached_fzv is not None:
            fzv = Var(cached_fzv, "-")
            print(f"Worker {worker_id}: Using cached fzv={cached_fzv:.4f} for zf={zf.gv('m')}, flux_avg={flux_avg.gv('MW/m2')}")
        else:
            lbl = 'fzv'
            bracket = (70, 84, 95)
            res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(plant, lbl), method='brent', tol=tol)
            if isinstance(res, spo.OptimizeResult):
                fzv = Var(res.x/100, "-")
            else:
                raise ValueError('Optimization for fzv did not converge')

        for rcv_nom_power in rcv_nom_powers:
            stime = time.time()

            plant.fzv = fzv
            plant.rcv_nom_power = rcv_nom_power
            plant.receiver_area = plant.rcv_nom_power / plant.rcv_eta_des / plant.flux_avg
            plant.rcv_type = 'TPR2D'

            print([zf, flux_avg, rcv_nom_power])

            # CSV cache lookup per your instructions (lines 236-240)
            # 1) Read CSV (done above) 2) Use zf, flux_avg, rcv_nom_power columns
            # 3) Compare with tolerance 4) If matched, retrieve; else run simulation
            cached_row = None
            used_cache = False
            if df_existing is not None and {'zf', 'flux_avg', 'rcv_nom_power'}.issubset(df_existing.columns):
                zf_val = np.round(zf.gv("m"), 1)
                flux_val = np.round(flux_avg.gv("MW/m2"), 1)
                power_val = np.round(rcv_nom_power.gv("MW"), 1)
                try:
                    zf_arr = df_existing['zf'].to_numpy()
                    flux_arr = df_existing['flux_avg'].to_numpy()
                    power_arr = df_existing['rcv_nom_power'].to_numpy()
                    mask = (
                        (np.round(zf_arr, 1) == zf_val)
                        & (np.round(flux_arr, 1) == flux_val)
                        & (np.round(power_arr, 1) == power_val)
                    )
                    matches = df_existing.loc[mask]
                    if not matches.empty:
                        cached_row = matches.iloc[-1]
                        used_cache = True
                except Exception as _e:
                    print(f"Worker {worker_id}: Cache lookup failed, will run simulation. Details: {_e}")

            output = None
            if not used_cache:
                output = plant.run_simulation(verbose=False)
            # Prepare result row aligned with CSV schema
            OUTPUT_COLS = [
                'receiver_area', 'sf_power_sim', 'hb_rmax', 'n_hels',
                'eta_rcv', 'eta_sf', 'eta_bdr', 'eta_tod', 'eta_StH',
                'hb_surface_area', 'tod_surface_area', "total_surface_area",
                'heat_stored', 'rad_flux_avg', 'rad_flux_max','temp_part_max',
                'tod_height','stg_height','stg_vol', 'land_prod',
                'lcoh', 'lcoe',
            ]
            OUTPUT_UNITS = [
                'm2', 'MW', 'm', '-',
                '-', '-', '-', '-', '-',
                'm2', 'm2', 'm2',
                'MW', 'MW/m2', 'MW/m2', 'K',
                'm', 'm', 'm3', 'MW/ha',
                'USD/MWh', 'USD/MWh',
            ]
            INPUT_DATA = [rcv_nom_power, zf, flux_avg, fzv]
            INPUT_COLS = ["rcv_nom_power", "zf", "flux_avg", "fzv"]
            INPUT_UNITS = ["MW", "m", "MW/m2", "-"]
            COLS_ALL = INPUT_COLS + OUTPUT_COLS

            if used_cache and (cached_row is not None):
                # Directly pass cached values to the new results
                row = [float(cached_row[col]) for col in COLS_ALL]
            else:
                # Convert units and build row
                row = [INPUT_DATA[i].gv(INPUT_UNITS[i]) for i in range(len(INPUT_DATA))]
                row += [
                    output[OUTPUT_COLS[i]].gv(OUTPUT_UNITS[i]) for i in range(len(OUTPUT_COLS))
                ]

            # Push row to the shared queue for the writer thread
            try:
                result_queue.put(row)
            except Exception as _e:
                print(f"Worker {worker_id}: Failed to enqueue result row: {_e}")
            
            if used_cache:
                print(f"Worker {worker_id}: {zf} - {flux_avg} - {rcv_nom_power} - LCOH = {row[-2]:.2f} USD/MWh (from cache)")
            else:
                print(f"Worker {worker_id}: {zf} - {flux_avg} - {rcv_nom_power} - LCOH = {row[-2]:.2f} USD/MWh, time = {time.time() - stime:.1f}s")
        
        return []
        
    except Exception as e:
        print(f"Worker {worker_id}: Failed for zf={zf}, flux_avg={flux_avg}: {e}")
        import traceback
        traceback.print_exc()
        return []


def optimization_one_var_quick_parallel(max_workers=None):
    """
    Parallel version of optimization_one_var_quick that runs simulations 
    for different (zf, flux_avg) combinations in parallel.
    
    Args:
        max_workers: Maximum number of parallel workers. Defaults to min(cpu_count()-1, 3)
                    to avoid memory overload (each simulation uses ~400MB).
    """
    DIR_RESULTS = os.path.join(DIR_PROJECT, "testing")
    os.makedirs(DIR_RESULTS, exist_ok=True)
    file_new = os.path.join(DIR_RESULTS, 'results_quick_optim_final.csv')

    # Memory-safe default: limit workers to prevent system overload
    if max_workers is None:
        max_workers = min(cpu_count() - 1, 4)
    
    print(f"Starting parallel parametric optimization with {max_workers} workers")
    print(f"Note: Each simulation uses ~400MB RAM. Total memory ~{max_workers * 400}MB")

    stime = time.time()

    tol = 1e-4
    zfs = Array(np.arange(80, 96, 5), "m")
    flux_avgs = Array(np.arange(1.3, 2.0, 0.1), "MW/m2")
    rcv_nom_powers = Array(np.arange(20, 46, 2), "MW")

    # Create list of all (zf, flux_avg) combinations to parallelize
    combinations = [(zf, flux_avg) for flux_avg in flux_avgs for zf in zfs]
    
    print(f"Total combinations to simulate: {len(combinations)}")
    print(f"Each combination will optimize fzv and run {len(rcv_nom_powers)} simulations")
    print(f"Total simulations: {len(combinations) * len(rcv_nom_powers)}")
    
    # Define CSV schema once here
    INPUT_COLS = ["rcv_nom_power", "zf", "flux_avg", "fzv"]
    OUTPUT_COLS = [
        'receiver_area', 'sf_power_sim', 'hb_rmax', 'n_hels',
        'eta_rcv', 'eta_sf', 'eta_bdr', 'eta_tod', 'eta_StH',
        'hb_surface_area', 'tod_surface_area', "total_surface_area",
        'heat_stored', 'rad_flux_avg', 'rad_flux_max', 'temp_part_max',
        'tod_height', 'stg_height', 'stg_vol', 'land_prod',
        'lcoh', 'lcoe',
    ]
    COLS_ALL = INPUT_COLS + OUTPUT_COLS

    # Prepare a manager queue and a writer thread for incremental appends
    manager = Manager()
    result_queue = manager.Queue(maxsize=1000)

    def writer_loop(queue, csv_path, columns):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(columns)
            while True:
                row = queue.get()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()

    writer_thread = threading.Thread(target=writer_loop, args=(result_queue, file_new, COLS_ALL), daemon=True)
    writer_thread.start()

    # Prepare arguments for parallel execution
    worker_args = [
        (zf, flux_avg, rcv_nom_powers, tol, DIR_RESULTS, i, result_queue)
        for i, (zf, flux_avg) in enumerate(combinations)
    ]

    # Run parallel simulations
    all_results = []  # no longer used for writing, kept for optional stats
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_args = {
            executor.submit(simulate_zf_flux_combination, args): args 
            for args in worker_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            zf, flux_avg, _, _, _, worker_id, _queue = args
            
            try:
                results = future.result()
                # Workers now stream rows directly; 'results' is unused
                completed += 1
                print(f"\n[Progress {completed}/{len(combinations)}] Completed zf={zf}, flux_avg={flux_avg}")
                print(f"  Returned {len(results)} simulation results")
                
            except Exception as e:
                print(f"\n[ERROR] Worker {worker_id} failed for zf={zf}, flux_avg={flux_avg}: {e}")
                import traceback
                traceback.print_exc()

    # Signal writer thread to finish and wait for it
    try:
        result_queue.put(None)
    except Exception:
        pass
    writer_thread.join()

    # Aggregate all results into single DataFrame and write to new file (skipped; we wrote incrementally)
    if all_results:
        print(f"\n{'='*60}")
        print(f"All parallel simulations completed!")
        print(f"Total successful simulations: {len(all_results)}")
        print(f"Total time: {time.time() - stime:.1f} seconds")
        print(f"{'='*60}\n")
        
        # Create DataFrame from all results (rows)
        INPUT_COLS = ["rcv_nom_power", "zf", "flux_avg", "fzv"]
        INPUT_UNITS = ["MW", "m", "MW/m2", "-"]
        OUTPUT_COLS = [
            'receiver_area', 'sf_power_sim', 'hb_rmax', 'n_hels',
            'eta_rcv', 'eta_sf', 'eta_bdr', 'eta_tod', 'eta_StH',
            'hb_surface_area', 'tod_surface_area', "total_surface_area",
            'heat_stored', 'rad_flux_avg', 'rad_flux_max', 'temp_part_max',
            'tod_height', 'stg_height', 'stg_vol', 'land_prod',
            'lcoh', 'lcoe',
        ]
        OUTPUT_UNITS = [
            'm2', 'MW', 'm', '-',
            '-', '-', '-', '-', '-',
            'm2', 'm2', 'm2',
            'MW', 'MW/m2', 'MW/m2', 'K',
            'm', 'm', 'm3', 'MW/ha',
            'USD/MWh', 'USD/MWh',
        ]
        
        COLS_ALL = INPUT_COLS + OUTPUT_COLS
        UNITS_ALL = INPUT_UNITS + OUTPUT_UNITS

        print(f"\nResults appended to: {file_new}")
        return None
    else:
        print("No results collected. All workers failed.")
        return None


def optimization_one_var_quick():
    """
    Original sequential version - kept for reference.
    Use optimization_one_var_quick_parallel() for faster execution.
    """
    plot = False
    save_detailed = False
    DIR_RESULTS = os.path.join(DIR_PROJECT, "testing")
    os.makedirs(DIR_RESULTS, exist_ok=True)
    file_CSTs = os.path.join(DIR_RESULTS,'TPR0D_quick_fzv_prcv.plk')
    file_df   = os.path.join(DIR_RESULTS,'TPR0D_quick_fzv_prcv.csv')

    stime = time.time()

    tol = 1e-3        #Solving the non-linear equation
    zfs   = Array(np.arange(25,76,5),"m")
    flux_avgs = Array([0.75,],"MW/m2")
    rcv_nom_powers = Array(np.arange(5,36,2), "MW")

    for (flux_avg,zf) in [(flux_avg,zf) for flux_avg in flux_avgs for zf in zfs]:

        print(f"Creating plant for {zf} and {flux_avg}.")
        plant = spr.CSPBeamDownParticlePlant(
            zf=zf,
            flux_avg=flux_avg,
            rcv_nom_power=Var(zf.gv("m")*0.35, "MW"),
            costs_in=spr.get_plant_costs(),
            rcv_type='TPR0D'
        )
        
        print("Obtaining initial optimized value for fzv")

        lbl = 'fzv'
        bracket = (75 , 84, 97)
        res = spo.minimize_scalar(f_Optim1D, bracket=bracket, args=(plant,lbl), method='brent', tol=tol)
        if isinstance(res, spo.OptimizeResult):
            fzv  = Var(res.x/100, "-")
        else:
            raise ValueError('Optimization for fzv did not converge')
        
        for rcv_nom_power in rcv_nom_powers:
            
            stime = time.time()

            plant.fzv = fzv
            plant.rcv_nom_power = rcv_nom_power
            plant.receiver_area = plant.rcv_nom_power / plant.rcv_eta_des / plant.flux_avg
            plant.rcv_type = 'TPR2D'

            case = 'zf{:.0f}_Q_avg{:.0f}_Prcv{:.1f}'.format(zf, flux_avg, rcv_nom_power)
            output = plant.run_simulation()

            date_sim = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            output['date_sim'] = date_sim
            if save_detailed:
                if os.path.isfile(file_CSTs):
                    CSTs = pickle.load(open(file_CSTs,'rb'))
                    CSTs.append(output)
                    pickle.dump(CSTs,open(file_CSTs,'wb'))
                else:
                    CSTs = [output,]
                    pickle.dump(CSTs,open(file_CSTs,'wb'))

            if os.path.isfile(file_df):
                data = pd.read_csv(file_df,index_col=0).values.tolist()
            else:
                data=[]
            
            OUTPUT_COLS = [
                'receiver_area', 'sf_power_sim', 'hb_rmax', 'n_hels',
                'eta_rcv', 'eta_sf', 'eta_bdr', 'eta_tod', 'eta_StH',
                'hb_surface_area', 'tod_surface_area', "total_surface_area",
                'heat_stored', 'rad_flux_avg', 'rad_flux_max','temp_part_max',
                'tod_height','stg_height','stg_vol', 'land_prod',
                'lcoh', 'lcoe',
            ]
            OUTPUT_UNITS = [
                'm2', 'MW', 'm', '-',
                '-', '-', '-', '-', '-',
                'm2', 'm2', 'm2',
                'MW', 'MW/m2', 'MW/m2', 'K',
                'm', 'm', 'm3', 'MW/ha',
                'USD/MWh', 'USD/MWh',
            ]
            INPUT_DATA = [rcv_nom_power, zf, flux_avg, fzv]
            INPUT_COLS = ["rcv_nom_power", "zf", "flux_avg", "fzv"]
            INPUT_UNITS = ["MW", "m", "MW/m2", "-"]
            DATA_ROW = (
                [INPUT_DATA[i].gv(INPUT_UNITS[i]) for i in range(len(INPUT_DATA))]
                + [output[OUTPUT_COLS[i]].gv(OUTPUT_UNITS[i]) for i in range(len(OUTPUT_COLS))]
            )
            data.append(DATA_ROW)
            COLS_ALL = INPUT_COLS + OUTPUT_COLS
            UNITS_ALL = INPUT_UNITS + OUTPUT_UNITS
            df = Frame(data,columns=COLS_ALL, units=UNITS_ALL)
            print('\t'.join('{:.3f}'.format(x) for x in data[-1]))
            
            pd.set_option('display.max_columns', None)
            df.to_csv(file_df)
            
            print(f"Simulation took {time.time()-stime} seconds.")
            
            if plot:
                try:
                    pass
                    # plot_detailed_hsf()
                    # plot_detailed_hb()
                    # plot_detailed_rcvr()
                except Exception as e:
                    print('It was not possible to create figures.')
                    print(e)
                    
            print(df)

def main():
    """
    Main execution function. 
    
    Set run_parallel=True to use the faster parallel version.
    Set run_parallel=False to use the original sequential version.
    """
    run_parallel = True
    
    if run_parallel:
        print("Running PARALLEL parametric optimization")
        print("=" * 60)
        # Run parallel version with up to 4 workers (memory-safe)
        df = optimization_one_var_quick_parallel(max_workers=4)
    else:
        print("Running SEQUENTIAL parametric optimization")
        print("=" * 60)
        # Run original sequential version
        optimization_one_var_quick()

if __name__ == "__main__":
    main()