"""
Parallel Bayesian optimization for CSP thermal system with fixed zf values.

This module runs separate Bayesian optimizations for each discrete zf value,
optimizing only the 3 continuous variables in parallel. Each zf gets its own
Bayesian optimizer with Gaussian Process surrogate that handles simulation noise.

Variables per optimization:
- zf (fixed): One value from np.arange(25, 76, 5) - Tower focal height [m]
- flux_avg (continuous): [0.5, 1.5] - Average flux [MW/m2]  
- rcv_nom_power (continuous): [5, 35] - Receiver nominal power [MW]
- fzv (continuous): [0.75, 0.97] - HB vertex height fraction [-]

Objective: Minimize LCOH [USD/MWh] for each fixed zf value

Key features:
- Gaussian Process models handle simulation noise naturally
- Parallel execution across multiple zf values
- Uncertainty quantification for each optimization
- Principled exploration/exploitation via acquisition functions
"""

import os
import sys
import time
import pickle
from typing import Tuple, Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt import dump, load

from antupy import Var, Array, Frame
from bdr_csp import spr

DIR_PROJECT = os.path.dirname(os.path.abspath(__file__))
DIR_RESULTS = os.path.join(DIR_PROJECT, "parallel_bayesian_thermal_optim")
os.makedirs(DIR_RESULTS, exist_ok=True)


class FixedZfBayesianOptimizer:
    """
    Bayesian optimizer for a single fixed zf value with 3 continuous variables.
    
    Uses Gaussian Process regression to model the objective function including
    noise estimates. Designed to be picklable for multiprocessing.
    """
    
    def __init__(self, 
                 zf_value: float,
                 n_calls: int = 30,
                 n_initial_points: int = 8,
                 noise_level: float = 0.1,
                 acquisition_function: str = 'EI',
                 worker_id: int = 0):
        """
        Initialize Bayesian optimizer for fixed zf value.
        
        Parameters
        ----------
        zf_value : float
            Fixed tower focal height value [m]
        n_calls : int
            Total number of objective function evaluations
        n_initial_points : int
            Number of initial random evaluations
        noise_level : float
            Estimated noise standard deviation (relative to objective scale)
        acquisition_function : str
            Acquisition function: 'EI', 'LCB', 'PI'
        worker_id : int
            Worker identifier for parallel execution
        """
        self.zf_value = zf_value
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.noise_level = noise_level
        self.acquisition_function = acquisition_function
        self.worker_id = worker_id
        
        # Bounds for continuous variables only (zf is fixed)
        self.bounds_continuous = {
            'flux_avg': (0.5, 1.5),
            'rcv_nom_power': (5.0, 35.0), 
            'fzv': (0.75, 0.97)
        }
        
        # Define search space for skopt (3D continuous space)
        self.search_space = [
            Real(*self.bounds_continuous['flux_avg'], name='flux_avg'),
            Real(*self.bounds_continuous['rcv_nom_power'], name='rcv_nom_power'),
            Real(*self.bounds_continuous['fzv'], name='fzv')
        ]
        
        # Evaluation tracking
        self.evaluations = []
        self.evaluation_count = 0
        self.iteration_count = 0
        
        # Results tracking - worker-specific files
        self.results_file = os.path.join(DIR_RESULTS, f"optimization_results_zf_{zf_value:.0f}.csv")
        self.optimizer_file = os.path.join(DIR_RESULTS, f"optimizer_checkpoint_zf_{zf_value:.0f}.pkl")
        
    def _create_plant(self, flux_avg: float, rcv_nom_power: float, fzv: float) -> spr.CSPBeamDownParticlePlant:
        """Create plant instance with fixed zf and given continuous parameters."""
        plant = spr.CSPBeamDownParticlePlant(
            zf=Var(self.zf_value, "m"),
            flux_avg=Var(flux_avg, "MW/m2"),
            rcv_nom_power=Var(rcv_nom_power, "MW"),
            fzv=Var(fzv, "-"),
            costs_in=spr.get_plant_costs(),
            rcv_type='TPR2D'
        )
        
        # Update receiver area based on design parameters
        plant.receiver_area = plant.rcv_nom_power / plant.rcv_eta_des / plant.flux_avg
        return plant
    
    def _expensive_evaluation(self, params: np.ndarray) -> float:
        """Run expensive CSP simulation with fixed zf."""
        flux_avg, rcv_nom_power, fzv = params
        
        print(f"🔥 Worker {self.worker_id} (zf={self.zf_value}m) - Eval {self.evaluation_count + 1}/{self.n_calls}: "
              f"flux={flux_avg:.3f}, power={rcv_nom_power:.1f}, fzv={fzv:.3f}")
        
        start_time = time.time()
        
        try:
            # Create plant with fixed zf and continuous parameters
            plant = self._create_plant(flux_avg, rcv_nom_power, fzv)
            results = plant.run_simulation(verbose=False, testing=False)
            
            # Extract LCOH from results
            if isinstance(results["costs_out"], dict):
                costs_out = results["costs_out"]
                if isinstance(costs_out["LCOH"], Var):
                    lcoh = costs_out["LCOH"].gv("USD/MWh")
                else:
                    raise ValueError("LCOH output is not valid.")
            else:
                raise ValueError("costs_out is not a valid dictionary.")

            eval_time = time.time() - start_time
            
            # Store detailed results
            detailed_result = {
                'iteration': self.iteration_count,
                'zf': self.zf_value,
                'flux_avg': flux_avg,
                'rcv_nom_power': rcv_nom_power,
                'fzv': fzv,
                'lcoh': lcoh,
                'evaluation_time': eval_time,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'worker_id': self.worker_id,
                'results': results
            }
            
            self._save_detailed_result(detailed_result)
            
            print(f"   ✅ Worker {self.worker_id}: LCOH = {lcoh:.2f} USD/MWh, time = {eval_time:.1f}s")
            
            # Store evaluation
            self.evaluations.append((params.copy(), lcoh, eval_time))
            self.evaluation_count += 1
            
            return lcoh
            
        except Exception as e:
            print(f"   ❌ Worker {self.worker_id} simulation failed: {e}")
            self.evaluation_count += 1
            return 1000.0  # Penalty value
    
    def _save_detailed_result(self, result: Dict[str, Any]):
        """Save detailed results to worker-specific CSV file."""
        row_data = {
            'iteration': result['iteration'],
            'zf': result['zf'],
            'flux_avg': result['flux_avg'], 
            'rcv_nom_power': result['rcv_nom_power'],
            'fzv': result['fzv'],
            'lcoh': result['lcoh'],
            'evaluation_time': result['evaluation_time'],
            'timestamp': result['timestamp'],
            'worker_id': result['worker_id']
        }
        
        # Add key simulation outputs if available
        if 'results' in result:
            costs_out = result['results'].get('costs_out', {})
            if isinstance(costs_out, dict):
                row_data.update({
                    'lcoe': costs_out.get('LCOE', np.nan),
                    'capital_cost': costs_out.get('C_capital', np.nan),
                    'land_productivity': costs_out.get('land_prod', np.nan)
                })
        
        # Append to worker-specific CSV
        df_row = pd.DataFrame([row_data])
        if os.path.exists(self.results_file):
            df_row.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(self.results_file, mode='w', header=True, index=False)
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """
        Normalize parameters to [0,1] range.
        
        Note: Not strictly needed for Bayesian optimization since skopt handles
        scaling internally, but kept for consistency.
        """
        normalized = np.zeros_like(params)
        
        # Normalize flux_avg
        flux_min, flux_max = self.bounds_continuous['flux_avg']
        normalized[0] = (params[0] - flux_min) / (flux_max - flux_min)
        
        # Normalize rcv_nom_power  
        power_min, power_max = self.bounds_continuous['rcv_nom_power']
        normalized[1] = (params[1] - power_min) / (power_max - power_min)
        
        # Normalize fzv
        fzv_min, fzv_max = self.bounds_continuous['fzv']
        normalized[2] = (params[2] - fzv_min) / (fzv_max - fzv_min)
        
        return normalized
    
    def _objective_function(self, params_skopt: List[float]) -> float:
        """
        Objective function wrapper for skopt.
        
        Parameters
        ----------
        params_skopt : list
            Parameters in skopt format: [flux_avg, rcv_nom_power, fzv]
            
        Returns
        -------
        lcoh : float
            Objective value (to be minimized)
        """
        flux_avg = params_skopt[0]
        rcv_nom_power = params_skopt[1]
        fzv = params_skopt[2]
        
        params = np.array([flux_avg, rcv_nom_power, fzv])
        
        # Increment iteration counter
        self.iteration_count += 1
        
        # Run expensive evaluation
        lcoh = self._expensive_evaluation(params)
        
        return lcoh
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run Bayesian optimization for fixed zf value.
        
        Returns
        -------
        best_params : np.ndarray
            Best parameters found [flux_avg, rcv_nom_power, fzv]
        best_value : float
            Best objective value (LCOH)
        """
        
        print(f"🚀 Worker {self.worker_id} starting Bayesian optimization for zf={self.zf_value}m")
        print(f"   Settings: {self.n_calls} calls, {self.n_initial_points} initial, noise={self.noise_level}, acq={self.acquisition_function}")
        
        start_time = time.time()
        
        try:
            # Run Bayesian optimization
            result = gp_minimize(
                func=self._objective_function,
                dimensions=self.search_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acquisition_function,
                noise=self.noise_level,
                random_state=42 + self.worker_id,  # Different seed per worker
                verbose=False,  # Suppress skopt output
                n_jobs=1
            )
            
            # Save optimization result
            dump(result, self.optimizer_file, store_objective=False)
            
            # Extract best parameters
            best_params = np.array(result.x)
            best_lcoh = result.fun
            
            total_time = time.time() - start_time
            
            print(f"🏆 Worker {self.worker_id} (zf={self.zf_value}m) completed!")
            print(f"   Best: flux={best_params[0]:.3f}, power={best_params[1]:.1f}MW, fzv={best_params[2]:.3f}")
            print(f"   Best LCOH: {best_lcoh:.2f} USD/MWh")
            print(f"   Total evaluations: {self.evaluation_count}")
            print(f"   Total time: {total_time/60:.1f} minutes")
            
            return best_params, best_lcoh
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id} (zf={self.zf_value}m) optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def optimize_single_zf_bayesian(args: Tuple[float, int, int, int, float, str, int]) -> Dict[str, Any]:
    """
    Function to run Bayesian optimization for a single zf value - designed for multiprocessing.
    
    Parameters
    ----------
    args : tuple
        (zf_value, worker_id, n_calls, n_initial_points, noise_level, acquisition_function, random_offset)
        
    Returns
    -------
    result : dict
        Optimization results for this zf value
    """
    zf_value, worker_id, n_calls, n_initial_points, noise_level, acquisition_function, random_offset = args
    
    try:
        optimizer = FixedZfBayesianOptimizer(
            zf_value=zf_value,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            noise_level=noise_level,
            acquisition_function=acquisition_function,
            worker_id=worker_id
        )
        
        best_params, best_lcoh = optimizer.optimize()
        
        return {
            'zf': zf_value,
            'worker_id': worker_id,
            'best_params': best_params,
            'best_lcoh': best_lcoh,
            'evaluation_count': optimizer.evaluation_count,
            'success': best_params is not None,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ Worker {worker_id} (zf={zf_value}) failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'zf': zf_value,
            'worker_id': worker_id,
            'best_params': None,
            'best_lcoh': None,
            'evaluation_count': 0,
            'success': False,
            'error': str(e)
        }


class ParallelBayesianOptimizer:
    """
    Parallel Bayesian optimization manager for multiple zf values.
    
    Coordinates parallel Bayesian optimization of multiple zf values, each optimizing
    over 3 continuous variables independently with Gaussian Process surrogates.
    """
    
    def __init__(self, 
                 zf_values: Optional[np.ndarray] = None,
                 max_workers: Optional[int] = None,
                 n_calls: int = 30,
                 n_initial_points: int = 8,
                 noise_level: float = 0.1,
                 acquisition_function: str = 'EI'):
        """
        Initialize parallel Bayesian optimizer.
        
        Parameters
        ----------
        zf_values : np.ndarray, optional
            Array of zf values to optimize. Default: np.arange(25, 76, 5)
        max_workers : int, optional
            Maximum number of parallel workers
        n_calls : int
            Total evaluations per zf value
        n_initial_points : int
            Initial random points per zf value
        noise_level : float
            Noise standard deviation for GP
        acquisition_function : str
            'EI', 'LCB', or 'PI'
        """
        self.zf_values = zf_values if zf_values is not None else np.arange(25, 76, 5)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.noise_level = noise_level
        self.acquisition_function = acquisition_function
        
        # Conservative default for memory-intensive simulations (~400MB each)
        if max_workers is None:
            n_zf_values = len(self.zf_values)
            available_cores = cpu_count()
            max_workers = min(3, available_cores, n_zf_values)
        
        self.max_workers = max_workers
        
        print(f"🔥 Parallel Bayesian Optimizer initialized:")
        print(f"   zf values: {list(self.zf_values)}")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Evaluations per zf: {self.n_calls}")
        print(f"   Initial random points: {self.n_initial_points}")
        print(f"   Noise level: {self.noise_level}")
        print(f"   Acquisition function: {self.acquisition_function}")
        
        # Results tracking
        self.results_summary_file = os.path.join(DIR_RESULTS, "parallel_optimization_summary.csv")
    
    def run_parallel_optimization(self) -> pd.DataFrame:
        """
        Run parallel Bayesian optimization for all zf values.
        
        Returns
        -------
        results_df : pd.DataFrame
            Summary of optimization results for all zf values
        """
        
        print(f"\n🚀 Starting parallel Bayesian optimization:")
        print(f"   {len(self.zf_values)} zf values")
        print(f"   {self.n_calls} evaluations per value")
        print(f"   Total expected evaluations: {len(self.zf_values) * self.n_calls}")
        print(f"   Estimated time: {len(self.zf_values) * self.n_calls * 100 / 60:.0f} minutes")
        
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        worker_args = [
            (zf_val, i, self.n_calls, self.n_initial_points, 
             self.noise_level, self.acquisition_function, i * 100) 
            for i, zf_val in enumerate(self.zf_values)
        ]
        
        results = []
        
        # Run parallel optimization
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            print(f"\n📊 Submitting {len(worker_args)} Bayesian optimization jobs...")
            
            # Submit all jobs
            future_to_zf = {
                executor.submit(optimize_single_zf_bayesian, args): args[0] 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_zf):
                zf_value = future_to_zf[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        print(f"✅ zf={zf_value:.0f}m completed: LCOH = {result['best_lcoh']:.2f} USD/MWh ({result['evaluation_count']} evals)")
                    else:
                        print(f"❌ zf={zf_value:.0f}m failed: {result['error']}")
                        
                except Exception as e:
                    print(f"❌ zf={zf_value:.0f}m exception: {e}")
                    results.append({
                        'zf': zf_value,
                        'worker_id': -1,
                        'best_params': None,
                        'best_lcoh': None,
                        'evaluation_count': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # Process results
        results_df = self._process_results(results, total_time)
        
        print(f"\n🏆 Parallel Bayesian optimization completed!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average time per zf: {total_time/len(self.zf_values):.1f} seconds")
        print(f"   Total evaluations: {sum(r['evaluation_count'] for r in results)}")
        print(f"   Results saved to: {self.results_summary_file}")
        
        return results_df
    
    def _process_results(self, results: List[Dict[str, Any]], total_time: float) -> pd.DataFrame:
        """Process and save optimization results."""
        
        # Convert results to DataFrame
        processed_results = []
        
        for result in results:
            if result['success'] and result['best_params'] is not None:
                processed_results.append({
                    'zf': result['zf'],
                    'flux_avg': result['best_params'][0],
                    'rcv_nom_power': result['best_params'][1],
                    'fzv': result['best_params'][2],
                    'lcoh': result['best_lcoh'],
                    'evaluation_count': result['evaluation_count'],
                    'worker_id': result['worker_id'],
                    'success': True
                })
            else:
                processed_results.append({
                    'zf': result['zf'],
                    'flux_avg': np.nan,
                    'rcv_nom_power': np.nan,
                    'fzv': np.nan,
                    'lcoh': np.nan,
                    'evaluation_count': result['evaluation_count'],
                    'worker_id': result['worker_id'],
                    'success': False
                })
        
        results_df = pd.DataFrame(processed_results)
        
        # Add summary statistics
        total_evaluations = results_df['evaluation_count'].sum()
        successful_optimizations = results_df['success'].sum()
        
        # Calculate best results safely
        if successful_optimizations > 0:
            successful_results = results_df[results_df['success'] == True]
            best_overall_lcoh = successful_results['lcoh'].min()
            best_idx = successful_results['lcoh'].idxmin()
            best_zf = successful_results.loc[best_idx, 'zf']
        else:
            best_overall_lcoh = np.nan
            best_zf = np.nan
        
        # Save summary
        summary_stats = {
            'total_time_minutes': total_time / 60,
            'total_evaluations': int(total_evaluations),
            'successful_optimizations': int(successful_optimizations),
            'failed_optimizations': len(results_df) - successful_optimizations,
            'best_overall_lcoh': best_overall_lcoh,
            'best_zf': best_zf,
            'noise_level': self.noise_level,
            'acquisition_function': self.acquisition_function
        }
        
        # Save detailed results
        results_df.to_csv(self.results_summary_file, index=False)
        
        # Save summary statistics
        summary_file = os.path.join(DIR_RESULTS, "optimization_summary_stats.csv")
        pd.DataFrame([summary_stats]).to_csv(summary_file, index=False)
        
        return results_df
    
    def analyze_results(self) -> None:
        """Analyze and display optimization results."""
        
        if not os.path.exists(self.results_summary_file):
            print("No results file found")
            return
        
        df = pd.read_csv(self.results_summary_file)
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful optimizations found")
            return
        
        print(f"\n📊 Parallel Bayesian Optimization Analysis:")
        print(f"   Successful optimizations: {len(successful_df)}/{len(df)}")
        print(f"   Total evaluations: {df['evaluation_count'].sum()}")
        print(f"   Average evaluations per zf: {df['evaluation_count'].mean():.1f}")
        
        # Best overall result
        best_idx = successful_df['lcoh'].idxmin()
        best_result = successful_df.iloc[best_idx]
        
        print(f"\n🏆 Best Overall Configuration:")
        print(f"   zf = {best_result['zf']:.0f} m")
        print(f"   flux_avg = {best_result['flux_avg']:.3f} MW/m2")
        print(f"   rcv_nom_power = {best_result['rcv_nom_power']:.1f} MW")
        print(f"   fzv = {best_result['fzv']:.3f}")
        print(f"   LCOH = {best_result['lcoh']:.2f} USD/MWh")
        
        # Top 5 results
        print(f"\n🥇 Top 5 Results:")
        top_5 = successful_df.nsmallest(5, 'lcoh')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. zf={row['zf']:.0f}m: LCOH={row['lcoh']:.2f} USD/MWh")
        
        # Statistics by zf value
        print(f"\n📈 Results by zf value:")
        for zf in sorted(successful_df['zf'].unique()):
            zf_result = successful_df[successful_df['zf'] == zf].iloc[0]
            print(f"   zf={zf:.0f}m: LCOH={zf_result['lcoh']:.2f} USD/MWh ({zf_result['evaluation_count']} evals)")


def run_parallel_bayesian_optimization(
    zf_values: Optional[np.ndarray] = None,
    max_workers: Optional[int] = None,
    n_calls: int = 30,
    n_initial_points: int = 8,
    noise_level: float = 0.1,
    acquisition_function: str = 'EI'
) -> pd.DataFrame:
    """
    Main function to run parallel Bayesian optimization.
    
    Parameters
    ----------
    zf_values : np.ndarray, optional
        Array of zf values to optimize
    max_workers : int, optional
        Maximum parallel workers
    n_calls : int
        Evaluations per zf value
    n_initial_points : int
        Initial random points per zf value
    noise_level : float
        Noise standard deviation for GP
    acquisition_function : str
        'EI', 'LCB', or 'PI'
        
    Returns
    -------
    results_df : pd.DataFrame
        Summary of results
    """
    
    # Create optimizer
    optimizer = ParallelBayesianOptimizer(
        zf_values=zf_values,
        max_workers=max_workers,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        noise_level=noise_level,
        acquisition_function=acquisition_function
    )
    
    # Run optimization
    results_df = optimizer.run_parallel_optimization()
    
    # Analyze results
    optimizer.analyze_results()
    
    return results_df


def merge_individual_results(zf_values: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
    """Merge individual worker CSV files into a single comprehensive dataset."""
    
    if zf_values is None:
        zf_values = np.arange(25, 76, 5)
    
    all_data = []
    
    for zf in zf_values:
        individual_file = os.path.join(DIR_RESULTS, f"optimization_results_zf_{zf:.0f}.csv")
        if os.path.exists(individual_file):
            df_individual = pd.read_csv(individual_file)
            all_data.append(df_individual)
            print(f"✅ Loaded {len(df_individual)} evaluations for zf={zf:.0f}m")
        else:
            print(f"⚠️  No results file found for zf={zf:.0f}m")
    
    if all_data:
        # Combine all individual results
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined results
        combined_file = os.path.join(DIR_RESULTS, "all_evaluations_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n📊 Combined Results:")
        print(f"   Total evaluations: {len(combined_df)}")
        print(f"   zf values covered: {sorted(combined_df['zf'].unique())}")
        print(f"   LCOH range: {combined_df['lcoh'].min():.2f} - {combined_df['lcoh'].max():.2f} USD/MWh")
        print(f"   Combined data saved to: {combined_file}")
        
        return combined_df
    else:
        print("❌ No individual result files found")
        return None


if __name__ == "__main__":
    print("🔥 CSP Parallel Bayesian Thermal Optimization")
    print("=" * 60)
    print("Running separate Bayesian optimizations for each zf value")
    print("Each uses Gaussian Process models to handle simulation noise")
    print("=" * 60)
    
    # Run parallel Bayesian optimization
    results_df = run_parallel_bayesian_optimization(
        zf_values=np.arange(25, 96, 5),  # 15 values: 25, 30, ..., 95
        max_workers=3,                   # Conservative for memory
        n_calls=40,                      # 40 evaluations per zf
        n_initial_points=8,              # 8 random initial points
        noise_level=0.1,                 # 10% noise assumption
        acquisition_function='EI'        # Expected Improvement
    )
    
    # Merge individual worker results
    print("\n" + "=" * 60)
    print("🔗 Merging Individual Results")
    combined_df = merge_individual_results()
    
    print("\n✅ Parallel Bayesian optimization complete!")
