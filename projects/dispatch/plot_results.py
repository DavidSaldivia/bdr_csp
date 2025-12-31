"""
Plot parametric dispatch results for Chilean (CL) or Australian (AU) cases.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_Y_VARS = ["lcoe", "payback_period", "roi"]
DIR_DISPATCH = os.path.dirname(os.path.abspath(__file__))


def load_and_plot(case: str, y_vars: list[str] = None) -> None:
    """Load results and plot."""
    if y_vars is None:
        y_vars = DEFAULT_Y_VARS
    
    results_dir = os.path.join(DIR_DISPATCH, f"parametric_dispatch_{case}")
    results_file = os.path.join(results_dir, "results.csv")
    
    df = pd.read_csv(results_file)
    stg_values = sorted(df["stg_cap"].unique())
    
    for y in y_vars:
        if y not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for stg in stg_values:
            sub = df[df["stg_cap"] == stg].sort_values("solar_multiple")
            ax.plot(sub["solar_multiple"], sub[y], marker="o", label=f"stg_cap={stg}")
        
        ax.set_xlabel("Solar Multiple [-]")
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        ax.legend(title="stg_cap", fontsize=9)
        fig.tight_layout()
        
        out_path = os.path.join(results_dir, f"plot_{y}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)



def main():
    # load_and_plot("cl", DEFAULT_Y_VARS)
    load_and_plot("au", DEFAULT_Y_VARS)

if __name__ == "__main__":
    main()
