"""
Quick plotting for parametric dispatch results.

Plots selected variables vs solar_multiple, colored by stg_cap.

Usage (from repo root):
    python projects/dispatch/parametric_dispatch/plot_parametric_dispatch.py

This reads `results.csv` in the same folder and writes PNGs alongside it.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_Y_VARS = ["lcoe", "payback_period", "roi"]
RESULTS_FILE = "results.csv"


def load_results(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find results file: {path}")
    df = pd.read_csv(path)
    required_cols = {"solar_multiple", "stg_cap"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def plot_variables(df: pd.DataFrame, y_vars: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stg_values = sorted(df["stg_cap"].unique())
    for y in y_vars:
        if y not in df.columns:
            print(f"[skip] Column '{y}' not in results; available: {list(df.columns)}")
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
        out_path = os.path.join(out_dir, f"plot_{y}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot parametric dispatch results")
    parser.add_argument(
        "--file",
        default=os.path.join(os.path.dirname(__file__), RESULTS_FILE),
        help="Path to results.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.dirname(__file__),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--vars",
        nargs="*",
        default=DEFAULT_Y_VARS,
        help="Y variables to plot",
    )
    args = parser.parse_args()

    df = load_results(args.file)
    plot_variables(df, args.vars, args.out)


if __name__ == "__main__":
    main()
