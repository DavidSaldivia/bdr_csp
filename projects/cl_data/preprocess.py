from __future__ import annotations
from dataclasses import dataclass, field
from os.path import isfile
from typing import Callable, Union
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


from bdr_csp import bdr, spr
from bdr_csp.dir import DIRECTORY

ParticleReceiver = spr.HPR0D | spr.HPR2D | spr.TPR2D
pd.set_option('display.max_columns', None)

DIR_MAIN = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DIR_DATA = DIRECTORY.DIR_DATA

vars_dict = {
    "temp_amb": {"file": "Tamb_2024.csv", "col_name": "AirTC", "units": "C"},
    "DNI": {"file": "DNI_2024.csv", "col_name": "DNI", "units": "W/m2"}
    
}

def preprocess_weather_data(var_list: list[str]) -> pd.DataFrame:
    
    DIR_WEATHER = os.path.join(DIR_DATA, "weather", "psda_2024")
    df_weather = pd.DataFrame()
    for var in var_list:
        if var not in vars_dict:
            raise ValueError(f"Variable '{var}' not recognized. Available variables: {list(vars_dict.keys())}")
        file_path = os.path.join(DIR_WEATHER, vars_dict[var]["file"])
        if not isfile(file_path):
            raise FileNotFoundError(f"Weather data file for '{var}' not found at {file_path}")
        df_var = pd.read_csv(file_path, sep=";", index_col=0, header=0, skiprows=[1,2])
        df_var.rename(columns={f"{vars_dict[var]["col_name"]}_Avg":var}, inplace=True)
        df_var.drop(
            columns= [f"{vars_dict[var]["col_name"]}_{text}" for text in ["Min", "Max", "Std"]], inplace=True
        )
        df_weather = pd.concat([df_weather, df_var], axis=1)
    # df_weather.index.name = "date"
    return df_weather


def preprocess_market_data(
        year: int = 2024,
        location: str = "crucero"
    ) -> pd.DataFrame:
    DIR_MARKET = os.path.join(DIR_DATA, "energy_market", "sen")
    FILE_MARKET = os.path.join(DIR_MARKET, f"{year}_{location}.csv")
    df_market = pd.read_csv(FILE_MARKET, sep=",", index_col=3, header=0)
    df_market.drop(["fecha", "hora"], inplace=True, axis=1)
    df_market.index = pd.to_datetime(df_market.index)
    return df_market

def preprocess_market_data_2(
        year: int = 2024,
        location: str = "crucero"
    ) -> pd.DataFrame:

    path = r"C:\Users\david\Downloads\datos-de-costos-marginales-en-linea.tsv"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # parses the JSON array

    df = pd.DataFrame(data, columns=["fecha", "barra", "cmg"])
    df["barra"] = df["barra"].str.lower()
    # If you want fecha as datetime:
    df["date"] = pd.to_datetime(df["fecha"])
    df2 = df[(df["barra"]==location)&(df["date"].dt.year==year)].copy()
    df2 = df2.rename(columns={"cmg":"SP"})
    return df2[["date","SP"]].set_index("date")


def analyze_regularity(df: pd.DataFrame) -> None:
    """Analyze time series regularity, gaps, and duplicates using index as date column."""
    df_sorted = df.sort_index()
    date_index = pd.to_datetime(df_sorted.index)
    date_diffs = date_index.to_series().diff()
    
    print(f"Total records: {len(df)}")
    print(f"Date range: {date_index.min()} to {date_index.max()}")
    print(f"\nTime intervals (most common):")
    print(date_diffs.value_counts().head())
    
    print(f"\nIrregular intervals (not the most common):")
    most_common_interval = date_diffs.mode()[0]
    irregular = date_diffs[date_diffs != most_common_interval]
    print(f"Count: {len(irregular)} ({len(irregular)/len(df)*100:.2f}%)")
    
    print(f"\nLargest gaps:")
    print(date_diffs.nlargest(5))
    
    # Duplicates analysis
    print(f"\n--- DUPLICATES ANALYSIS ---")
    dup_mask = date_index.duplicated(keep=False)
    n_dup = dup_mask.sum()
    print(f"Duplicate timestamps: {n_dup} records ({n_dup/len(df)*100:.2f}%)")
    
    if n_dup > 0:
        df_dups = df_sorted[dup_mask]
        print(f"\nDuplicate groups: {dup_mask.sum() // date_index[dup_mask].nunique():.0f} duplicates per timestamp on average")
        
        # Show value differences in duplicates
        numeric_cols = df_dups.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\nValue ranges in duplicate rows:")
            for col in numeric_cols:
                dup_groups = df_dups.groupby(df_dups.index)[col]
                min_max_diff = dup_groups.apply(lambda x: x.max() - x.min())
                print(f"  {col}: max difference = {min_max_diff.max():.4f}, avg difference = {min_max_diff.mean():.4f}")



def _localize_and_convert_index(df: pd.DataFrame, src_tz: str, dst_tz: str) -> pd.DataFrame:
    """Return a copy with index localized from src_tz and converted to dst_tz.

    - Treats a naive datetime index as being in src_tz.
    - Handles DST with ambiguous="infer" and nonexistent="shift_forward".
    """
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is None:
        # Try to infer DST transitions; if it fails, fall back to standard-time assumption
        try:
            idx = idx.tz_localize(src_tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            try:
                idx = idx.tz_localize(src_tz, ambiguous=False, nonexistent="shift_forward")
            except Exception:
                # Last resort: localize without disambiguation; user can inspect later
                idx = idx.tz_localize(src_tz)
    else:
        # If index already timezone-aware, assume it's in src_tz conceptually and convert
        idx = idx.tz_convert(src_tz)
    idx = idx.tz_convert(dst_tz)
    out.set_index(idx, inplace=True)
    out.index.name = "date"
    out.sort_index(inplace=True)
    return out


def merge_and_resample(
    df_weather: pd.DataFrame,
    df_market: pd.DataFrame,
    freq: str = "30min",
    target_tz: str = "America/Santiago",
) -> pd.DataFrame:
    """Align timezones, average duplicates, merge, and resample to a regular grid.

    Steps:
    1) Localize df_weather from UTC → target_tz and df_market from target_tz → target_tz.
    2) Average any duplicate timestamps within each dataframe.
    3) Merge on the (timezone-aware) index.
    4) Reindex to a regular `freq` grid and interpolate missing values in time.
    """
    # 1) Align timezones
    weather_tz = _localize_and_convert_index(df_weather, src_tz="UTC", dst_tz=target_tz)
    market_tz = _localize_and_convert_index(df_market, src_tz=target_tz, dst_tz=target_tz)

    # 2) Average duplicates within each dataframe
    weather_tz = weather_tz.groupby(level=0).mean(numeric_only=True)
    market_tz = market_tz.groupby(level=0).mean(numeric_only=True)

    # 3) Merge on index
    df_merged = pd.concat([weather_tz, market_tz], axis=1)

    # Safety: if any duplicates remain after concat, average them too
    df_merged = df_merged.groupby(level=0).mean(numeric_only=True)

    # 4) Build regular time index in the target timezone
    new_index = pd.date_range(
        start=df_merged.index.min(),
        end=df_merged.index.max(),
        freq=freq,
        tz=target_tz,
    )

    # Reindex and interpolate in time
    df_resampled = df_merged.reindex(new_index)
    df_resampled = df_resampled.interpolate(method="time", limit_direction="both")

    return df_resampled


def plot_scatter(df: pd.DataFrame, x_col: str = "temp_amb", y_col: str = "DNI", save_path: str = None) -> None:
    """Create scatter plot of two variables from the dataframe."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5, s=10)
    plt.xlabel(f"{x_col.replace('_', ' ').title()}")
    plt.ylabel(f"{y_col}")
    plt.title(f"{y_col} vs {x_col.replace('_', ' ').title()}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_monthly_boxplots(df: pd.DataFrame, variables: list[str], save_dir: str) -> None:
    """Create monthly boxplots for specified variables."""
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    
    for var in variables:
        if var not in df_copy.columns:
            print(f"Warning: {var} not found in dataframe")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        df_copy.boxplot(column=var, by='month', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel(var)
        ax.set_title(f'Monthly Distribution of {var}')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"monthly_boxplot_{var}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_hourly_boxplots(df: pd.DataFrame, variables: list[str], save_dir: str) -> None:
    """Create hourly boxplots for specified variables."""
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    
    for var in variables:
        if var not in df_copy.columns:
            print(f"Warning: {var} not found in dataframe")
            continue
            
        fig, ax = plt.subplots(figsize=(14, 6))
        df_copy.boxplot(column=var, by='hour', ax=ax)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(var)
        ax.set_title(f'Hourly Distribution of {var}')
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"hourly_boxplot_{var}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def main():
    df_weather = preprocess_weather_data(["temp_amb", "DNI"])
    analyze_regularity(df_weather)
    df_market = preprocess_market_data_2(2024, "crucero")
    analyze_regularity(df_market)
    
    # Merge and resample to 30-minute intervals
    df_combined = merge_and_resample(df_weather, df_market, freq="30min")
    print(df_combined.head())
    print(f"\nShape: {df_combined.shape}")
    print(f"Missing values:\n{df_combined.isnull().sum()}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "tsg_data_exploration")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_scatter(df_combined, x_col="DNI", y_col="SP", 
                save_path=os.path.join(output_dir, "scatter_DNI_vs_SP.png"))
    
    plot_monthly_boxplots(df_combined, ["DNI", "SP"], output_dir)
    plot_hourly_boxplots(df_combined, ["DNI", "SP"], output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()