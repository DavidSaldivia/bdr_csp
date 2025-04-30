import os

import pandas as pd
import polars as pl


DATA_DIR = os.path.abspath(os.path.join(os.sep, "users","david","downloads"))

def main():

    print(DATA_DIR)

    data_file = os.path.join(DATA_DIR, "2025_02_EN_LINEA.tsv")
    # df1 = pd.read_csv(data_file, sep="\t")

    df1 = pl.read_csv(data_file, separator="\t")
    print(df1)


    data_file = os.path.join(DATA_DIR, "2024_10_REAL-PRE.tsv")
    df2 = pd.read_csv(data_file, sep="\t")
    print(df2)

    pass


if __name__ == "__main__":
    main()