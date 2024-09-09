#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Create abstract dataframes, saving the year, and combine the title and
description."""


import glob
from typing import List

import pandas as pd


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a saved scopus df and processes it via:
    1. Combining title and description into abstracts
    2. Extracting year from coverDate
    """
    df["abstracts"] = df["title"] + " " + df["description"].fillna("")
    df["year"] = pd.to_datetime(df["coverDate"]).dt.year
    return df[["abstracts", "year"]]


def process_file(file_path: str) -> pd.DataFrame:
    """Process each df."""
    df = pd.read_pickle(file_path)
    return process_dataframe(df)


def main() -> None:
    """Process all files and save the result."""
    all_files: List[str] = glob.glob("abstracts_results*.pkl")
    result: pd.DataFrame = pd.concat(
        (process_file(f) for f in all_files), ignore_index=True
    )
    result.to_pickle("abstracts_combined.pkl")


if __name__ == "__main__":
    main()
