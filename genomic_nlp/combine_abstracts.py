#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Collection of utilities for combining and writing out abstracts for
Word2Vec."""


import argparse
import glob
import os
from pathlib import Path
import pickle
from typing import List

import pandas as pd


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the DataFrame by combining title and description into abstracts
    and extracting the year.
    """
    df["abstracts"] = df["title"] + " " + df["description"].fillna("")
    df["year"] = pd.to_datetime(df["coverDate"]).dt.year
    return df[["abstracts", "year"]]


def process_file(file_path: str) -> pd.DataFrame:
    """Process each pickle file and return the processed DataFrame."""
    df = pd.read_pickle(file_path)
    return process_dataframe(df)


def _chunk_locator(path: str, prefix: str) -> List[str]:
    """Locate all chunk files matching the given prefix within the specified
    path.
    """
    pattern = os.path.join(path, f"{prefix}*.pkl")
    return glob.glob(pattern)


def _concat_chunks(filenames: List[str]) -> List[pd.DataFrame]:
    """Concatenate chunks of abstracts from multiple pickle files into a list of
    DataFrames.
    """
    combined = []
    for file in filenames:
        try:
            with open(file, "rb") as f:
                df = pickle.load(f)
                if isinstance(df, pd.DataFrame):
                    combined.append(df)
                else:
                    print(f"Warning: {file} does not contain a DataFrame. Skipping.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return combined


def _combine_chunks(path: str, prefix: str) -> pd.DataFrame:
    """Combine multiple chunked pickle files into a single DataFrame."""
    filenames = _chunk_locator(path, prefix)
    if not filenames:
        raise FileNotFoundError(
            f"No files found with prefix '{prefix}' in directory '{path}'."
        )

    print(f"Combining {len(filenames)} chunks for prefix '{prefix}'.")
    if dataframes := _concat_chunks(filenames):
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        print("No valid DataFrames to combine.")
        return pd.DataFrame()


def flatten_abstract(abstract: List[List[str]]) -> List[str]:
    """Flatten a list of sentences (each a list of tokens) into a single list of
    tokens.
    """
    return [word for sentence in abstract for word in sentence]


def write_temporal_abstracts(
    df: pd.DataFrame,
    outdir: Path,
    year: int,
    column: str,
) -> None:
    """Write chunks of abstracts to a text file for Word2Vec, filtering by year.

    Each sentence is written on a new line. Only abstracts with 'year' <= input
    year are included.
    """

    with open(outdir / f"{column}+{year}", "w", encoding="utf-8") as output:

        # ensure df has the required columns
        if not {"year", column}.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns 'year' and "
                f"'{column}' to write out abstracts."
            )

        # temporal split
        filtered_df = df[df["year"] <= year]
        print(f" - {len(filtered_df)} abstracts after filtering by year.")

        for _, row in filtered_df.iterrows():
            processed_abstract: List[List[str]] = row[column]
            for sentence in processed_abstract:
                line = " ".join(sentence)
                output.write(f"{line}\n")
    print(f"Writing out abstracts for {year} complete.")


def main() -> None:
    """Main execution flow."""
    working_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data"
    abstracts = "processed_abstracts_w2v_combined.pkl"
    outdir = Path(working_dir) / "combined"
    df = pd.read_pickle(f"{working_dir}/{abstracts}")

    # write out abstracts from 2003 to 2023
    for year in range(2003, 2024):

        # set up directory
        year_outdir = outdir / str(year)
        os.makedirs(year_outdir, exist_ok=True)

        # write out with and without genes
        write_temporal_abstracts(df, year_outdir, year, "processed_abstracts_w2v")
        write_temporal_abstracts(
            df, year_outdir, year, "processed_abstracts_w2v_nogenes"
        )

    # parser = argparse.ArgumentParser(
    #     description="Combine abstract chunks and write sentences to text files for Word2Vec."
    # )
    # parser.add_argument(
    #     "--abstracts_dir",
    #     type=str,
    #     help="Directory containing abstract chunks.",
    #     default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    # )
    # parser.add_argument(
    #     "--year",
    #     type=int,
    #     required=True,
    #     help="Cutoff year. Only abstracts up to and including this year will be processed.",
    # )
    # args = parser.parse_args()

    # combined_df = _combine_chunks(args.abstracts_dir, "processed_abstracts_w2v_")
    # combined_df.to_pickle(f"{args.abstracts_dir}/processed_abstracts_w2v_combined.pkl")

    # del combined_df

    # combined_df = _combine_chunks(args.abstracts_dir, "processed_abstracts_finetune_")
    # combined_df.to_pickle(
    #     f"{args.abstracts_dir}/processed_abstracts_finetune_combined.pkl"
    # )

    # combined_df = _combine_chunks(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    #     "processed_abstracts_w2v_",
    # )


if __name__ == "__main__":
    main()
