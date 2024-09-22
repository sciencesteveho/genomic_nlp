#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Collection of utilities for combining and writing out abstracts."""


import argparse
import glob
import os
import pickle
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


def _concat_chunks(filenames: List[str]) -> List[List[str]]:
    """Concatenates chunks of abstracts"""
    combined = []
    combined += [pickle.load(open(file, "rb")) for file in filenames]
    return combined


def _chunk_locator(path: str, prefix: str) -> List[str]:
    """Returns abstract chunks matching a specific prefix"""
    pattern = f"{path}/{prefix}_*.pkl"
    return glob.glob(pattern)


def _combine_chunks(path: str, prefix: str) -> List[List[str]]:
    """Combines chunks of abstracts"""
    filenames = _chunk_locator(path, prefix)
    print(f"Combining chunks of abstracts: {filenames}")
    return _concat_chunks(filenames)


def flatten_abstract(abstract: List[str]) -> List[str]:
    """Flatten a potentially nested abstract."""
    if isinstance(abstract, list) and (abstract and isinstance(abstract[0], list)):
        return [word for sentence in abstract for word in sentence]
    return abstract


def write_chunks_to_text(args: argparse.Namespace, prefix: str) -> None:
    """Write chunks of abstracts to text files"""
    filenames = _chunk_locator(args.abstracts_dir, prefix)
    with open(f"{args.abstracts_dir}/combined/{prefix}_combined.txt", "w") as output:
        for filename in filenames:
            with open(filename, "rb") as file:
                abstracts = pickle.load(file)
                for abstract in abstracts:
                    flattened_abstract = flatten_abstract(abstract)
                    line = " ".join(flattened_abstract) + "\n"
                    output.write(line)


def prepare_and_load_abstracts(args: argparse.Namespace) -> None:
    """Combine chunked abstracts if they do not exist"""

    def combine_chunks(suffix: str) -> None:
        """Combine chunks of abstracts if they do not exist"""
        filename = f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_{suffix}_combined.pkl"
        if not os.path.isfile(filename):
            print(f"Combining abstract chunks for {filename}")
            with open(filename, "wb") as f:
                pickle.dump(
                    _combine_chunks(
                        f"{args.abstracts_dir}",
                        f"tokens_cleaned_abstracts_{suffix}",
                    ),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    file_suffixes = ["casefold", "remove_genes"]
    for suffix in file_suffixes:
        combine_chunks(suffix)


"""
Bert text should split by abstract.
W2V text should split by sentence.
In [38]: test.columns
Out[38]: Index(['cleaned_abstracts', 'year', 'processed_abstracts_finetune'], dtype='object')
"""

# # prepare abstracts by writing chunks out to text file
# print("Writing out cleaned_corpus...")
# write_chunks_to_text(args, "tokens_cleaned_abstracts_casefold")
# print("Writing gene_remove corpus...")
# write_chunks_to_text(args, "tokens_cleaned_abstracts_remove_genes")
# print("Abstracts written! Instantiating object...")


# def main() -> None:
#     """Process all files and save the result."""
#     all_files: List[str] = glob.glob("abstracts_results*.pkl")
#     result: pd.DataFrame = pd.concat(
#         (process_file(f) for f in all_files), ignore_index=True
#     )
#     result.to_pickle("abstracts_combined.pkl")


# if __name__ == "__main__":
#     main()
