#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Collection of utilities for combining and writing out abstracts for
Word2Vec."""


import argparse
import glob
import os
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


def write_chunks_to_text(args: argparse.Namespace, prefix: str) -> None:
    """Write chunks of abstracts to a text file for Word2Vec, filtering by year.

    Each sentence is written on a new line. Only abstracts with 'year' <= input
    year are included.
    """
    combined_dir = os.path.join(args.abstracts_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    output_file = os.path.join(combined_dir, f"{prefix}_{args.year}_combined.txt")

    filenames = _chunk_locator(args.abstracts_dir, prefix)
    if not filenames:
        print(
            f"No files found with prefix '{prefix}'"
            f"in directory '{args.abstracts_dir}'."
        )
        return

    print(f"Writing combined abstracts to '{output_file}' up to year {args.year}...")

    with open(output_file, "w", encoding="utf-8") as output:
        for filename in filenames:
            print(f"Processing file: {filename}")
            try:
                df = pd.read_pickle(filename)

                # ensure df has the required columns
                if not {"year", "processed_abstracts_w2v"}.issubset(df.columns):
                    print(f"Skipping {filename}: Missing required columns.")
                    continue

                # temporal split
                filtered_df = df[df["year"] <= args.year]
                print(f" - {len(filtered_df)} abstracts after filtering by year.")

                for _, row in filtered_df.iterrows():
                    processed_abstract: List[List[str]] = row[
                        "processed_abstracts_finetune"
                    ]
                    for sentence in processed_abstract:
                        line = " ".join(sentence)
                        output.write(f"{line}\n")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Writing out abstracts for {args.year} complete.")


def prepare_and_load_abstracts(args: argparse.Namespace) -> None:
    """Combine chunked abstracts into single pickle files if they do not exist.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    combined_dir = os.path.join(args.abstracts_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    def combine_chunks(suffix: str) -> None:
        """Combine chunks of abstracts with a given suffix into a single pickle
        file.

        Args:
            suffix (str): Suffix identifying the chunk files to combine.
        """
        prefix = f"tokens_cleaned_abstracts_{suffix}"
        combined_filename = os.path.join(combined_dir, f"{prefix}_combined.pkl")
        if not os.path.isfile(combined_filename):
            print(
                f"Combining abstract chunks for suffix '{suffix}' into '{combined_filename}'."
            )
            combined_df = _combine_chunks(args.abstracts_dir, prefix)
            if not combined_df.empty:
                try:
                    with open(combined_filename, "wb") as f:
                        pickle.dump(combined_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Successfully combined and saved to '{combined_filename}'.")
                except Exception as e:
                    print(f"Error saving combined file '{combined_filename}': {e}")
            else:
                print(f"No data to combine for suffix '{suffix}'.")
        else:
            print(f"Combined file already exists: '{combined_filename}'.")

    # define the suffixes to process
    file_suffixes = ["casefold", "remove_genes"]
    for suffix in file_suffixes:
        combine_chunks(suffix)


def main() -> None:
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description="Combine abstract chunks and write sentences to text files for Word2Vec."
    )
    parser.add_argument(
        "--abstracts_dir",
        type=str,
        required=True,
        help="Directory containing abstract chunks.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    )
    # parser.add_argument(
    #     "--year",
    #     type=int,
    #     required=True,
    #     help="Cutoff year. Only abstracts up to and including this year will be processed.",
    # )
    args = parser.parse_args()

    combined_df = _combine_chunks(args.abstracts_dir, "processed_abstracts_w2v_")
    combined_df.to_pickle(f"{args.abstracts_dir}/processed_abstracts_w2v_combined.pkl")

    del combined_df

    combined_df = _combine_chunks(args.abstracts_dir, "processed_abstracts_finetune_")
    combined_df.to_pickle(
        f"{args.abstracts_dir}/processed_abstracts_finetune_combined.pkl"
    )

    # combined_df = _combine_chunks(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    #     "processed_abstracts_w2v_",
    # )


if __name__ == "__main__":
    main()
