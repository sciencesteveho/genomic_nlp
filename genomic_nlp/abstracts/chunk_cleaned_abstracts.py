#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Chunk abstracts into smaller parts for parallel processing."""


import argparse
from math import ceil
import pickle

import pandas as pd  # type: ignore


def _get_relevant_abstracts(abstract_file: str) -> pd.DataFrame:
    """Get abstracts classified as relevant and reset the index."""
    abstracts_df = pd.read_pickle(abstract_file)
    return abstracts_df.loc[abstracts_df["predictions"] == 1].reset_index(drop=True)


def chunk_corpus(corpus: pd.DataFrame, parts: int, output_base_path: str) -> None:
    """Splits the corpus into a specified number of parts and saves each part as
    a pickle file.

    Args:
        corpus (list): List of documents to be split into chunks.
        parts (int): Number of parts to split the corpus into.
        output_base_path (str): Base output path for the chunks.
    """
    chunk_size = ceil(len(corpus) / parts)  # get size of each chunk
    for idx in range(parts):
        start_idx = idx * chunk_size
        end_idx = min((idx + 1) * chunk_size, len(corpus))
        batch = corpus.iloc[start_idx:end_idx]
        output_path = f"{output_base_path}_part_{idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(batch, f)


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified_abstracts_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/classification",
    )
    parser.add_argument(
        "--classified_abstracts",
        type=str,
        default="abstracts_mlp_classified_tfidf_40000.pkl",
    )
    parser.add_argument("--num_parts", type=int, default=20)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="abstracts_mlp_classified_tfidf_40000_chunk",
    )
    args = parser.parse_args()

    # get full paths to files
    classified_abstracts = (
        f"{args.classified_abstracts_dir}/{args.classified_abstracts}"
    )
    output_base_path = f"{args.output_path}/{args.output_filename}"

    # get relevant abstracts
    corpus = _get_relevant_abstracts(classified_abstracts)

    # chunk the corpus
    chunk_corpus(corpus=corpus, parts=args.num_parts, output_base_path=output_base_path)


if __name__ == "__main__":
    main()
