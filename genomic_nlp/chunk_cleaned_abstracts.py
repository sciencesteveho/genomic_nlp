#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Chunk abstracts."""


import argparse
from math import ceil
import pickle
from typing import List

import more_itertools  # type: ignore
import pandas as pd  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore


def _get_relevant_abstracts(abstract_file: str) -> List[str]:
    """Get abstracts classified as relevant"""
    abstracts_df = pd.read_pickle(abstract_file)
    return abstracts_df.loc[abstracts_df["predictions"] == 1]["abstracts"].to_list()


def chunk_corpus(corpus: List[str], parts: int, output_base_path: str) -> None:
    """Splits the corpus into a specified number of parts and saves each part as
    a pickle file.

    Args:
        corpus (list): List of documents to be split into chunks.
        parts (int): Number of parts to split the corpus into.
        output_base_path (str): Base output path for the chunks.
    """
    chunk_size = ceil(len(corpus) / parts)  # get size of each chunk

    for idx, batch in enumerate(more_itertools.chunked(corpus, chunk_size)):
        output_path = f"{output_base_path}_part_{idx}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(batch, f)


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified_abstracts",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data/abstracts_logistic_classified_tfidf_40000.pkl",
    )
    parser.add_argument("--num_parts", type=int, default=10)
    parser.add_argument(
        "--output_base_path",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data/abstracts_logistic_classified_tfidf_40000_chunk",
    )
    args = parser.parse_args()

    # get relevant abstracts
    corpus = _get_relevant_abstracts(args.classified_abstracts)

    # chunk the corpus
    chunk_corpus(
        corpus=corpus, parts=args.num_parts, output_base_path=args.output_base_path
    )


if __name__ == "__main__":
    main()
