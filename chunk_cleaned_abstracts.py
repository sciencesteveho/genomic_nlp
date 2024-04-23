#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Chunk abstracts"""


import argparse
import pickle
from typing import List

import more_itertools  # type: ignore
import pandas as pd  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore
from math import ceil


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
    # get size of each chunk
    chunk_size = ceil(len(corpus) / parts)
    
    for idx, batch in enumerate(more_itertools.chunked(corpus, chunk_size)):
        output_path = f"{output_base_path}_part_{idx}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(batch, f)


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevant_abstracts", type=str, default='../data/relevant_abstracts.pkl')
    parser.add_argument("--num_parts", type=int, default=10)
    parser.add_argument("--output_base_path", type=str, default="../data/abstracts_classified_tfidf_20000_chunk")
    args = parser.parse_args()
    
    # load abstracts
    with open(args.relevant_abstracts, 'rb') as f:
        corpus = pickle.load(f)
    
    # chunk the corpus
    chunk_corpus(
        corpus=corpus,
        parts=args.num_parts,
        output_base_path=args.output_base_path
    )


if __name__ == "__main__":
    main()
