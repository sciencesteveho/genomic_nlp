#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse
import pickle

import more_itertools
import pandas as pd
import spacy
from tqdm import tqdm


def main() -> None:
    """Main function"""
    # abstracts = "/scratch/remills_root/remills/stevesho/bio_nlp/nlp/classification/abstracts_classified_tfidf_20000.pkl"
    # abstracts = pd.read_pickle(abstracts)
    # abstracts = list(abstracts.loc[abstracts["predictions"] == 1]["abstracts"])
    # for idx, batch in enumerate(more_itertools.batched(abstracts, 250000)):
    #     with open(
    #         f"data/abstracts_classified_tfidf_20000_chunk_{idx}.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump(batch, f)

    # perform tokenization per batched chunk
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--idx",
        type=int,
    )
    args = parser.parse_args()

    with open(
        f"data/abstracts_classified_tfidf_20000_chunk_{args.idx}.pkl",
        "rb",
    ) as file:
        abstracts = pickle.load(file)

    # start tokenization
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("sentencizer")
    n_process = 4
    batch_size = 256

    dataset_tokens = [
        [word.text for word in sentence]
        for doc in tqdm(
            nlp.pipe(
                abstracts,
                n_process=n_process,
                batch_size=batch_size,
                disable=["parser", "tagger", "ner", "lemmatizer"],
            ),
            total=len(abstracts),
        )
        for sentence in doc.sents
    ]

    with open(f"data/tokens_from_cleaned_abstracts_chunk_{args.idx}.pkl", "wb") as f:
        pickle.dump(dataset_tokens, f)


if __name__ == "__main__":
    main()
