#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Abstract df concatenation, cleaning with regular expressions, and relevancy
classification"""

import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils import (
    SUBLIST,
    SUBLIST_INITIAL,
    SUBLIST_TOKEN_ZERO,
    SUBLIST_TOKEN_ONE,
    SUBLIST_POST,
    SUBLIST_TITLE,
    time_decorator,
)


def _abstract_retrieval_concat(data_path: str) -> None:
    """Take abstract outputs and combine into a single pd.series. Only needs to
    be done initially after downloading abstracts"""
    files = os.listdir(data_path)
    frames = [
        pd.read_pickle((os.path.join(data_path, file)), compression=None)
        for file in files
        if not file.startswith(".")
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(f"{data_path}/abstracts_combined.pkl")


class AbstractCollection:
    """Collection of scientific abstracts.

    # Properties
        abstracts
        date

    # Methods
        parse_abstracts_for_regex
        abstract_processing
        check_cleaning_fidelity_and_save

    # Helpers
        GENE_REPLACEMENTS
        SUBLIST
    """

    GENE_REPLACEMENTS = {
        " WAS ": " wasgene ",
        " SHE ": " shegene ",
        " IMPACT ": " impactgene ",
        " MICE ": " micegene ",
        " REST ": " restgene ",
        " SET ": " setgene ",
        " MET ": " metgene ",
        " CA2 ": " ca2gene ",
        " ATM ": " atmgene ",
        " MB ": " mbgene ",
        " PIGS ": " pigsgene ",
        " CAT ": " catgene ",
        " COIL ": " coilgene ",
    }

    SUBLIST_SPACES = ["-", "\s+"]

    def __init__(self, abstracts) -> None:
        """Initialize the class"""
        self.abstracts = [abstract for abstract in abstracts.tolist() if abstract]

    # @time_decorator
    def _abstract_cleaning(self):
        """Lorem Ipsum"""
        cleaned_abstracts = []
        for abstract in self.abstracts:
            for pattern in SUBLIST:
                abstract = re.sub(pattern, "", abstract)
            for pattern in SUBLIST_INITIAL:
                abstract = re.sub(pattern, "", abstract, flags=re.IGNORECASE)
            tokens = abstract.split(". ")
            if tokens[0].startswith("©"):
                for pattern in SUBLIST_TOKEN_ZERO:
                    abstract = re.sub(pattern, r"\4", abstract)
                abstract = re.sub("^©(.*?)\.", "", abstract)
            elif (("©") in tokens[0]) and (not tokens[0].startswith("©")):
                for pattern in SUBLIST_TOKEN_ONE:
                    abstract = re.sub(pattern, r"\4", abstract)
                abstract = re.sub(
                    "^[0-9][0-9][0-9][0-9](.*?)©(.*?)the (authors|author|author(s))",
                    "",
                    abstract,
                    flags=re.IGNORECASE,
                )
            for idx in (-2, -1):
                tokens = abstract.split(". ")
                try:
                    if "©" in tokens[idx]:
                        abstract = ". ".join(tokens[:idx]) + "."
                except:
                    pass
            else:
                abstract = abstract
            for pattern in SUBLIST_POST:
                abstract = re.sub(pattern, "", abstract)
            cleaned_abstracts.append(abstract)
        cleaned_abstracts = set([i for i in cleaned_abstracts if i])
        return cleaned_abstracts

    def process_abstracts(self):
        """Process abstracts through regex, NER, and classification"""
        self.cleaned_abstracts = self._abstract_cleaning()


def main():
    """Processing pipeline"""
    abstract_path = "/scratch/remills_root/remills/stevesho/bio_nlp/nlp/abstracts"
    abstract_file = f"{abstract_path}/abstracts_combined.pkl"

    if not os.path.exists(abstract_file):
        try:
            _abstract_retrieval_concat(data_path=abstract_path)
        except FileExistsError:
            pass

    abstractcollectionObj = AbstractCollection(abstracts=pd.read_pickle(abstract_file))

    # run processing!
    abstractcollectionObj.process_abstracts()

    # save
    with open(f"{abstract_path}/cleaned_abstracts.pkl", "wb") as f:
        pickle.dump(abstractcollectionObj.cleaned_abstracts, f)


if __name__ == "__main__":
    main()
