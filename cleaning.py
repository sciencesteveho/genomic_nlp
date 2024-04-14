#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Abstract dataframe concatenation and cleaning with regular expressions."""


import argparse
import contextlib
import os
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils import _abstract_retrieval_concat
from utils import SUBLIST
from utils import SUBLIST_INITIAL
from utils import SUBLIST_POST
from utils import SUBLIST_TOKEN_ONE
from utils import SUBLIST_TOKEN_ZERO


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
        """Initialize the class, only adding abstracts that are not empty"""
        self.abstracts = abstracts['title'].astype(str) + ". " + abstracts['abstract'].astype(str)

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
                with contextlib.suppress(Exception):
                    if "©" in tokens[idx]:
                        abstract = ". ".join(tokens[:idx]) + "."
            abstract = abstract
            for pattern in SUBLIST_POST:
                abstract = re.sub(pattern, "", abstract)
            cleaned_abstracts.append(abstract)
        return {i for i in cleaned_abstracts if i}

    def clean_abstracts(self) -> None:
        """Process abstracts through regex, NER, and classification"""
        self.cleaned_abstracts = self._abstract_cleaning()


def main() -> None:
    """Processing pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/nlp/abstracts",
        help="Path to abstracts",
    )
    args = parser.parse_args()

    working_path = Path(args.path)
    abstract_file = working_path / "abstracts_combined.pkl"

    if not os.path.exists(abstract_file):
        with contextlib.suppress(FileExistsError):
            _abstract_retrieval_concat(data_path=working_path, save=True)

    print(f"Abstract file found or created without issue {abstract_file}")

    abstractcollectionObj = AbstractCollection(
        abstracts=pd.read_pickle(abstract_file)
    )

    # run processing!
    print("Cleaning abstracts...")
    abstractcollectionObj.clean_abstracts()

    # save
    with open(working_path / "cleaned_abstracts.pkl", "wb") as f:
        pickle.dump(abstractcollectionObj.cleaned_abstracts, f)


if __name__ == "__main__":
    main()
