#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Abstract df concatenation, cleaning with regular expressions, and relevancy
classification"""


import contextlib
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils import _abstract_retrieval_concat
from utils import SUBLIST
from utils import SUBLIST_INITIAL
from utils import SUBLIST_POST
from utils import SUBLIST_TITLE
from utils import SUBLIST_TOKEN_ONE
from utils import SUBLIST_TOKEN_ZERO
from utils import time_decorator


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
        self.abstracts = [abstract for abstract in abstracts if abstract]

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

    def process_abstracts(self) -> None:
        """Process abstracts through regex, NER, and classification"""
        self.cleaned_abstracts = self._abstract_cleaning()


def main(path: str) -> None:
    """Processing pipeline"""
    abstract_file = f"{path}/abstracts_combined.pkl"

    if not os.path.exists(abstract_file):
        with contextlib.suppress(FileExistsError):
            _abstract_retrieval_concat(data_path=path, save=True)
    abstractcollectionObj = AbstractCollection(abstracts=pd.read_pickle(abstract_file))

    # run processing!
    abstractcollectionObj.process_abstracts()

    # save
    with open(f"{path}/cleaned_abstracts.pkl", "wb") as f:
        pickle.dump(abstractcollectionObj.cleaned_abstracts, f)


if __name__ == "__main__":
    main(path="/scratch/remills_root/remills/stevesho/bio_nlp/nlp/abstracts")
