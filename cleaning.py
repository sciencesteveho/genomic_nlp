#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Abstract df concatenation, cleaning with regular expressions, and relevancy
classification"""


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

    def __init__(self, abstracts: pd.Series) -> None:
        """Initialize the class, only adding abstracts that are not empty"""
        self.abstracts = abstracts[abstracts.notnull()]

    # @time_decorator
    def _abstract_cleaning(self):
        """Clean abstracts through a series of regex substitutions."""
        cleaned_abstracts = self.abstracts.copy()
        for pattern in SUBLIST:
            cleaned_abstracts = cleaned_abstracts.str.replace(pattern, "", regex=True)
        for pattern in SUBLIST_INITIAL:
            cleaned_abstracts = cleaned_abstracts.str.replace(
                pattern, "", flags=re.IGNORECASE, regex=True
            )
        tokens = cleaned_abstracts.str.split(". ")
        mask = tokens.str[0].str.startswith("©")
        cleaned_abstracts.loc[mask] = cleaned_abstracts.loc[mask].str.replace(
            SUBLIST_TOKEN_ZERO, r"\4", regex=True
        )
        cleaned_abstracts.loc[mask] = cleaned_abstracts.loc[mask].str.replace(
            "^©(.*?)\.", "", regex=True
        )
        mask = (~tokens.str[0].str.startswith("©")) & (tokens.str[0].str.contains("©"))
        cleaned_abstracts.loc[mask] = cleaned_abstracts.loc[mask].str.replace(
            SUBLIST_TOKEN_ONE, r"\4", regex=True
        )
        cleaned_abstracts.loc[mask] = cleaned_abstracts.loc[mask].str.replace(
            "^[0-9][0-9][0-9][0-9](.*?)©(.*?)the (authors|author|author(s))",
            "",
            flags=re.IGNORECASE,
            regex=True,
        )
        tokens = cleaned_abstracts.str.split(". ")
        cleaned_abstracts.loc[tokens.str[-2].str.contains("©")] = tokens.loc[
            tokens.str[-2].str.contains("©")
        ].apply(lambda x: ". ".join(x[:-2]) + ".", axis=1)
        cleaned_abstracts.loc[tokens.str[-1].str.contains("©")] = tokens.loc[
            tokens.str[-1].str.contains("©")
        ].apply(lambda x: ". ".join(x[:-1]) + ".", axis=1)
        for pattern in SUBLIST_POST:
            cleaned_abstracts = cleaned_abstracts.str.replace(pattern, "", regex=True)
        return cleaned_abstracts.dropna().unique()

    def process_abstracts(self) -> None:
        """Process abstracts through regex, NER, and classification"""
        self.cleaned_abstracts = self._abstract_cleaning()


def main(path: str) -> None:
    """Processing pipeline"""
    path = Path(path)
    abstract_file = path / "abstracts_combined.pkl"

    if not os.path.exists(abstract_file):
        with contextlib.suppress(FileExistsError):
            _abstract_retrieval_concat(data_path=path, save=True)

    abstractcollectionObj = AbstractCollection(abstracts=pd.read_pickle(abstract_file))

    # run processing!
    abstractcollectionObj.process_abstracts()

    # save
    with open(path / "cleaned_abstracts.pkl", "wb") as f:
        pickle.dump(abstractcollectionObj.cleaned_abstracts, f)


if __name__ == "__main__":
    main(path="/nfs/turbo/remillsscr/stevesho/nlp/abstracts")
