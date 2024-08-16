#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Abstract dataframe concatenation and cleaning with regular expressions."""


import contextlib
import re
from typing import Union

import pandas as pd

from constants import SUBLIST
from constants import SUBLIST_INITIAL
from constants import SUBLIST_POST
from constants import SUBLIST_TOKEN_ONE
from constants import SUBLIST_TOKEN_ZERO


class AbstractCleaner:
    """Collection of scientific abstracts.

    Attributes:
        abstracts: A pandas Series or DataFrame containing abstracts.

    Methods
    ----------
        clean_abstracts: Process abstracts with regex.

    # Helpers
        GENE_REPLACEMENTS -- A dictionary of gene replacements for genes that look like names.
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

    def __init__(self, abstracts: Union[pd.Series, pd.DataFrame]) -> None:
        """Initialize the class, only adding abstracts that are not empty"""
        self.abstracts = abstracts
        if isinstance(abstracts, pd.DataFrame):
            self.abstracts = (
                self.abstracts["title"].astype(str)
                + ". "
                + self.abstracts["description"].astype(str)
            )

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
        """Process abstracts with regex"""
        self.cleaned_abstracts = self._abstract_cleaning()
