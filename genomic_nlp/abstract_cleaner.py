#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Abstract dataframe concatenation and cleaning with regular expressions."""


import argparse
import contextlib
import re
from typing import Union

import pandas as pd
from tqdm import tqdm  # type: ignore

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
        """Initialize the class. Dataframe should have both `abstracts` and
        `year` columns.
        """
        if (
            not isinstance(abstracts, pd.DataFrame)
            or "abstracts" not in abstracts.columns
            or "year" not in abstracts.columns
        ):
            raise ValueError(
                "Input must be a DataFrame with columns 'abstracts' and 'year'."
            )
        self.abstracts = abstracts

    def _abstract_cleaning(self, abstract: Union[str, float]) -> str:
        """Clean a single abstract using regex patterns."""
        try:
            if not isinstance(abstract, str):
                if pd.isna(abstract):
                    return ""
                abstract = str(abstract)

            for pattern in SUBLIST:
                abstract = re.sub(pattern, "", abstract)
            for pattern in SUBLIST_INITIAL:
                abstract = re.sub(pattern, "", abstract, flags=re.IGNORECASE)

            tokens = abstract.split(". ")
            if tokens[0].startswith("©"):
                for pattern in SUBLIST_TOKEN_ZERO:
                    abstract = re.sub(pattern, r"\4", abstract)
                abstract = re.sub("^©(.*?)\.", "", abstract)
            elif ("©" in tokens[0]) and (not tokens[0].startswith("©")):
                for pattern in SUBLIST_TOKEN_ONE:
                    abstract = re.sub(pattern, r"\4", abstract)
                abstract = re.sub(
                    r"^[0-9]{4}(.*?)©(.*?)the (authors|author|author\(s\))",
                    "",
                    abstract,
                    flags=re.IGNORECASE,
                )

            for idx in (-2, -1):
                tokens = abstract.split(". ")
                with contextlib.suppress(IndexError):
                    if "©" in tokens[idx]:
                        abstract = ". ".join(tokens[:idx]) + "."
            for pattern in SUBLIST_POST:
                abstract = re.sub(pattern, "", abstract)

            return abstract.strip()
        except Exception as e:
            print(f"\nError: {str(e)}")
            print(f"Abstract: {abstract}...")
            return ""

    def clean_abstracts(self) -> pd.DataFrame:
        """Process abstracts with regex and return a cleaned DataFrame."""
        tqdm.pandas(desc="Cleaning abstracts")
        cleaned_abstracts = self.abstracts["abstracts"].progress_apply(
            lambda x: self._abstract_cleaning(x)
        )
        cleaned_df = pd.DataFrame(
            {"cleaned_abstracts": cleaned_abstracts, "year": self.abstracts["year"]}
        )
        cleaned_df = cleaned_df[cleaned_df["cleaned_abstracts"].str.strip() != ""]

        return cleaned_df


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean abstracts.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the input file (CSV).",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/abstracts",
    )
    args = parser.parse_args()

    abstractcollectionObj = AbstractCleaner(
        pd.read_pickle(f"{args.path}/abstracts_combined.pkl")
    )
    cleaned_abstracts = abstractcollectionObj.clean_abstracts()
    cleaned_abstracts.to_pickle(f"{args.path}/abstracts_cleaned.pkl")


if __name__ == "__main__":
    main()
