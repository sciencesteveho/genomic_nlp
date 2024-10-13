#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Create dictionaries of gene synonyms from the hgnc complete set. The
dictionaries are used downstream to combine different gene embeddings into a
single representative gene embedding.

HGNC complete set was downloaded from:
    https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt
"""


import csv
from pathlib import Path
from typing import Dict, List, Set, Union


def create_synonym_dictionary(hgnc: Path, casefold: bool = True) -> Dict[str, Set[str]]:
    """Map gene symbols to potential synonyms from the hgnc complete set. We add
    the gene name, alias, and previous symbols as synonyms.
    """
    synonyms: Dict[str, Set[str]] = {}

    with open(hgnc, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # skip header
        for line in reader:

            # set gene symbol as key
            key = line[1].casefold() if casefold else line[1]
            synonyms[key] = set()

            # add name alias symbol, previous symbol
            for idx in [2, 8, 10]:
                if line[idx]:
                    synonym = formatter(name=line[idx], casefold=casefold)
                    _add_values(synonyms, key, synonym)

    return synonyms


def formatter(name: str, casefold: bool = True) -> Union[str, List[str]]:
    """Format a string to be used as a key in a dictionary."""
    REPLACE_SYMBOLS = {
        "(": "",
        ")": "",
        ",": "",
        '"': "",
        "/": "_",
    }

    name = name.casefold() if casefold else name
    for symbol, replacement in REPLACE_SYMBOLS.items():
        name = name.replace(symbol, replacement)
    return name.split("|") if "|" in name else name


def _add_values(
    synonyms: Dict[str, Set[str]], key: str, values: Union[str, List[str]]
) -> None:
    """Add values to a key in a dictionary accounting for type."""
    if isinstance(values, list):
        for value in values:
            synonyms[key].add(value)
    else:
        synonyms[key].add(values)
