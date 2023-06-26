#! /usr/bin/env python -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] to-do

"""Download testing set abstracts from Scopus API"""

import argparse
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from pybliometrics.scopus import ScopusSearch

from utils import (
    SUBLIST,
    SUBLIST_INITIAL,
    SUBLIST_TOKEN_ZERO,
    SUBLIST_TOKEN_ONE,
    SUBLIST_POST,
    SUBLIST_TITLE,
)


relevant_source_ids = [
    21677,
    22101,
    22245,
    22266,
]  # AJHG, Human Genetics, Human Molecular Genetics, European Journal of Human Genetics  

irrelevant_source_ids = [  
    11500153511,
    19881,
    21100812579,
    21537,
    14316,
    21100838559,
]  # ACS Nano, Advanced Materials, Nature Energy, Environmental Science & Technology, Harvard Law Review, Fashion and Textiles

def make_directories(dir: str) -> None:
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


def _abstract_cleaning(abstract):
    """Lorem Ipsum"""
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
    return abstract


def clean_dict(d):
    """Lorem Ipsum"""
    new_dict = {}
    for key, value in d.items():
        for pattern in SUBLIST_TITLE:
            try:
                key = re.sub(pattern, "", key)
            except TypeError:
                key = ''
        new_dict[_abstract_cleaning(value)] = key
    return new_dict


def main() -> None:
    """Download some abstracts!"""
    make_directories("abstracts")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive",
        help="Positive or negative training set",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()
    
    if args.positive:
        source_ids = relevant_source_ids
    else:
        source_ids = irrelevant_source_ids
        
    scopus_general = ScopusSearch(
        f"EISSN({' OR '.join(source_ids)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re))",
        cursor=True,
        refresh=False,
        verbose=True,
        download=True,
    )
    year = f"{args.start}_{args.end}"

    # save as title plus abstract
    df = pd.DataFrame(pd.DataFrame(scopus_general.results))
    subdf = df[df['description'].str.len() > 1].reset_index()  # filter out empty descriptions
    subdf["combined"] = subdf["title"].astype(str) + ". " + subdf["description"].astype(str)
    subdf["combined"].to_pickle(f"abstracts/abstracts_{year}.pkl")

    # save as a dict for matching
    # ab_dict = dict(zip(df.title, df.description)) 
    # with open(f'abstract_dicts/abstract_retrieval_{year}_dict.pkl', 'wb') as output:
    #     pickle.dump(clean_dict(ab_dict), output)


if __name__ == "__main__":
    main()
