#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Mine abstracts from scopus API."""


import argparse
from pathlib import Path
from typing import Union

import pandas as pd
from pybliometrics.scopus import ScopusSearch  # type: ignore

from genomic_nlp.utils.common import dir_check_make
from genomic_nlp.utils.constants import GENERAL_SEARCH_TERMS
from genomic_nlp.utils.constants import TEST_SET_JOURNALS


def create_scopus_search(
    query: str,
    start_year: int,
    end_year: Union[int, None] = None,
    interval: bool = False,
) -> ScopusSearch:
    """Creates a ScopusSearch object with the specified query, start year, end
    year, and interval.

    Args:
        query (str): The search query.
        start_year (int): The start year for filtering the search results.
        end_year (int): The end year for filtering the search results.
        interval (bool): Flag indicating whether to include a range of years or
        a single year.

    Returns:
        ScopusSearch: The created ScopusSearch object.

    Examples:
        >>> create_scopus_search("machine learning", 2010, 2020, True)
        <ScopusSearch object at 0x7f9a3a2b5a90>
    """
    if interval:
        query += f" AND (PUBYEAR > {start_year}) AND (PUBYEAR < {end_year})"
    else:
        query += f" AND (PUBYEAR = {start_year})"
    return ScopusSearch(query, cursor=True, refresh=False, verbose=True, download=True)


def main() -> None:
    """Download some abstracts!"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working_dir",
        help="working directory",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/nlp",
        required=False,
    )
    parser.add_argument(
        "--interval",
        help="search over an internal of time",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1898,
        help="random seed to use (default: 1898)",
        required=False,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2023,
        help="random seed to use (default: 2023)",
        required=False,
    )
    args = parser.parse_args()

    working_dir = Path(args.working_dir)
    abstract_dir = working_dir / "abstracts"
    dir_check_make(abstract_dir)

    search_query = f"TITLE-ABS-KEY({' OR '.join(GENERAL_SEARCH_TERMS)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re))"
    scopus_general = create_scopus_search(
        query=search_query,
        start_year=args.start,
        end_year=args.end or None,
        interval=args.interval or False,
    )
    year = f"{args.start}_{args.end}" if args.interval else args.start

    # save all abstracts w/ metadata
    df = pd.DataFrame(pd.DataFrame(scopus_general.results))
    df.to_pickle(abstract_dir / f"abstracts_results_{year}.pkl")

    # save just title and abstract
    subdf = df[
        df["description"].str.len() > 1
    ].reset_index()  # filter out empty descriptions
    subdf["combined"] = (
        subdf["title"].astype(str) + ". " + subdf["description"].astype(str)
    )
    subdf["combined"].to_pickle(abstract_dir / f"abstracts_{year}.pkl")

    # save subset where publicationName matches journal in the test set
    testdf = df[df["publicationName"].isin(TEST_SET_JOURNALS)].reset_index()
    testdf.to_pickle(abstract_dir / "test" / f"abstracts_testset_{year}.pkl")

    # save as a dict for matching
    # ab_dict = dict(zip(df.title, df.description))
    # with open(f'abstract_dicts/abstract_retrieval_{year}_dict.pkl', 'wb') as output:
    #     pickle.dump(clean_abstract_collection(ab_dict), output)


if __name__ == "__main__":
    main()
