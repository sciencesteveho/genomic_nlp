#! /usr/bin/env python -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] to-do

"""Mine abstracts from scopus API"""


import argparse
import contextlib
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from pybliometrics.scopus import ScopusSearch

from utils import dir_check_make
from utils import SUBLIST
from utils import SUBLIST_INITIAL
from utils import SUBLIST_POST
from utils import SUBLIST_TITLE
from utils import SUBLIST_TOKEN_ONE
from utils import SUBLIST_TOKEN_ZERO

GENERAL_SEARCH_TERMS = [
    "ATAC-seq",
    "ChIA-PET",
    "DNA",
    "DNase",
    "GWAS",
    "Hi-C",
    "Pseudogene",
    "QTL",
    "RNA",
    "RNAi",
    "Repli-seq",
    "SNPs",
    "WGBS",
    "ChIP-seq",
    "chromatid",
    "chromatin",
    "eCLIP",
    "eQTL",
    "epigenetics",
    "epigenome",
    "epigenomic",
    "epigenomics",
    "gene",
    "genes",
    "genetic",
    "genetics",
    "genome",
    "genomic",
    "genomics",
    "genotype",
    "haplotype",
    "lncRNA",
    "lncRNAs",
    "mRNA",
    "methylation",
    "noncoding",
    "phenotype",
    "polymerase",
    "proteome",
    "retrotransposon",
    "sRNAs",
    "telomerase",
    "transcription",
    "transcriptional",
    "transcriptome",
    "transcriptomic",
    "transcriptomics",
    "transposon",
    "tRNA",
    "{allele}",
    "{chromatin modification}",
    "{3D chromatin interactions}",
    "{DNA accessibility}",
    "{DNA condensation}",
    "{DNA damage}",
    "{DNA elements}",
    "{DNA polymerase}",
    "{DNA repair}",
    "{DNA replication}",
    "{DNA sequencing}",
    "{DNA supercoiling}",
    "{RNA decay}",
    "{RNA interference}",
    "{RNA modification}",
    "{RNA polymerase}",
    "{RNA processing}",
    "{RNA replication}",
    "{chromatin remodeling}",
    "{chromatin states}",
    "{chromosome condensation}",
    "{chromosome segregation}",
    "{cis-regulatory}",
    "{copy number variation}",
    "{distal enhancers}",
    "{dna damage}",
    "{dna recombination}",
    "{functional genomics}",
    "{gene expression}",
    "{gene function}",
    "{genetic mutation}",
    "{gene regulation}",
    "{gene regulatory network}",
    "{genetic mechanism}",
    "{genome sequencing}",
    "{histone}",
    "{long non-coding RNA}",
    "{massively parallel reporter assays}",
    "{massively parallel sequencing}",
    "{messenger RNA}",
    "{microRNA}",
    "{noncoding elements}",
    "{next generation sequencing}",
    "{open chromatin}",
    "{origin of replication}",
    "{polyA RNA}",
    "{post-transcriptional modification}",
    "{post-translational modification}",
    "{protein activation}",
    "{protein coding}",
    "{protein decay}",
    "{protein modification}",
    "{protein translation}",
    "{protein-coding}",
    "{regulatory element}",
    "{regulatory elements}",
    "{repeat DNA}",
    "{repetitive DNA}",
    "{RNA binding}",
    "{segmental duplication}",
    "{short hairpin RNA}",
    "{single nucleotide polymorphism}",
    "{small RNA}",
    "{small inhibitory RNA}",
    "{tandem repeat}",
    "{topologically associating domains}",
    "{trans-regulatory}",
    "{transcription factors}",
    "{transcription factor}",
    "{transcriptional modification}",
    "{transcriptional regulation}",
    "{transcriptional regulation}",
    "{translational modification}",
    "{translational regulation}",
]

TEST_SET_JOURNALS = [
    "American Journal of Human Genetics",
    "Human Genetics",
    "Human Molecular Genetics",
    "European Journal of Human Genetics",
]


def create_scopus_search(
    query: str, start_year: int, end_year: int, interval: bool
) -> ScopusSearch:
    """Creates a ScopusSearch object with the specified query, start year, end year, and interval.

    Args:
        query (str): The search query.
        start_year (int): The start year for filtering the search results.
        end_year (int): The end year for filtering the search results.
        interval (bool): Flag indicating whether to include a range of years or a single year.

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
    dir_check_make("abstracts")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval",
        help="search over an internal of time",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--year",
        type=int,
        default=1898,
        help="year abstracts are published",
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

    search_query = f"TITLE-ABS-KEY({' OR '.join(GENERAL_SEARCH_TERMS)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re)"
    scopus_general = create_scopus_search(
        search_query, args.start, args.end, args.interval
    )
    year = f"{args.start}_{args.end}" if args.interval else args.year

    # save all abstracts w/ metadata
    df = pd.DataFrame(pd.DataFrame(scopus_general.results))
    df.to_pickle(f"abstracts/abstracts_results_{year}.pkl")

    # save just title and abstract
    subdf = df[
        df["description"].str.len() > 1
    ].reset_index()  # filter out empty descriptions
    subdf["combined"] = (
        subdf["title"].astype(str) + ". " + subdf["description"].astype(str)
    )
    subdf["combined"].to_pickle(f"abstracts/abstracts_{year}.pkl")

    # save subset where publicationName matches journal in the test set
    testdf = df[df["publicationName"].isin(TEST_SET_JOURNALS)].reset_index()
    testdf.to_pickle(f"abstracts/test/abstracts_testset_{year}.pkl")

    # save as a dict for matching
    # ab_dict = dict(zip(df.title, df.description))
    # with open(f'abstract_dicts/abstract_retrieval_{year}_dict.pkl', 'wb') as output:
    #     pickle.dump(clean_abstract_collection(ab_dict), output)


if __name__ == "__main__":
    main()
