#! /usr/bin/env python -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] to-do

"""Mine abstracts from scopus API"""

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


general_search_terms = [
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

    if args.interval:
        scopus_general = ScopusSearch(
            f"TITLE-ABS-KEY({' OR '.join(general_search_terms)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re) AND (PUBYEAR > {args.start}) AND (PUBYEAR < {args.end}))",
            cursor=True,
            refresh=False,
            verbose=True,
            download=True,
        )
        year = f"{args.start}_{args.end}"
    else:
        scopus_general = ScopusSearch(
            f"TITLE-ABS-KEY({' OR '.join(general_search_terms)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re) AND (PUBYEAR = {args.year}))",
            cursor=True,
            refresh=False,
            verbose=True,
            download=True,
        )
        year = args.year

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
