#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for bio-genetics-NLP"""


import contextlib
from datetime import timedelta
import functools
import glob
import inspect
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from tqdm import tqdm  # type: ignore

SUBLIST = [
    "\[Figure not available\: see fulltext.\]\.",
    "(Authors [0-9][0-9][0-9][0-9],|Author\(s\) [0-9][0-9][0-9][0-9],|Author\(s\), [0-9][0-9][0-9][0-9]| Group, a division of|)( under exclusive licence to|[0-9][0-9][0-9][0-9]|) Macmillan Publishers (Ltd|Limited, part of Springer Nature|Limited)",
    "Copyright [0-9][0-9][0-9][0-9] by Annual Reviews. ",
    " Biological Reviews",
    "(Copyright [0-9][0-9][0-9][0-9] by the|Copyright [0-9][0-9][0-9][0-9] the|Copyright [0-9][0-9][0-9][0-9]|exclusive licensee) American Association for the Advancement of Science.",
    "Elsevier (Science Inc.|Science|)",
    "Copyright \(C\) [0-9][0-9][0-9][0-9]",
    "Copyright [0-9][0-9][0-9][0-9]",
    "\(C\) [0-9][0-9][0-9][0-9]",
    "Proceedings of the National Academy of Sciences of the United States of America",
    "National Academy of Sciences",
    "The (Author\(s\)|Authors|Author) [0-9][0-9][0-9][0-9]\. ",
    "This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution \(CC BY\) license \(http://creativecommons.org/licenses/by/4.0/\), ",
    "This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution \(CC BY\) license \(http://creativecommons.org/licenses/by/4.0/\)",
    "This( article| )is an (Open Access|Open-Access|open-access|open access) article (distributed under|under)( the CC BY-NC-ND license \(http://creativecommons.org/licenses/by-nc-nd/4.0/\)|  the CC BY license \(http://creativecommons.org/licenses/by/3.0/\).| |)the terms of the Creative (Commons|Commns) (Attribution Non-Commercial |Attribution |)License \(http(s|)://( creativecommons|creativecommons).org/( |)licenses/(by-nc-nd|by-nc|by)/(2.5|2.0|3.0|4.0|[0-9]\.[0-9]|[0-9]\. [0-9])(/uk/|/|)\)(, | |.|)(which permits unrestricted non-commercial use, distribution, and reproduction in any medium, provided the original work is properly cited|which permits unrestricted non-commercial use, distribution, and reproduction in any medium, provided the original work is properly cited attributed| |)(\.|\,| |)",
    "http://creativecommons.org/licenses/by/4.0/",
    "This article is distributed under the terms of the Creative Commons Attribution 4.0 International License (), which permits unrestricted use, distribution, and reproduction in any medium, provided you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The Creative Commons Public Domain Dedication waiver \(http://creativecommons.org/publicdomain/zero/1.0/\) applies to the data made available in this article, unless otherwise stated.",
    "et al.; licensee BioMed Central Ltd. which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly credited. The Creative Commons Public Domain Dedication waiver \(http://creativecommons.org/publicdomain/zero/1.0/\) applies to the data made available in this article, unless otherwise stated.",
    "This is an open (access article|article) (distributed under|under) the (terms if the Creative Commons Attribution Non-Commercial License|)(CC BY-NC-ND|CC BY-NC|\(CC BY-NC\)|CC BY|)( license | |)\(http://creativecommons.org/licenses/by-nc-nd/4.0/\)",
    "This article is distributed under the terms of the Creative Commons Attribution 4.0 International License \(\), which permits unrestricted use, distribution, and reproduction in any medium, provided you give appropriate credit to the original author\(s\) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The Creative Commons Public Domain Dedication waiver \(http://creativecommons.org/publicdomain/zero/1.0/\) applies to the data made available in this article, unless otherwise stated.",
    "This is an open access article distributed under the terms of the Creative Commons Attribution \(CC BY-NC\) license \(https://creativecommons.org/licenses/by-nc/4.0/\).",
    "After six months it is available under a Creative Commons License \(Attribution-Noncommercial-Share Alike 4\.0 International license, as described at https://creativecommons.org/licenses/by-nc-sa/4.0/\)",
    "This is an article distributed under the terms of the Creative Commons Attribution Non-Commercial License \(http://creativecommons.org/licenses/by-nc-nd/4.0/\)",
    "video abstract",
    "Office in association with Oxford University Press.*?\.",
    "Published by Oxford University Press.*?\.",
    " Oxford University Press",
    "Proc. Natl. Acad. Sci.",
    "Published by Cold Spring Harbor Laboratory Press.( |)",
    "[a-zA-Z]+ published by Wiley Periodicals, Inc. on behalf.*?\.",
    "[a-zA-Z]+ published by Wiley Periodicals, Inc.",
    "Wiley Periodicals, Inc",
    "et al.",
    "This is a U.S. government work and not under copyright protection in the U.S(.|); foreign copyright protection may apply.",
    "Limited, part of Springer Nature.",
    " , part of Springer Nature.",
    " LLC, part of Springer Nature.",
    " Germany, part of Springer Nature.",
    " Author\(s\), under exclusive licence to Springer Nature America, Inc.",
    " Author\(s\), under exclusive licence to Springer Nature Limited.",
    " ., part of Springer Nature.",
    " KK, part of Springer Nature.",
    "Austria, part of Springer Nature.",
    "AG, part of Springer Nature.",
    " SAS, part of Springer Nature.",
    " \(C\) [0-9][0-9][0-9][0-9] by The American Society of Hematology.",
    " Official journal of the American Society of Gene  &  Cell Therapy",
    "  by The American Society of Hematology.",
    "Published under exclusive license by The American Society for Biochemistry and Molecular Biology, Inc.",
    "Transplantation and the American Society of Transplant Surgeons",
    "This article is distributed by The American Society for Cell Biology under license from the author\(s\).",
    "[a-zA-Z]+ published by John Wiley  &  Sons.*?\.",
    "; licensee Biomed Central( Ltd.|.)",
    "( |)(For permissions|For Permissions|Permissions)(,|) please email: (J|j)ournals.permissions(@oup.com.|@oxfordjournals.org.)",
    "For any queries, please email at epub@benthamscience.org.",
    "For any queries, please email at epub@benthamscience.net.",
    "http[s]?://[^\s]+[^.( \n)]",
    "www[^\s]+[^.( \n)]",
    "�",
    "wiley periodicals inc",
    "This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.",
]

SUBLIST_INITIAL = [
    "( All|All) (rights of any nature whatsoever reserved.|rights of reproduction in any form reserved.|rights rights reserved.|rights reserved.|rights reserved|rights resreved.|rights reseserved.|rights reseved.|rights received.|rights reserve.|rights.|rights)",
    "(, some| some) (rights reserved|rights reserved;)",
    "(Published with license by Taylor  &  Francis)( Group, LLC. | Group, LLC |. | )",
]
SUBLIST_TOKEN_ZERO = [
    '(^© ([0-9]*))(.*?)([a-z0-9A-Z!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+)',
    '(^©([0-9]*))(.*?)([a-z0-9A-Z!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+)',
]
SUBLIST_TOKEN_ONE = [
    '^Copyright(.*?)© ([0-9]*)(.*?)([a-z0-9A-Z!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+)',
    '^Copyright(.*?)©([0-9]*)(.*?)([a-z0-9A-Z!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+)',
    '^, Copyright(.*?)©([0-9]*)(.*?)([a-z0-9A-Z!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+\ [a-z0-9!@#$&()\\-`+,/"]+)',
]
SUBLIST_POST = [
    "^[A-Z]{1}[a-z]+ et al\. ",
    "This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited. ",
    "Limited, trading as Taylor  &  Francis Group. ",
    "<.{3}>",
    "</.{3}>",
]
SUBLIST_TITLE = ["<.{3}>", "</.{3}>"]

FILTER_TERMS = [
    "promoter",
    "enhancer",
    "transcription factor",
    "transcriptional regulator",
    "transcriptional repressor",
    "gene",
    "protein",
    "transcript",
    "chromatin",
    "chromosome",
    "mRNA",
    "DNA",
    "RNA",
    "mechanism",
    "mechanisms",
    "messenger",
    "signaling",
    "signal",
    "pathway",
    "pathways",
    "receptor",
    "receptors",
    "cell",
    "cells",
    "cellular",
    "eQTL",
    "QTL",
    "ChIP",
    "epigenomic",
    "epigenome",
    "lncRNA",
    "noncoding",
    "polymerase",
    "retrotransposon",
    "telomerase",
    "transcription",
    "transcriptional",
    "transcriptome",
    "transposon",
    "DNA accessibility",
    "DNA damage",
    "DNA element" "DNA methylation",
    "DNA repair",
    "DNA sequence",
    "DNA structure",
    "3D chromatin",
    "3D genome",
    "chromatin accessibility",
    "cis-regulatory element",
    "chromatin conformation",
    "chromatin domain",
    "chromatin loop",
    "chromatin modification",
    "chromatin organization",
    "histone",
    "histone modification",
    "histone variant",
    "nucleosome",
    "H3K4me3",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "microRNA",
    "post-translational modification",
    "protein complex",
    "noncoding elements",
    "noncoding RNA",
    "noncoding RNAs",
    "super enhancer",
    "super-enhancer",
    "super-enhancers",
    "transcriptional modification",
    "transcriptional regulation",
    "transcriptional regulator",
    "transcriptional repressor",
    "cell function",
    "cell type",
    "cell types",
    "cellular function",
    "biological process",
    "biological processes",
    "cellular component",
    "biological mechanism",
    "biological mechanisms",
    "biological pathway",
    "multicellular organism",
    "multicellular organisms",
    "eukaryotic",
    "vertebrate",
    "signaling pathway",
    "signaling pathways",
    "mammalian",
    "human",
    "humans",
    "tissue",
    "tissues",
    "tissue-specific",
    "tissue-specificity",
    "tissue-specific expression",
    "controlling",
    "target",
    "targets",
    "targeting",
    "targeted",
    "cell",
    "key role",
    "genome regulation",
    "structural variation",
    "polymorphism",
    "gene expression",
    "gene expression regulation",
    "activated",
    "key signaling",
    "key role",
    "key roles",
    "key function",
    "genome structure",
    "genome organization",
    "genome regulation",
    "genome-wide",
    "genome-wide association",
    "genome-wide association study",
    "critical roles",
    "biological functions",
    "biological function",
    "biological processes",
    "biological process",
    "biological mechanisms",
    "gene target",
    "gene targets",
    "core human",
    "core promoter",
    "loss",
    "loss of function",
    "inhibitor",
    "inhibits",
    "role",
    "function",
    "contribution",
    "critical",
    "involved",
    "evidence",
    "lacked",
    "present",
    "precise",
    "process",
    "explored",
    "demonstrated",
    "proposed",
    "involvement",
    "down-regulated",
    "down-regulation",
    "regulates",
]

SPECIFIC_TERMS = [
    "central dogma",
    "gene expression",
    "mRNA",
    "mRNA stability",
    "DNA replication",
    "transcription",
    "translation",
    "transcriptional regulation",
    "reverse transcription",
    "RNA replication",
    "post-translational modification",
    "methylation",
    "epigenetic",
    "model",
    "models",
    "cell cycle",
    "regulator",
    "mechanism",
    "mammalian",
    "human",
    "histones",
    "chromatin",
    "promoter",
    "enhancer",
    "required",
    "necessary",
    "essential",
    "dependent",
]

COPY_GENES = {
    "WAS": "wasgene",
    "SHE": "shegene",
    "IMPACT": "impactgene",
    "MICE": "micegene",
    "REST": "restgene",
    "SET": "setgene",
    "MET": "metgene",
    "GC": "gcgene",
    "ATM": "atmgene",
    "MB": "mbgene",
    "PIGS": "pigsgene",
    "CAT": "catgene",
    "COIL": "coilgene",
}


def _concat_chunks(filenames: List[str]) -> List[List[str]]:
    """Concatenates chunks of abstracts"""
    combined = []
    combined += [pickle.load(open(file, "rb")) for file in filenames]
    return combined


def _chunk_locator(path: str, prefix: str) -> List[str]:
    """Returns abstract chunks matching a specific prefix"""
    pattern = f"{path}/{prefix}_*.pkl"
    return glob.glob(pattern)


def _combine_chunks(path: str, prefix: str) -> List[List[str]]:
    """Combines chunks of abstracts"""
    filenames = _chunk_locator(path, prefix)
    print(f"Combining chunks of abstracts: {filenames}")
    return _concat_chunks(filenames)


def dir_check_make(dir_path: Union[str, Path]) -> None:
    """Check if a directory exists, if not, create it."""
    Path(dir_path).mkdir(exist_ok=True)


def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    """Decorator to time functions.

    Args:
        print_args (bool, optional): Whether to print the function arguments.
        Defaults to False. display_arg (str, optional): The argument to display
        in the print statement. Defaults to "".

    Returns:
        Callable: The decorated function.
    """

    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            result = function(*args, **kwargs)
            end_time = time.monotonic()
            args_to_print = list(fxn_args.values()) if print_args else display_arg
            print(
                f"Finished {function.__name__} {args_to_print} - Time: {timedelta(seconds=end_time - start_time)}"
            )
            return result

        return _execute

    return _time_decorator_func


def filter_abstract_by_terms(string: str, substr: str, matches, remove, keep):
    filtered = []
    for s in tqdm(string):
        if keep == "match":
            if len(substr.intersection(s.split())) >= matches:
                if len(remove) > 0:
                    if len(remove.intersection(s.split())) == 0:
                        filtered.append(s)
                else:
                    filtered.append(s)
        elif keep == "remove ":
            if len(substr.intersection(s.split())) <= matches:
                filtered.append(s)
        else:
            raise ValueError("keep must be either 'match' or 'remove'")
    return filtered


def _abstract_retrieval_concat(
    data_path: Union[Path, str], save: bool = True
) -> pd.DataFrame:
    """Take abstract outputs and combine into a single pd.series. Only needs to
    be done initially after downloading abstracts"""
    frames = [
        pd.read_pickle(file, compression=None)
        for file in glob.glob(f"{data_path}/*.pkl")
    ]
    df = pd.concat(frames, ignore_index=True)
    if save:
        with open(f"{data_path}/abstracts_combined.pkl", "wb") as f:
            df.to_pickle(f)
    return df


def _random_subset_abstract_printer(n: int, abstracts: List) -> None:
    """Prints N random abstracts"""
    for num in random.sample(range(len(abstracts)), n):
        print(abstracts[num])


def _listdir_isfile_wrapper(dir_path: str) -> List[str]:
    return [file.name for file in Path(dir_path).iterdir() if file.is_file()]


def is_number(entry: str) -> bool:
    """
    # Arguments
        entry: the string to be checked
    # Returns
        True for float, false for other
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False


def avg_len(lst: List[str]) -> int:
    """Takes the average length of elements in a list

    Args:
        lst (_type_): _description_

    Returns:
        int
    """
    total_lengths = [len(i) for i in lst]
    return int(sum(total_lengths)) / len(total_lengths) if total_lengths else 0


def dict_from_gene_symbol_and_name_list(gene_file_path):
    """Takes a tab delimited file organized as 'symbol'\t''name' and
    parses as a dictionary, removing entries with values in remove_words,
    which includes duplicates."

    # Arguments
        gene_file_path: filepath for gene tab file

    # Returns
        dictionary of values
    """
    remove_words = {"novel transcript", ""}
    namedict = {}
    with open(gene_file_path) as f:
        for line in f:
            symbol, name = line.strip().split("\t")
            name = re.sub(r"[^\w\s]|_", "", name).replace("  ", " ").strip().lower()
            namedict[name] = symbol.lower()

    # Find duplicates
    duplicates = {
        symbol for symbol, count in Counter(namedict.values()).items() if count > 1
    }
    remove_words.update(duplicates)

    # Filter out remove_words and duplicates
    return {
        name: symbol for name, symbol in namedict.items() if symbol not in remove_words
    }
    # remove_words = ["novel transcript", ""]
    # namedict = {}
    # with open(gene_file_path) as f:
    #     for line in f:
    #         line = line.strip("\n")
    #         a, b = line.split("\t")
    #         b = "".join(e for e in b if e.isalnum() or e in string.whitespace)
    #         b = re.sub("  ", " ", b)
    #         b = b.rstrip()
    #         b = re.sub(" ", "_", b)
    #         namedict.update({b.lower(): a.lower()})
    # dup_list = list(namedict.values())
    # val_dupes = set([item for item in dup_list if dup_list.count(item) > 1])
    # for dupe in val_dupes:
    #     remove_words.append(dupe)
    # set(remove_words)
    # return {key: value for key, value in namedict.items() if value not in remove_words}
