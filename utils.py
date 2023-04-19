#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for bio-genetics-NLP"""

from datetime import timedelta
import functools
import inspect
import pickle
import random
import time
from tqdm import tqdm
from typing import Any, Callable, List

import pandas as pd


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


def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            try:
                result = function(*args, **kwargs)
                return result
            except Exception as error:
                result = str(error)
                raise
            finally:
                end_time = time.monotonic()
                if print_args == True:
                    print(
                        f"Finished {function.__name__} {[val for val in fxn_args.values()]} - Time: {timedelta(seconds=end_time - start_time)}"
                    )
                else:
                    print(
                        f"Finished {function.__name__} {display_arg} - Time: {timedelta(seconds=end_time - start_time)}"
                    )

        return _execute

    return _time_decorator_func


def Filter(string, substr):
    filtered_list = []
    for s in tqdm(string):
        common = substr.intersection(s.split())
        if len(common) >= 9:
            filtered_list.append(s)
    return filtered_list


def _random_subset_abstract_printer(n: int, abstracts: List) -> None:
    """Prints N random abstracts"""
    for num in random.sample(range(0, len(abstracts)), n):
        print(abstracts[num])


abstracts = []
for num in [1, 2, 3, 4, 5]:
    abstracts.append(pd.read_pickle(f"gene_filtered_abstracts_{num}.pkl"))

abstracts = [x for abstract in abstracts for x in abstract]
abstracts = list(set(abstracts))

_random_subset_abstract_printer(
    25, abstracts
)

genes = [
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


with open('cleaned_abstracts.pkl', 'rb') as file:
    abstracts = pickle.load(file)

# get a subset of genes, and a subset of abstracts
for num in [1, 2, 3, 4, 5]:
    # subset_genes = sample(genes, 1000)
    subset_abs = random.sample(abstracts, 1000000)
    geneset = set(genes)

    # gene_filtered_abstracts = Filter(subset_abs, subset_genes)
    # gene_filtered_abstracts = Filter(subset_abs, genes)
    gene_filtered_abstracts = Filter(subset_abs, geneset)
    print(len(gene_filtered_abstracts))

    with open(f"gene_filtered_abstracts_{num}.pkl", "wb") as output:
        pickle.dump(gene_filtered_abstracts, output)
