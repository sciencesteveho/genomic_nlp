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

import flair  # type: ignore
from flair.data import Sentence  # type: ignore
from flair.models import EntityMentionLinker  # type: ignore
from flair.nn import Classifier  # type: ignore
from tqdm import tqdm  # type: ignore

from genomic_nlp.abstracts.gene_entity_normalization import replace_symbols
from genomic_nlp.utils.normalize_provenance import load_flair_models


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


def create_disease_synonym_dictionary(
    ctd: Path, casefold: bool = True
) -> Dict[str, Set[str]]:
    """Map disease identifiers to potential synonyms from the CTD disease
    vocabulary. We add the disease name, and any alternate identifiers as
    synonyms.
    """
    synonyms: Dict[str, Set[str]] = {}

    with open(ctd, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:

            # skip header, #
            if line[0].startswith("#"):
                continue

            # set disease identifier as key
            key = (
                replace_symbols(line[0]).casefold()
                if casefold
                else replace_symbols(line[0])
            )
            synonyms[key] = set()

            # add name and alternate identifiers
            for idx in [7]:
                if line[idx]:
                    synonym = formatter(name=line[idx], casefold=casefold)
                    _add_values(synonyms, key, synonym)

    return synonyms


def formatter(name: str, casefold: bool = True) -> Union[str, List[str]]:
    """Format a string to be used as a key in a dictionary."""
    name = replace_symbols(name)
    name = name.casefold() if casefold else name
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


def _extract_normalized_name(linked_value: str) -> str:
    """Extract the normalized name from the linked identifier."""
    try:
        return linked_value.split("/name=", 1)[1].split(" (")[0]
    except IndexError:
        return linked_value  # fallback


def add_flair_normalized_genes(
    gene_synonyms: Dict[str, Set[str]],
    tagger: Classifier,
    gene_linker: EntityMentionLinker,
    batch_size: int = 1024,
) -> Dict[str, Set[str]]:
    """Use Flair to normalize the gene name (key) and add it to the synonym
    set.
    """
    gene_symbols = list(gene_synonyms.keys())
    num_genes = len(gene_symbols)
    normalized_count = 0

    for start_idx in tqdm(
        range(0, num_genes, batch_size), desc="Normalizing Gene Symbols"
    ):
        end_idx = min(start_idx + batch_size, num_genes)
        batch_symbols = gene_symbols[start_idx:end_idx]

        # create sentences for flair
        sentences = [
            Sentence(f"The related gene symbol is {symbol}.")
            for symbol in batch_symbols
        ]

        # tag and link entities
        tagger.predict(sentences)
        gene_linker.predict(sentences)

        # extract and add normalized names to synonym sets
        for i, sentence in enumerate(sentences):
            original_symbol = batch_symbols[i]
            normalized_symbol = original_symbol  # fallback if no normalization found

            spans = sentence.get_spans("link")
            for span in spans:
                if label := span.get_label("link"):
                    normalized_symbol = _extract_normalized_name(str(label))
                    normalized_count += 1
                    break

            gene_synonyms[original_symbol].add(normalized_symbol)

    print(f"Normalization complete. Total normalized symbols added: {normalized_count}")
    return gene_synonyms
