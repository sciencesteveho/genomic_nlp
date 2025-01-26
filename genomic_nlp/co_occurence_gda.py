#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Use flair2
"""


import itertools
import multiprocessing as mp
import pickle
from typing import Dict, List, Set, Tuple

import flair  # type: ignore
from flair.data import Sentence  # type: ignore
from flair.models import EntityMentionLinker  # type: ignore
from flair.nn import Classifier  # type: ignore
import pandas as pd
import torch
from tqdm import tqdm  # type: ignore

from genomic_nlp.utils.common import gencode_genes


def combine_synonyms(
    synonyms: Dict[str, Set[str]], genes: Set[str]
) -> Dict[str, Set[str]]:
    """Combine synonyms and genes into a single set."""
    # first, casefold gene names
    genes = {gene.casefold() for gene in genes}

    # loop through synonyms. if gene is in synonym set, remove it from {genes}.
    # otherwise, add it to the synonym dictionary as its own entry
    for syn_set in synonyms.values():
        for syn in syn_set:
            if syn in genes:
                genes.remove(syn)

    # add leftover genes to its own synonym set
    for gene in genes:
        synonyms[gene] = {gene}

    return synonyms


def create_alias_to_gene_mapping(gene_aliases: Dict[str, Set[str]]) -> Dict[str, str]:
    """Create a mapping from each alias to its gene symbol."""
    alias_to_gene = {}
    for gene, aliases in gene_aliases.items():
        for alias in aliases:
            alias_to_gene[alias] = gene
    return alias_to_gene


def detect_genes_with_synonyms(
    tokenized_sentences: List[List[str]], alias_to_gene: Dict[str, str]
) -> Set[str]:
    """Flatten the tokenized sentences into one list of tokens, intersect with
    alias_to_gene keys, and map those matches back to canonical gene symbols.
    """
    # flatten all sentences into one token list
    tokens = [token for sent in tokenized_sentences for token in sent]
    token_uniq = set(tokens)

    # intersect with our known aliases
    valid_aliases = token_uniq.intersection(alias_to_gene.keys())
    return {alias_to_gene[a] for a in valid_aliases}


def _extract_normalized_name(linked_value: str) -> str:
    """Extract the normalized name from the linked identifier."""
    try:
        return linked_value.split("/name=", 1)[1].split(" (")[0]
    except IndexError:
        print(f"Error extracting normalized name from {linked_value}")


def detect_diseases_with_flair(
    tokenized_sentences: List[List[str]],
    disease_tagger: Classifier,
    disease_linker: EntityMentionLinker,
) -> Set[str]:
    """Detect diseases in a list of tokenized sentences using flair."""
    flair_sents = [
        Sentence(" ".join(sent_tokens)) for sent_tokens in tokenized_sentences
    ]

    # tag and link
    disease_tagger.predict(flair_sents)
    disease_linker.predict(flair_sents)

    found_diseases = set()
    for sentence in flair_sents:
        for span in sentence.get_spans("link"):
            label = span.get_label("link")
            if label is not None:
                norm_name = _extract_normalized_name(str(label.value))
                found_diseases.add(norm_name)

    return found_diseases


def collect_gene_disease_edges(
    gene_set: Set[str], disease_set: Set[str]
) -> Set[Tuple[str, str]]:
    """For each gene in gene_set, pair it with each disease in disease_set."""
    return set(itertools.product(gene_set, disease_set))


def process_abstract_file(
    chunk_idx: int, alias_to_gene: Dict[str, str]
) -> Set[Tuple[str, str]]:
    """Process one chunk of abstracts"""
    path = f"/ocean/projects/bio210019p/stevesho/genomic_nlp/data/processed_abstracts_w2v_chunk_{chunk_idx}.pkl"
    abstracts = pd.read_pickle(path)

    # load flair models
    disease_tagger = Classifier.load("hunflair2")
    disease_linker = EntityMentionLinker.load("disease-linker")

    gd_edges = set()

    for _, row in abstracts.iterrows():
        tokenized_sentences = row["processed_abstracts_w2v"]

        # detect genes
        gene_occurences = detect_genes_with_synonyms(tokenized_sentences, alias_to_gene)

        # disease flair
        disease_occurences = detect_diseases_with_flair(
            tokenized_sentences, disease_tagger, disease_linker
        )

        # combine to get gene-disease edges
        if len(gene_occurences) > 0 and len(disease_occurences) > 0:
            gd_edges.update(
                collect_gene_disease_edges(gene_occurences, disease_occurences)
            )

    return gd_edges


def write_edges_to_file(edge_set: Set[Tuple[str, str]], filename: str) -> None:
    """Write out edges to tsv."""
    with open(filename, "w") as file:
        for a, b in edge_set:
            file.write(f"{a}\t{b}\n")


def main() -> None:
    """Get gene-disease edges!"""
    # force GPU if desired
    flair.device = torch.device("cuda:0")

    working_directory = "/ocean/projects/bio210019p/stevesho/genomic_nlp"
    genes = gencode_genes(
        f"{working_directory}/reference_files/gencode.v45.basic.annotation.gtf"
    )

    with open(
        f"{working_directory}/embeddings/gene_synonyms.pkl",
        "rb",
    ) as file:
        hgnc_synonyms = pickle.load(file)

    combined_genes = combine_synonyms(hgnc_synonyms, genes)
    alias_to_gene = create_alias_to_gene_mapping(combined_genes)

    # get gda for chunk 0
    gda_edges = process_abstract_file(0, alias_to_gene)


if __name__ == "__main__":
    main()
