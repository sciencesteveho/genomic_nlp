#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Detect gene-disease associations (GDAs) from scientific abstracts, using a
token/alias-based approach for genes and a HunFlair2 for diseases.
"""


import multiprocessing as mp
import pickle
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm  # type: ignore

from genomic_nlp.utils.common import gencode_genes


def gene_disease_edges_per_abstract(
    gene_sets: List[Set[str]], disease_sets: List[Set[str]]
) -> Set[Tuple[str, str]]:
    """Generate gene-disease edges for a single abstract."""
    all_pairs = set()
    for gset, dset in zip(gene_sets, disease_sets):
        # cartesian product for that abstract
        for g in gset:
            for d in dset:
                all_pairs.add((g, d))
    return all_pairs


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


def mentions_per_abstract(
    abstracts: pd.DataFrame, alias_to_entity: Dict[str, str]
) -> List[Set[str]]:
    """Loop through tokenized abstracts and create a sublist of mentioned genes
    within the abstract. Gene mentions are based on tokens that either map to
    the gene symbol or synonyms from the HGNC complete set.
    """
    alias = set(alias_to_entity.keys())

    def process_abstract(abstract_sentences: List[List[str]]) -> Set[str]:
        """Vectorized matching"""
        tokens = [
            token for sentence in abstract_sentences for token in sentence
        ]  # flatten the list of sentences
        matches = set(tokens) & alias
        mentions = {alias_to_entity[token] for token in matches}
        return mentions if len(mentions) > 2 else set()

    relations = abstracts["processed_abstracts_w2v"].apply(process_abstract)
    return [ents for ents in relations if ents]


def process_abstract_file(
    args: Tuple[int, Dict[str, str], Dict[str, str], int]
) -> Set[Tuple[str, str]]:
    """Process a single abstract file and return gene edges."""
    num, alias_to_gene, alias_to_disease, year = args
    abstracts = pd.read_pickle(
        f"/ocean/projects/bio210019p/stevesho/genomic_nlp/data/processed_abstracts_w2v_chunk_{num}.pkl"
    )
    # only keep abstracts from the specified year
    abstracts = abstracts[abstracts["year"] <= year]
    gene_relationships = mentions_per_abstract(abstracts, alias_to_gene)
    disease_relationships = mentions_per_abstract(abstracts, alias_to_disease)
    return gene_disease_edges_per_abstract(gene_relationships, disease_relationships)


def extract_gda_edges_from_abstracts(
    alias_to_gene: Dict[str, str],
    alias_to_disease: Dict[str, str],
    year: int,
    index_end: int = 20,
) -> Set[Tuple[str, str]]:
    """Extract gene edges from abstracts using multiprocessing."""
    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_abstract_file,
                    [
                        (num, alias_to_gene, alias_to_disease, year)
                        for num in range(index_end)
                    ],
                ),
                total=index_end,
            )
        )

    gene_edges: Set[Tuple[str, str]] = set()
    for result in results:
        gene_edges.update(result)
    return gene_edges


def write_edges_to_file(edge_set: Set[Tuple[str, str]], filename: str) -> None:
    """Write gene pairs to file from a set of edges."""
    with open(filename, "w") as file:
        for edge in edge_set:
            file.write(f"{edge[0]}\t{edge[1]}\n")


def main() -> None:
    """Main function"""
    working_directory = "/ocean/projects/bio210019p/stevesho/genomic_nlp"
    genes = gencode_genes(
        f"{working_directory}/reference_files/gencode.v45.basic.annotation.gtf"
    )

    with open(
        f"{working_directory}/embeddings/gene_synonyms.pkl",
        "rb",
    ) as file:
        hgnc_synonyms = pickle.load(file)

    with open(
        f"{working_directory}/training_data/disease/disease_synonyms.pkl",
        "rb",
    ) as file:
        disease_synonyms = pickle.load(file)

    combined_genes = combine_synonyms(hgnc_synonyms, genes)
    alias_to_gene = create_alias_to_gene_mapping(combined_genes)
    alias_to_disease = create_alias_to_gene_mapping(disease_synonyms)

    # run text extraction for each year model
    for year in range(2003, 2023 + 1):
        # for year in range(1):
        # year = 2007
        print(f"Processing year {year}")
        gene_edges = extract_gda_edges_from_abstracts(
            alias_to_gene=alias_to_gene, alias_to_disease=alias_to_disease, year=year
        )

        # write to text file
        write_edges_to_file(
            gene_edges,
            f"{working_directory}/training_data/disease/gda_co_occurence_{year}.tsv",
        )


if __name__ == "__main__":
    main()
