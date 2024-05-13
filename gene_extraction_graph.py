#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Code to extract genes from scientific abstracts. Individual genes are nodes:
if genes appear in the same abstract, and edge is created between genes."""


import argparse
import csv
import pickle
from typing import Any, Dict, List, Set, Tuple, Union

import pybedtools  # type: ignore
from tqdm import tqdm  # type: ignore

# from utils import COPY_GENES

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


def gencode_genes(gtf: str) -> Set[str]:
    """_summary_

    Args:
        entity_file (str): _description_
        genes (Set[str]): _description_
        type (str, optional): _description_. Defaults to "gene".

    Returns:
        Set[str]: _description_
    """

    def gene_symbol_from_gencode(gencode_ref: pybedtools.BedTool) -> Set[str]:
        """Returns deduped set of genes from a gencode gtf. Written for the gencode
        45 and avoids header"""
        return {
            line[8].split('gene_name "')[1].split('";')[0]
            for line in gencode_ref
            if not line[0].startswith("#") and "gene_name" in line[8]
        }

    print("Grabbing genes from GTF")
    gtf = pybedtools.BedTool(gtf)
    genes = list(gene_symbol_from_gencode(gtf))

    for key in COPY_GENES:
        genes.remove(key)
        genes.append(COPY_GENES[key])
    return set(genes)


def gene_mentions_per_abstract(
    abstracts: List[List[str]], genes: Set[str]
) -> List[Set[str]]:
    """Loop through tokenized abstracts and create a sublist of mentioned genes
    within the abstract."""
    gene_relations = []

    for abstract in abstracts:
        gene_mentions: List[str] = []
        for sentence in abstract:
            gene_mentions.extend(token for token in sentence if token in genes)
        if len(set(gene_mentions)) > 2:
            gene_relations.append(set(gene_mentions))

    return gene_relations


def collect_gene_edges(gene_sets: List[Set[str]]) -> Set[Tuple[str, str]]:
    """Store gene pairs in a set to avoid duplicates."""
    edges = set()
    for gene_set in gene_sets:
        genes = list(gene_set)
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                edges.add((genes[i], genes[j]))
    return edges


def write_gene_edges_to_file(edge_set: Set[Tuple[str, str]], filename: str) -> None:
    """Write gene pairs to file from a set of edges."""
    with open(filename, "w") as file:
        for edge in edge_set:
            file.write(f"{edge[0]}\t{edge[1]}\n")


def extract_gene_edges_from_abstracts(
    index_end: int, genes: Set[str]
) -> Set[Tuple[str, str]]:
    """Extract gene edges from abstracts."""
    gene_edges: Set[Tuple[str, str]] = set()
    for num in range(index_end):
        with open(
            f"tokens_cleaned_abstracts_remove_punct_finetune_{num}.pkl", "rb"
        ) as f:
            abstracts = pickle.load(f)
        gene_relations = gene_mentions_per_abstract(abstracts, genes)
        gene_edges.update(collect_gene_edges(gene_relations))
    return gene_edges


def _load_ppi(ppi_file: str) -> Set[Tuple[str, str]]:
    """Load PPIs from HuRI edge file."""
    ppi = set()
    with open(ppi_file, "r") as file:
        for line in file:
            protein1, protein2 = line.strip().split()
            if protein1 != protein2:
                ppi.add((protein1, protein2))
    return ppi


def _load_reference(reference_file: str) -> Dict[str, str]:
    """Load a biomart file to map ensembl gene ids to gene symbols."""
    reference = {}
    with open(reference_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)
        for line in reader:
            if line[1] != "":
                reference[line[0]] = line[1]

    return reference


def _map_proteins_to_gene_symbols(
    ppi: Set[Tuple[str, str]], reference: Dict[str, str]
) -> Set[Tuple[str, str]]:
    """Map proteins to gene symbols."""
    return {
        (reference[protein1], reference[protein2])
        for protein1, protein2 in ppi
        if protein1 in reference and protein2 in reference
    }


def main() -> None:
    """Main function"""
    genes = gencode_genes("gencode.v45.basic.annotation.gtf")
    gene_edges = extract_gene_edges_from_abstracts(10, genes=genes)

    # write to text file
    write_gene_edges_to_file(gene_edges, "text_extracted_gene_edges.tsv")
    # unique_genes = {gene for edge in gene_edges for gene in edge}
    # len(unique_genes)  # 20785

    # load gene uniq gene edges to check for overlap
    gene_edges_uniq = set()
    with open("text_extracted_gene_edges.tsv", "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            gene_edges_uniq.add(line[0])
            gene_edges_uniq.add(line[1])

    # load gene edges
    with open("text_extracted_gene_edges.tsv", "r") as file:
        reader = csv.reader(file, delimiter="\t")
        gene_edges = {(line[0], line[1]) for line in reader}

    ppi = _load_ppi("HI-union.tsv")
    unique_proteins = {gene for edge in ppi for gene in edge}
    with open("uniq_proteins.txt", "w") as file:
        for protein in unique_proteins:
            file.write(f"{protein}\n")

    reference = _load_reference("biomart_ppi.txt")
    mapped_ppi = _map_proteins_to_gene_symbols(ppi, reference)

    unique_mapped_proteins = {gene for edge in mapped_ppi for gene in edge}

    huri_only = mapped_ppi - gene_edges


if __name__ == "__main__":
    main()
