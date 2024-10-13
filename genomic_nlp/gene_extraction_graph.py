#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to extract genes from scientific abstracts. Individual genes are nodes:
if genes appear in the same abstract, an edge is created between genes."""


import multiprocessing as mp
import pickle
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm  # type: ignore

from utils import gencode_genes


def combine_synonyms(
    synonyms: Dict[str, Set[str]], genes: Set[str]
) -> Dict[str, Set[str]]:
    """Combine synonyms and genes into a single set. For each synonym and gene,
    add itself to its own set to make a comprehensive list.
    """
    for gene, syn_set in synonyms.items():
        syn_set.add(gene)

    # add genes not in the synonym set
    for gene in genes:
        if gene not in synonyms:
            synonyms[gene] = {gene}

    return synonyms


def create_alias_to_gene_mapping(gene_aliases: Dict[str, Set[str]]) -> Dict[str, str]:
    """Create a mapping from each alias to its gene symbol."""
    alias_to_gene = {}
    for gene, aliases in gene_aliases.items():
        for alias in aliases:
            alias_to_gene[alias] = gene
    return alias_to_gene


def gene_mentions_per_abstract(
    abstracts: pd.DataFrame, alias_to_gene: Dict[str, str]
) -> List[Set[str]]:
    """Loop through tokenized abstracts and create a sublist of mentioned genes
    within the abstract. Gene mentions are based on tokens that either map to
    the gene symbol or synonyms from the HGNC complete set.
    """
    alias = set(alias_to_gene.keys())

    def process_abstract(abstract_sentences: List[List[str]]) -> Set[str]:
        """Vectorized matching"""
        tokens = [
            token for sentence in abstract_sentences for token in sentence
        ]  # flatten the list of sentences
        matches = set(tokens) & alias
        gene_mentions = {alias_to_gene[token] for token in matches}
        return gene_mentions if len(gene_mentions) > 2 else set()

    gene_relations = abstracts["processed_abstracts_w2v"].apply(process_abstract)
    return [genes for genes in gene_relations if genes]


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


def process_abstract_file(args: Tuple[int, Dict[str, str]]) -> Set[Tuple[str, str]]:
    """Process a single abstract file and return gene edges."""
    num, alias_to_gene = args
    abstracts = pd.read_pickle(
        f"/ocean/projects/bio210019p/stevesho/genomic_nlp/data/processed_abstracts_w2v_chunk_{num}.pkl"
    )
    gene_relationships = gene_mentions_per_abstract(abstracts, alias_to_gene)
    return collect_gene_edges(gene_relationships)


def extract_gene_edges_from_abstracts(
    index_end: int,
    alias_to_gene: Dict[str, str],
) -> Set[Tuple[str, str]]:
    """Extract gene edges from abstracts using multiprocessing."""
    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_abstract_file,
                    [(num, alias_to_gene) for num in range(index_end)],
                ),
                total=index_end,
            )
        )

    gene_edges: Set[Tuple[str, str]] = set()
    for result in results:
        gene_edges.update(result)
    return gene_edges


def main() -> None:
    """Main function"""
    genes = gencode_genes(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/reference_files/gencode.v45.basic.annotation.gtf"
    )

    with open(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl",
        "rb",
    ) as file:
        hgnc_synonyms = pickle.load(file)

    combined_genes = combine_synonyms(hgnc_synonyms, genes)
    alias_to_gene = create_alias_to_gene_mapping(combined_genes)

    gene_edges = extract_gene_edges_from_abstracts(20, alias_to_gene=alias_to_gene)

    # write to text file
    write_gene_edges_to_file(
        gene_edges,
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi/text_extracted_gene_edges_syns.tsv",
    )


if __name__ == "__main__":
    main()


# unique_genes = {gene for edge in gene_edges for gene in edge}
# len(unique_genes)  # 20785

# # load gene uniq gene edges to check for overlap
# gene_edges_uniq = set()
# with open("text_extracted_gene_edges.tsv", "r") as file:
#     reader = csv.reader(file, delimiter="\t")
#     for line in reader:
#         gene_edges_uniq.add(line[0])
#         gene_edges_uniq.add(line[1])

# # load gene edges
# with open("text_extracted_gene_edges.tsv", "r") as file:
#     reader = csv.reader(file, delimiter="\t")
#     gene_edges = {(line[0], line[1]) for line in reader}

# ppi = _load_ppi("HI-union.tsv")
# unique_proteins = {gene for edge in ppi for gene in edge}
# with open("uniq_proteins.txt", "w") as file:
#     for protein in unique_proteins:
#         file.write(f"{protein}\n")

# reference = _load_reference("biomart_ppi.txt")
# mapped_ppi = _map_proteins_to_gene_symbols(ppi, reference)

# unique_mapped_proteins = {gene for edge in mapped_ppi for gene in edge}
# huri_only = mapped_ppi - gene_edges
