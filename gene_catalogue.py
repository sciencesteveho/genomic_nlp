#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Build gene catalogues from edge files and filter catalogues to only contain
genes relevant to our experimentally derived positive training examples."""


from enum import Enum
from pathlib import Path
import pickle
from typing import Set, Tuple

import networkx as nx  # type: ignore


class EdgeCatalogue(Enum):
    """Enum class to handle different edge types."""

    EXPERIMENTAL = "experimentally_derived_edges.pkl"
    ALL_POSITIVE = "all_positive_edges.pkl"
    NEGATIVE = "negative_edges.pkl"
    TEXT_EXTRACTION = "text_extracted_edges.tsv"


def graph_from_edgelist(edge_file: Path) -> nx.Graph:
    """Create a graph from an edge list."""
    return nx.read_edgelist(edge_file, delimiter="\t", create_using=nx.Graph())


def prune_node_without_gene_connection(graph: nx.Graph, genes: Set[str]) -> nx.Graph:
    """Remove nodes that are not connected to any gene."""
    connected_to_gene = set()
    for component in nx.connected_components(graph):
        if any(gene in component for gene in genes):
            connected_to_gene.update(component)

    nodes_to_remove = set(graph) - connected_to_gene
    graph.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes not connected to any gene.")
    return graph


def genes_from_catalogue(catalogue: str, data_path: Path) -> Set[str]:
    """Get a uniq set of genes from an edge catalogue."""
    try:
        edge_file = EdgeCatalogue(catalogue)
    except ValueError as e:
        valid_catalogues = ", ".join([item.value for item in EdgeCatalogue])
        raise ValueError(
            f"Invalid edge catalogue. Choose from {valid_catalogues}."
        ) from e

    if edge_file == EdgeCatalogue.TEXT_EXTRACTION:
        with open(data_path / edge_file.value, "r") as file:
            edges = {tuple(line.strip().split("\t")) for line in file}
    else:
        with open(data_path / edge_file.value, "rb") as file:
            edges = pickle.load(file)

    return uniq_from_tuple(edges)


def uniq_from_tuple(edges: Set[Tuple[str, ...]]) -> Set[str]:
    """Return uniq genes from a tuple of edges."""
    return {item for tuple_pair in edges for item in tuple_pair}


def main() -> None:
    """Main function to generate relevant gene catalogue."""
    # load required edges
    data_path = Path("/ocean/projects/bio210019p/stevesho/nlp/training_data")
    experimental_edges = genes_from_catalogue(
        catalogue=EdgeCatalogue.EXPERIMENTAL.value, data_path=data_path
    )

    text_extracted_graph = graph_from_edgelist(
        data_path / EdgeCatalogue.TEXT_EXTRACTION.value
    )
    filtered_text_extracted_graph = prune_node_without_gene_connection(
        graph=text_extracted_graph, genes=experimental_edges
    )

    # make gene catalogue and save
    gene_catalogue = experimental_edges | {filtered_text_extracted_graph.nodes()}
    with open(data_path / "gene_catalogue.pkl", "wb") as file:
        pickle.dump(gene_catalogue, file)


if __name__ == "__main__":
    main()
