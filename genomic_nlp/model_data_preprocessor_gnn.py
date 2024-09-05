# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to preprocess interaction data for GNN models."""


import argparse
import csv
from pathlib import Path
import random
from typing import Dict, Set, Tuple

import torch
from torch_geometric.data import Data  # type: ignore

from model_data_preprocessor import _load_pickle
from model_data_preprocessor import casefold_pairs


class GNNDataPreprocessor:
    """Preprocess data for GNN link prediction model. Load edge list, filter
    edges for those with embeddings.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Initialize a GNNDataPreprocessor object. Load data and
        embeddings.
        """
        self.data_dir = data_dir
        self.gene_embeddings = _load_pickle(args.embeddings)
        self.pos_pairs_with_source = _load_pickle(args.positive_pairs_file)
        self.pair_to_source = {
            (pair[0].lower(), pair[1].lower()): pair[2]
            for pair in self.pos_pairs_with_source
        }
        self.positive_pairs = casefold_pairs(self.pos_pairs_with_source)
        text_edges = casefold_pairs(
            [
                tuple(row)
                for row in csv.reader(open(args.text_edges_file), delimiter="\t")
            ]
        )
        self.edges = set(text_edges)
        print("Data and embeddings loaded!")

        # hardcoded, cause to be honest, I'm lazy right now
        edge_dir = Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data")
        go_edges = _load_pickle(edge_dir / "go_graph.pkl")
        string_edges = _load_pickle(edge_dir / "string_graph.pkl")
        self.avoid_edges = go_edges.union(string_edges)

    def filter_pairs_for_prior_knowledge(
        self,
        positive_pairs: Set[Tuple[str, str]],
    ) -> Set[Tuple[str, str]]:
        """Filter the gene pairs into two sets: those with prior knowledge, and
        those without. Gene pairs with prior knowledge are those that exist in the
        text_edges set, which contains edges extracted from the literature.
        """
        pos_train = []
        pos_test = []

        print(f"Total positive pairs: {len(positive_pairs)}")

        for pair in positive_pairs:
            if pair in self.edges or (pair[1], pair[0]) in self.edges:
                pos_train.append(pair)
            else:
                pos_test.append(pair)

        print(f"Positive pairs with prior knowledge: {len(pos_train)}")
        print(f"Positive pairs without prior knowledge: {len(pos_test)}")
        return set(pos_test)

    def preprocess_data(self) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """Preprocess data for GNN link prediction model."""
        # filter edges to keep only those with embeddings
        filtered_edges = self._filter_edges_for_embeddings()
        unique_genes = self._uniq_filtered_genes(filtered_edges)

        # get test edges
        filtered_test_edges = self._filter_test_edges_for_embeddings(unique_genes)

        # filter edges for prior knowledge
        final_test_edges = self.filter_pairs_for_prior_knowledge(
            positive_pairs=filtered_test_edges
        )

        # generate negative samples
        negative_samples = self._negative_sampling(
            unique_genes=unique_genes,
            positive_edges=filtered_edges,
            num_samples=len(filtered_edges) + len(final_test_edges),
        )

        # assign negative samples to training and test sets
        negative_train_samples = set(
            random.sample(negative_samples, len(filtered_edges))
        )
        negative_test_samples = negative_samples - negative_train_samples

        # map genes to indices
        node_mapping = self._map_genes_to_indices(filtered_edges)

        # get node features
        x = self._get_node_features(node_mapping)

        # convert (str) edges to tensor of indices
        edge_index = self._get_edge_index_tensor(
            edges=filtered_edges, node_mapping=node_mapping
        )
        neg_edge_index = self._get_edge_index_tensor(
            edges=negative_train_samples, node_mapping=node_mapping
        )
        positive_test_edges = self._get_edge_index_tensor(
            edges=final_test_edges, node_mapping=node_mapping
        )
        negative_test_edges = self._get_edge_index_tensor(
            edges=negative_test_samples, node_mapping=node_mapping
        )

        # split into train and validation sets
        train_pos_edge_index, val_pos_edge_index = self._split_edge_index(
            edge_index=edge_index, percent=0.8
        )
        train_neg_edge_index, val_neg_edge_index = self._split_edge_index(
            edge_index=neg_edge_index, percent=0.8
        )

        # add an option to train on full dataset
        all_pos_edge_index = torch.cat(
            [train_pos_edge_index, val_pos_edge_index], dim=1
        )
        all_neg_edge_index = torch.cat(
            [train_neg_edge_index, val_neg_edge_index], dim=1
        )

        # create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            all_pos_edge_index=all_pos_edge_index,
            all_neg_edge_index=all_neg_edge_index,
            train_pos_edge_index=train_pos_edge_index,
            train_neg_edge_index=train_neg_edge_index,
            val_pos_edge_index=val_pos_edge_index,
            val_neg_edge_index=val_neg_edge_index,
        )

        return data, positive_test_edges, negative_test_edges

    @staticmethod
    def _split_edge_index(
        edge_index: torch.Tensor, percent: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split edge index into two tensors by ratio."""
        num_edges = edge_index.shape[1]
        num_train = int(num_edges * percent)

        # shuffle indices
        indices = torch.randperm(num_edges)

        # split indices
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        return edge_index[:, train_indices], edge_index[:, val_indices]

    def _filter_edges_for_embeddings(self) -> Set[Tuple[str, str]]:
        """Filter edges to only keep those with embeddings."""
        return {
            (edge[0], edge[1])
            for edge in self.edges
            if edge[0] in self.gene_embeddings and edge[1] in self.gene_embeddings
        }

    def _uniq_filtered_genes(self, edges: Set[Tuple[str, str]]) -> Set[str]:
        """Get unique genes from filtered edges."""
        return {gene for edge in edges for gene in edge}

    def _negative_sampling(
        self,
        unique_genes: Set[str],
        positive_edges: Set[Tuple[str, str]],
        num_samples: int,
    ) -> Set[Tuple[str, str]]:
        """Generate negative samples for training. To avoid sampling edges that
        seem negative but are actually positive, we avoid creating edges that
        overlap with STRING, GO, or our test set.
        """
        negative_samples: Set[Tuple[str, str]] = set()
        genes = list(unique_genes)

        while len(negative_samples) < num_samples:
            gene1, gene2 = random.sample(genes, 2)
            edge = (
                (gene1, gene2) if gene1 < gene2 else (gene2, gene1)
            )  # avoid duplicates

            if (
                edge not in positive_edges
                and edge not in self.avoid_edges
                and edge not in negative_samples
            ):
                negative_samples.add(edge)

        return negative_samples

    def _get_node_features(self, node_mapping: Dict[str, int]) -> torch.Tensor:
        """Fill out node feature matrix via retrieving embeddings."""
        return torch.tensor(
            [self.gene_embeddings[gene] for gene in node_mapping], dtype=torch.float
        )

    def _filter_test_edges_for_embeddings(
        self, unique_genes: Set[str]
    ) -> Set[Tuple[str, str]]:
        """Filter edges to only keep those with embeddings."""
        return {
            (edge[0], edge[1])
            for edge in self.positive_pairs
            if edge[0] in unique_genes and edge[1] in unique_genes
        }

    @staticmethod
    def _map_genes_to_indices(edges: Set[Tuple[str, str]]) -> Dict[str, int]:
        """Map genes to indices."""
        unique_genes = {gene for edge in edges for gene in edge}
        return {gene: idx for idx, gene in enumerate(unique_genes)}

    @staticmethod
    def _get_edge_index_tensor(
        edges: Set[Tuple[str, str]],
        node_mapping: Dict[str, int],
    ) -> torch.Tensor:
        """Convert (str) edges to tensor of indices."""
        return torch.tensor(
            [[node_mapping[e1], node_mapping[e2]] for e1, e2 in edges], dtype=torch.long
        ).t()
