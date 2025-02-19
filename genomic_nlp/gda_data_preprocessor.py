# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Preprocess interaction data (gene-disease associations) for GNN link
prediction model.

1. Loads gene-disease pairs.
2. Splits the positive edges into train/test/validation sets.
3. Generates negative samples (only gene-disease pairs) for split.
4. Maps nodes to indices.
5. Converts edges into torch tensors.
"""


import csv
import random
from typing import Dict, Set, Tuple

import torch
from torch_geometric.data import Data  # type: ignore

from genomic_nlp.interaction_data_preprocessor import _load_pickle


class GDADataPreprocessor:
    """Preprocess data for GNN link prediction model."""

    def __init__(
        self,
        embedding_file: str,
        text_edges_file: str,
    ) -> None:
        """Initialize a GNNDataPreprocessor object. Load data and
        embeddings.
        """
        self.gene_embeddings = _load_pickle(embedding_file)

        text_edges = [
            tuple(row) for row in csv.reader(open(text_edges_file), delimiter="\t")
        ]

        self.edges = set(text_edges)
        print("Data and embeddings loaded!")

    def preprocess_data(self) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """Preprocess data for GNN link prediction model."""
        # partition data into train/val/test 70/15/15
        all_pos_edges = list(self.edges)
        random.shuffle(all_pos_edges)
        total = len(all_pos_edges)
        train_count = int(total * 0.70)
        val_count = int(total * 0.15)

        train_edges = set(all_pos_edges[:train_count])
        val_edges = set(all_pos_edges[train_count : train_count + val_count])
        test_edges = set(all_pos_edges[train_count + val_count :])

        # 1:1 negative sampling
        total_neg_samples = total
        all_neg_edges = list(
            self._negative_sampling(
                unique_genes={gene for gene, _ in self.edges},
                unique_diseases={disease for _, disease in self.edges},
                positive_edges=self.edges,
                num_samples=total_neg_samples,
            )
        )
        random.shuffle(all_neg_edges)
        train_neg = set(all_neg_edges[:train_count])
        val_neg = set(all_neg_edges[train_count : train_count + val_count])
        test_neg = set(all_neg_edges[train_count + val_count :])

        # map nodes to indices and get node features
        node_mapping = self._map_nodes_to_indices(self.edges)
        x = self._get_node_features(node_mapping)

        # compute gda node indices
        gene_node_indices = sorted({node_mapping[gene] for gene, _ in self.edges})
        disease_node_indices = sorted(
            {node_mapping[disease] for _, disease in self.edges}
        )

        # convert edges to PyG tensors
        train_pos_edge_index = self._get_edge_index_tensor(
            edges=train_edges, node_mapping=node_mapping
        )
        val_pos_edge_index = self._get_edge_index_tensor(
            edges=val_edges, node_mapping=node_mapping
        )
        test_pos_edge_index = self._get_edge_index_tensor(
            edges=test_edges, node_mapping=node_mapping
        )

        train_neg_edge_index = self._get_edge_index_tensor(
            edges=train_neg, node_mapping=node_mapping
        )
        val_neg_edge_index = self._get_edge_index_tensor(
            edges=val_neg, node_mapping=node_mapping
        )
        test_neg_edge_index = self._get_edge_index_tensor(
            edges=test_neg, node_mapping=node_mapping
        )

        # build PyG data object
        data = Data(
            x=x,
            edge_index=train_pos_edge_index,
            train_pos_edge_index=train_pos_edge_index,
            train_neg_edge_index=train_neg_edge_index,
            val_pos_edge_index=val_pos_edge_index,
            val_neg_edge_index=val_neg_edge_index,
        )

        # add gene and disease node indices
        data.gene_nodes = torch.tensor(gene_node_indices, dtype=torch.long)
        data.disease_nodes = torch.tensor(disease_node_indices, dtype=torch.long)

        return data, test_pos_edge_index, test_neg_edge_index

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

    def _negative_sampling(
        self,
        unique_genes: Set[str],
        unique_diseases: Set[str],
        positive_edges: Set[Tuple[str, str]],
        num_samples: int,
    ) -> Set[Tuple[str, str]]:
        """Generate negative gene-disease pairs for training. Samples one gene
        and one disease at random and accepts the pair if it is not in the
        positive set.
        """
        negative_samples: Set[Tuple[str, str]] = set()
        genes = list(unique_genes)
        diseases = list(unique_diseases)

        while len(negative_samples) < num_samples:
            gene = random.choice(genes)
            disease = random.choice(diseases)
            edge = (gene, disease)
            if edge not in positive_edges and edge not in negative_samples:
                negative_samples.add(edge)
        return negative_samples

    def _get_node_features(self, node_mapping: Dict[str, int]) -> torch.Tensor:
        """Fill out node feature matrix via retrieving embeddings."""
        return torch.tensor(
            [self.gene_embeddings[gene] for gene in node_mapping], dtype=torch.float
        )

    @staticmethod
    def _map_nodes_to_indices(edges: Set[Tuple[str, str]]) -> Dict[str, int]:
        """Map all nodes from the provided edges to unique indices."""
        unique_nodes = {node for edge in edges for node in edge}
        return {node: idx for idx, node in enumerate(unique_nodes)}

    @staticmethod
    def _get_edge_index_tensor(
        edges: Set[Tuple[str, str]],
        node_mapping: Dict[str, int],
    ) -> torch.Tensor:
        """Convert (str) edges to tensor of indices."""
        return torch.tensor(
            [[node_mapping[e1], node_mapping[e2]] for e1, e2 in edges], dtype=torch.long
        ).t()
