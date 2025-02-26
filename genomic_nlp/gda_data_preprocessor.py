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
import pickle
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data  # type: ignore
from tqdm import tqdm  # type: ignore


class GDADataPreprocessor:
    """Preprocess data for GNN link prediction model."""

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        text_edges_file: str,
        disease_synonyms_file: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/disease_synonyms.pkl",
        gene_synonyms_file: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl",
    ) -> None:
        """Initialize a GNNDataPreprocessor object. Load data and
        embeddings.
        """
        # load gene and diseases names
        with open(disease_synonyms_file, "rb") as f:
            disease_names = pickle.load(f)

        with open(gene_synonyms_file, "rb") as f:
            gene_names = pickle.load(f)

        # load gene-disease pairs
        with open(text_edges_file, "r") as file:
            text_edges = [tuple(row) for row in csv.reader(file, delimiter="\t")]

        # load embeddings and get edges
        self.embeddings = embeddings

        # filter out edges with missing embeddings
        self.edges = {
            (gene, disease)
            for gene, disease in text_edges
            if gene in self.embeddings and disease in self.embeddings
        }

        # only keep genes and diseases with embeddings
        self.available_diseases = {
            disease for disease in disease_names if disease in self.embeddings
        }
        self.available_genes = {gene for gene in gene_names if gene in self.embeddings}
        print("Data and embeddings loaded!")
        print(f"Total filtered edges: {len(self.edges)}")
        print(f"Available genes: {len(self.available_genes)}")
        print(f"Available diseases: {len(self.available_diseases)}")

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
                positive_edges=self.edges,
                num_samples=total_neg_samples,
            )
        )
        random.shuffle(all_neg_edges)
        neg_total = len(all_neg_edges)
        neg_train_count = int(neg_total * 0.70)
        neg_val_count = int(neg_total * 0.15)

        train_neg = set(all_neg_edges[:neg_train_count])
        val_neg = set(all_neg_edges[neg_train_count : neg_train_count + neg_val_count])
        test_neg = set(all_neg_edges[neg_train_count + neg_val_count :])

        print("train_pos has", len(train_edges), "edges")
        print("val_pos has", len(val_edges), "edges")
        print("test_pos has", len(test_edges), "edges")
        print("train_neg has", len(train_neg), "edges")
        print("val_neg has", len(val_neg), "edges")
        print("test_neg has", len(test_neg), "edges")

        # map nodes to indices and get node features
        all_current_edges = (
            train_edges.union(val_edges)
            .union(test_edges)
            .union(train_neg)
            .union(val_neg)
            .union(test_neg)
        )
        node_mapping = self._map_nodes_to_indices(all_current_edges)
        x = self._get_node_features(node_mapping)

        # inverse node mapping for later
        inv_node_mapping = [""] * len(node_mapping)
        for node_str, idx in node_mapping.items():
            inv_node_mapping[idx] = node_str

        # compute gda node indices
        gene_node_indices = sorted(
            {
                node_mapping[gene]
                for gene, _ in self.edges
                if gene in self.available_genes
            }
        )
        disease_node_indices = sorted(
            {
                node_mapping[disease]
                for _, disease in self.edges
                if disease in self.available_diseases
            }
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
        data.inv_node_mapping = inv_node_mapping

        # sanity check
        print("Total edges read from file:", len(all_pos_edges))  # ~50M
        print("Unique edges after set:", len(self.edges))
        print("Number of unique nodes after map:", len(node_mapping))
        print("node feature matrix shape:", x.shape)
        print("Gene nodes:", data.gene_nodes.shape)
        print("Disease nodes:", data.disease_nodes.shape)

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
        positive_edges: Set[Tuple[str, str]],
        num_samples: int,
    ) -> Set[Tuple[str, str]]:
        """Generate negative gene-disease pairs for training. Samples one gene
        and one disease at random and accepts the pair if it is not in the
        positive set.
        """
        genes = list(self.available_genes)
        diseases = list(self.available_diseases)
        num_genes = len(genes)
        num_diseases = len(diseases)

        # create dictionaries to map genes/diseases to indices
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        disease_to_idx = {disease: idx for idx, disease in enumerate(diseases)}

        # precompute a sorted hash for each positive edge
        pos_hashes = set()
        for gene_str, disease_str in positive_edges:
            if gene_str in gene_to_idx and disease_str in disease_to_idx:
                g_idx = gene_to_idx[gene_str]
                d_idx = disease_to_idx[disease_str]
                pos_hashes.add(g_idx * num_diseases + d_idx)

        # the maximum number of distinct negatives possible
        max_possible_neg = (num_genes * num_diseases) - len(pos_hashes)
        if num_samples > max_possible_neg:
            print(
                f"Warning: Requested {num_samples} negatives but only {max_possible_neg} "
                "distinct negatives exist. Clamping to the maximum possible."
            )
            num_samples = max_possible_neg

        negative_samples: List[Tuple[str, str]] = []
        used_hashes = set(pos_hashes)
        batch_size = 50000

        with tqdm(total=num_samples, desc="Generating negatives") as pbar:
            while len(negative_samples) < num_samples:
                # generate a batch of random indices
                candidate_gene_indices = np.random.randint(
                    0, num_genes, size=batch_size
                )
                candidate_disease_indices = np.random.randint(
                    0, num_diseases, size=batch_size
                )

                # check not in positive and also not used
                for g_idx, d_idx in zip(
                    candidate_gene_indices, candidate_disease_indices
                ):
                    neg_hash = g_idx * num_diseases + d_idx
                    if neg_hash not in used_hashes:
                        # add if frensh
                        used_hashes.add(neg_hash)
                        negative_samples.append((genes[g_idx], diseases[d_idx]))
                        pbar.update(1)

                        # stop if we have enough
                        if len(negative_samples) >= num_samples:
                            break

        return set(negative_samples)

    def _get_node_features(self, node_mapping: Dict[str, int]) -> torch.Tensor:
        """Fill out node feature matrix via retrieving embeddings."""
        print("DEBUG: gene_embeddings has size", len(self.embeddings))
        print("DEBUG: node_mapping has size", len(node_mapping))

        # check for missing embeddings
        missing = sum(node_str not in self.embeddings for node_str in node_mapping)
        print("DEBUG: # missing embeddings:", missing)

        features = np.array([self.embeddings[gene] for gene in node_mapping])
        return torch.from_numpy(features).float()

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
        edge_index = np.array(
            [[node_mapping[e1], node_mapping[e2]] for e1, e2 in edges]
        )
        return torch.from_numpy(edge_index).long().t()
