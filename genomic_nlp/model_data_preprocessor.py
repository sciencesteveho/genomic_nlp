#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to preprocess interaction data for models."""


import argparse
from collections import defaultdict
import csv
import pickle
import random
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch


class InteractionDataPreprocessor:
    """Preprocess data for baseline models. Load gene pairs, embeddings, filter,
    and statify the train and test sets by source.
    """

    def __init__(
        self,
        args,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Instantiate a BaselineDataPreprocessor object. Load data and
        embeddings.
        """
        self.data_dir = data_dir
        self.gene_embeddings = _unpickle_dict(args.embeddings)
        self.pos_pairs_with_source = _unpickle_dict(args.positive_pairs_file)
        self.pair_to_source = {
            (pair[0].lower(), pair[1].lower()): pair[2]
            for pair in self.pos_pairs_with_source
        }
        self.positive_pairs = casefold_pairs(self.pos_pairs_with_source)
        self.negative_pairs = casefold_pairs(_unpickle_dict(args.negative_pairs_file))
        text_edges = casefold_pairs(
            [
                tuple(row)
                for row in csv.reader(open(args.text_edges_file), delimiter="\t")
            ]
        )
        self.text_edges = set(text_edges)
        print("Data and embeddings loaded!")

    def load_and_preprocess_data(self) -> Tuple[
        List[Tuple[str, str]],
        List[Tuple[str, str]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[str, str]],
        List[Tuple[str, str]],
    ]:
        """Load all data for training!"""
        print("Filtering pairs for those with embeddings")
        self.positive_pairs, self.negative_pairs = self.filter_pairs_for_embeddings()

        print("Splitting pairs into train and test sets")
        pos_train, pos_test = self.filter_pairs_for_prior_knowledge()

        print("Getting negative samples.")
        neg_train, neg_test = self.split_negative_pairs(len(pos_train), len(pos_test))

        print("Preparing training data and targets.")
        train_features, train_targets = self.prepare_data_and_targets(
            pos_train, neg_train
        )
        test_features, test_targets = self.prepare_data_and_targets(pos_test, neg_test)

        return (
            self.positive_pairs,
            self.negative_pairs,
            train_features,
            train_targets,
            test_features,
            test_targets,
            pos_test,
            neg_test,
        )

    def prepare_stratified_test_data(
        self,
        pos_test: List[Tuple[str, str]],
        test_features: np.ndarray,
        neg_test: List[Tuple[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Strafity the test data by source."""
        stratified_test_data: Dict[str, Any] = defaultdict(
            lambda: {"features": [], "targets": []}
        )
        for i, pair in enumerate(pos_test):
            sources = self.pair_to_source.get(pair, ("unknown",))
            for source in sources:
                stratified_test_data[source]["features"].append(test_features[i])
                stratified_test_data[source]["targets"].append(1)

        for source in stratified_test_data:
            n_pos = len(stratified_test_data[source]["features"])
            neg_indices = np.random.choice(len(neg_test), n_pos, replace=False)
            stratified_test_data[source]["features"].extend(
                test_features[len(pos_test) + i] for i in neg_indices
            )
            stratified_test_data[source]["targets"].extend([0] * n_pos)

        for source in stratified_test_data:
            stratified_test_data[source]["features"] = np.array(
                stratified_test_data[source]["features"]
            )
            stratified_test_data[source]["targets"] = np.array(
                stratified_test_data[source]["targets"]
            )
        return dict(stratified_test_data)

    def filter_pairs_for_embeddings(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Filter pairs to only include those with embeddings."""
        filtered_positive_pairs = [
            pair
            for pair in self.positive_pairs
            if all(gene in self.gene_embeddings for gene in pair)
        ]
        filtered_negative_pairs = [
            pair
            for pair in self.negative_pairs
            if all(gene in self.gene_embeddings for gene in pair)
        ]
        return self.balance_filtered_pairs(
            filtered_positive_pairs, filtered_negative_pairs
        )

    def filter_pairs_for_prior_knowledge(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Filter the gene pairs into two sets: those with prior knowledge, and
        those without. Gene pairs with prior knowledge are those that exist in the
        text_edges set, which contains edges extracted from the literature.
        """
        pos_train = []
        pos_test = []

        print(f"Total positive pairs: {len(self.positive_pairs)}")

        for pair in self.positive_pairs:
            if pair in self.text_edges or (pair[1], pair[0]) in self.text_edges:
                pos_train.append(pair)
            else:
                pos_test.append(pair)

        print(f"Positive pairs with prior knowledge: {len(pos_train)}")
        print(f"Positive pairs without prior knowledge: {len(pos_test)}")
        return pos_train, pos_test

    def split_negative_pairs(
        self, n_train: int, n_test: int
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Split negative pairs into train and test sets without overlap."""
        if len(self.negative_pairs) < n_train + n_test:
            raise ValueError(
                "Not enough negative pairs to split into train and test sets"
            )
        shuffled_pairs = random.sample(self.negative_pairs, len(self.negative_pairs))
        return shuffled_pairs[:n_train], shuffled_pairs[n_train : n_train + n_test]

    def prepare_data_and_targets(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature data and target labels from gene pairs."""
        data = []
        targets = []
        for pair in positive_pairs + negative_pairs:
            gene1, gene2 = pair
            vec1 = self.gene_embeddings[gene1]
            vec2 = self.gene_embeddings[gene2]
            data.append(np.concatenate([vec1, vec2]))
            targets.append(1 if pair in positive_pairs else 0)
        return np.array(data), np.array(targets)

    @staticmethod
    def balance_filtered_pairs(
        filtered_positive_pairs: List[Tuple[str, str]],
        filtered_negative_pairs: List[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Adjust pairs to have the same size."""
        if len(filtered_positive_pairs) > len(filtered_negative_pairs):
            filtered_positive_pairs = random.sample(
                filtered_positive_pairs, len(filtered_negative_pairs)
            )
        elif len(filtered_negative_pairs) > len(filtered_positive_pairs):
            filtered_negative_pairs = random.sample(
                filtered_negative_pairs, len(filtered_positive_pairs)
            )
        return filtered_positive_pairs, filtered_negative_pairs


class GNNDataPreprocessor:
    """Preprocess data for GNN link prediction model. Load edge list, filter
    edges for those with embeddings.

    Create negative samples, but ensure negative samples don't exist in STRING or GO.
    Initialize a PyG Data object with node features and edge index.
    Prepare test edges for evaluation.
    """

    def __init__(
        self,
        args,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Initialize a GNNDataPreprocessor object. Load data and
        embeddings.
        """
        self.data_dir = data_dir
        self.gene_embeddings = _unpickle_dict(args.embeddings)
        self.pos_pairs_with_source = _unpickle_dict(args.positive_pairs_file)
        self.pair_to_source = {
            (pair[0].lower(), pair[1].lower()): pair[2]
            for pair in self.pos_pairs_with_source
        }
        self.positive_pairs = casefold_pairs(self.pos_pairs_with_source)
        self.negative_pairs = casefold_pairs(_unpickle_dict(args.negative_pairs_file))
        text_edges = casefold_pairs(
            [
                tuple(row)
                for row in csv.reader(open(args.text_edges_file), delimiter="\t")
            ]
        )
        self.edges = set(text_edges)
        print("Data and embeddings loaded!")

    def _filter_edges_for_embeddings(self) -> Set[Tuple[str, str]]:
        """Filter edges to only keep those with embeddings."""
        return {
            (edge[0], edge[1])
            for edge in self.edges
            if edge[0] in self.gene_embeddings and edge[1] in self.gene_embeddings
        }

    def preprocess_data(self) -> Data:
        """Preprocess data for GNN link prediction model."""
        # Filter edges to keep only those with embeddings
        filtered_edges = self._filter_edges_for_embeddings()

        # Create node mapping and feature matrix
        unique_genes = set([gene for edge in filtered_edges for gene in edge])
        node_mapping = {gene: i for i, gene in enumerate(unique_genes)}
        x = torch.tensor(
            [self.gene_embeddings[gene] for gene in node_mapping], dtype=torch.float
        )

        # Convert edge list to tensor
        edge_index = torch.tensor(
            [[node_mapping[e1], node_mapping[e2]] for e1, e2 in filtered_edges],
            dtype=torch.long,
        ).t()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        # Add negative samples
        data.neg_edge_index = self.negative_sampling(edge_index, num_nodes=x.size(0))

        return data

    def prepare_test_edges(self, data: Data) -> torch.Tensor:
        node_mapping = {gene: i for i, gene in enumerate(data.x)}
        test_edge_index = torch.tensor(
            [
                [node_mapping[e1], node_mapping[e2]]
                for e1, e2 in self.test_edges
                if e1 in node_mapping and e2 in node_mapping
            ],
            dtype=torch.long,
        ).t()
        return test_edge_index

    @staticmethod
    def negative_sampling(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Simple negative sampling implementation
        num_neg_samples = edge_index.size(1)
        neg_edge_index = torch.randint(
            0, num_nodes, (2, num_neg_samples), dtype=torch.long
        )

        # Remove self-loops and duplicates
        mask = neg_edge_index[0] != neg_edge_index[1]
        neg_edge_index = neg_edge_index[:, mask]
        neg_edge_index = torch.unique(neg_edge_index, dim=1)

        # Ensure no overlap with positive edges
        pos_edge_set = set(map(tuple, edge_index.t().tolist()))
        neg_edge_index = neg_edge_index[
            :, [tuple(e) not in pos_edge_set for e in neg_edge_index.t().tolist()]
        ]

        return neg_edge_index


def _unpickle_dict(pickle_file: str) -> Any:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def casefold_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Casefold gene pairs."""
    return [(pair[0].casefold(), pair[1].casefold()) for pair in pairs]
