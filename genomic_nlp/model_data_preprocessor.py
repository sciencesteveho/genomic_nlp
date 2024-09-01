# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to preprocess interaction data for models."""


import argparse
from collections import defaultdict
import csv
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import train_test_split_edges  # type: ignore


class InteractionDataPreprocessor:
    """Preprocess data for baseline models. Load gene pairs, embeddings, filter,
    and statify the train and test sets by source.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Instantiate a BaselineDataPreprocessor object. Load data and
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
        self.negative_pairs = casefold_pairs(_load_pickle(args.negative_pairs_file))
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

        # create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
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


def _load_pickle(pickle_file: Any) -> Any:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def casefold_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Casefold gene pairs."""
    return [(pair[0].casefold(), pair[1].casefold()) for pair in pairs]


class OncogenicDataPreprocessor:
    """Preprocess data for oncogenicity prediction models."""

    def __init__(
        self,
        args: argparse.Namespace,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Instantiate an OncogenicDataPreprocessor object. Load data and embeddings."""
        self.data_dir = Path(data_dir)
        self.gene_embeddings = _load_pickle(args.embeddings)
        self.cancer_genes = self.get_positive_test_set()
        print("Embeddings loaded.")

    def get_positive_test_set(self) -> Set[str]:
        """Get the positive test set of cancer related genes."""
        cosmic_genes = self._load_cosmic()
        ncg_genes = self._load_ncg()
        return cosmic_genes.union(ncg_genes)

    def _load_cosmic(self) -> Set[str]:
        """Load COSMIC gene_symbols."""
        data = pd.read_csv(
            self.data_dir / "Cosmic_CancerGeneCensus_v100_GRCh38.tsv",
            delimiter="\t",
            header=[0],
        )
        return set(data["GENE_SYMBOL"].str.lower())

    def _load_ncg(self) -> Set[str]:
        """Load NCG gene_symbols."""
        data = pd.read_csv(
            self.data_dir / "NCG_cancerdrivers_annotation_supporting_evidence.tsv",
            delimiter="\t",
            header=[0],
        )
        return set(data["symbol"].str.lower())

    def format_data_and_targets(
        self,
        cancer_genes: Set[str],
        negative_samples: Set[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature data and target labels for cancer related genes."""
        data = []
        targets = []
        for pair in positive_pairs + negative_pairs:
            gene1, gene2 = pair
            vec1 = self.gene_embeddings[gene1]
            vec2 = self.gene_embeddings[gene2]
            data.append(np.concatenate([vec1, vec2]))
            targets.append(1 if pair in positive_pairs else 0)
        return np.array(data), np.array(targets)

    def preprocess_data(self) -> None:
        """Preprocess data for oncogenicity prediction models."""
        # filter cancer genes for those with embeddings
        cancer_genes = {
            gene for gene in self.cancer_genes if gene in self.gene_embeddings
        }

        # create negative samples, same size as cancer genes
        # any gene not in cancer_genes is considered a negative sample
        total_samples = len(cancer_genes)
        negative_samples = [
            gene for gene in self.gene_embeddings if gene not in cancer_genes
        ]
        matched_negative_samples = {random.sample(negative_samples, total_samples)}
