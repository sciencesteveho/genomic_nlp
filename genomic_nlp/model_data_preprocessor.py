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
        """Split negative pairs into train and test sets without overlap, based
        on the length of the positive pairs.
        """
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


def _load_pickle(pickle_file: Any) -> Any:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def casefold_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Casefold gene pairs."""
    return [(pair[0].casefold(), pair[1].casefold()) for pair in pairs]


class CancerGeneDataPreprocessor:
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

        # hardcoded, to fix later
        self.resource_dir = Path(
            "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/cancer"
        )

    def get_positive_test_set(self) -> Set[str]:
        """Get the positive test set of cancer related genes."""
        cosmic_genes = self._load_cosmic()
        ncg_genes = self._load_ncg()
        return cosmic_genes.union(ncg_genes)

    def _load_cosmic(self) -> Set[str]:
        """Load COSMIC gene_symbols."""
        data = pd.read_csv(
            self.resource_dir / "Cosmic_CancerGeneCensus_v100_GRCh38.tsv",
            delimiter="\t",
            header=[0],
        )
        return set(data["GENE_SYMBOL"].str.lower())

    def _load_ncg(self) -> Set[str]:
        """Load NCG gene_symbols."""
        data = pd.read_csv(
            self.resource_dir / "NCG_cancerdrivers_annotation_supporting_evidence.tsv",
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
        data, targets = [], []
        for gene in cancer_genes.union(negative_samples):
            vec = self.gene_embeddings[gene]
            data.append(vec)
            targets.append(1 if gene in cancer_genes else 0)
        return np.array(data), np.array(targets)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
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
        matched_negative_samples = set(random.sample(negative_samples, total_samples))

        print(f"Number of cancer genes: {len(cancer_genes)}")
        print(f"Number of negative samples: {len(matched_negative_samples)}")

        # format data and targets for models
        return self.format_data_and_targets(cancer_genes, matched_negative_samples)
