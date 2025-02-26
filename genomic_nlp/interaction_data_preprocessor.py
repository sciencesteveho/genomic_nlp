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


def load_unique_pairs(file_path: str) -> Set[Tuple[str, ...]]:
    """Load pairs for training."""
    unique_pairs: Set[Tuple[str, ...]] = set()
    with open(file_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            gene1, gene2 = row
            pair = tuple(sorted([gene1, gene2]))
            unique_pairs.add(pair)
    return unique_pairs


class InteractionDataPreprocessor:
    """Preprocess data for baseline models. Load gene pairs, embeddings, filter,
    and statify the train and test sets by source.

    Our split is designed as follows:
    Train set:
        All text-derived edges (that have embeddings) as positive.
        An equal number of randomly sampled negative pairs that do not exist in
        a catalogue of known interactions + the genes are not within 100kb on
        the linear reference genome.

    Test set:
        All experiment pairs that have embeddings. These are further stratified
        so we can see performance on all pairs, performance on known pairs, and
        performance on unknown pairs (pairs that are not in the text-derived
        edges).
    """

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        positive_train_file: str,
        negative_train_file: str,
        positive_test_file: str,
        negative_test_file: str,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Instantiate a BaselineDataPreprocessor object. Load data and
        embeddings.

        1. We load the embeddings and pair sets
            model embeddings
            testing data - experimentally verified interactions
            training data - text-derived interactions
        """
        self.data_dir = data_dir
        self.gene_embeddings = embeddings

        # load experimental pairs
        self.test_pairs_with_source = _load_pickle(positive_test_file)

        # casefold pairs and create a mapping of pairs to source
        self.test_pairs_to_source = {
            (pair[0].lower(), pair[1].lower()): pair[2]
            for pair in self.test_pairs_with_source
        }
        self.positive_test_pairs_original_case = [
            tuple(pair[:2]) for pair in self.test_pairs_with_source
        ]
        self.positive_test_pairs = casefold_pairs(self.test_pairs_with_source)
        self.negative_test_pairs = casefold_pairs(_load_pickle(negative_test_file))

        # load training pairs
        self.positive_training_pairs = load_unique_pairs(positive_train_file)
        self.negative_training_pairs = _load_pickle(negative_train_file)
        self.negative_training_pairs = casefold_pairs(self.negative_training_pairs)
        print("Data and embeddings loaded!")

    def load_and_preprocess_data(self):
        """Load all data for training!

        2. We filter the pairs for those that have embeddings
        3. We filter the test set to remove pairs that are in the training set
        4. We remove any pairs present in both the negative training and
           negative test sets
        4. We return the training and test sets

            Returns a dictionary of the format:
            {
                'train_features': np.ndarray,
                'train_targets': np.ndarray,
                'test_features': np.ndarray,
                'test_targets': np.ndarray,
                'test_pairs_known': List[Tuple[str,str]],
                'test_pairs_unknown': List[Tuple[str,str]],
              }
        """
        print("Filtering pairs for those with embeddings")
        train_positive_filtered = self._filter_pairs_for_embeddings(
            self.positive_training_pairs, self.gene_embeddings
        )
        train_negative_filtered = self._filter_pairs_for_embeddings(
            self.negative_training_pairs, self.gene_embeddings
        )
        test_positive_filtered = self._filter_pairs_for_embeddings(
            set(self.positive_test_pairs), self.gene_embeddings
        )
        test_negative_filtered = self._filter_pairs_for_embeddings(
            set(self.negative_test_pairs), self.gene_embeddings
        )

        print("Filtering pairs for prior knowledge.")
        pos_test = self.filter_pairs_for_prior_knowledge(
            train_positive_filtered=train_positive_filtered,
            test_positive_filtered=test_positive_filtered,
        )

        print("Filtering negative pairs to avoid data leakage.")
        neg_train, neg_test = self.split_negative_pairs(
            train_negative_filtered=train_negative_filtered,
            test_negative_filtered=test_negative_filtered,
            pos_train_examples=len(train_positive_filtered),
            pos_test_examples=len(pos_test),
        )

        print("Preparing training data and targets.")
        train_features, train_targets = self.prepare_data_and_targets(
            positive_pairs=list(train_positive_filtered), negative_pairs=neg_train
        )

        # prepare test data and targets BEFORE processing original case pairs
        print("\n--- Before processing original case pairs ---")
        print(f"Length of pos_test: {len(pos_test)}")
        print(f"Length of neg_test: {len(neg_test)}")
        print("Preparing test features and targets...")
        test_features, test_targets = self.prepare_data_and_targets(
            positive_pairs=pos_test, negative_pairs=neg_test
        )
        print(f"Length of test_features: {len(test_features)}")
        print(f"Length of test_targets: {len(test_targets)}")

        # get the original case gene pairs for the filtered positive test set
        original_case_map = {}
        for original_case_pair in self.positive_test_pairs_original_case:
            normalized_pair = self.normalize_pair(original_case_pair)
            original_case_map[normalized_pair] = original_case_pair

        # ensure original_case_pos_test_pairs maintains exact same order as
        # pos_test
        original_case_pos_test_pairs = []
        for pair in pos_test:
            normalized_pair = self.normalize_pair(pair)
            if normalized_pair in original_case_map:
                original_case_pos_test_pairs.append(original_case_map[normalized_pair])
            else:
                print(
                    f"Warning: Normalized pair {normalized_pair} not found in original case map. Using casefolded pair."
                )
                original_case_pos_test_pairs.append(
                    pair
                )  # Fallback to casefolded if original not found

        if len(original_case_pos_test_pairs) != len(pos_test):
            print(
                f"WARNING: Length mismatch! original_case_pos_test_pairs: {len(original_case_pos_test_pairs)}, pos_test: {len(pos_test)}"
            )
            original_case_pos_test_pairs = pos_test

        # combine positive and negative test pairs to maintain order with features
        test_gene_pairs = original_case_pos_test_pairs + list(neg_test)

        print("\n--- After processing original case pairs ---")
        print(f"Length of test_gene_pairs: {len(test_gene_pairs)}")
        print(f"Length of test_features: {len(test_features)}")
        print(f"Length of test_targets: {len(test_targets)}")

        # verify final lengths match - CRITICAL CHECK
        if len(test_gene_pairs) != len(test_features):
            print(
                f"CRITICAL ERROR: Final length mismatch! test_gene_pairs: {len(test_gene_pairs)}, test_features: {len(test_features)}"
            )
            if len(test_gene_pairs) > len(test_features):
                test_gene_pairs = test_gene_pairs[: len(test_features)]
            else:
                test_features = test_features[: len(test_gene_pairs)]
                test_targets = test_targets[: len(test_gene_pairs)]
                print(
                    "WARNING: test_features was longer than test_gene_pairs. Truncating features and targets to match pairs. Investigate data preprocessing!"
                )

        return (
            train_features,
            train_targets,
            test_features,
            test_targets,
            test_gene_pairs,
        )

    def filter_pairs_for_prior_knowledge(
        self,
        train_positive_filtered: Set[Tuple[str, str]],
        test_positive_filtered: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Given the test pairs, filter them to remove any pairs that overlap
        with the training pairs."""
        normalized_train = {
            self.normalize_pair(pair) for pair in train_positive_filtered
        }
        pos_test: List[Tuple[str, str]] = []

        # only keep pairs that are not in the training set
        for pair in test_positive_filtered:
            normalized = self.normalize_pair(pair)
            if normalized not in normalized_train:
                pos_test.append(pair)

        print(f"Total test positive pairs after filtering: {len(pos_test)}")
        return pos_test

    def split_negative_pairs(
        self,
        train_negative_filtered: Set[Tuple[str, str]],
        test_negative_filtered: Set[Tuple[str, str]],
        pos_train_examples: int,
        pos_test_examples: int,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Ensure no overlap between negative training and test sets."""
        # check number of train and test negatives
        print(f"Total train negative pairs: {len(train_negative_filtered)}")
        print(f"Total test negative pairs: {len(test_negative_filtered)}")
        normalized_test = {self.normalize_pair(pair) for pair in test_negative_filtered}
        filtered_train: List[Tuple[str, str]] = []

        filtered_train.extend(
            pair
            for pair in train_negative_filtered
            if self.normalize_pair(pair) not in normalized_test
        )
        print(f"Total train negative pairs after filtering: {len(filtered_train)}")
        print(f"Total test negative pairs after filtering: {len(normalized_test)}")
        deduped_test = [pair for pair in normalized_test if len(pair) == 2]

        # sample train and test to be commensurate
        if len(filtered_train) > pos_train_examples:
            filtered_train = random.sample(filtered_train, pos_train_examples)  # type: ignore

        if len(normalized_test) > pos_test_examples:
            deduped_test = random.sample(deduped_test, pos_test_examples)  # type: ignore

        # ensure len of test
        return filtered_train, deduped_test

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
            sources = self.test_pairs_to_source.get(pair, ("unknown",))
            for source in sources:
                stratified_test_data[source]["features"].append(test_features[i])
                stratified_test_data[source]["targets"].append(1)

        for source in stratified_test_data:
            n_pos = len(stratified_test_data[source]["features"])
            neg_indices = np.random.choice(len(neg_test), n_pos, replace=False)
            stratified_test_data[source]["features"].extend(
                test_features[neg_index] for neg_index in neg_indices
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

    def prepare_data_and_targets(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature data and target labels from gene pairs."""
        # check that positive and negative pairs have values
        print(f"Positive pairs: {len(positive_pairs)}")
        print(f"Negative pairs: {len(negative_pairs)}")

        all_pairs = positive_pairs + negative_pairs
        genes1, genes2 = zip(*all_pairs)

        # get embeddings for all genes at once
        vecs1 = np.array([self.gene_embeddings[gene] for gene in genes1])
        vecs2 = np.array([self.gene_embeddings[gene] for gene in genes2])

        # concatenate
        data = np.hstack((vecs1, vecs2))

        # create targets array
        targets = np.zeros(len(all_pairs))
        targets[: len(positive_pairs)] = 1

        # check that targets have two classes
        assert len(np.unique(targets)) == 2
        print(f"Classes: {np.unique(targets)}")
        return data, targets

    @staticmethod
    def _filter_pairs_for_embeddings(
        pairs: Set[Tuple[str, str]], gene_embeddings: Dict[str, np.ndarray]
    ) -> Set[Tuple[str, str]]:
        """Filter pairs of genes or proteins to ensure they have valid
        embeddings.
        """
        return {pair for pair in pairs if all(gene in gene_embeddings for gene in pair)}

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

    @staticmethod
    def randomly_split_gene_list(genes: List[str]) -> Tuple[List[str], List[str]]:
        """Randomly split a list of genes into two distinct sets."""
        random.shuffle(genes)
        split_index = len(genes) // 2
        return genes[:split_index], genes[split_index:]

    @staticmethod
    def normalize_pair(pair: Tuple[str, ...]) -> Tuple[str, ...]:
        """Normalize a pair of genes."""
        return tuple(sorted(gene.casefold() for gene in pair))


def _load_pickle(pickle_file: Any) -> Any:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def casefold_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Casefold gene pairs."""
    return [(pair[0].casefold(), pair[1].casefold()) for pair in pairs]
