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
        test_features, test_targets = self.prepare_data_and_targets(
            positive_pairs=pos_test, negative_pairs=neg_test
        )

        return (
            train_features,
            train_targets,
            test_features,
            test_targets,
            pos_test,
            neg_test,
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
        deduped_test = list(normalized_test)

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


class CancerGeneDataPreprocessor:
    """Preprocess data for oncogenicity prediction models.

    For each year (i.e. 2003) we get all of the cancer drivers that happen from
    that year, and before. Then, we get all of the cancer drivers after that
    year.
    """

    def __init__(self, gene_embeddings: Dict[str, np.ndarray]) -> None:
        """Instantiate an OncogenicDataPreprocessor object. Load data and embeddings."""
        self.gene_embeddings = gene_embeddings

        # hardcoded, to fix later
        self.resource_dir = Path(
            "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/cancer"
        )

        self.cancer_genes = self._get_known_cancer_genes()

        # load provenance data
        self.provenance = self._load_provenance()

        # get genes that appear in the entire provenance dataset (any year)
        self.discovered_any_time = set(self.provenance["Gene"].str.lower())

    def _get_known_cancer_genes(self) -> Set[str]:
        """Get the positive test set of cancer related genes.

        Positive genes are are the union of
        - COSMIC
        - NCG
        - Intogen
        """
        cosmic_genes = self._load_cosmic()
        ncg_genes = self._load_ncg()
        intogen_genes = self._load_intogen()
        return cosmic_genes.union(ncg_genes).union(intogen_genes)

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

    def _load_intogen(self) -> Set[str]:
        """Load Intogen cancer driver genes."""
        data = pd.read_csv(
            self.resource_dir / "Compendium_Cancer_Genes.tsv",
            delimiter="\t",
            header=[0],
        )
        return set(data["SYMBOL"].str.lower())

    def _load_provenance(self) -> pd.DataFrame:
        """Load provenance data."""
        data = pd.read_csv(
            f"{self.resource_dir}/cancer_normalized.txt",
            sep="\s+",
            names=["Gene", "Year"],
        )
        return data.sort_values("Year")

    def get_cancer_genes_by_year(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get cancer genes by year."""
        # get genes from year, inclusive
        genes_before = self.provenance[self.provenance["Year"] <= year]
        after = self.provenance[self.provenance["Year"] > year]
        return genes_before, after

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

    def preprocess_data_by_year(
        self, year: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For each year in 2001+=2019, we train a separate model"""
        # get cancer genes by year
        before, after = self.get_cancer_genes_by_year(year)

        # filter cancer genes for those with embeddings
        cancer_genes = {
            gene for gene in self.cancer_genes if gene in self.gene_embeddings
        }

        # split into training and test sets
        # training = year inclusive and all previous
        # test = years after
        pos_train = set(before["Gene"].str.lower())
        pos_test = set(after["Gene"].str.lower())

        # generate negative samples
        # any sample not in the positive set or known cancer gene
        # but if it's in the provenance in the future, it's fine
        all_genes = set(self.gene_embeddings.keys())
        negative_train = (all_genes - pos_train) - cancer_genes

        # negative samples for the test set
        # true negatives - so not in cancer genes, and never discovered
        negative_test = (all_genes - self.discovered_any_time) - cancer_genes

        # sample negatives
        neg_train_samples = set(random.sample(negative_train, len(pos_train)))
        neg_test_samples = set(random.sample(negative_test, len(pos_test)))

        train_features, train_targets = self.format_data_and_targets(
            pos_train, neg_train_samples
        )

        test_features, test_targets = self.format_data_and_targets(
            pos_test, neg_test_samples
        )

        return (
            train_features,
            train_targets,
            test_features,
            test_targets,
        )


def _load_pickle(pickle_file: Any) -> Any:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def casefold_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Casefold gene pairs."""
    return [(pair[0].casefold(), pair[1].casefold()) for pair in pairs]
