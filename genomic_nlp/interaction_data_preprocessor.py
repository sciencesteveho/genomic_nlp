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
        args: argparse.Namespace,
        data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    ) -> None:
        """Instantiate a BaselineDataPreprocessor object. Load data and
        embeddings.
        """
        self.data_dir = data_dir

        # load embeddings
        self.gene_embeddings = _load_pickle(args.embeddings)
        self.gene_embeddings = casefold_embeddings(self.gene_embeddings)

        # load experimental pairs
        self.pos_pairs_with_source = _load_pickle(args.positive_pairs_file)

        # casefold pairs and create a mapping of pairs to source
        self.pair_to_source = {
            (pair[0].lower(), pair[1].lower()): pair[2]
            for pair in self.pos_pairs_with_source
        }
        self.positive_pairs = casefold_pairs(self.pos_pairs_with_source)
        self.negative_pairs = casefold_pairs(_load_pickle(args.negative_pairs_file))

        # load co_occurrence pairs
        self.text_edges = set(
            casefold_pairs(
                [
                    tuple(row)
                    for row in csv.reader(open(args.text_edges_file), delimiter="\t")
                ]
            )
        )
        print("Data and embeddings loaded!")

    def load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load all data for training!

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
        train_pos = self._filter_pairs_for_embeddings(
            set(self.positive_pairs), self.gene_embeddings
        )
        train_neg = self._filter_pairs_for_embeddings(
            set(self.negative_pairs), self.gene_embeddings
        )

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

    def filter_pairs_for_prior_knowledge(
        self,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Filter the gene pairs into two sets: those with prior knowledge, and
        those without. Gene pairs with prior knowledge are those that exist in the
        text_edges set, which contains edges extracted from the literature.

        There's a large imbalance between the number of test / train examples,
        so we additionally split the positive pairs (test set) 50% and ensure it
        is split without gene leakage.
        """
        pos_train: List[Tuple[str, str]] = []
        pos_test: List[Tuple[str, str]] = []
        prior_knowledge_pairs = []
        no_prior_knowledge_pairs = []

        print(f"Total positive pairs: {len(self.positive_pairs)}")

        # separate pairs with and without prior knowledge
        for pair in self.positive_pairs:
            if pair in self.text_edges or (pair[1], pair[0]) in self.text_edges:
                prior_knowledge_pairs.append(pair)
            else:
                no_prior_knowledge_pairs.append(pair)

        print(f"Positive pairs with prior knowledge: {len(prior_knowledge_pairs)}")
        print(
            f"Positive pairs without prior knowledge: {len(no_prior_knowledge_pairs)}"
        )

        # get gene list
        genes: Set[str] = set()
        for pair in prior_knowledge_pairs:
            genes.add(pair[0])
            genes.add(pair[1])

        # split genes into disjoint sets
        train_genes, test_genes = self.randomly_split_gene_list(list(genes))

        # assign pairs to train or test based on gene sets
        # exclude pairs with genes in both sets
        for pair in no_prior_knowledge_pairs:
            if pair[0] in train_genes and pair[1] in train_genes:
                pos_train.append(pair)
            elif pair[0] in test_genes and pair[1] in test_genes:
                pos_test.append(pair)

        pos_train.extend(prior_knowledge_pairs)
        print(f"Total training positive pairs after splitting: {len(pos_train)}")
        print(f"Total testing positive pairs after splitting: {len(pos_test)}")

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
