# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to preprocess oncogenic prediction data for models."""


from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


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
        positive_samples: Set[str],
        negative_samples: Set[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create feature data and target labels for cancer related genes."""
        data = []
        targets = []
        gene_names = []
        for gene in sorted(positive_samples.union(negative_samples)):
            data.append(self.gene_embeddings[gene])
            targets.append(1 if gene in positive_samples else 0)
            gene_names.append(gene)
        return np.array(data), np.array(targets), gene_names

    def preprocess_data_by_year(
        self,
        year: int,
        horizon: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str]]:
        """For each year in 2001+=2019, we train a separate model.
        For test data due to size imbalance, we only consider a fixed lookahead
        window, testing the models's ability to capture cancer genes with
        provenance over the next 3 years.

        If it's the last models (2018 and 2019), we add any genes in
        self.cancer_genes that are not in the training set to the test set.
        """
        # filter cancer genes for those with embeddings
        cancer_genes = {
            gene for gene in self.cancer_genes if gene in self.gene_embeddings
        }

        # get boundaries
        start_test_year = year + 1
        end_test_year = year + horizon if horizon else 2020

        # split provenance data
        train_df = self.provenance[self.provenance["Year"] <= year]
        test_df = self.provenance[
            (self.provenance["Year"] > year)
            & (self.provenance["Year"] <= end_test_year)
        ]

        # filter cancer genes for those with embeddings
        train_genes = set(train_df["Gene"].str.lower()) & set(
            self.gene_embeddings.keys()
        )
        test_genes = set(test_df["Gene"].str.lower()) & set(self.gene_embeddings.keys())

        # get train and test set
        pos_train = train_genes
        pos_test = test_genes

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

        # print to check
        print("Year:", year)
        print("Pos train:", len(pos_train))
        print("Pos test:", len(pos_test))
        print("Neg train:", len(neg_train_samples))
        print("Neg test:", len(neg_test_samples))

        train_features, train_targets, train_gene_names = self.format_data_and_targets(
            pos_train, neg_train_samples
        )
        test_features, test_targets, test_gene_names = self.format_data_and_targets(
            pos_test, neg_test_samples
        )

        return (
            train_features,
            train_targets,
            train_gene_names,
            test_features,
            test_targets,
            test_gene_names,
        )
