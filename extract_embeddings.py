#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models.

3. Make a dictionary of HGNC snynonyms for genes in gene_catalogue.
4. Extract word2vec embeddings for (a) casefolded gene catalogue (b) concatenated gene catalogue w/ synonyms.


# wget https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt

"""


from pathlib import Path
import pickle
from typing import Dict, Set

from gensim.models import Word2Vec  # type: ignore
import numpy as np


class Word2VecEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(self, model: str, model_path: Path, data_path: Path) -> None:
        """Instantiate the embedding extractor class."""
        self.model_path = model_path
        self.data_path = data_path

        # load model
        self.model = Word2Vec.load(model_path / model)

    def extract_embeddings(self, genes: list) -> Dict[str, np.ndarray]:
        """Extract embeddings from natural language processing models."""
        embeddings = {}
        for gene in genes:
            try:
                gene_vector = self.model.wv[gene]
                embeddings[gene] = gene_vector
            except KeyError:
                print(f"Warning: gene {gene} not in vocabulary.")
                embeddings[gene] = np.zeros(self.model.vector_size)
        return embeddings

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Save dictionary of embeddings."""
        pickle.dump(embeddings, open(self.data_path / "word2vec_embeddings.pkl", "wb"))


class DeBERTaEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(self, model: str, model_path: Path, data_path: Path) -> None:
        """Instantiate the embedding extractor class."""
        if model not in ["word2vec", "deberta", "biobert"]:
            raise ValueError(
                "Invalid model name. \
                Choose from `word2vec`, `deberta`, or `biobert`."
            )
        self.model = model
        self.model_path = model_path
        self.data_path = data_path

    def extract_embeddings(self, genes: list) -> None:
        """Extract embeddings from natural language processing models."""
        pass

    def save_embeddings(self, embeddings: dict) -> None:
        """Save dictionary of embeddings."""
        pass

    def load_model(self) -> None:
        """Load model for embeddings extraction."""
        pass


def casefold_genes(genes: Set[str]) -> Set[str]:
    """Casefold all genes."""
    return {gene.casefold() for gene in genes}
