#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""


from pathlib import Path
import pickle
from typing import Dict, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np


class Word2VecEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(
        self,
        model_path: str,
        # data_path: str,
        synonyms: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Instantiate the embedding extractor class."""
        self.model_path = model_path
        # self.data_path = data_path
        if synonyms:
            self.synonyms = synonyms

        # load model
        self.model = Word2Vec.load(model_path)

    def extract_embeddings(
        self, genes: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract embeddings from natural language processing models."""
        embeddings = {}
        synonym_embeddings = {}

        for gene in genes:
            gene_vector = self._get_gene_vector(gene)
            embeddings[gene] = gene_vector
            synonym_embeddings[gene] = self._get_synonym_embedding(gene, gene_vector)

        return embeddings, synonym_embeddings

    def _get_gene_vector(self, gene: str) -> np.ndarray:
        """Get the embedding vector for a gene."""
        try:
            return self.model.wv[gene]
        except KeyError:
            print(f"Warning: gene {gene} not in vocabulary. Populating with zeros.")
            return np.zeros(self.model.vector_size)

    def _get_synonym_embedding(self, gene: str, gene_vector: np.ndarray) -> np.ndarray:
        """Get the synonym-enhanced embedding for a gene by average over the
        gene vector and all synonym vectors.
        """
        if gene not in self.synonyms:
            return gene_vector

        if synonym_vectors := [
            self.model.wv[synonym]
            for synonym in self.synonyms[gene]
            if synonym in self.model.wv
        ]:
            return np.mean([gene_vector] + synonym_vectors, axis=0)
        else:
            return gene_vector

    # def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
    #     """Save dictionary of embeddings."""
    #     pickle.dump(embeddings, open(self.data_path / "word2vec_embeddings.pkl", "wb"))


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


def filter_zero_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out key: value pairs where the value (embedding) consists of all
    zeroes.
    """
    return {key: value for key, value in embeddings.items() if np.any(value != 0)}


gene_catalogue_file = (
    "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/gene_catalogue.pkl"
)
synonyms_file = (
    "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl"
)
w2v_model = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v/word2vec_300_dimensions_2024-08-13.model"

with open(gene_catalogue_file, "rb") as file:
    gene_catalogue = pickle.load(file)

with open(synonyms_file, "rb") as file:
    synonyms = pickle.load(file)

genes = casefold_genes(gene_catalogue)
w2v_extractor = Word2VecEmbeddingExtractor(model_path=w2v_model, synonyms=synonyms)

embeddings, synonym_embeddings = w2v_extractor.extract_embeddings(list(genes))

with open("w2v_embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)

with open("w2v_synonym_embeddings.pkl", "wb") as file:
    pickle.dump(synonym_embeddings, file)

embeddings = filter_zero_embeddings(embeddings)
synonym_embeddings = filter_zero_embeddings(synonym_embeddings)

with open("w2v_filtered_embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)
with open("w2v_filtered_synonym_embeddings.pkl", "wb") as file:
    pickle.dump(synonym_embeddings, file)
