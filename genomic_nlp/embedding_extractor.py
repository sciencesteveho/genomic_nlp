#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""

from pathlib import Path
import pickle
from typing import Dict, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm  # type: ignore
from transformers import AutoModel  # type: ignore
from transformers import AutoTokenizer  # type: ignore


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

    def __init__(self, model_path: str, max_length=512, batch_size=32):
        """Instantiate the embedding extractor class."""
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get three different types of embeddings from a DeBERTa model:

        1. Averaged embeddings - average over all token embeddings
        2. CLS embeddings - embeddings of the [CLS] token
        3. Attention-weighted embeddings - embeddings weighted by attention weights
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_attentions=True, output_hidden_states=True
            )

        last_hidden_states = outputs.last_hidden_state
        attention_weights = outputs.attentions[-1]

        averaged_embeddings = last_hidden_states.mean(dim=1)
        cls_embeddings = last_hidden_states[:, 0, :]
        attention_weights = attention_weights.mean(dim=1).mean(dim=1)
        attention_weighted_embeddings = (
            last_hidden_states * attention_weights.unsqueeze(-1)
        ).sum(dim=1)

        return averaged_embeddings, cls_embeddings, attention_weighted_embeddings

    def process_dataset(
        self, dataset: IterableDataset
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a dataset to extract embeddings."""
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

        averaged_embeddings = []
        cls_embeddings = []
        attention_weighted_embeddings = []

        for batch in tqdm(dataloader, desc="Processing batches"):
            avg_emb, cls_emb, att_emb = self.get_embeddings(batch["input_ids"])

            averaged_embeddings.append(avg_emb.cpu())
            cls_embeddings.append(cls_emb.cpu())
            attention_weighted_embeddings.append(att_emb.cpu())

        return (
            torch.cat(averaged_embeddings, dim=0),
            torch.cat(cls_embeddings, dim=0),
            torch.cat(attention_weighted_embeddings, dim=0),
        )


def casefold_genes(genes: Set[str]) -> Set[str]:
    """Casefold all genes."""
    return {gene.casefold() for gene in genes}


def filter_zero_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out key: value pairs where the value (embedding) consists of all
    zeroes.
    """
    return {key: value for key, value in embeddings.items() if np.any(value != 0)}


# gene_catalogue_file = (
#     "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/gene_catalogue.pkl"
# )
# synonyms_file = (
#     "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl"
# )
# w2v_model = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v/word2vec_300_dimensions_2024-08-13.model"

# with open(gene_catalogue_file, "rb") as file:
#     gene_catalogue = pickle.load(file)

# with open(synonyms_file, "rb") as file:
#     synonyms = pickle.load(file)

# genes = casefold_genes(gene_catalogue)
# w2v_extractor = Word2VecEmbeddingExtractor(model_path=w2v_model, synonyms=synonyms)

# embeddings, synonym_embeddings = w2v_extractor.extract_embeddings(list(genes))

# with open("w2v_embeddings.pkl", "wb") as file:
#     pickle.dump(embeddings, file)

# with open("w2v_synonym_embeddings.pkl", "wb") as file:
#     pickle.dump(synonym_embeddings, file)

# embeddings = filter_zero_embeddings(embeddings)
# synonym_embeddings = filter_zero_embeddings(synonym_embeddings)

# with open("w2v_filtered_embeddings.pkl", "wb") as file:
#     pickle.dump(embeddings, file)
# with open("w2v_filtered_synonym_embeddings.pkl", "wb") as file:
#     pickle.dump(synonym_embeddings, file)
