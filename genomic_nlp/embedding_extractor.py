#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""

import os
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from safetensors.torch import load_file  # type: ignore
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm  # type: ignore
from transformers import AutoConfig  # type: ignore
from transformers import AutoModel  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from streaming_corpus import EmbeddingExtractorStreamingCorpus


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

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 32):
        """Instantiate the embedding extractor class."""
        config_path = os.path.dirname(model_path)
        config = AutoConfig.from_pretrained(config_path)
        self.model = AutoModel.from_config(config)
        statedict = load_file(model_path)

        # remove module prefix from torch distributed
        adjusted_states = {re.sub(r"^module\.", "", k): v for k, v in statedict.items()}
        model_keys = set(self.model.statedict().keys())
        adjusted_states = {k: v for k, v in adjusted_states.items() if k in model_keys}
        self.model.load_state_dict(adjusted_states, strict=False)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get three different types of embeddings from a DeBERTa model:

        1. Averaged embeddings - average over all token embeddings
        2. CLS embeddings - embeddings of the [CLS] token
        3. Attention-weighted embeddings - embeddings weighted by attention weights
        """
        inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }

        # get outputs from model - last_hidden_state, attention, hidden_states
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

        return (
            averaged_embeddings.cpu().numpy(),
            cls_embeddings.cpu().numpy(),
            attention_weighted_embeddings.cpu().numpy(),
        )

    def process_dataset(
        self, dataset: EmbeddingExtractorStreamingCorpus
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process a dataset to extract embeddings."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

        # initialize dictionary to store embeddings
        embeddings: Dict[str, Dict[str, List[np.ndarray]]] = {}

        # process batches and get embeddings
        for batch in tqdm(dataloader, desc="Processing batches"):
            avg_emb, cls_emb, att_emb = self.get_embeddings(batch)

            for i, gene in enumerate(batch["gene"]):
                if gene not in embeddings:
                    embeddings[gene] = {
                        "averaged": [],
                        "cls": [],
                        "attention_weighted": [],
                    }

                embeddings[gene]["averaged"].append(avg_emb[i])
                embeddings[gene]["cls"].append(cls_emb[i])
                embeddings[gene]["attention_weighted"].append(att_emb[i])

        # initialize new dicts to avoid type errors
        averaged_embeddings: Dict[str, np.ndarray] = {}
        cls_embeddings: Dict[str, np.ndarray] = {}
        attention_weighted_embeddings: Dict[str, np.ndarray] = {}

        # average embedding types across occurences
        for gene, gene_embeddings in embeddings.items():
            averaged_embeddings[gene] = np.mean(gene_embeddings["averaged"], axis=0)
            cls_embeddings[gene] = np.mean(gene_embeddings["cls"], axis=0)
            attention_weighted_embeddings[gene] = np.mean(
                gene_embeddings["attention_weighted"], axis=0
            )

        return averaged_embeddings, cls_embeddings, attention_weighted_embeddings


def casefold_genes(genes: Set[str]) -> Set[str]:
    """Casefold all genes."""
    return {gene.casefold() for gene in genes}


def filter_zero_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out key: value pairs where the value (embedding) consists of all
    zeroes.
    """
    return {key: value for key, value in embeddings.items() if np.any(value != 0)}
