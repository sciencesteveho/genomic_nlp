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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm  # type: ignore
from transformers import AutoConfig  # type: ignore
from transformers import AutoModel  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformers import DebertaV2Config  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore

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

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 64):
        """instantiate the embedding extractor class."""
        # print directory contents
        model_dir = os.path.dirname(model_path)
        print(f"Contents of {model_dir}:")
        for item in os.listdir(model_dir):
            print(item)

        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "model.safetensors")

        # load the configuration
        config = DebertaV2Config.from_pretrained(config_path)

        # initialize the model with the configuration
        full_model = DebertaV2ForMaskedLM(config)

        # load the state dict
        old_states = load_file(model_path)

        # remove the 'module.' prefix if it exists
        new_states = {k.replace("module.", ""): v for k, v in old_states.items()}

        # load the weights
        missing, unexpected = full_model.load_state_dict(new_states, strict=False)

        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")

        # extract the base model
        self.model = full_model.deberta
        # self.model = torch.compile(self.model)  # compile to improve performance

        self.max_length = max_length
        self.batch_size = batch_size
        self.amp_dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")

    @torch.inference_mode()
    def get_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get three different types of embeddings from a DeBERTa model:

        1. Averaged embeddings - average over all token embeddings
        2. CLS embeddings - embeddings of the [CLS] token
        3. Attention-weighted embeddings - embeddings weighted by attention weights
        """
        inputs = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
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
        self, dataset: EmbeddingExtractorStreamingCorpus, total_tokens: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process a dataset to extract embeddings."""
        TOTAL_BATCHES = 3088709 // self.batch_size

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=5,
            pin_memory=True,
            prefetch_factor=10,
            persistent_workers=True,
        )

        embeddings: Dict[str, Dict[str, Any]] = {
            "averaged": {},
            "cls": {},
            "attention_weighted": {},
        }

        # pre-allocate CUDA tensors for accumulating embeddings
        accumulated_embeddings = {
            emb_type: torch.zeros(
                (total_tokens, 768), dtype=torch.float32, device=self.device
            )
            for emb_type in embeddings
        }
        counts = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)

        with tqdm(total=TOTAL_BATCHES, desc="Processing batches") as pbar:
            for batch in dataloader:
                avg_emb, cls_emb, att_emb = self.get_embeddings(batch)

                gene_indices = torch.tensor(
                    [dataset.gene_to_index[gene] for gene in batch["gene"]],
                    device=self.device,
                )

                for emb_type, emb in zip(
                    ["averaged", "cls", "attention_weighted"],
                    [avg_emb, cls_emb, att_emb],
                ):
                    accumulated_embeddings[emb_type].index_add_(0, gene_indices, emb)

                counts.index_add_(
                    0, gene_indices, torch.ones_like(gene_indices, dtype=torch.int32)
                )

                pbar.update(1)

        # average the embeddings
        for emb_type in embeddings:
            avg_embeddings = (
                (accumulated_embeddings[emb_type] / counts.unsqueeze(1)).cpu().numpy()
            )
            embeddings[emb_type] = {
                gene: avg_embeddings[i] for i, gene in enumerate(dataset.genes)
            }

        return (
            embeddings["averaged"],
            embeddings["cls"],
            embeddings["attention_weighted"],
        )


def casefold_genes(genes: Set[str]) -> Set[str]:
    """Casefold all genes."""
    return {gene.casefold() for gene in genes}


def filter_zero_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out key: value pairs where the value (embedding) consists of all
    zeroes.
    """
    return {key: value for key, value in embeddings.items() if np.any(value != 0)}
