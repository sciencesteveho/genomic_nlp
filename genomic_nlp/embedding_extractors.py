#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""


import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import h5py  # type: ignore
import numpy as np
from safetensors.torch import load_file  # type: ignore
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Config  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore


class Word2VecEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(
        self,
        model_path: str,
        # data_path: str,lgit
        synonyms: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Initialize the embedding extractor class."""
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


class TokenizedDataset(IterableDataset):
    """A dataset for efficiently extracting gene embeddings from abstracts."""

    def __init__(
        self,
        abstract_file: str,
        tokenizer: DebertaV2Tokenizer,
        genes: Set[str],
        max_length: int = 512,
    ):
        """Initialize the dataset."""
        self.file_path = abstract_file
        self.tokenizer = tokenizer
        self.genes_of_interest = {gene.lower() for gene in genes}
        self.max_length = max_length
        self.total_abstracts = self._count_abstracts()

    def _count_abstracts(self) -> int:
        """Count the number of abstracts in the file."""
        with open(self.file_path, "r") as f:
            return sum(1 for _ in f)

    def __iter__(
        self,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]]:
        """Iterate over abstracts, yielding tokenized inputs and gene positions."""
        worker_info = torch.utils.data.get_worker_info()
        start_position = 0
        end_position = None

        if worker_info:
            start_position = worker_info.id * self._shard_size(worker_info.num_workers)
            end_position = start_position + self._shard_size(worker_info.num_workers)

        with open(self.file_path, "r", encoding="utf-8") as file_iterator:
            if start_position > 0:
                for _ in range(start_position):
                    next(file_iterator)

            pbar = tqdm(total=self.total_abstracts, desc="Processing abstracts")
            for line_number, line in enumerate(file_iterator):
                if end_position is not None and line_number >= end_position:
                    break

                abstract = line.strip()
                words = abstract.lower().split()
                gene_positions = [
                    (word, i)
                    for i, word in enumerate(words)
                    if word in self.genes_of_interest
                ]

                tokenized = self.tokenizer(
                    abstract,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # adjust gene positions based on tokenization
                adjusted_positions = []
                for gene, pos in gene_positions:
                    token_pos = self.tokenizer.encode(
                        " ".join(words[:pos]),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    if len(token_pos) < self.max_length - 1:  # account for [CLS]
                        adjusted_positions.append((gene, len(token_pos)))

                yield (dict(tokenized.items()), adjusted_positions)
                pbar.update(1)
            pbar.close()

    def _shard_size(self, num_workers: int) -> int:
        """Estimate shard size based on the total number of workers"""
        return math.ceil(self.total_abstracts / num_workers)


class DeBERTaEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(
        self,
        model_path: str,
        dataset: TokenizedDataset,
        tokenizer: DebertaV2Tokenizer,
        batch_size: int = 16,
        chunk_size: int = 8,
    ):
        """Initialize the embedding extractor class."""
        model_dir = Path(model_path)
        config_path = model_dir / "config.json"
        model_path = str(model_dir / "model.safetensors")

        # load model and initialize with pretrained state dict
        config = DebertaV2Config.from_pretrained(config_path)
        full_model = DebertaV2ForMaskedLM(config)
        model_state = self._rename_state_dict_keys(load_file(model_path))

        # load the weights
        missing, unexpected = full_model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = full_model.deberta
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        print("Model loaded successfully.")

        self.vocab_size = len()

    def _rename_state_dict_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rename state dict keys to remove the 'module.' prefix."""
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
