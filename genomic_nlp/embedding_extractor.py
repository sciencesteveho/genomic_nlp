#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""


from collections import defaultdict
import os
from pathlib import Path
import pickle
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from safetensors.torch import load_file  # type: ignore
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Config  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore


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

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 8):
        """instantiate the embedding extractor class."""
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

        self.model = full_model.deberta
        self.max_length = max_length
        self.batch_size = batch_size
        self.amp_dtype = torch.float16  # mixed precision for speed gains
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")
        print(f"Model hidden size: {config.hidden_size}")
        print(f"Max length: {self.max_length}")

    def _rename_state_dict_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rename state dict keys to remove the 'module.' prefix."""
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    @torch.inference_mode()
    def extract_embeddings(
        self,
        tokenized_abstracts: List[List[int]],
        gene_occurrences: Dict[str, List[Tuple[int, int]]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get three different types of embeddings from a DeBERTa model:

        1. Averaged embeddings - average over all token embeddings
        2. CLS embeddings - embeddings of the [CLS] token
        3. Attention-weighted embeddings - embeddings weighted by attention weights
        """
        embeddings: Dict[str, Dict[str, List[torch.Tensor]]] = {
            "averaged": defaultdict(list),
            "cls": defaultdict(list),
            "attention_weighted": defaultdict(list),
        }

        # batched processing
        all_occurrences = [
            (gene, abstract_idx, token_idx)
            for gene, occurrences in gene_occurrences.items()
            for abstract_idx, token_idx in occurrences
        ]

        for i in tqdm(
            range(0, len(all_occurrences), self.batch_size), desc="Processing batches"
        ):
            batch = all_occurrences[i : i + self.batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(tokenized_abstracts[abstract_idx][: self.max_length])
                    for _, abstract_idx, _ in batch
                ],
                batch_first=True,
                padding_value=0,
            ).to(
                self.device
            )  # pad to max length
            attention_mask = torch.ones_like(input_ids)
            token_indices = torch.tensor([token_idx for _, _, token_idx in batch]).to(
                self.device
            )

            outputs = self.model(
                input_ids, attention_mask=attention_mask, output_attentions=True
            )
            last_hidden_state = outputs.last_hidden_state
            attentions = outputs.attentions[-1]  # final layer attention

            for j, (gene, _, _) in enumerate(batch):
                embeddings["averaged"][gene].append(
                    last_hidden_state[j, token_indices[j]]
                )
                embeddings["cls"][gene].append(last_hidden_state[j, 0])

                attention_weights = attentions[j, :, token_indices[j], :].mean(dim=0)
                weighted_embedding = (
                    last_hidden_state[j] * attention_weights.unsqueeze(-1)
                ).sum(dim=0)
                embeddings["attention_weighted"][gene].append(weighted_embedding)

        # average embeddings for each gene
        final_embeddings: Dict[str, Dict[str, torch.Tensor]] = {
            "averaged": {},
            "cls": {},
            "attention_weighted": {},
        }

        for embed_type in final_embeddings:
            for gene, gene_embeddings in embeddings[embed_type].items():
                if gene_embeddings:
                    final_embeddings[embed_type][gene] = torch.stack(
                        gene_embeddings
                    ).mean(dim=0)
                else:
                    print(
                        f"Warning: No embeddings found for gene {gene} in {embed_type}"
                    )

        return final_embeddings

    def process_chunks_for_embeddings(
        self, chunk_files: List[str]
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int]]:
        """Process a list of chunked and pre-tokenized abstracts one at a time
        and extract gene embeddings to an `all_embeddings` dictionary`. When all
        abstracts have been processed, return the dictionary.
        """
        all_embeddings: DefaultDict[str, DefaultDict[str, List[torch.Tensor]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        gene_counts: DefaultDict[str, int] = defaultdict(int)

        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            with open(chunk_file, "rb") as f:
                tokenized_abstracts, gene_occurrences = pickle.load(f)

            chunk_embeddings = self.extract_embeddings(
                tokenized_abstracts, gene_occurrences
            )

            for embed_type in chunk_embeddings:
                for gene, embedding in chunk_embeddings[embed_type].items():
                    all_embeddings[embed_type][gene].append(embedding)
                    gene_counts[gene] += 1

        # average embeddings across all chunks
        final_embeddings: Dict[str, Dict[str, torch.Tensor]] = {
            "averaged": {},
            "cls": {},
            "attention_weighted": {},
        }

        for embed_type in final_embeddings:
            for gene, embeddings in all_embeddings[embed_type].items():
                if embeddings:
                    final_embeddings[embed_type][gene] = torch.stack(embeddings).mean(
                        dim=0
                    )
                else:
                    print(
                        f"Warning: No embeddings found for gene {gene} in {embed_type}"
                    )

        return final_embeddings, dict(gene_counts)
