#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""


from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
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

    def _count_abstracts(self) -> int:
        """Count the number of abstracts in the file."""
        with open(self.file_path, "r") as f:
            return sum(1 for _ in f)

    def __iter__(
        self,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], List[Tuple[str, int]]]]:
        """Iterate over abstracts, yielding tokenized inputs and gene positions."""
        with open(self.file_path, "r") as f:
            for line in tqdm(
                iterable=f, total=self._count_abstracts(), desc="Processing abstracts"
            ):
                abstract = line.strip()
                words = abstract.lower().split()
                if gene_positions := [
                    (word, i)
                    for i, word in enumerate(words)
                    if word in self.genes_of_interest
                ]:
                    encoded = self.tokenizer.encode_plus(
                        abstract,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield encoded, gene_positions


class DeBERTaEmbeddingExtractor:
    """Extract embeddings from natural language processing models."""

    def __init__(
        self,
        model_path: str,
        dataset: TokenizedDataset,
        batch_size: int = 32,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = full_model.deberta
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def _rename_state_dict_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rename state dict keys to remove the 'module.' prefix."""
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    @torch.inference_mode()
    def extract_embeddings(self) -> Dict[str, np.ndarray]:
        """Extract gene embeddings from the model. We extract averaged
        embeddings - because DeBERTa models create embeddings per context, we
        average them to get one representation.
        """
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

        gene_embeddings: Dict[str, Any] = {}
        gene_counts: Dict[str, int] = {}

        with torch.no_grad():
            for batch, gene_positions_batch in tqdm(
                iterable=dataloader, total=len(dataloader), desc="Extracting embeddings"
            ):
                input_ids = batch["input_ids"].squeeze(1).to(self.device)
                attention_mask = batch["attention_mask"].squeeze(1).to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

                for i, gene_positions in enumerate(gene_positions_batch):
                    for gene, position in gene_positions:
                        if gene not in gene_embeddings:
                            gene_embeddings[gene] = torch.zeros(
                                hidden_states.size(-1), device=self.device
                            )
                            gene_counts[gene] = 0

                        gene_embeddings[gene] += hidden_states[i, position]
                        gene_counts[gene] += 1

        # average the embeddings
        return {
            gene: (embedding / gene_counts[gene]).cpu().numpy()
            for gene, embedding in gene_embeddings.items()
        }
