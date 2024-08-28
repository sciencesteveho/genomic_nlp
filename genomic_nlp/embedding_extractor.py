#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from natural language processing models."""


from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from safetensors.torch import load_file  # type: ignore
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Config  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore

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
            if k in ["input_ids", "attention_mask"]
        }

        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Attention mask shape: {inputs['attention_mask'].shape}")

        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            outputs = self.model(
                **inputs, output_attentions=True, output_hidden_states=True
            )
            last_hidden_states = outputs.last_hidden_state
            attention_weights = outputs.attentions[-1]

            print(f"Last hidden states shape: {last_hidden_states.shape}")
            print(f"Attention weights shape: {attention_weights.shape}")

            averaged_embeddings = last_hidden_states.mean(dim=1)
            cls_embeddings = last_hidden_states[:, 0, :]
            attention_weights = attention_weights.mean(dim=1).mean(dim=1)
            attention_weighted_embeddings = (
                last_hidden_states * attention_weights.unsqueeze(-1)
            ).sum(dim=1)

        return averaged_embeddings, cls_embeddings, attention_weighted_embeddings

    def process_chunk(
        self,
        tokenized_file: str,
        output_file: str,
    ) -> None:
        """Process a single chunk of data and save embeddings."""
        # get number of samples for tqdm
        with open(tokenized_file, "rb") as f:
            tokenized_abstracts, _ = pickle.load(f)
        total_examples = len(tokenized_abstracts)
        del tokenized_abstracts

        dataset = EmbeddingExtractorStreamingCorpus(
            [tokenized_file],
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate_batch,
        )

        embeddings: Dict[str, Dict[str, Any]] = {
            "averaged": {},
            "cls": {},
            "attention_weighted": {},
        }

        processed_examples = 0
        pbar = tqdm(
            total=total_examples, desc=f"Processing {Path(tokenized_file).name}"
        )

        for batch in dataloader:
            print(f"Batch size: {len(batch['gene'])}")
            avg_emb, cls_emb, att_emb = self.get_embeddings(batch)

            for gene, avg, cls, att in zip(
                batch["gene"],
                avg_emb.cpu().numpy(),
                cls_emb.cpu().numpy(),
                att_emb.cpu().numpy(),
            ):
                if gene not in embeddings["averaged"]:
                    embeddings["averaged"][gene] = []
                    embeddings["cls"][gene] = []
                    embeddings["attention_weighted"][gene] = []

                embeddings["averaged"][gene].append(avg)
                embeddings["cls"][gene].append(cls)
                embeddings["attention_weighted"][gene].append(att)

            processed_examples += len(batch["gene"])
            pbar.update(len(batch["gene"]))

        pbar.close()

        # average embeddings for genes with multiple occurrences
        for emb_type, value in embeddings.items():
            for gene in value:
                embeddings[emb_type][gene] = (
                    np.mean(embeddings[emb_type][gene], axis=0)
                    if len(embeddings[emb_type][gene]) > 1
                    else embeddings[emb_type][gene][0]
                )

        # save embeddings
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)

    def process_all_chunks(
        self,
        tokenized_files: List[str],
        output_dir: str,
    ) -> None:
        """Process all chunks of data and save embeddings."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, tokenized_file in enumerate(tokenized_files):
            output_file = output_path / f"deberta_embeddings_chunk_{i}.pkl"
            self.process_chunk(tokenized_file, str(output_file))

    def combine_embeddings(
        self,
        embedding_files: List[str],
        output_file: str,
    ) -> None:
        """Combine embeddings across chunks."""
        combined_embeddings: Dict[str, Dict[str, Any]] = {
            "averaged": {},
            "cls": {},
            "attention_weighted": {},
        }

        for file in tqdm(embedding_files, desc="Combining embeddings"):
            with open(file, "rb") as f:
                chunk_embeddings = pickle.load(f)

            for emb_type in combined_embeddings:
                for gene, emb in chunk_embeddings[emb_type].items():
                    if gene not in combined_embeddings[emb_type]:
                        combined_embeddings[emb_type][gene] = []
                    combined_embeddings[emb_type][gene].append(emb)

        # average embeddings for genes with multiple occurrences
        for emb_type, value in combined_embeddings.items():
            for gene in value:
                combined_embeddings[emb_type][gene] = (
                    np.mean(combined_embeddings[emb_type][gene], axis=0)
                    if len(combined_embeddings[emb_type][gene]) > 1
                    else combined_embeddings[emb_type][gene][0]
                )
        # save combined embeddings
        with open(output_file, "wb") as f:
            pickle.dump(combined_embeddings, f)

    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of tokenized examples."""

        def pad_and_truncate(
            tensor: Union[List[int], torch.Tensor], target_len: int
        ) -> torch.Tensor:
            if isinstance(tensor, list):
                tensor = torch.tensor(tensor)
            if len(tensor) > target_len:
                return tensor[:target_len]
            return torch.nn.functional.pad(
                tensor, (0, target_len - len(tensor)), value=0
            )

        try:
            collated_batch: Dict[str, Any] = {
                "gene": [item["gene"] for item in batch],
                "input_ids": [
                    pad_and_truncate(item["input_ids"], self.max_length)
                    for item in batch
                ],
                "attention_mask": [
                    pad_and_truncate(item["attention_mask"], self.max_length)
                    for item in batch
                ],
            }

            collated_batch["input_ids"] = torch.stack(collated_batch["input_ids"])
            collated_batch["attention_mask"] = torch.stack(
                collated_batch["attention_mask"]
            )

            print("Collated batch shapes:")
            print(f"input_ids: {collated_batch['input_ids'].shape}")
            print(f"attention_mask: {collated_batch['attention_mask'].shape}")

            return collated_batch
        except Exception as e:
            print(f"Error in collate_batch: {str(e)}")
            print("Batch contents:")
            for i, item in enumerate(batch):
                print(f"Item {i}:")
                for k, v in item.items():
                    print(
                        f"  {k}: {type(v)}, {len(v) if hasattr(v, '__len__') else 'N/A'}"
                    )
            raise

    # def process_dataset(
    #     self,
    #     tokenized_files: List[str],
    #     total_genes: int,
    # ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    #     """Process a dataset to extract embeddings."""
    #     TOTAL_BATCHES = 3088709 // self.batch_size

    #     embeddings: Dict[str, Dict[str, Any]] = {
    #         "averaged": {},
    #         "cls": {},
    #         "attention_weighted": {},
    #     }

    #     # prepare the dataset and dataloader
    #     dataset = EmbeddingExtractorStreamingCorpus(tokenized_files)
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=self.batch_size,
    #         num_workers=5,
    #         pin_memory=True,
    #         prefetch_factor=10,
    #         persistent_workers=True,
    #     )

    #     # pre-allocate CUDA tensors for accumulating embeddings
    #     accumulated_embeddings = {
    #         emb_type: torch.zeros(
    #             (total_genes, 768), dtype=torch.float32, device=self.device
    #         )
    #         for emb_type in embeddings
    #     }
    #     counts = torch.zeros(total_genes, dtype=torch.int32, device=self.device)

    #     # process dataset and extract embeddings
    #     with tqdm(total=TOTAL_BATCHES, desc="Processing batches") as pbar:
    #         for batch in dataloader:
    #             avg_emb, cls_emb, att_emb = self.get_embeddings(batch)

    #             gene_indices = torch.tensor(
    #                 [dataset.gene_to_index[gene] for gene in batch["gene"]],
    #                 device=self.device,
    #             )

    #             for emb_type, emb in zip(
    #                 ["averaged", "cls", "attention_weighted"],
    #                 [avg_emb, cls_emb, att_emb],
    #             ):
    #                 accumulated_embeddings[emb_type].index_add_(0, gene_indices, emb)

    #             counts.index_add_(
    #                 0, gene_indices, torch.ones_like(gene_indices, dtype=torch.int32)
    #             )

    #             pbar.update(1)

    #     # average the embeddings
    #     for emb_type in embeddings:
    #         avg_embeddings = (
    #             (accumulated_embeddings[emb_type] / counts.unsqueeze(1)).cpu().numpy()
    #         )
    #         embeddings[emb_type] = {
    #             gene: avg_embeddings[i] for gene, i in dataset.gene_to_index.items()
    #         }

    #     return (
    #         embeddings["averaged"],
    #         embeddings["cls"],
    #         embeddings["attention_weighted"],
    #     )
