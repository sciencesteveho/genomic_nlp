#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementations of streaming corpus classes for DeBERTa fine-tuning and
embedding extraction."""


import logging
import math
import mmap
import os
import pickle
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import PreTrainedTokenizer  # type: ignore


class SimpleStreamingCorpus(IterableDataset):
    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        with open(self.file_path, "r") as f:
            for line in f:
                if line.strip():
                    encoded = self.tokenizer.encode_plus(
                        line.strip(),
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    yield {
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                    }


class StreamingCorpus(IterableDataset):
    """Custom streaming dataset for abstracts."""

    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ):
        """Initialize the streaming corpus class."""
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_size = os.path.getsize(file_path)
        self.num_lines = 3889578
        self.current_position = 0
        self.iteration_count = 0
        logging.info(
            f"Initialized StreamingCorpus with file: {file_path}, size: {self.file_size}"
        )

    def __len__(self) -> int:
        """Return the number of lines in the file."""
        return self.num_lines

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Create an iterator over the corpus."""
        self.iteration_count += 1
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # distribute data across processes for distributed training
        if torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0

        # calculate the range for this worker and process
        per_worker = self.file_size // (num_workers * num_replicas)
        start = (rank * num_workers + worker_id) * per_worker
        end = (
            start + per_worker
            if (rank * num_workers + worker_id + 1) < (num_workers * num_replicas)
            else self.file_size
        )

        # reset position if we've reached the end of the file
        if self.current_position >= end:
            self.current_position = start

        logging.info(
            f"Iteration {self.iteration_count}: Worker {worker_id} (rank {rank}) processing range: {self.current_position} - {end}"
        )

        return self.read_abstracts(self.current_position, end)

    def read_abstracts(self, start: int, end: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Read and tokenize abstracts from the file."""
        with open(self.file_path, "r") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.seek(start)
            count = 0
            while mm.tell() < end:
                if line := mm.readline().decode().strip():
                    try:
                        encoded = self.tokenizer.encode_plus(
                            line,
                            max_length=self.max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                        result = {
                            "input_ids": encoded["input_ids"].squeeze(0),
                            "attention_mask": encoded["attention_mask"].squeeze(0),
                        }
                        self.current_position = mm.tell()
                        count += 1
                        if count % 1000 == 0:
                            logging.info(
                                f"Processed {count} abstracts. Current position: {self.current_position}"
                            )
                        yield result
                    except Exception as e:
                        logging.error(
                            f"Error processing abstract at position {mm.tell()}: {str(e)}"
                        )
            mm.close()
        logging.info(
            f"Finished processing {count} abstracts. Final position: {self.current_position}"
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Implement __getitem__ to make the dataset work with DataLoader.
        This is not efficient for large datasets, but helps with debugging.
        """
        with open(self.file_path, "r") as f:
            for i, line in enumerate(f):
                if i == index:
                    encoded = self.tokenizer.encode_plus(
                        line.strip(),
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    return {
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                    }
        raise IndexError("Index out of range")


class RobustDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability
        )

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        logging.info(f"DataCollator received {len(features)} features")
        if not features:
            logging.warning("DataCollator received an empty list of features")
            return {}

        try:
            # Filter out empty features
            valid_features = [
                f for f in features if f and "input_ids" in f and "attention_mask" in f
            ]

            if not valid_features:
                logging.warning("No valid features found after filtering")
                return {}

            # log details about each feature
            for i, feature in enumerate(valid_features):
                logging.info(f"Feature {i} keys: {feature.keys()}")
                for key, value in feature.items():
                    logging.info(f"Feature {i} {key} shape: {value.shape}")

            batch = self.data_collator(valid_features)
            logging.info(f"Successfully collated batch with keys: {batch.keys()}")
            for key, value in batch.items():
                logging.info(f"Collated batch {key} shape: {value.shape}")
            return batch
        except Exception as e:
            logging.error(f"Error in DataCollator: {str(e)}")
            logging.error(
                f"First feature: {features[0] if features else 'No features'}"
            )
            raise


class FinetuneStreamingCorpus(IterableDataset):
    """Class to create a Hugging Face dataset object from text corpus as an
    iterable
    """

    def __init__(self, dataset_file, tokenizer, data_collator, max_length=512):
        """Instantiate the streaming corpus class."""
        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.max_length = max_length

    def __iter__(self):
        """Iterate over the dataset file and yield tokenized examples"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start_position = 0
            end_position = None  # read until the end of the file
        else:
            # calculate start and end positions for this worker's shard
            start_position = worker_info.id * self._shard_size(worker_info.num_workers)
            end_position = start_position + self._shard_size(worker_info.num_workers)

        with open(self.dataset_file, "r", encoding="utf-8") as file_iterator:
            if start_position > 0:
                # skip lines up to the start position
                for _ in range(start_position):
                    next(file_iterator)

            for line_number, line in enumerate(file_iterator):
                if end_position is not None and line_number >= end_position:
                    break

                abstract = line.strip()
                tokenized = self.tokenizer(
                    abstract,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                yield {k: v.squeeze(0) for k, v in tokenized.items()}

    def _shard_size(self, num_workers):
        """Estimate shard size based on the total number of workers"""
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        return math.ceil(total_lines / num_workers)


class EmbeddingExtractorStreamingCorpus(IterableDataset):
    """Creates a Hugging Face dataset object for embedding extraction from text
    corpus.
    """

    def __init__(
        self,
        tokenized_files: List[str],
        max_length: int = 512,
        context_window: int = 128,
        batch_size: int = 16,
    ) -> None:
        """Instantiate the streaming corpus class."""
        self.tokenized_files = tokenized_files
        self.max_length = max_length
        self.context_window = context_window
        self.batch_size = batch_size
        self.gene_to_index: Dict[str, int] = {}
        self.load_gene_occurrences()

    def load_gene_occurrences(self) -> None:
        """Load the gene occurence dictionary from the tokenized files."""
        all_genes: set = set()
        for file in self.tokenized_files:
            with open(file, "rb") as f:
                _, gene_occurrences = pickle.load(f)
                all_genes.update(gene_occurrences.keys())

        self.gene_to_index = {gene: i for i, gene in enumerate(sorted(all_genes))}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset file and yield tokenized examples"""
        if worker_info := torch.utils.data.get_worker_info():
            per_worker = len(self.tokenized_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else None
            )
            files_to_process = self.tokenized_files[start:end]
        else:
            files_to_process = self.tokenized_files  # single worker

        for file in files_to_process:
            with open(file, "rb") as f:
                tokenized_abstracts, gene_occurrences = pickle.load(f)
            for abstract_idx, tokens in enumerate(tokenized_abstracts):
                yield from self.process_tokenized_abstract(
                    tokens, abstract_idx, gene_occurrences
                )

    def process_tokenized_abstract(
        self,
        tokens: List[int],
        abstract_idx: int,
        gene_occurrences: Dict[str, List[Tuple[int, int]]],
    ) -> Iterator[Dict[str, Any]]:
        """Process a tokenized abstract to extract gene embeddings."""
        all_occurrences = [
            (gene, idx)
            for gene, occs in gene_occurrences.items()
            for _, idx in occs
            if _ == abstract_idx
        ]

        for i in range(0, len(all_occurrences), self.batch_size):
            batch_occurrences = all_occurrences[i : i + self.batch_size]
            if not batch_occurrences:
                continue

            genes, token_indices = zip(*batch_occurrences)

            context_starts = np.maximum(
                0, np.array(token_indices) - self.context_window // 2
            )
            context_ends = np.minimum(
                len(tokens), np.array(token_indices) + self.context_window // 2
            )

            contexts = [
                tokens[start:end] for start, end in zip(context_starts, context_ends)
            ]

            padded_contexts = np.zeros((len(contexts), self.max_length), dtype=int)
            attention_masks = np.zeros((len(contexts), self.max_length), dtype=int)

            for i, context in enumerate(contexts):
                if len(context) > self.max_length:
                    context = context[: self.max_length]
                padded_contexts[i, : len(context)] = context
                attention_masks[i, : len(context)] = 1

            yield {
                "gene": genes,
                "input_ids": torch.tensor(padded_contexts),
                "attention_mask": torch.tensor(attention_masks),
            }

    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of tokenized examples."""
        return {
            "gene": [item["gene"] for item in batch],
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }

    def pad_or_truncate(self, token_ids: List[int]) -> List[int]:
        """Pad or truncate the token ids to the max length."""
        if len(token_ids) > self.max_length:
            return token_ids[: self.max_length]
        else:
            return token_ids + [0] * (self.max_length - len(token_ids))

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        print(
            f"Item {idx} shapes: input_ids: {item['input_ids'].shape}, attention_mask: {item['attention_mask'].shape}"
        )
        return item
