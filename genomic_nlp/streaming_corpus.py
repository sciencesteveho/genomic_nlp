#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementations of streaming corpus classes for DeBERTa fine-tuning and
embedding extraction."""


import math
import pickle
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer  # type: ignore


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
    corpus
    """

    def __init__(
        self,
        tokenized_files: List[str],
        max_length: int = 512,
        context_window: int = 128,
    ) -> None:
        """Instantiate the streaming corpus class."""
        self.tokenized_files = tokenized_files
        self.max_length = max_length
        self.context_window = context_window
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
            self.tokenized_files = self.tokenized_files[start:end]

        for file in self.tokenized_files:
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
        for gene, occurrences in gene_occurrences.items():
            for abs_idx, token_idx in occurrences:
                if abs_idx == abstract_idx:
                    context_start = max(0, token_idx - self.context_window // 2)
                    context_end = min(len(tokens), token_idx + self.context_window // 2)
                    context = tokens[context_start:context_end]

                    padded_context = self.pad_or_truncate(context)
                    attention_mask = [1] * len(context) + [0] * (
                        self.max_length - len(context)
                    )

                    yield {
                        "gene": gene,
                        "input_ids": torch.tensor(padded_context),
                        "attention_mask": torch.tensor(attention_mask),
                    }

    def pad_or_truncate(self, token_ids: List[int]) -> List[int]:
        """Pad or truncate the token ids to the max length."""
        if len(token_ids) > self.max_length:
            return token_ids[: self.max_length]
        else:
            return token_ids + [0] * (self.max_length - len(token_ids))
