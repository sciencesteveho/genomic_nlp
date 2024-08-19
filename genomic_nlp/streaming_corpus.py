#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementations of streaming corpus classes for DeBERTa fine-tuning and
embedding extraction."""


import math
import re
from typing import Any, Dict, Iterator, List

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
        dataset_file: str,
        tokenizer: PreTrainedTokenizer,
        genes: List[str],
        max_length: int = 512,
        context_window: int = 128,
    ) -> None:
        """Instantiate the streaming corpus class."""
        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.genes = genes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset file and yield tokenized examples"""
        worker_info = torch.utils.data.get_worker_info()

        with open(self.dataset_file, "r", encoding="utf-8") as file:
            if worker_info:
                file_size = file.seek(0, 2)  # file size
                chunk_size = file_size // worker_info.num_workers
                start = worker_info.id * chunk_size  # find start position for worker
                end = (
                    start + chunk_size
                    if worker_info.id < worker_info.num_workers - 1
                    else file_size
                )

                # move to start position
                file.seek(start)
                if start > 0:
                    file.readline()

                for line in iter(file.readline, ""):
                    if file.tell() > end:
                        break
                    yield from self.process_line(line)
            else:
                # process entire file if single worker
                for line in file:
                    yield from self.process_line(line)

    def process_line(self, line: str) -> Iterator[Dict[str, Any]]:
        """Process a line of text to extract embeddings"""
        # tokenize the abstract
        abstract = line.strip()
        tokenized_abstract = self.tokenizer(
            abstract, padding=False, truncation=False, return_tensors="pt"
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            tokenized_abstract["input_ids"][0]
        )

        for i, token in enumerate(tokens):
            if token.casefold() in (gene.casefold() for gene in self.genes):
                gene_mention = token
                gene_name = next(
                    gene for gene in self.genes if gene.casefold() == token.casefold()
                )

                # extract context
                context_start = max(0, i - self.context_window // 2)
                context_end = min(len(tokens), i + self.context_window // 2)

                # get tokenized context
                context_slice = {
                    k: v[0, context_start:context_end]
                    for k, v in tokenized_abstract.items()
                }

                # pad or truncate the context to the required length
                padded_context = self.tokenizer.pad(
                    context_slice,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                yield {
                    "gene": gene_name,
                    "mention": gene_mention,
                    "context": self.tokenizer.decode(context_slice["input_ids"]),
                    **{k: v.squeeze(0) for k, v in padded_context.items()},
                }
