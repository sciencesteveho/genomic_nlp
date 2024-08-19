#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementations of streaming corpus classes for DeBERTa fine-tuning and
embedding extraction."""


import math

import torch
from torch.utils.data import IterableDataset


class FinetuneStreamingCorpus(IterableDataset):
    """Class to create a Hugging Face dataset object from text corpus as an iterable"""

    def __init__(self, dataset_file, tokenizer, data_collator, max_length=512):
        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.max_length = max_length

    def __iter__(self):
        """Iterate over the dataset file and yield tokenized examples"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the full iterator
            start_position = 0
            end_position = None  # Read until the end of the file
        else:
            # Calculate start and end positions for this worker's shard
            start_position = worker_info.id * self._shard_size(worker_info.num_workers)
            end_position = start_position + self._shard_size(worker_info.num_workers)

        # Opens the file, ensuring it's closed after iteration
        with open(self.dataset_file, "r", encoding="utf-8") as file_iterator:
            if start_position > 0:
                # Skip lines up to the start position
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
