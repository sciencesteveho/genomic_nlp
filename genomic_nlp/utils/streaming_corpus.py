#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementations of streaming corpus classes for DeBERTa fine-tuning and
embedding extraction."""


import linecache
import logging
import math
import os
import pickle
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer  # type: ignore


class MLMTextDataset(Dataset):
    """Map-style dataset for mlm."""

    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ) -> None:
        """Initialize the MLMTextDataset."""
        super().__init__()
        self.file_path: str = file_path
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_length

        logging.info(f"Counting lines in: {file_path}")
        with open(file_path, "rb") as f:
            self.num_lines: int = 0
            chunk_size: int = 1024 * 1024
            while chunk := f.read(chunk_size):
                self.num_lines += chunk.count(b"\n")

        logging.info(f"{self.num_lines} lines found in {file_path}.")

    def __len__(self) -> int:
        """Return the number of lines in the file."""
        return self.num_lines

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get the tokenized example for a given index."""
        raw_line: str = linecache.getline(self.file_path, idx + 1)
        if not raw_line:
            raise IndexError(f"line index {idx} out of range in {self.file_path}")

        line: str = raw_line.strip()
        if not line:
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }

        encoded = self.tokenizer.encode_plus(
            line,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
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
        self.num_lines = 3_370_369

        self.current_position = 0
        self.iteration_count = 0

        # simple caching dict: line_index -> tokenized result
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}

        logging.info(
            f"Initialized StreamingCorpus with file: {file_path}, total lines: {self.num_lines}"
        )

    def __len__(self) -> int:
        """Return the number of lines in the file."""
        return self.num_lines

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Create an iterator over the corpus."""

        self.iteration_count += 1

        # worker info from DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # figure out how many replicas and which rank.
        if torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 1
            rank = 0

        # distribute lines
        total_shards = num_workers * num_replicas
        lines_per_shard = self.num_lines // total_shards

        # shard index from [0..(total_shards-1)]
        shard_index = rank * num_workers + worker_id
        start_line = shard_index * lines_per_shard
        # if last shard, go to the end, else just do + lines_per_shard
        if shard_index == (total_shards - 1):
            end_line = self.num_lines
        else:
            end_line = start_line + lines_per_shard

        if self.current_position < start_line or self.current_position >= end_line:
            self.current_position = start_line

        logging.info(
            f"Iteration {self.iteration_count}: Worker {worker_id}, rank {rank} "
            f"processing lines {self.current_position} to {end_line}"
        )

        return self.read_abstracts(self.current_position, end_line)

    def read_abstracts(self, start: int, end: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Read and tokenize abstracts from the file."""
        count = 0

        for i in range(start, end):
            if i in self.cache:
                yield self.cache[i]
                count += 1
                continue

            # linecache is 1-based, so use i+1
            raw_line = linecache.getline(self.file_path, i + 1)
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line:
                continue

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
                self.cache[i] = result
                count += 1

                if count % 1000 == 0:
                    logging.info(
                        f"Processed {count} abstracts. Current line index: {i}"
                    )
                yield result

            except Exception as e:
                logging.error(f"Error processing abstract at line {i}: {str(e)}")

        logging.info(
            f"Finished processing {count} abstracts. Final line index: {end - 1}"
        )
        self.current_position = end


# class StreamingCorpus(IterableDataset):
#     """Custom streaming dataset for abstracts."""

#     def __init__(
#         self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512
#     ):
#         """Initialize the streaming corpus class."""
#         self.file_path = file_path
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.file_size = os.path.getsize(file_path)
#         self.num_lines = 3889578
#         self.current_position = 0
#         self.iteration_count = 0
#         self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
#         logging.info(
#             f"Initialized StreamingCorpus with file: {file_path}, size: {self.file_size}"
#         )

#     def __len__(self) -> int:
#         """Return the number of lines in the file."""
#         return self.num_lines

#     def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
#         """Create an iterator over the corpus."""
#         self.iteration_count += 1
#         worker_info = torch.utils.data.get_worker_info()
#         num_workers = worker_info.num_workers if worker_info else 1
#         worker_id = worker_info.id if worker_info else 0

#         # distribute data across processes for distributed training
#         if torch.distributed.is_initialized():
#             num_replicas = torch.distributed.get_world_size()
#             rank = torch.distributed.get_rank()
#         else:
#             num_replicas = 1
#             rank = 0

#         # calculate the range for this worker and process
#         per_worker = self.file_size // (num_workers * num_replicas)
#         start = (rank * num_workers + worker_id) * per_worker
#         end = (
#             start + per_worker
#             if (rank * num_workers + worker_id + 1) < (num_workers * num_replicas)
#             else self.file_size
#         )

#         # reset position if we've reached the end of the file
#         if self.current_position >= end:
#             self.current_position = start

#         logging.info(
#             f"Iteration {self.iteration_count}: Worker {worker_id} (rank {rank}) processing range: {self.current_position} - {end}"
#         )

#         return self.read_abstracts(self.current_position, end)

#     def read_abstracts(self, start: int, end: int) -> Iterator[Dict[str, torch.Tensor]]:
#         """Read and tokenize abstracts from the file."""
#         count = 0
#         for i in range(start, end):
#             if i in self.cache:
#                 yield self.cache[i]
#                 count += 1
#                 continue

#             if line := linecache.getline(self.file_path, i).strip():
#                 try:
#                     encoded = self.tokenizer.encode_plus(
#                         line,
#                         max_length=self.max_length,
#                         padding="max_length",
#                         truncation=True,
#                         return_tensors="pt",
#                     )
#                     result = {
#                         "input_ids": encoded["input_ids"].squeeze(0),
#                         "attention_mask": encoded["attention_mask"].squeeze(0),
#                     }
#                     self.cache[i] = result
#                     count += 1
#                     if count % 1000 == 0:
#                         logging.info(
#                             f"Processed {count} abstracts. Current position: {i}"
#                         )
#                     yield result
#                 except Exception as e:
#                     logging.error(f"Error processing abstract at line {i}: {str(e)}")
#         logging.info(f"Finished processing {count} abstracts. Final position: {end-1}")


# class FinetuneStreamingCorpus(IterableDataset):
#     """Class to create a Hugging Face dataset object from text corpus as an
#     iterable
#     """

#     def __init__(self, dataset_file, tokenizer, data_collator, max_length=512):
#         """Instantiate the streaming corpus class."""
#         self.dataset_file = dataset_file
#         self.tokenizer = tokenizer
#         self.data_collator = data_collator
#         self.max_length = max_length

#     def __iter__(self):
#         """Iterate over the dataset file and yield tokenized examples"""
#         worker_info = torch.utils.data.get_worker_info()

#         if worker_info is None:
#             start_position = 0
#             end_position = None  # read until the end of the file
#         else:
#             # calculate start and end positions for this worker's shard
#             start_position = worker_info.id * self._shard_size(worker_info.num_workers)
#             end_position = start_position + self._shard_size(worker_info.num_workers)

#         with open(self.dataset_file, "r", encoding="utf-8") as file_iterator:
#             if start_position > 0:
#                 # skip lines up to the start position
#                 for _ in range(start_position):
#                     next(file_iterator)

#             for line_number, line in enumerate(file_iterator):
#                 if end_position is not None and line_number >= end_position:
#                     break

#                 abstract = line.strip()
#                 tokenized = self.tokenizer(
#                     abstract,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=self.max_length,
#                     return_tensors="pt",
#                 )
#                 yield {k: v.squeeze(0) for k, v in tokenized.items()}

#     def _shard_size(self, num_workers):
#         """Estimate shard size based on the total number of workers"""
#         with open(self.dataset_file, "r", encoding="utf-8") as f:
#             total_lines = sum(1 for _ in f)

#         return math.ceil(total_lines / num_workers)


# class EmbeddingExtractorStreamingCorpus(IterableDataset):
#     """Creates a Hugging Face dataset object for embedding extraction from text
#     corpus.
#     """

#     def __init__(
#         self,
#         tokenized_files: List[str],
#         max_length: int = 512,
#         context_window: int = 128,
#         batch_size: int = 16,
#     ) -> None:
#         """Instantiate the streaming corpus class."""
#         self.tokenized_files = tokenized_files
#         self.max_length = max_length
#         self.context_window = context_window
#         self.batch_size = batch_size
#         self.gene_to_index: Dict[str, int] = {}
#         self.load_gene_occurrences()

#     def load_gene_occurrences(self) -> None:
#         """Load the gene occurence dictionary from the tokenized files."""
#         all_genes: set = set()
#         for file in self.tokenized_files:
#             with open(file, "rb") as f:
#                 _, gene_occurrences = pickle.load(f)
#                 all_genes.update(gene_occurrences.keys())

#         self.gene_to_index = {gene: i for i, gene in enumerate(sorted(all_genes))}

#     def __iter__(self) -> Iterator[Dict[str, Any]]:
#         """Iterate over the dataset file and yield tokenized examples"""
#         if worker_info := torch.utils.data.get_worker_info():
#             per_worker = len(self.tokenized_files) // worker_info.num_workers
#             start = worker_info.id * per_worker
#             end = (
#                 start + per_worker
#                 if worker_info.id < worker_info.num_workers - 1
#                 else None
#             )
#             files_to_process = self.tokenized_files[start:end]
#         else:
#             files_to_process = self.tokenized_files  # single worker

#         for file in files_to_process:
#             with open(file, "rb") as f:
#                 tokenized_abstracts, gene_occurrences = pickle.load(f)
#             for abstract_idx, tokens in enumerate(tokenized_abstracts):
#                 yield from self.process_tokenized_abstract(
#                     tokens, abstract_idx, gene_occurrences
#                 )

#     def process_tokenized_abstract(
#         self,
#         tokens: List[int],
#         abstract_idx: int,
#         gene_occurrences: Dict[str, List[Tuple[int, int]]],
#     ) -> Iterator[Dict[str, Any]]:
#         """Process a tokenized abstract to extract gene embeddings."""
#         all_occurrences = [
#             (gene, idx)
#             for gene, occs in gene_occurrences.items()
#             for _, idx in occs
#             if _ == abstract_idx
#         ]

#         for i in range(0, len(all_occurrences), self.batch_size):
#             batch_occurrences = all_occurrences[i : i + self.batch_size]
#             if not batch_occurrences:
#                 continue

#             genes, token_indices = zip(*batch_occurrences)

#             context_starts = np.maximum(
#                 0, np.array(token_indices) - self.context_window // 2
#             )
#             context_ends = np.minimum(
#                 len(tokens), np.array(token_indices) + self.context_window // 2
#             )

#             contexts = [
#                 tokens[start:end] for start, end in zip(context_starts, context_ends)
#             ]

#             padded_contexts = np.zeros((len(contexts), self.max_length), dtype=int)
#             attention_masks = np.zeros((len(contexts), self.max_length), dtype=int)

#             for i, context in enumerate(contexts):
#                 if len(context) > self.max_length:
#                     context = context[: self.max_length]
#                 padded_contexts[i, : len(context)] = context
#                 attention_masks[i, : len(context)] = 1

#             yield {
#                 "gene": genes,
#                 "input_ids": torch.tensor(padded_contexts),
#                 "attention_mask": torch.tensor(attention_masks),
#             }

#     def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Collate a batch of tokenized examples."""
#         return {
#             "gene": [item["gene"] for item in batch],
#             "input_ids": torch.stack([item["input_ids"] for item in batch]),
#             "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
#         }

#     def pad_or_truncate(self, token_ids: List[int]) -> List[int]:
#         """Pad or truncate the token ids to the max length."""
#         if len(token_ids) > self.max_length:
#             return token_ids[: self.max_length]
#         else:
#             return token_ids + [0] * (self.max_length - len(token_ids))

#     def __getitem__(self, idx):
#         item = super().__getitem__(idx)
#         print(
#             f"Item {idx} shapes: input_ids: {item['input_ids'].shape}, attention_mask: {item['attention_mask'].shape}"
#         )
#         return item
