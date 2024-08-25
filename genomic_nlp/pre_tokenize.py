#!/usr/bin/env python
# -*- coding: utf-8 -*-'


"""Pre-tokenize text to speed up embedding extraction."""


import multiprocessing as mp
import os
import pickle
from typing import List, Tuple

import psutil  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    return psutil.cpu_count(logical=False) - 1


def tokenize_text(text: str, tokenizer: DebertaV2Tokenizer) -> List[int]:
    """function to tokenize text"""
    return tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=512
    )


def process_and_save_chunk(
    chunk: List[str], tokenizer: DebertaV2Tokenizer, output_file: str
) -> None:
    """function to process a chunk of text and save it"""
    tokenized_chunk = [tokenize_text(line.strip(), tokenizer) for line in chunk]
    with open(output_file, "wb") as f:
        pickle.dump(tokenized_chunk, f)


def main() -> None:
    """main func"""
    model_name: str = "microsoft/deberta-v3-base"
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    input_file: str = (
        f"{data_dir}/tokens_cleaned_abstracts_casefold_finetune_combined_onlygenetokens_nosyn_debertaext.txt"
    )

    # read all lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()

    # num_processes: int = get_physical_cores()
    num_processes = 8

    # chunk abstracts
    chunk_size: int = len(lines) // num_processes
    chunks: List[List[str]] = [
        lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)
    ]

    # prepare arguments for multiprocessing
    chunk_args: List[Tuple[List[str], DebertaV2Tokenizer, str]] = [
        (chunk, tokenizer, f"{data_dir}/tokenized_chunk_{i}.pkl")
        for i, chunk in enumerate(chunks)
    ]

    # tokenize and save chunks
    with mp.Pool(processes=num_processes) as pool:
        list(
            tqdm(
                pool.starmap(process_and_save_chunk, chunk_args),
                total=len(chunks),
            )
        )

    print(f"Tokenized data saved to {data_dir}")


if __name__ == "__main__":
    main()
