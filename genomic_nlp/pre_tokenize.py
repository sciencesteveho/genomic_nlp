#!/usr/bin/env python
# -*- coding: utf-8 -*-'


"""Pre-tokenize text to speed up embedding extraction."""


import multiprocessing as mp
import pickle
from typing import List

import psutil  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    return psutil.cpu_count(logical=False) - 1


def tokenize_text(
    text: str,
    tokenizer: DebertaV2Tokenizer,
) -> list:
    """function to tokenize text"""
    return tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=512
    )


def process_chunk(
    chunk: list,
    tokenizer: DebertaV2Tokenizer,
) -> list:
    """function to process a chunk of text"""
    return [tokenize_text(line.strip(), tokenizer) for line in chunk]


def main() -> None:
    """main func"""
    model_name = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/"

    input_file = f"{data_dir}/tokens_cleaned_abstracts_casefold_finetune_combined_onlygenetokens_nosyn_debertaext.txt"
    output_file = f"{data_dir}/tokenized_abs_deberta.pkl"

    # read all lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # determine number of processes
    num_processes = get_physical_cores()

    # split the data into chunks
    chunk_size = len(lines) // num_processes
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # use multiprocessing to tokenize
    with mp.Pool(processes=num_processes) as pool:
        tokenized_chunks = list(
            tqdm(
                pool.imap(lambda chunk: process_chunk(chunk, tokenizer), chunks),
                total=len(chunks),
            )
        )

    # flatten the list of chunks
    tokenized_data = [item for sublist in tokenized_chunks for item in sublist]

    # save the tokenized data
    with open(output_file, "wb") as f:
        pickle.dump(tokenized_data, f)

    print(f"tokenized data saved to {output_file}")


if __name__ == "__main__":
    main()
