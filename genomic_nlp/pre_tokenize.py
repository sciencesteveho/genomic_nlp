#!/usr/bin/env python
# -*- coding: utf-8 -*-'


"""Pre-tokenize text to speed up embedding extraction."""


import multiprocessing as mp
import os
import pickle
from typing import Dict, List, Tuple

import psutil  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore


def load_tokens(filename: str) -> List[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f]


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
    chunk: List[str],
    tokenizer: DebertaV2Tokenizer,
    genes: List[str],
    output_file: str,
) -> None:
    """function to process a chunk of text and save it"""
    tokenized_chunk: List[List[int]] = []
    gene_occurrences: Dict[str, List[Tuple[int, int]]] = {gene: [] for gene in genes}

    for abstract_idx, abstract in enumerate(chunk):
        tokens = tokenizer.encode(
            abstract.strip(), add_special_tokens=True, truncation=True, max_length=512
        )
        tokenized_chunk.append(tokens)

        for token_idx, token_id in enumerate(tokens):
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            if token.casefold() in (gene.casefold() for gene in genes):
                gene = next(g for g in genes if g.casefold() == token.casefold())
                gene_occurrences[gene].append((abstract_idx, token_idx))

    with open(output_file, "wb") as f:
        pickle.dump((tokenized_chunk, gene_occurrences), f)


def main() -> None:
    """main func"""
    model_name: str = "microsoft/deberta-v3-base"
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    data_dir: str = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    input_file: str = (
        f"{data_dir}/tokens_cleaned_abstracts_casefold_finetune_combined_onlygenetokens_nosyn_debertaext.txt"
    )

    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"

    # load genes of interest
    genes = load_tokens(token_file)

    # read all lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()

    # set up chunks by cores
    num_processes = get_physical_cores()
    chunk_size: int = len(lines) // num_processes

    # process chunks
    chunks: List[List[str]] = [
        lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)
    ]

    # prepare arguments for multiprocessing
    chunk_args: List[Tuple[List[str], DebertaV2Tokenizer, List[str], str]] = [
        (chunk, tokenizer, genes, f"{data_dir}/tokenized_chunk_{i}.pkl")
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
