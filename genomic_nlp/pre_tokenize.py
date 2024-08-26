#!/usr/bin/env python
# -*- coding: utf-8 -*-'


"""Pre-tokenize text to speed up embedding extraction."""


import multiprocessing as mp
import pickle
from typing import Dict, List, Set, Tuple

import psutil  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    return psutil.cpu_count(logical=False) - 1


def tokenize_text(text: str, tokenizer: DebertaV2Tokenizer) -> List[int]:
    """Function to tokenize text"""
    return tokenizer.encode(
        text, add_special_tokens=True, truncation=True, max_length=512
    )


def process_and_save_chunk(
    abstract_chunk: List[str],
    tokenizer: DebertaV2Tokenizer,
    genes: Set[str],
    output_file: str,
) -> None:
    """Function to process a chunk of text and save it"""
    tokenized_abstracts: List[List[int]] = []
    gene_occurrences: Dict[str, List[Tuple[int, int]]] = {gene: [] for gene in genes}

    batch_size = 32
    for chunk_start in tqdm(
        range(0, len(abstract_chunk), batch_size), desc="Processing abstracts"
    ):
        abstract_batch = abstract_chunk[chunk_start : chunk_start + batch_size]
        encoded_batch = tokenizer.batch_encode_plus(
            abstract_batch,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        for batch_index, (input_ids, token_type_ids) in enumerate(
            zip(encoded_batch["input_ids"], encoded_batch["token_type_ids"])
        ):
            abstract_index = chunk_start + batch_index
            token_ids = input_ids.tolist()
            tokenized_abstracts.append(token_ids)

            # match genes
            decoded_text = tokenizer.decode(token_ids)
            lowercase_text = decoded_text.lower()
            for gene in genes:
                start_position = 0
                while True:
                    gene_position = lowercase_text.find(gene, start_position)
                    if gene_position == -1:
                        break
                    token_index = len(
                        tokenizer.encode(
                            decoded_text[:gene_position], add_special_tokens=False
                        )
                    )
                    gene_occurrences[gene].append((abstract_index, token_index))
                    start_position = gene_position + len(gene)

    with open(output_file, "wb") as output:
        pickle.dump((tokenized_abstracts, gene_occurrences), output)


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
    # num_processes = get_physical_cores()
    num_processes = 7
    chunk_size: int = len(lines) // num_processes

    # process chunks
    chunks: List[List[str]] = [
        lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)
    ]

    # prepare arguments for multiprocessing
    chunk_args: List[Tuple[List[str], DebertaV2Tokenizer, Set[str], str]] = [
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
