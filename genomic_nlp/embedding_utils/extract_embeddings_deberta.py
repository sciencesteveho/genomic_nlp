#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import glob
from pathlib import Path
import pickle
from typing import Any, Dict, List, Set

from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore

from embedding_extractors import DeBERTaEmbeddingExtractor
from embedding_extractors import TokenizedDataset


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def main() -> None:
    """Main function"""
    # set up paths
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta"
    output_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings"
    output_file = f"{output_dir}/deberta_averaged_embeddings.pkl"
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    abstract_file = "tokens_cleaned_abstracts_casefold_finetune_combined_onlygenetokens_nosyn_debertaext.txt"

    # load genes of interest
    genes = load_tokens(token_file)

    # set up tokenizer
    tokenizer_name = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_name)

    # prepare dataset
    abstracts = TokenizedDataset(
        abstract_file=f"{data_dir}/{abstract_file}",
        tokenizer=tokenizer,
        genes=genes,
    )

    # instantiate extractor
    extractor = DeBERTaEmbeddingExtractor(
        model_path=model_path,
        dataset=abstracts,
    )
    embeddings = extractor.extract_embeddings()

    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)


if __name__ == "__main__":
    main()
