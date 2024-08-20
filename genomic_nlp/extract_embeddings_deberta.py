#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import pickle
from typing import List

import torch
from tqdm import tqdm  # type: ignore

from embedding_extractor import DeBERTaEmbeddingExtractor
from streaming_corpus import EmbeddingExtractorStreamingCorpus


def load_tokens(filename: str) -> List[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return [line.strip().lower() for line in f]


def main() -> None:
    """Main function"""

    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    gene_tokens = load_tokens(token_file)

    # abstracts_dir = f"{args.root_dir}/data"
    trained_model = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta/model.safetensors"
    abstracts = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"

    extractor = DeBERTaEmbeddingExtractor(trained_model)
    dataset = EmbeddingExtractorStreamingCorpus(
        dataset_file=abstracts, tokenizer=extractor.tokenizer, genes=gene_tokens
    )
    avg_emb, cls_emb, att_emb = extractor.process_dataset(dataset)

    with open(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_avg_embeddings.pkl",
        "wb",
    ) as file:
        pickle.dump(avg_emb, file)

    with open(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_cls_embeddings.pkl",
        "wb",
    ) as file:
        pickle.dump(cls_emb, file)

    with open(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_att_embeddings.pkl",
        "wb",
    ) as file:
        pickle.dump(att_emb, file)


if __name__ == "__main__":
    main()
