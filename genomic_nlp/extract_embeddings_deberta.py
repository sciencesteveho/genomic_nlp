#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import glob
import pickle
from typing import List

from tqdm import tqdm  # type: ignore

from embedding_extractor import DeBERTaEmbeddingExtractor
from pre_tokenize import load_tokens


def main() -> None:
    """Main function"""
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta"
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"

    # load tokens to get the total number of genes
    gene_tokens = load_tokens(token_file)
    total_genes = len(gene_tokens)

    # load tokenized files
    tokenized_files = glob.glob(f"{data_dir}/tokenized_chunk_*.pkl")

    extractor = DeBERTaEmbeddingExtractor(model_path=model_path)
    avg_emb, cls_emb, att_emb = extractor.process_dataset(
        tokenized_files=tokenized_files, total_genes=total_genes
    )

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
