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
    output_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings"

    # load tokens to get the total number of genes
    # gene_tokens = load_tokens(token_file)
    # total_genes = len(gene_tokens)

    # load tokenized files
    tokenized_files = glob.glob(f"{data_dir}/tokenized_chunk_*.pkl")
    tokenized_files = [tokenized_files[0]]  # testing first for now

    extractor = DeBERTaEmbeddingExtractor(model_path=model_path)
    extractor.process_all_chunks(tokenized_files=tokenized_files, output_dir=output_dir)

    # with open(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_avg_embeddings.pkl",
    #     "wb",
    # ) as file:
    #     pickle.dump(avg_emb, file)

    # with open(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_cls_embeddings.pkl",
    #     "wb",
    # ) as file:
    #     pickle.dump(cls_emb, file)

    # with open(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/deberta_att_embeddings.pkl",
    #     "wb",
    # ) as file:
    #     pickle.dump(att_emb, file)


if __name__ == "__main__":
    main()
