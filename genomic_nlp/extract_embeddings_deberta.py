#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import pickle

import torch
from tqdm import tqdm  # type: ignore

from embedding_extractor import DeBERTaEmbeddingExtractor
from gene_extraction_graph import combine_synonyms
from streaming_corpus import EmbeddingExtractorStreamingCorpus
from utils import gencode_genes


def main() -> None:
    """Main function"""
    genes = gencode_genes(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/gencode.v45.basic.annotation.gtf"
    )

    with open(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms_nocasefold.pkl",
        "rb",
    ) as file:
        hgnc_synonyms = pickle.load(file)

    combined_genes = combine_synonyms(hgnc_synonyms, genes)
    gene_tokens = list(combined_genes)

    # abstracts_dir = f"{args.root_dir}/data"
    trained_model = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta/model.safetensors"
    abstracts = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
