#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import glob
from pathlib import Path
import pickle
from typing import Dict, List

import torch
from tqdm import tqdm  # type: ignore

from embedding_extractor import DeBERTaEmbeddingExtractor


def main() -> None:
    """Main function"""
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta"
    output_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings"

    # load tokenized files
    tokenized_files = glob.glob(f"{data_dir}/tokenized_chunk_*.pkl")
    tokenized_files = [tokenized_files[0]]  # testing first for now

    # instantiate the embedding extractor
    extractor = DeBERTaEmbeddingExtractor(model_path=model_path)

    with torch.inference_mode():
        final_embeddings: Dict[str, Dict[str, torch.Tensor]]
        gene_counts: Dict[str, int]
        final_embeddings, gene_counts = extractor.process_chunks_for_embeddings(
            tokenized_files
        )

    for embed_type, embeddings in final_embeddings.items():
        with open(Path(output_dir) / f"{embed_type}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    with open(Path(output_dir) / "gene_counts.pkl", "wb") as f:
        pickle.dump(gene_counts, f)


if __name__ == "__main__":
    main()
