#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from a DeBERTa v3 model."""


import glob
from pathlib import Path
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from embedding_extractor import DeBERTaEmbeddingExtractor
from embedding_extractor import TokenizedDataset


def extract_embeddings(
    model_path: str, tokenized_files: List[str], output_dir: str
) -> None:
    """Extract embeddings."""
    extractor = DeBERTaEmbeddingExtractor(model_path)
    dataset = TokenizedDataset(tokenized_files)
    dataloader = DataLoader(
        dataset, batch_size=extractor.batch_size, num_workers=4, pin_memory=True
    )

    embeddings: Dict[str, Any] = {"averaged": {}, "cls": {}, "attention_weighted": {}}

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        genes = batch.pop("gene")
        avg_emb, cls_emb, att_emb = extractor.get_embeddings(batch)

        for i, gene in enumerate(genes):
            if gene not in embeddings["averaged"]:
                embeddings["averaged"][gene] = []
                embeddings["cls"][gene] = []
                embeddings["attention_weighted"][gene] = []

            embeddings["averaged"][gene].append(avg_emb[i])
            embeddings["cls"][gene].append(cls_emb[i])
            embeddings["attention_weighted"][gene].append(att_emb[i])

    # average embeddings for each gene
    for emb_type in embeddings:
        for gene in embeddings[emb_type]:
            embeddings[emb_type][gene] = np.mean(embeddings[emb_type][gene], axis=0)

    for emb_type, embeddings in embeddings.items():
        with open(Path(output_dir) / f"{emb_type}_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)


def main() -> None:
    """Main function"""
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta"
    output_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings"

    # load tokenized files
    tokenized_files = glob.glob(f"{data_dir}/tokenized_chunk_*.pkl")
    tokenized_files = [tokenized_files[0]]  # testing first for now
    print(f"Tokenized files: {tokenized_files}")

    # extract embeddings
    extract_embeddings(
        model_path=model_path, tokenized_files=tokenized_files, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
