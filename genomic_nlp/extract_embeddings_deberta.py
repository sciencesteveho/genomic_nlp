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


def load_existing_embeddings(output_file: str) -> Dict[str, List[np.ndarray]]:
    """Load existing embeddings from a file."""
    try:
        with open(output_file, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}


def save_embeddings(embeddings: Dict[str, List[np.ndarray]], output_file: str) -> None:
    """Save embeddings to a file, merging with existing embeddings if present.

    Args:
        embeddings (Dict[str, List[np.ndarray]]): New embeddings to save.
        output_file (str): Path to the output file.
    """
    existing_embeddings = load_existing_embeddings(output_file)

    for gene, emb_list in embeddings.items():
        if gene in existing_embeddings:
            existing_embeddings[gene].extend(emb_list)
        else:
            existing_embeddings[gene] = emb_list

    with open(output_file, "wb") as f:
        pickle.dump(existing_embeddings, f)


def extract_embeddings(
    model_path: str, tokenized_files: List[str], output_file: str
) -> None:
    """Extract embeddings from tokenized files and save them.

    Args:
        model_path (str): Path to the DeBERTa model.
        tokenized_files (List[str]): List of paths to tokenized files.
        output_file (str): Path to the output file for saving embeddings.
    """
    extractor = DeBERTaEmbeddingExtractor(model_path)
    dataset = TokenizedDataset(tokenized_files)
    dataloader = DataLoader(dataset, batch_size=extractor.batch_size, num_workers=4)
    embeddings: Dict[str, List[np.ndarray]] = {}

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        genes = batch.pop("gene")
        input_ids = batch["input_ids"]
        emb = extractor.get_embeddings({"input_ids": input_ids})
        for i, gene in enumerate(genes):
            if gene not in embeddings:
                embeddings[gene] = []
            embeddings[gene].append(emb[i])

        if len(embeddings) > 10000:
            save_embeddings(embeddings, output_file)
            embeddings = {}

    save_embeddings(embeddings, output_file)


def process_final_embeddings(output_file: str) -> None:
    """Process the final embeddings by averaging them for each gene.

    Args:
        output_file (str): Path to the file containing embeddings.
    """
    embeddings = load_existing_embeddings(output_file)
    averaged_embeddings = {
        gene: np.mean(emb_list, axis=0) for gene, emb_list in embeddings.items()
    }

    with open(output_file, "wb") as f:
        pickle.dump(averaged_embeddings, f)

    # embeddings: Dict[str, Any] = {"averaged": {}, "cls": {}, "attention_weighted": {}}

    # for batch in tqdm(dataloader, desc="Extracting embeddings"):
    #     genes = batch.pop("gene")
    #     avg_emb, cls_emb, att_emb = extractor.get_embeddings(batch)

    #     for i, gene in enumerate(genes):
    #         if gene not in embeddings["averaged"]:
    #             embeddings["averaged"][gene] = []
    #             embeddings["cls"][gene] = []
    #             embeddings["attention_weighted"][gene] = []

    #         embeddings["averaged"][gene].append(avg_emb[i])
    #         embeddings["cls"][gene].append(cls_emb[i])
    #         embeddings["attention_weighted"][gene].append(att_emb[i])

    # # average embeddings for each gene
    # for emb_type in embeddings:
    #     for gene in embeddings[emb_type]:
    #         embeddings[emb_type][gene] = np.mean(embeddings[emb_type][gene], axis=0)

    # for emb_type, embeddings in embeddings.items():
    #     with open(Path(output_dir) / f"{emb_type}_embeddings.pkl", "wb") as f:
    #         pickle.dump(embeddings, f)


def main() -> None:
    """Main function"""
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined"
    model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta"
    output_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings"
    output_file = f"{output_dir}/deberta_averaged_embeddings.pkl"

    # load tokenized files
    tokenized_files = glob.glob(f"{data_dir}/tokenized_chunk_*.pkl")
    tokenized_files = [tokenized_files[0]]  # testing first for now
    print(f"Tokenized files: {tokenized_files}")

    # extract embeddings
    extract_embeddings(
        model_path=model_path, tokenized_files=tokenized_files, output_file=output_file
    )
    process_final_embeddings(output_file)


if __name__ == "__main__":
    main()
