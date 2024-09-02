#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Embedding extraction on abstract corpus via Bioformer."""


from collections import defaultdict
import os
from pathlib import Path
import pickle
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForTokenClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore


def load_abstracts(file_path: str) -> List[str]:
    """Load abstracts from text file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def get_gene_embeddings(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    abstract: str,
    max_length: int = 512,
) -> Dict[str, List[np.ndarray]]:
    """Tokenize abstracts and extract embeddings for anything NER tags as a
    gene.
    """
    gene_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)

    # tokenize the abstract
    inputs = tokenizer(
        abstract,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # get full sequence hidden states
    hidden_states: torch.Tensor = outputs.hidden_states[-1][0]

    # predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)[0]

    # extract embeddings for tokens classified as genes
    current_gene = []
    current_embeddings = []
    for i, (token, pred) in enumerate(zip(inputs["input_ids"][0], predictions)):
        if pred in [0, 1]:  # 0 for "B-bio", 1 for "I-bio"
            current_gene.append(token)
            current_embeddings.append(hidden_states[i])
        elif current_gene:
            gene_name = tokenizer.decode(current_gene).strip()
            gene_embedding = torch.mean(torch.stack(current_embeddings), dim=0)
            gene_embeddings[gene_name].append(gene_embedding.cpu().numpy())
            current_gene = []
            current_embeddings = []

    # handle case where gene mention is at the end of the sequence
    if current_gene:
        gene_name = tokenizer.decode(current_gene).strip()
        gene_embedding = torch.mean(torch.stack(current_embeddings), dim=0)
        gene_embeddings[gene_name].append(gene_embedding.cpu().numpy())

    return gene_embeddings


def process_abstracts(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    abstracts: List[str],
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Process abstracts in batches and return average embeddings for each gene."""
    all_gene_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)

    with tqdm(
        total=len(abstracts), desc="Processing abstracts", unit="abstract"
    ) as pbar:
        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i : i + batch_size]
            for abstract in batch:
                embeddings = get_gene_embeddings(model, tokenizer, abstract)
                for gene, embs in embeddings.items():
                    all_gene_embeddings[gene].extend(embs)
                pbar.update(1)

    return {gene: np.mean(embs, axis=0) for gene, embs in all_gene_embeddings.items()}


def main() -> None:
    """Get those embeddings!"""
    model_name = "bioformers/bioformer-8L-bc2gm"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, output_hidden_states=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    abstracts_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
    abstracts = load_abstracts(abstracts_file)

    gene_embeddings = process_abstracts(model, tokenizer, abstracts)

    np.savez(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/bioformer/gene_bioformer_embeddings.npy",
        **gene_embeddings,
    )

    print(f"Generated embeddings for {len(gene_embeddings)} genes.")
    print(f"Embedding shape: {next(iter(gene_embeddings.values())).shape}")


if __name__ == "__main__":
    main()
