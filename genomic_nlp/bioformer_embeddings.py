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
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForTokenClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore


class AbstractDataset(Dataset):
    """Dataset for efficient loading and tokenization of abstracts."""

    def __init__(
        self, abstracts: List[str], tokenizer: AutoTokenizer, max_length: int = 512
    ):
        """Initialize dataset."""
        self.abstracts = abstracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return number of abstracts."""
        return len(self.abstracts)

    def __getitem__(self, idx):
        """Return tokenized abstract."""
        return self.tokenizer(
            self.abstracts[idx],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )


def load_abstracts(file_path: str) -> List[str]:
    """Load abstracts from text file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def get_gene_embeddings(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, List[np.ndarray]]:
    """Tokenize abstracts and extract embeddings for anything NER tags as a
    gene.
    """
    gene_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    inputs = {k: v.squeeze(1).to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # full hidden states
    hidden_states = outputs.hidden_states[-1]

    # predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)

    # process each abstract
    for abstract_inputs, abstract_predictions, abstract_hidden_states in zip(
        inputs["input_ids"], predictions, hidden_states
    ):
        current_gene = []
        current_embeddings = []
        for k, (token, pred) in enumerate(zip(abstract_inputs, abstract_predictions)):
            if pred in [0, 1]:  # 0 for "B-bio", 1 for "I-bio"
                current_gene.append(token.item())
                current_embeddings.append(abstract_hidden_states[k])
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

    dataset = AbstractDataset(abstracts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    device = next(model.parameters()).device

    with tqdm(
        total=len(abstracts), desc="Processing abstracts", unit="abstract"
    ) as pbar:
        for batch in dataloader:
            embeddings = get_gene_embeddings(model, tokenizer, batch, device)
            for gene, embs in embeddings.items():
                all_gene_embeddings[gene].extend(embs)
            pbar.update(batch["input_ids"].size(0))

    return {gene: np.mean(embs, axis=0) for gene, embs in all_gene_embeddings.items()}


def main() -> None:
    """Get those embeddings!"""
    model_name = "bioformers/bioformer-8L-bc2gm"
    bioformer_dir = (
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/bioformer"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, output_hidden_states=True
    )

    # set up device(s)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs")
        model = DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    abstracts_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
    abstracts = load_abstracts(abstracts_file)

    # set batch size
    batch_size_per_gpu = 16
    total_batch_size = batch_size_per_gpu * num_gpus

    gene_embeddings = process_abstracts(
        model=model,
        tokenizer=tokenizer,
        abstracts=abstracts,
        batch_size=total_batch_size,
    )

    np.savez(
        f"{bioformer_dir}/gene_bioformer_embeddings.npy",
        **gene_embeddings,
    )

    print(f"Generated embeddings for {len(gene_embeddings)} genes.")
    print(f"Embedding shape: {next(iter(gene_embeddings.values())).shape}")

    # save gene names to a text file
    with open(f"{bioformer_dir}/identified_genes.txt", "w") as f:
        for gene in gene_embeddings.keys():
            f.write(f"{gene}\n")


if __name__ == "__main__":
    main()
