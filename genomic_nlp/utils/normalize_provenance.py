#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Use Flair/HunFlair2 to normalize gene and disease names in GDA and cancer
provenance data."""


from typing import Tuple

import flair  # type: ignore
from flair.data import Sentence  # type: ignore
from flair.models import EntityMentionLinker  # type: ignore
from flair.nn import Classifier  # type: ignore
import pandas as pd
import torch  # type: ignore
from tqdm import tqdm  # type: ignore

from genomic_nlp.abstracts.gene_entity_normalization import extract_normalized_name


def load_flair_models() -> Tuple[Classifier, EntityMentionLinker, EntityMentionLinker]:
    """Load flair models for normalizing gene and disease entities."""
    return (
        Classifier.load("hunflair2"),
        EntityMentionLinker.load("gene-linker"),
        EntityMentionLinker.load("disease-linker"),
    )


def normalize_cancer_provenance_batched(
    provenance: str,
    tagger: Classifier,
    gene_linker: EntityMentionLinker,
    out_file: str,
    batch_size: int = 128,
) -> None:
    """Normalize cancer provenance data using Flair in batches.

    Args:
        provenance (str): Path to input provenance data
        tagger (Classifier): hunflair2 tagger
        gene_linker (EntityMentionLinker): hunflair2 linker
        out_file (str): Where to write the output TSV
        batch_size (int): How many rows to process per batch
        sub_batch_size (int): Mini-batch size when calling predict
    """
    df = pd.read_csv(provenance, sep="\t", header=None, names=["gene_symbol", "year"])
    num_rows = len(df)

    normalized_symbols = [None] * num_rows

    # keep track of counts to check normalization
    normed = 0
    for start_idx in tqdm(
        range(0, num_rows, batch_size), desc="Processing Cancer Batches"
    ):
        end_idx = min(start_idx + batch_size, num_rows)
        batch_df = df.iloc[start_idx:end_idx]
        symbols = batch_df["gene_symbol"].tolist()

        sentences = [
            Sentence(f"The related gene symbol is {symbol}.") for symbol in symbols
        ]

        # tag and link genes
        tagger.predict(sentences)
        gene_linker.predict(sentences)

        # normalize gene symbols
        for i, sentence in enumerate(sentences):
            norm_symbol = symbols[i]
            spans = sentence.get_spans("link")
            for span in spans:
                if label := span.get_label("link"):
                    norm_symbol = extract_normalized_name(str(label))
                    normed += 1
                    break
            normalized_symbols[start_idx + i] = norm_symbol

    df["normalized_symbol"] = normalized_symbols
    df[["normalized_symbol", "year"]].to_csv(
        out_file, sep="\t", index=False, header=False
    )
    print("Cancer provenance normalization complete.")
    print("Normalized:", normed)


def normalize_gda_provenance(
    provenance: str,
    tagger: Classifier,
    gene_linker: EntityMentionLinker,
    disease_linker: EntityMentionLinker,
    out_file: str,
    batch_size: int = 128,
) -> None:
    """Normalize gene-disease association provenance data using flair.

    We are only concerned with columns 1, 4, and 6 (0-indexed).

    Writes out the normalized data to a text file.
    """
    gda_df = pd.read_csv(
        provenance,
        sep="\t",
        header=None,
        usecols=[1, 4, 6],
        names=["gene_symbol", "disease_name", "year"],
    )

    num_rows = len(gda_df)
    normalized_genes = [None] * num_rows
    normalized_diseases = [None] * num_rows

    # keep track of counts to check normalization
    normed = 0
    print(f"Total rows to process: {num_rows}")
    for start_idx in tqdm(
        range(0, num_rows, batch_size), desc="Processing GDA Batches"
    ):
        end_idx = min(start_idx + batch_size, num_rows)
        batch_df = gda_df.iloc[start_idx:end_idx]
        genes = batch_df["gene_symbol"].tolist()
        diseases = batch_df["disease_name"].tolist()

        gene_sentences = [Sentence(f"The symbol is {gene}.") for gene in genes]
        disease_sentences = [
            Sentence(f"The pertinent name is {disease}.") for disease in diseases
        ]

        # tag and link
        tagger.predict(gene_sentences)
        gene_linker.predict(gene_sentences)

        # normalize gene symbols
        for i, sentence in enumerate(gene_sentences):
            norm_gene = genes[i]
            spans = sentence.get_spans("link")
            for span in spans:
                if label := span.get_label("link"):
                    norm_gene = extract_normalized_name(str(label))
                    break
            normalized_genes[start_idx + i] = norm_gene

        # tag and link diseases
        tagger.predict(disease_sentences)
        disease_linker.predict(disease_sentences)

        # 9) Extract normalized disease names
        for i, sentence in enumerate(disease_sentences):
            norm_disease = diseases[i]
            spans = sentence.get_spans("link")
            for span in spans:
                if label := span.get_label("link"):
                    norm_disease = extract_normalized_name(str(label))
                    normed += 1
                    break
            normalized_diseases[start_idx + i] = norm_disease

    gda_df["normalized_gene_symbol"] = normalized_genes
    gda_df["normalized_disease_name"] = normalized_diseases
    gda_df[["normalized_gene_symbol", "normalized_disease_name", "year"]].to_csv(
        out_file, sep="\t", index=False, header=False
    )
    print("GDA provenance normalization complete.")
    print("Normalized:", normed)


def main() -> None:
    """Main function to add publication year to gene-disease associations."""
    # force GPU
    flair.device = torch.device("cuda:0")

    # load flair models
    tagger, gene_linker, disease_linker = load_flair_models()

    # normalize cancer provenance data
    cancer_provenance = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/cancer/cancer_drivers_pmid_with_year_deduped.txt"
    cancer_out = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/cancer/cancer_normalized.txt"
    normalize_cancer_provenance_batched(
        provenance=cancer_provenance,
        tagger=tagger,
        gene_linker=gene_linker,
        out_file=cancer_out,
    )

    # normalize GDA provenance data
    gda_provenance = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/Gene-RD-Provenance_V2.1_with_year.txt"
    gda_out = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/gda_normalized.txt"
    normalize_gda_provenance(
        gda_provenance, tagger, gene_linker, disease_linker, gda_out
    )


if __name__ == "__main__":
    main()
