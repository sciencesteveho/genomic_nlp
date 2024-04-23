# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Tokenization, token clean-up, and gene removal. Model training for word
embeddings for bio-nlp model!"""


import argparse
from datetime import date
import logging
import os
import pickle
import re
from typing import Any, List, Set

import pandas as pd  # type: ignore
from progressbar import ProgressBar  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import COPY_GENES
from utils import dir_check_make
from utils import is_number
from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def _get_relevant_abstracts(abstract_file: str) -> List[str]:
    """Get abstracts classified as relevant"""
    abstracts_df = pd.read_pickle(abstract_file)
    return abstracts_df.loc[abstracts_df["predictions"] == 1]["abstracts"].to_list()


def gene_symbol_from_gencode(gencode_ref: pybedtools.BedTool) -> Set[str]:
    """Returns deduped set of genes from a gencode gtf. Written for the gencode
    45 and avoids header"""
    return {
        line[8].split('gene_name "')[1].split('";')[0]
        for line in gencode_ref
        if not line[0].startswith("#") and "gene_name" in line[8]
    }


def normalization_list(entity_file: str, type: str = "gene") -> Set[str]:
    """_summary_

    Args:
        entity_file (str): _description_
        genes (Set[str]): _description_
        type (str, optional): _description_. Defaults to "gene".

    Returns:
        Set[str]: _description_
    """

    # def handle_ents(entity_file:) -> Set[str]:
    #     """Remove gene tokens"""
    #     ents = [entity[0].casefold() for entity in entity_file if entity not in genes]
    #     return set(ents)

    def handle_gene() -> Set[str]:
        """Remove copy genes from gene list"""
        for key in COPY_GENES:
            genes.remove(key)
            genes.append(COPY_GENES[key])
        return set(genes)

    type_handlers = {
        # "ents": handle_ents,
        "gene": handle_gene,
    }

    print("Grabbing genes from GTF")
    gtf = pybedtools.BedTool(entity_file)
    genes = [gene.lower() for gene in gene_symbol_from_gencode(gtf)]

    if type not in type_handlers:
        raise ValueError("type must be either 'gene' or 'ents'")

    # return type_handlers[type](entity_file)
    return type_handlers[type]()
    

class DocumentProcessor:
    """Object class to process a chunk of abstracts before model training.
    
    Attributes:
        root_dir: root directory for the project
        abstracts: list of abstracts
        date: date of processing

    Methods
    ----------
    _make_directories:
        Make directories for processing
    tokenization:
        Tokenize the abstracts using spaCy
    exclude_punctuation_tokens_replace_standalone_numbers:
        Removes standalone symbols if they exist as tokens. Replaces numbers with a number based symbol
    remove_entities_in_tokenized_corpus:
        Remove genes in gene_list from tokenized corpus
    processing_pipeline:
        Runs the nlp pipeline

    # Helpers
        EXTRAS -- set of extra characters to remove
    
    Examples:
    ----------
    >>> documentProcessor = DocumentProcessor(
        root_dir=root_dir,
        abstracts=abstracts,
        date=date.today(),
    )
    
    >>> documentProcessor.processing_pipeline(gene_gtf=args.gene_gtf)
    """
    EXTRAS = set(
        [
            ".",
            "\\",
            "-",
            "/",
            "©",
            "~",
            "*",
            "&",
            "#",
            "# ",
            "'",
            '"',
            "^",
            "$",
            "|",
            "“",
            "”",
            "(",
            ")",
            "[",
            "]",
            "′′",
            "!",
            "'",
            "''",
            "+",
            "'s",
            "?",
            "& ",
            "@",
            "@ ",
            "\*\*",
            "±",
            "®",
            "â",
            "Å",
        ]
    )

    def __init__(
        self,
        root_dir,
        abstracts,
        date,
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.abstracts = abstracts
        self.date = date

    def _make_directories(self) -> None:
        """Make directories for processing"""
        for dir in [
            "data",
            "models/gram_models",
            "models/sentence_models",
            "models/w2v_models",
        ]:
            dir_check_make(dir)

    @time_decorator(print_args=False)
    def tokenization(self, use_gpu: bool = False) -> List[List[str]]:
        """Tokenize the abstracts using spaCy.
        Args:
            use_gpu (bool, optional): Flag to indicate whether to use GPU for
            processing. Defaults to False.

        Returns:
            list: Tokens extracted from the cleaned abstracts.
        """
        nlp = spacy.load("en_core_sci_scibert" if use_gpu else "en_core_sci_md")
        nlp.add_pipe("sentencizer")

        if use_gpu:
            spacy.require_gpu()
            n_process = 1
            batch_size = 32
        else:
            n_process = 4
            batch_size = 500

        dataset_tokens = []
        for doc in tqdm(
            nlp.pipe(
                self.abstracts,
                n_process=n_process,
                batch_size=batch_size,
                disable=["parser", "tagger", "ner", "lemmatizer"],
            ),
            total=len(self.abstracts),
        ):
            dataset_tokens.extend(
                [[word.text for word in sentence] for sentence in doc.sents]
            )

        self._save_wrapper(
            dataset_tokens,
            f"{self.root_dir}/data/tokens_from_cleaned_abstracts_{self.date}.pkl",
        )

        return dataset_tokens

    @time_decorator(print_args=False)
    def exclude_punctuation_tokens_replace_standalone_numbers(
        self, abstracts: List[List[str]]
    ) -> List[List[str]]:
        """Removes standalone symbols if they exist as tokens. Replaces
        numbers with a number based symbol.
        """
        pbar = ProgressBar()
        new_corpus = []
        for sentence in pbar(abstracts):
            new_sentence = [
                "<nUm>" if is_number(token) else token
                for token in sentence
                if token not in self.EXTRAS
            ]
            new_corpus.append(new_sentence)

        self._save_wrapper(
            new_corpus,
            f"{self.root_dir}/data/tokens_from_cleaned_abstracts_remove_punct{self.date}.pkl",
        )

        return new_corpus

    @time_decorator(print_args=False)
    def remove_entities_in_tokenized_corpus(
        self, entity_list: Set[str], abstracts: List[List[str]]
    ) -> List[List[str]]:
        """Remove genes in gene_list from tokenized corpus

        # Arguments
            gene_list: genes from GTF
        """
        return [
            [token for token in sentence if token not in entity_list]
            for sentence in abstracts
        ]

    def processing_pipeline(self, gene_gtf: str) -> None:
        """Runs the entire pipeline for word2vec model training"""
        # prepare genes for removal
        genes = normalization_list(gene_gtf, "gene")

        # tokenize abstracts
        # abstracts = self.tokenization(use_gpu=True)
        abstracts_file_path = (
            f"{self.root_dir}/data/tokens_from_cleaned_abstracts_{self.date}.pkl"
        )
        abstracts = self._check_before_processing(
            abstracts_file_path, self.tokenization, use_gpu=False
        )

        # remove punctuation and standardize numbers with replacement
        abstracts_standard_file_path = f"{self.root_dir}/data/tokens_from_cleaned_abstracts_remove_punct{self.date}.pkl"
        abstracts_standard = self._check_before_processing(
            abstracts_standard_file_path,
            self.exclude_punctuation_tokens_replace_standalone_numbers,
            abstracts=abstracts,
        )

    @staticmethod
    def _save_wrapper(obj: Any, filename: str) -> None:
        """Save object to file"""
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def _check_before_processing(file_path, process_func, *args, **kwargs):
        if not os.path.exists(file_path):
            data = process_func(*args, **kwargs)
        else:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        return data


def main() -> None:
    """Main function"""
    # load classified abstracts
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified_abstracts", type=str)
    parser.add_argument("--gene_gtf", type=str)
    args = parser.parse_args()

    # get relevant abstracts
    root_dir = "/ocean/projects/bio210019p/stevesho/nlp"
    relevant_abstracts = f"{root_dir}/data/relevant_abstracts.pkl"
    if not os.path.exists(relevant_abstracts):
        abstracts = _get_relevant_abstracts(abstract_file=args.classified_abstracts)
        with open(relevant_abstracts, "wb") as output:
            pickle.dump(abstracts, output)
    else:
        with open(relevant_abstracts, "rb") as f:
            abstracts = pickle.load(f)
            
    documentProcessor = DocumentProcessor(
        root_dir=root_dir,
        abstracts=abstracts,
        date=date.today(),
    )
    
    documentProcessor.processing_pipeline(gene_gtf=args.gene_gtf)

if __name__ == "__main__":
    main()
