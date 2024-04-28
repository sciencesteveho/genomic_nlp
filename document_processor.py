# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup"""


import argparse
import os
import pickle
from typing import List, Optional, Set, Union

from progressbar import ProgressBar  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import COPY_GENES
from utils import dir_check_make
from utils import is_number
from utils import time_decorator


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


class ChunkedDocumentProcessor:
    """Object class to process a chunk of abstracts before model training.

    Attributes:
        root_dir: root directory for the project
        abstracts: list of abstracts
        date: date of processing
        lemmatizer: bool to apply lemmatization, which gets root stem of words
        word2vec: bool to apply additional word2vec processing steps

    Methods
    ----------
    _make_directories:
        Make directories for processing
    tokenization:
        Tokenize the abstracts using spaCy
    exclude_punctuation_tokens_replace_standalone_numbers:
        Removes standalone symbols if they exist as tokens. Replaces numbers
        with a number based symbol
    remove_entities_in_tokenized_corpus:
        Remove genes in gene_list from tokenized corpus
    processing_pipeline:
        Runs the nlp pipeline

    # Helpers
        EXTRAS -- set of extra characters to remove

    Examples:
    ----------
    >>> documentProcessor = ChunkedDocumentProcessor(
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
        root_dir: str,
        abstracts: Union[List[str], List[List[str]]],
        chunk: int,
        lemmatizer: bool,
        word2vec: bool,
        gene_gtf: str,
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.abstracts = abstracts
        self.chunk = chunk
        self.lemmatizer = lemmatizer
        self.word2vec = word2vec
        self.gene_gtf = gene_gtf

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
    def tokenization(self, abstracts: List[str], use_gpu: bool = False) -> None:
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

        n_process = 1 if use_gpu else 4
        batch_size = 32 if use_gpu else 500

        word_attr = "lemma_" if self.lemmatizer else "text"
        disable_pipes = ["parser", "tagger", "ner"]
        if not self.lemmatizer:
            disable_pipes.append("lemmatizer")

        dataset_tokens = []
        for doc in tqdm(
            nlp.pipe(
                abstracts,
                n_process=n_process,
                batch_size=batch_size,
                disable=disable_pipes,
            ),
            total=len(self.abstracts),
        ):
            dataset_tokens.extend(
                [
                    [getattr(word, word_attr) for word in sentence]
                    for sentence in doc.sents
                ]
            )

        self.abstracts = dataset_tokens

    @time_decorator(print_args=False)
    def exclude_punctuation_tokens_replace_standalone_numbers(
        self, abstracts: List[List[str]]
    ) -> None:
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

        self.abstracts = new_corpus

    @time_decorator(print_args=False)
    def _remove_entities_in_tokenized_corpus(
        self, entity_list: Set[str], abstracts: List[List[str]]
    ) -> None:
        """Remove genes in gene_list from tokenized corpus

        # Arguments
            entity_list: genes from GTF
        """
        self.abstracts = [
            [token for token in sentence if token not in entity_list]
            for sentence in abstracts
        ]

    def _save_processed_abstracts_checkpoint(self, outname: str) -> None:
        """
        Save processed abstracts to a pickle file after cleaning and lemmatization.

        Returns:
            None
        """
        outname += "_lemmatized.pkl" if self.lemmatizer else ".pkl"
        with open(outname, "wb") as output:
            pickle.dump(self.abstracts, output)

    def processing_pipeline(self) -> None:
        """Runs the initial cleaning pipeline."""
        # tokenize abstracts
        self.tokenization(abstracts=self.abstracts, use_gpu=False)

        # remove punctuation and standardize numbers with replacement
        self.exclude_punctuation_tokens_replace_standalone_numbers(
            abstracts=self.abstracts
        )

        # save the cleaned abstracts
        self._save_processed_abstracts_checkpoint(
            outname=f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_punct_{self.chunk}"
        )

        if not self.word2vec:
            return

        # additional processing steps
        genes = normalization_list(
            entity_file=self.gene_gtf,
            type="gene",
        )
        self._remove_entities_in_tokenized_corpus(
            abstracts=self.abstracts, entity_list=genes
        )

        self._save_processed_abstracts_checkpoint(
            outname=f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_genes_{self.chunk}"
        )


def main() -> None:
    """Main function"""
    # load classified abstracts
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, required=True)
    parser.add_argument(
        "--root_dir", type=str, default="/ocean/projects/bio210019p/stevesho/nlp"
    )
    parser.add_argument(
        "--gene_gtf", type=str, default="../data/gencode.v45.basic.annotation.gtf"
    )
    parser.add_argument("--lemmatizer", action="store_true")
    parser.add_argument("--prep_word2vec", action="store_true")
    args = parser.parse_args()

    # get relevant abstracts
    with open(
        f"{args.root_dir}/data/abstracts_classified_tfidf_20000_chunk_part_{args.chunk}.pkl",
        "rb",
    ) as f:
        abstracts = pickle.load(f)

    # instantiate document processor
    documentProcessor = ChunkedDocumentProcessor(
        root_dir=args.root_dir,
        abstracts=abstracts,
        chunk=args.chunk,
        lemmatizer=args.lemmatizer,
        word2vec=args.prep_word2vec,
        gene_gtf=args.gene_gtf,
    )

    # run processing pipeline
    documentProcessor.processing_pipeline()


if __name__ == "__main__":
    main()
