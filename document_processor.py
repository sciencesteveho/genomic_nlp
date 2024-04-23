# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup"""


import argparse
import os
import pickle
from typing import List, Union

from progressbar import ProgressBar  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import dir_check_make
from utils import is_number
from utils import time_decorator


class ChunkedDocumentProcessor:
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
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.abstracts = abstracts
        self.chunk = chunk
        self.lemmatizer = lemmatizer

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

        word_attr = 'lemma_' if self.lemmatizer else 'text'
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
                [[getattr(word, word_attr) for word in sentence] for sentence in doc.sents]
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

    def processing_pipeline(self) -> None:
        """Runs the initial cleaning pipeline."""
        # tokenize abstracts
        self.tokenization(abstracts=self.abstracts, lemmatizer=False, use_gpu=False)

        # remove punctuation and standardize numbers with replacement
        self.exclude_punctuation_tokens_replace_standalone_numbers(
            abstracts = self.abstracts
        )

        outname = f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_punct_{self.chunk}"
        outname += "_lemmatized.pkl" if self.lemmatizer else ".pkl"
        with open(outname, "wb") as output:
            pickle.dump(self.abstracts, output)

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
    parser.add_argument("--chunk", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/ocean/projects/bio210019p/stevesho/nlp")
    parser.add_argument("--lemmatizer", action="store_true")
    args = parser.parse_args()

    # get relevant abstracts
    with open(f'{args.root_dir}/data/abstracts_classified_tfidf_20000_chunk_part_{args.chunk}.pkl', "rb") as f:
        abstracts = pickle.load(f)
    
    # instantiate document processor
    documentProcessor = ChunkedDocumentProcessor(
        root_dir=args.root_dir,
        abstracts=abstracts,
        chunk=args.chunk,
        lemmatizer=args.lemmatizer,
    )
    
    # run processing pipeline
    documentProcessor.processing_pipeline()


if __name__ == "__main__":
    main()
