# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a word2vec model with a processed corpus"""


import argparse
from datetime import date
import glob
import logging
import os
import pickle
from typing import cast, Dict, List, Tuple

from fse.models import uSIF  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
from gensim.models.phrases import Phraser  # type: ignore
from gensim.models.phrases import Phrases  # type: ignore
import pandas as pd  # type: ignore
from progressbar import ProgressBar  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def _concat_chunks(filenames: List[str]) -> List[List[str]]:
    """Concatenates chunks of abstracts"""
    combined = []
    combined += [pickle.load(open(file, "rb")) for file in filenames]
    return combined


def _chunk_locator(path: str, prefix: str) -> List[str]:
    """Returns abstract chunks matching a specific prefix"""
    pattern = f"{path}/{prefix}_*.pkl"
    return glob.glob(pattern)


def _combine_chunks(path: str, prefix: str) -> List[List[str]]:
    """Combines chunks of abstracts"""
    filenames = _chunk_locator(path, prefix)
    print(f"Combining chunks of abstracts: {filenames}")
    return _concat_chunks(filenames)


def prepare_and_load_abstracts(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    """Prepare chunked abstracts for processing"""

    def combine_chunks(suffix: str) -> None:
        """Combine chunks of abstracts if they do not exist"""
        filename = f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_{suffix}_combined.pkl"
        if not os.path.isfile(filename):
            print(f"Combining abstract chunks for {filename}")
            with open(filename, "wb") as f:
                pickle.dump(
                    _combine_chunks(
                        f"{args.abstracts_dir}",
                        f"tokens_cleaned_abstracts_{suffix}",
                    ),
                    f,
                )

    file_suffixes = ["remove_punct", "remove_genes"]
    for suffix in file_suffixes:
        combine_chunks(suffix)

    abstracts_without_genes = pickle.load(
        open(
            f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_remove_genes_combined.pkl",
            "rb",
        )
    )
    abstracts = pickle.load(
        open(
            f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_remove_punct_combined.pkl",
            "rb",
        )
    )

    return abstracts_without_genes, abstracts


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, savedir: str):
        self.epoch = 0
        self.savedir = savedir

    def on_epoch_end(self, model: Word2Vec) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"{self.savedir}/model_epoch{self.epoch}.pkl")
        self.epoch += 1


class Word2VecCorpus:
    """Object class to process a chunk of abstracts before model training.

    Attributes:
        abstracts
        date
        min_count
        workers
        dimensions
        sample
        alpha
        min_alpha
        negative
        sg
        hs
        epochs
        sentence_model
        abstracts_without_entities
        gram_corpus
        model

    Methods
    ----------
    _gram_generator:
        Iterates through prefix list to generate n-grams from 2-8!
    _normalize_gene_name_to_symbol:
        Looks for grams in corpus that are equivalent to gene names and
        converts them to gene symbols for training
    _build_vocab_and_train:
        Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object init

    # Helpers
        GRAMLIST - list of n-gram prefixes
        GRAMDICT - dictionary of n-gram models

    Examples:
    ----------
    """

    GRAMLIST = ["bigram", "trigram", "quadgram", "quintigram"]
    GRAMDICT = dict.fromkeys(GRAMLIST)

    def __init__(
        self,
        root_dir,
        abstract_dir,
        date,
        min_count,
        dimensions,
        workers,
        sample,
        alpha,
        min_alpha,
        negative,
        sg,
        hs,
        epochs,
        sentence_model,
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.abstract_dir = abstract_dir
        self.date = date
        self.min_count = min_count
        self.dimensions = dimensions
        self.workers = workers
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.sg = sg
        self.hs = hs
        self.epochs = epochs
        self.sentence_model = sentence_model

        # make directories for saved models
        os.makedirs(f"{self.root_dir}/models/gram_models", exist_ok=True)

    @time_decorator(print_args=False)
    def _gram_generator(
        self,
        abstracts_without_entities: List[List[str]],
        abstracts: List[List[str]],
        minimum: int,
        score: int,
    ) -> None:
        """Iterates through prefix list to generate n-grams from 2-8!

        # Arguments
            minimum:
            score:
        """
        self.GRAMDICT = cast(dict, self.GRAMDICT)
        maxlen = len(self.GRAMDICT) - 1

        for index in range(maxlen + 1):
            if index == 0:
                source_sentences = abstracts_without_entities
                source_main = abstracts
            else:
                source_sentences = self.GRAMDICT[self.GRAMLIST[index - 1]][1]
                source_main = self.GRAMDICT[self.GRAMLIST[index - 1]][2]

            print(f"Generating {self.GRAMLIST[index]} grams")
            gram_model = Phrases(source_sentences, min_count=minimum, threshold=score)
            gram_model_phraser = Phraser(gram_model)
            gram_model.save(
                f"{self.root_dir}/models/gram_models/{self.GRAMLIST[index]}_model_{self.date}.pkl"
            )
            gram_sentence = (
                gram_model_phraser[sentence] for sentence in source_sentences
            )
            gram_main = (gram_model_phraser[sentence] for sentence in source_main)

            self.GRAMDICT[self.GRAMLIST[index]] = [gram_model, gram_sentence, gram_main]

        quintgram_main = self.GRAMDICT[self.GRAMLIST[maxlen]][2]
        self.abstracts = quintgram_main

    def _normalize_gene_name_to_symbol(
        self, gene_dict: Dict[str, str], corpus: List[List[str]]
    ) -> List[List[str]]:
        """Looks for grams in corpus that are equivalent to gene names and
        converts them to gene symbols for training.
        """
        pbar = ProgressBar()
        return [
            [gene_dict.get(token, token) for token in sentence]
            for sentence in pbar(corpus)
        ]

    @time_decorator(print_args=False)
    def _build_vocab_and_train(self) -> None:
        """Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object instantiation.
        """
        # avg_len = averageLen(self.gram_corpus_gene_standardized)

        model = Word2Vec(
            **{
                attr: getattr(self, attr)
                for attr in [
                    "min_count",
                    "window",
                    "size",
                    "workers",
                    "sample",
                    "alpha",
                    "min_alpha",
                    "negative",
                    "sg",
                    "hs",
                ]
            }
        )  # init word2vec class with alpha values from Tshitoyan et al.

        model.build_vocab(self.abstracts)  # build vocab

        model.train(
            self.abstracts,
            total_examples=model.corpus_count,
            epochs=30,
            report_delay=15,
            compute_loss=True,
            callbacks=[EpochSaver(savedir=f"{self.root_dir}/models")],
        )

        model.save(
            f"{self.root_dir}/models/w2v_models/word2vec_{self.dimensions}d_{self.date}.model"
        )


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a word2vec model")
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory for data",
        default="/ocean/projects/bio210019p/stevesho/nlp",
    )
    parser.add_argument(
        "--abstracts_dir",
        type=str,
        help="Path to abstracts",
        default="/ocean/projects/bio210019p/stevesho/nlp/data",
    )
    args = parser.parse_args()

    # prepare abstracts by combining chunks
    abstracts_without_genes, abstracts = prepare_and_load_abstracts(args)

    # instantiate object
    modelprocessingObj = Word2VecCorpus(
        root_dir=args.root_dir,
        abstract_dir=args.abstracts_dir,
        date=date.today(),
        min_count=5,
        dimensions=250,
        workers=8,
        sample=0.0001,
        alpha=0.01,
        min_alpha=0.0001,
        negative=15,
        sg=1,
        hs=1,
        epochs=30,
        sentence_model=uSIF,
    )

    # build gram models
    modelprocessingObj._gram_generator(
        abstracts_without_entities=abstracts_without_genes,
        abstracts=abstracts,
        minimum=5,
        score=50,
    )

    # train word2vec
    modelprocessingObj._build_vocab_and_train()


if __name__ == "__main__":
    main()
