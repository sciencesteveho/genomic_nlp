#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a word2vec model with a processed corpus"""


import argparse
import logging
import os
from typing import Any

from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
from gensim.models.phrases import Phraser  # type: ignore
from gensim.models.phrases import Phrases  # type: ignore
from gensim.models.word2vec import LineSentence  # type: ignore
import smart_open  # type: ignore

from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class IterableCorpus:
    """Takes a text file and returns a generator of sentences."""

    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        with smart_open.open(self.filename, "r", encoding="utf-8") as file:
            for line in file:
                yield line.rstrip("\n").split()


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, savedir: str):
        self.epoch = 0
        self.savedir = savedir

    def on_epoch_end(self, model: Any) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"{self.savedir}/model_epoch{self.epoch}.pkl")
        self.epoch += 1


class Word2VecCorpus:
    """Class to handle word2vec training.

    Attributes:
        abstracts
        date
        min_count
        workers
        vector_size
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
        Iterates through prefix list to generate n-grams up to 4-grams
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
    # instantiate object
    >>> modelprocessingObj = Word2VecCorpus(
        root_dir=args.root_dir,
        abstract_dir=args.abstracts_dir,
        date=date.today(),
        min_count=10,
        vector_size=300,
        window=12,
        workers=32,
        sample=0.0001,
        alpha=0.005,
        min_alpha=0.0001,
        negative=15,
        sg=1,
        hs=0,
        epochs=30,
    )

    # build gram models
    >>> modelprocessingObj._gram_generator(
        minimum=5,
        score=50,
    )

    # train word2vec
    >>> modelprocessingObj._build_vocab_and_train()
    """

    GRAMLIST = ["bigram", "trigram", "quadgram", "quintigram"]
    GRAMDICT = dict.fromkeys(GRAMLIST)

    def __init__(
        self,
        model_dir: str,
        abstract_dir: str,
        year: int,
        min_count: int,
        vector_size: int,
        window: int,
        workers: int,
        sample: float,
        alpha: float,
        min_alpha: float,
        negative: int,
        sg: int,
        hs: int,
        epochs: int = 30,
    ):
        """Initialize the class"""
        self.model_dir = model_dir
        self.abstract_dir = abstract_dir
        self.year = year
        self.min_count = min_count
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.sample = sample
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.sg = sg
        self.hs = hs
        self.epochs = epochs

        # make directories for saved models
        for dir in ["gram_models", "epochs", "data"]:
            os.makedirs(f"{self.model_dir}/{dir}", exist_ok=True)

        self.gram_dir = f"{self.model_dir}/gram_models"
        self.epoch_dir = f"{self.model_dir}/epochs"
        self.data_dir = f"{self.model_dir}/data"

        self.corpus = f"{self.abstract_dir}/processed_abstracts_w2v_{self.year}.txt"
        self.corpus_nogenes = (
            f"{self.abstract_dir}/processed_abstracts_w2v_nogenes_{self.year}.txt"
        )
        self.corpus_phrased = f"{self.data_dir}/corpus_phrased.txt"

    @time_decorator(print_args=False)
    def _gram_generator(
        self,
        minimum: int,
        score: int,
    ) -> None:
        """Iterates through prefix list to generate n-grams.

        # Arguments
            minimum:
            score:
        """
        source_stream = IterableCorpus(self.corpus_nogenes)
        source_main = IterableCorpus(self.corpus)

        # generate and train n-gram models
        phrases = Phrases(source_stream, min_count=minimum, threshold=score)
        bigram = Phraser(phrases)
        trigram = Phraser(
            Phrases(bigram[source_stream], min_count=minimum, threshold=score)
        )
        quintgram = Phraser(
            Phrases(
                trigram[trigram[bigram[source_stream]]],
                min_count=minimum,
                threshold=score,
            )
        )

        # save models
        bigram.save(f"{self.gram_dir}/bigram_model.pkl")
        trigram.save(f"{self.gram_dir}/trigram_model.pkl")
        quintgram.save(f"{self.gram_dir}/quintigram_model.pkl")

        # transform corpus and write out to text
        with open(self.corpus_phrased, "w", encoding="utf-8") as transformed:
            for sent in source_main:
                transformed.write(" ".join(quintgram[trigram[bigram[sent]]]) + "\n")

    @time_decorator(print_args=False)
    def _build_vocab_and_train(self) -> None:
        """Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object instantiation.
        """
        vocab_corpus = IterableCorpus(self.corpus_phrased)

        model = Word2Vec(
            **{
                attr: getattr(self, attr)
                for attr in [
                    "min_count",
                    "window",
                    "workers",
                    "sample",
                    "vector_size",
                    "alpha",
                    "min_alpha",
                    "negative",
                    "sg",
                    "hs",
                ]
            }
        )  # init word2vec class with alpha values from Tshitoyan et al.

        model.build_vocab(vocab_corpus)  # build vocab

        model.train(
            corpus_file=self.corpus_phrased,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=self.epochs,
            report_delay=15,
            compute_loss=True,
            callbacks=[EpochSaver(savedir=f"{self.epoch_dir}")],
        )

        model.save(
            f"{self.model_dir}/word2vec_{self.vector_size}_dimensions_{self.year}.model"
        )


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a word2vec model")
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory for data",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp",
    )
    parser.add_argument(
        "--abstracts_dir",
        type=str,
        help="Path to abstracts",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to save models",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Cutoff year. Only abstracts up to and including this year will be processed.",
    )
    args = parser.parse_args()
    print("Arguments parsed. Preparing abstracts...")

    # instantiate object
    modelprocessingObj = Word2VecCorpus(
        model_dir=f"{args.model_dir}/{args.year}",
        abstract_dir=f"{args.abstracts_dir}/{args.year}",
        year=args.year,
        min_count=10,
        vector_size=300,
        window=12,
        workers=24,
        sample=0.0001,
        alpha=0.005,
        min_alpha=0.0001,
        negative=15,
        sg=1,
        hs=0,
        epochs=30,
    )
    print("Model initialized. Generating grams...")

    # build gram models
    modelprocessingObj._gram_generator(
        minimum=5,
        score=50,
    )
    print("Grams generated. Training word2vec model...")

    # train word2vec
    modelprocessingObj._build_vocab_and_train()


if __name__ == "__main__":
    main()
