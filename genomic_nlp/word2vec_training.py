#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a word2vec model with a processed corpus"""


import argparse
from datetime import date
import logging
import os
import pickle
from typing import List

from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
from gensim.models.phrases import Phraser  # type: ignore
from gensim.models.phrases import Phrases  # type: ignore
from gensim.models.word2vec import LineSentence  # type: ignore
import smart_open  # type: ignore

from utils import _chunk_locator
from utils import _combine_chunks
from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def flatten_abstract(abstract: List[str]) -> List[str]:
    """Flatten a potentially nested abstract."""
    if isinstance(abstract, list) and (abstract and isinstance(abstract[0], list)):
        return [word for sentence in abstract for word in sentence]
    return abstract


def write_chunks_to_text(args: argparse.Namespace, prefix: str) -> None:
    """Write chunks of abstracts to text files"""
    filenames = _chunk_locator(args.abstracts_dir, prefix)
    with open(f"{args.abstracts_dir}/combined/{prefix}_combined_2.txt", "w") as output:
        for filename in filenames:
            with open(filename, "rb") as file:
                abstracts = pickle.load(file)
                for abstract in abstracts:
                    flattened_abstract = flatten_abstract(abstract)
                    line = " ".join(flattened_abstract) + "\n"
                    output.write(line)


def prepare_and_load_abstracts(args: argparse.Namespace) -> None:
    """Combine chunked abstracts if they do not exist"""

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
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    file_suffixes = ["casefold", "remove_genes"]
    for suffix in file_suffixes:
        combine_chunks(suffix)


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


class IterableCorpus:
    """Takes a text file and returns a generator of sentences"""

    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        with smart_open.open(self.filename, "r", encoding="utf-8") as file:
            for line in file:
                yield line.rstrip("\n").split()


class Word2VecCorpus:
    """Object class to process a chunk of abstracts before model training.

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
    # instantiate object
    >>> modelprocessingObj = Word2VecCorpus(
        root_dir=args.root_dir,
        abstract_dir=args.abstracts_dir,
        date=date.today(),
        min_count=5,
        vector_size=200,
        window=8,
        workers=24,
        sample=0.0001,
        alpha=0.01,
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
        root_dir: str,
        abstract_dir: str,
        date: date,
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
        epochs: int,
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.abstract_dir = abstract_dir
        self.date = date
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
        for dir in ["models", "w2v"]:
            os.makedirs(f"{self.root_dir}/{dir}", exist_ok=True)

    @time_decorator(print_args=False)
    def _gram_generator(
        self,
        minimum: int,
        score: int,
    ) -> None:
        """Iterates through prefix list to generate n-grams from 2-8!

        # Arguments
            minimum:
            score:
        """
        source_stream = IterableCorpus(
            "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_remove_genes_combined.txt"
        )
        source_main = IterableCorpus(
            "/ocean/projects/bio210019p/stevesho/genomic_nlp/data/combined/tokens_cleaned_abstracts_remove_genes_combined.txt"
        )

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
        bigram.save(f"{self.root_dir}/models/gram_models/bigram_model_{self.date}.pkl")
        trigram.save(
            f"{self.root_dir}/models/gram_models/trigram_model_{self.date}.pkl"
        )
        quintgram.save(
            f"{self.root_dir}/models/gram_models/quintigram_model_{self.date}.pkl"
        )

        # transform corpus and write out to text
        with open(
            f"{self.root_dir}/data/corpus_phrased.txt", "w", encoding="utf-8"
        ) as transformed:
            for sent in source_main:
                transformed.write(" ".join(quintgram[trigram[bigram[sent]]]) + "\n")

    @time_decorator(print_args=False)
    def _build_vocab_and_train(self) -> None:
        """Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object instantiation.
        """
        # avg_len = averageLen(self.gram_corpus_gene_standardized)
        vocab_corpus = IterableCorpus(f"{self.root_dir}/data/corpus_phrased.txt")
        corpus = f"{self.root_dir}/data/corpus_phrased.txt"

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
            corpus_file=corpus,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=30,
            report_delay=15,
            compute_loss=True,
            callbacks=[EpochSaver(savedir=f"{self.root_dir}/models")],
        )

        model.save(
            f"{self.root_dir}/models/w2v/word2vec_{self.vector_size}_dimensions_{self.date}.model"
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
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
    )
    args = parser.parse_args()
    print("Arguments parsed. Preparing abstracts...")

    # prepare abstracts by writing chunks out to text file
    print("Writing out cleaned_corpus...")
    # write_chunks_to_text(args, "tokens_cleaned_abstracts_casefold")
    print("Writing gene_remove corpus...")
    write_chunks_to_text(args, "tokens_cleaned_abstracts_remove_genes")
    print("Abstracts written! Instantiating object...")

    # instantiate object
    modelprocessingObj = Word2VecCorpus(
        root_dir=args.root_dir,
        abstract_dir=args.abstracts_dir,
        date=date.today(),
        min_count=5,
        vector_size=300,
        window=8,
        workers=24,
        sample=0.001,
        alpha=0.01,
        min_alpha=0.0001,
        negative=10,
        sg=1,
        hs=0,
        epochs=30,
    )
    print("Model initialized. Generating grams...")

    # build gram models
    # modelprocessingObj._gram_generator(
    #     minimum=5,
    #     score=50,
    # )
    # print("Grams generated. Training word2vec model...")

    # train word2vec
    modelprocessingObj._build_vocab_and_train()


if __name__ == "__main__":
    main()
