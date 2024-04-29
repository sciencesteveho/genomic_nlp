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
from typing import cast, Dict, Generator, List, Tuple

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


def _write_chunks_to_text(args: argparse.Namespace, prefix: str) -> None:
    """Write chunks of abstracts to text files"""
    filenames = _chunk_locator(args.abstracts_dir, prefix)
    with open(f"{args.abstracts_dir}/combined/{prefix}_combined.txt", "w") as output:
        for filename in filenames:
            with open(filename, "rb") as file:
                abstracts = pickle.load(file)
                for abstract in abstracts:
                    line = " ".join(abstract) + "\n"
                    output.write(line)


def read_abstracts_line_by_line(
    text_file_path: str,
) -> Generator[List[str], None, None]:
    """Stream corpus from text file line by line"""
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        for line in text_file:
            # split line back into tokens and yield the list of tokens
            yield line.rstrip("\n").split()


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

    file_suffixes = ["remove_punct", "remove_genes"]
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
        # abstracts_without_entities: List[List[str]],
        # abstracts: List[List[str]],
        minimum: int,
        score: int,
    ) -> None:
        """Iterates through prefix list to generate n-grams from 2-8!

        # Arguments
            minimum:
            score:
        """
        source_stream = read_abstracts_line_by_line(
            "/ocean/projects/bio210019p/stevesho/nlp/data/combined/tokens_cleaned_abstracts_remove_genes_combined.txt"
        )
        source_main = read_abstracts_line_by_line(
            "/ocean/projects/bio210019p/stevesho/nlp/data/combined/tokens_cleaned_abstracts_remove_punct_combined.txt"
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

        # self.GRAMDICT = cast(dict, self.GRAMDICT)
        # maxlen = len(self.GRAMDICT) - 1

        # for index in range(maxlen + 1):
        #     if index == 0:
        #         source_sentences = source_stream
        #         source_main = source_main
        #     else:
        #         source_sentences = self.GRAMDICT[self.GRAMLIST[index - 1]][1]
        #         source_main = self.GRAMDICT[self.GRAMLIST[index - 1]][2]

        #     print(f"Generating {self.GRAMLIST[index]} grams")
        #     gram_model = Phrases(source_sentences, min_count=minimum, threshold=score)
        #     gram_model_phraser = Phraser(gram_model)
        #     gram_model.save(
        #         f"{self.root_dir}/models/gram_models/{self.GRAMLIST[index]}_model_{self.date}.pkl"
        #     )
        #     gram_sentence = (
        #         gram_model_phraser[sentence] for sentence in source_sentences
        #     )
        #     gram_main = (gram_model_phraser[sentence] for sentence in source_main)

        #     self.GRAMDICT[self.GRAMLIST[index]] = [gram_model, gram_sentence, gram_main]

        # quintgram_main = self.GRAMDICT[self.GRAMLIST[maxlen]][2]
        # self.abstracts = quintgram_main

    # def _normalize_gene_name_to_symbol(
    #     self, gene_dict: Dict[str, str], corpus: List[List[str]]
    # ) -> List[List[str]]:
    #     """Looks for grams in corpus that are equivalent to gene names and
    #     converts them to gene symbols for training.
    #     """
    #     pbar = ProgressBar()
    #     return [
    #         [gene_dict.get(token, token) for token in sentence]
    #         for sentence in pbar(corpus)
    #     ]

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

        # model.build_vocab(self.abstracts)  # build vocab

        # model.train(
        #     self.abstracts,
        #     total_examples=model.corpus_count,
        #     epochs=30,
        #     report_delay=15,
        #     compute_loss=True,
        #     callbacks=[EpochSaver(savedir=f"{self.root_dir}/models")],
        # )

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
    print("Arguments parsed. Preparing abstracts...")

    # prepare abstracts by writing chunks out to text file
    _write_chunks_to_text(args, "tokens_cleaned_abstracts_remove_punct")
    print("Writing out cleaned_corpus...")
    _write_chunks_to_text(args, "tokens_cleaned_abstracts_remove_genes")
    print("Writing gene_remove corpus...")

    # prepare_and_load_abstracts(args)
    # print("Abstracts chunked. Loading...")

    # # load abstracts
    # with open(
    #     f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_remove_punct_combined.pkl",
    #     "rb",
    # ) as file:
    #     abstracts = pickle.load(file)

    # with open(
    #     f"{args.abstracts_dir}/combined/tokens_cleaned_abstracts_remove_genes_combined.pkl",
    #     "rb",
    # ) as file:
    #     abstracts_without_genes = pickle.load(file)
    # print("Abstracts loaded. Initializing model.")

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
    print("Model initialized. Generating grams...")

    # build gram models
    modelprocessingObj._gram_generator(
        # abstracts_without_entities=abstracts_without_genes,
        # abstracts=abstracts,
        minimum=5,
        score=50,
    )
    print("Grams generated. Training word2vec model...")

    # train word2vec
    # modelprocessingObj._build_vocab_and_train()


if __name__ == "__main__":
    main()
