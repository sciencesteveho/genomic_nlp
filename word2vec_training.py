#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a word2vec model with a processed corpus"""


import argparse
from collections import Counter
from datetime import date
import logging
from typing import Any, Dict, List, Set

from fse.models import uSIF  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
from gensim.models.phrases import Phraser  # type: ignore
from gensim.models.phrases import Phrases  # type: ignore
import pandas as pd  # type: ignore
from progressbar import ProgressBar  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import COPY_GENES
from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


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


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"models/model_epoch{self.epoch}.pkl")
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
    _remove_entities_in_tokenized_corpus:
        Remove genes in gene_list from tokenized corpus
    _gram_generator:
        Iterates through prefix list to generate n-grams from 2-8!
    _normalize_gene_name_to_symbol:
        Looks for grams in corpus that are equivalent to gene names and
        converts them to gene symbols for training
    _build_vocab_and_train:
        Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object init
        
    # Helpers
        PREFS - list of n-gram prefixes
        GRAMDICT - dictionary of n-gram models
        GRAMLIST - list of n-gram models
    
    Examples:
    ----------
    """
    PREFS = ["bigram", "trigram", "quadgram", "quintigram"]
    GRAMDICT = dict.fromkeys(PREFS)
    GRAMLIST = list(GRAMDICT)

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
        
    def _concat_chunks(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenates chunks of abstracts"""
        return pd.concat(chunks, axis=0)
        
    @time_decorator(print_args=False)
    def _remove_entities_in_tokenized_corpus(
        self, entity_list: Set[str], abstracts: List[List[str]]
    ) -> List[List[str]]:
        """Remove genes in gene_list from tokenized corpus

        # Arguments
            entity_list: genes from GTF
        """
        return [
            [token for token in sentence if token not in entity_list]
            for sentence in abstracts
        ]

    @time_decorator(print_args=False)
    def _gram_generator(
        self,
        abstracts_without_entities: List[List[str]],
        abstracts: List[List[str]],
        minimum: int,
        score: int,
    ):
        """Iterates through prefix list to generate n-grams from 2-8!

        # Arguments
            minimum:
            score:
        """
        maxlen = len(self.GRAMDICT) - 1

        for index in range(maxlen + 1):
            if index == 0:
                source_sentences = abstracts_without_entities
                source_main = abstracts
            else:
                source_sentences = self.GRAMDICT[self.GRAMLIST[index - 1]][1]
                source_main = self.GRAMDICT[self.GRAMLIST[index - 1]][2]

            gram_model = Phrases(source_sentences, min_count=minimum, threshold=score)
            gram_model_phraser = Phraser(gram_model)
            gram_sentence = (
                gram_model_phraser[sentence] for sentence in source_sentences
            )
            gram_main = (gram_model_phraser[sentence] for sentence in source_main)

            self.GRAMDICT[self.GRAMLIST[index]] = [gram_model, gram_sentence, gram_main]

        quintgram_main = self.GRAMDICT[self.GRAMLIST[maxlen]][2]

        gram_model.save(
            f"{self.root_dir}/models/gram_models/{self.GRAMLIST[maxlen]}_model_{self.date}.pkl"
        )

        self._save_wrapper(
            gram_model, f"{self.root_dir}/data/gram_model_{self.date}.pkl"
        )

        return quintgram_main

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
        according to parameters set during object init
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
            callbacks=[EpochSaver()],
        )

        model.save(
            f"{self.root_dir}/models/w2v_models/word2vec_{self.dimensions}d_{self.date}.model"
        ) 


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a word2vec model")
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Root directory for data"
    )
    parser.add_argument(
        "--abstracts_dir", type=str, required=True, help="Path to abstracts"
    )
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()