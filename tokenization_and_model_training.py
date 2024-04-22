# sourcery skip: do-not-use-staticmethod
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Tokenization, token clean-up, and gene removal. Model training for word
embeddings for bio-nlp model!"""


import argparse
from collections import Counter
from datetime import date
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Set

from fse.models import uSIF  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
from gensim.models.phrases import Phraser  # type: ignore
from gensim.models.phrases import Phrases  # type: ignore
import pandas as pd
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


def dict_from_gene_symbol_and_name_list(gene_file_path):
    """Takes a tab delimited file organized as 'symbol'\t''name' and
    parses as a dictionary, removing entries with values in remove_words,
    which includes duplicates."

    # Arguments
        gene_file_path: filepath for gene tab file

    # Returns
        dictionary of values
    """
    remove_words = {"novel transcript", ""}
    namedict = {}
    with open(gene_file_path) as f:
        for line in f:
            symbol, name = line.strip().split("\t")
            name = re.sub(r"[^\w\s]|_", "", name).replace("  ", " ").strip().lower()
            namedict[name] = symbol.lower()

    # Find duplicates
    duplicates = {
        symbol for symbol, count in Counter(namedict.values()).items() if count > 1
    }
    remove_words.update(duplicates)

    # Filter out remove_words and duplicates
    return {
        name: symbol for name, symbol in namedict.items() if symbol not in remove_words
    }
    # remove_words = ["novel transcript", ""]
    # namedict = {}
    # with open(gene_file_path) as f:
    #     for line in f:
    #         line = line.strip("\n")
    #         a, b = line.split("\t")
    #         b = "".join(e for e in b if e.isalnum() or e in string.whitespace)
    #         b = re.sub("  ", " ", b)
    #         b = b.rstrip()
    #         b = re.sub(" ", "_", b)
    #         namedict.update({b.lower(): a.lower()})
    # dup_list = list(namedict.values())
    # val_dupes = set([item for item in dup_list if dup_list.count(item) > 1])
    # for dupe in val_dupes:
    #     remove_words.append(dupe)
    # set(remove_words)
    # return {key: value for key, value in namedict.items() if value not in remove_words}


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"models/model_epoch{self.epoch}.pkl")
        self.epoch += 1


class ProcessWord2VecModel:
    """An object containing the trained Word2Vec Model and intermediate files

    # Properties
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

    # Methods
        tokenization
        exclude_punctuation_tokens_replace_standalone_numbers
        named_entity_recognition
        remove_genes_in_tokenized_corpus
        gram_gepator
        initialize_build_vocab_and_train_word2vec_model
        generate_sentence_embeddings
        save

    # Helpers
        PREFS
        GRAMDICT
        GRAMLIST
        EXTRAS
    """

    PREFS = ["bigram", "trigram", "quadgram", "quintigram"]
    GRAMDICT = dict.fromkeys(PREFS)
    GRAMLIST = list(GRAMDICT)
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
        self.abstracts = abstracts
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

    @time_decorator(print_args=False)
    def gram_generator(
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

    def normalize_gene_name_to_symbol(
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
    def initialize_build_vocab_and_train_word2vec_model(self) -> None:
        """Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object init
        """
        # avg_len = averageLen(self.gram_corpus_gene_standardized)
        self.gram_corpus_gene_standardized = self.abstracts

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

        model.build_vocab(self.gram_corpus_gene_standardized)  # build vocab

        model.train(
            self.gram_corpus_gene_standardized,
            total_examples=model.corpus_count,
            epochs=30,
            report_delay=15,
            compute_loss=True,
            callbacks=[EpochSaver()],
        )

        model.save(
            f"{self.root_dir}/models/w2v_models/word2vec_{self.dimensions}d_{self.date}.model"
        )

    def processing_pipeline(self, gene_gtf: str) -> None:
        """Runs the entire pipeline for word2vec model training"""
        # prepare genes for removal
        genes = normalization_list(gene_gtf, "gene")

        # tokenize abstracts
        # abstracts = self.tokenization(use_gpu=True)
        abstracts = self.tokenization(use_gpu=False)

        # # remove punctuation and standardize numbers with replacement
        # abstracts_standard = self.exclude_punctuation_tokens_replace_standalone_numbers(
        #     abstracts=abstracts
        # )

        # # remove genes so they are not used for gram generation
        # abstracts_without_entities = self.remove_entities_in_tokenized_corpus(
        #     entity_list=genes, abstracts=abstracts_standard
        # )

        # # generate ngrams
        # self.gram_generator(
        #     abstracts_without_entities=abstracts_without_entities,
        #     abstracts=self.abstracts,
        #     min_count=50,
        #     threshold=30,
        # )

        # # train model for 30 epochs
        # self.initialize_build_vocab_and_train_word2vec_model()

    @staticmethod
    def _save_wrapper(obj: Any, filename: str) -> None:
        """Save object to file"""
        with open(filename, "wb") as f:
            pickle.dump(obj, f)


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

    # instantiate object
    modelprocessingObj = ProcessWord2VecModel(
        root_dir=root_dir,
        abstracts=abstracts,
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

    # run pipeline!
    modelprocessingObj.processing_pipeline(
        gene_gtf=args.gene_gtf,
    )


if __name__ == "__main__":
    main()
