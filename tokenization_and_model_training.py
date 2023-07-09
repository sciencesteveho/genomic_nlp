#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Tokenization, token clean-up, and gene removal. Model training for word
embeddings for bio-nlp model!"""

from datetime import date
import gc
import logging
import pickle
import re
import string
from typing import Dict, List

from fse.models import uSIF
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.phrases import Phraser
from gensim.models.phrases import Phrases
import pandas as pd
from progressbar import ProgressBar
import pybedtools
import spacy
from tqdm import tqdm

from utils import COPY_GENES
from utils import dir_check_make
from utils import is_number
from utils import time_decorator

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def gene_symbol_from_gencode(gencode_ref) -> List[str]:
    """Returns deduped list of gencode V43 genes"""
    return [
        line[8].split('gene_name "')[1].split('";')[0]
        for line in gencode_ref
        if "gene_name" in line[8]
    ]


def normalization_list(entity_file: str, type: str = "gene"):
    """Uses gtf_parser to parse a GTF to a dataframe. Grabs a list
    of gene_names in the GTf, removes duplicates, and adds fixers
    for the weirdly named genes.

    # Arguments
        entity_file: either a list of tuples or GTF file
        type: ents(scispaCy entities), gene(GTF)
    # Returns
        gene_names_list: list of unique genes from GTF
    """
    if type == "gene":
        print("Grabbing genes from GTF")
        gtf = pybedtools.BedTool(entity_file)
        genes = [gene.lower() for gene in gene_symbol_from_gencode(gtf)]
        for key in COPY_GENES:
            genes.remove(key)
            genes.append(COPY_GENES[key])
        return set(genes)
    elif type == "ents":
        pass
    else:
        raise ValueError("type must be either 'gene' or 'ents'")

    # elif type == "ents":
    #     ent_list = [entity[0].casefold() for entity in entity_file]
    #     return set(ent_list)


def dict_from_gene_symbol_and_name_list(gene_file_path):
    """Takes a tab delimited file organized as 'symbol'\t''name' and
    parses as a dictionary, removing entries with values in remove_list,
    which includes duplicates."

    # Arguments
        gene_file_path: filepath for gene tab file

    # Returns
        dictionary of values
    """
    remove_list = ["novel transcript", ""]
    namedict = {}
    with open(gene_file_path) as f:
        for line in f:
            line = line.strip("\n")
            a, b = line.split("\t")
            b = "".join(e for e in b if e.isalnum() or e in string.whitespace)
            b = re.sub("  ", " ", b)
            b = b.rstrip()
            b = re.sub(" ", "_", b)
            namedict.update({b.lower(): a.lower()})
    dup_list = list(namedict.values())
    val_dupes = set([item for item in dup_list if dup_list.count(item) > 1])
    for dupe in val_dupes:
        remove_list.append(dupe)
    set(remove_list)
    return {key: value for key, value in namedict.items() if value not in remove_list}


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print("Save model number {}.".format(self.epoch))
        model.save("models/model_epoch{}.pkl".format(self.epoch))
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
        processing_and_tokenization
        exclude_punctuation_tokens_replace_standalone_numbers
        named_entity_recognition
        remove_genes_in_tokenized_corpus
        gram_generator
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
    def processing_and_tokenization(self, use_gpu: bool = False):
        """Takes each token, splits into sentences, and tokenizes
        each sentence, before saving to a file
        """
        if use_gpu:
            spacy.prefer_gpu()
            nlp = spacy.load("en_core_sci_scibert")
        else:
            nlp = spacy.load("en_core_sci_lg")

        dataset_tokens = []
        for doc in tqdm(nlp.pipe(self.abstracts), total=len(self.abstracts)):
            sentences = [i for i in doc.sents]
            split_tokens = [
                [word.text for word in sentence] for sentence in sentences
            ]  # generates list of tokenized sentences
            dataset_tokens.extend(split_tokens)

        with open(f"data/tokens_from_cleaned_abstracts_{self.date}.pkl", "wb") as f:
            pickle.dump(dataset_tokens, f)

        return dataset_tokens

    @time_decorator(print_args=False)
    def exclude_punctuation_tokens_replace_standalone_numbers(self):
        """Removes standalone symbols if they exist as tokens. Replaces
        numbers with a number based symbol.
        """
        pbar = ProgressBar()
        new_corpus = []
        for sentence in pbar(self.abstracts):
            new_sentence = [token for token in sentence if token not in self.EXTRAS]
            new_sentence_2 = []
            for new_token in new_sentence:
                if is_number(new_token) is True:
                    new_token = "<nUm>"
                    new_sentence_2.append(new_token)
                else:
                    new_sentence_2.append(new_token)
            new_corpus.append(new_sentence_2)

        with open(
            f"data/tokens_from_cleaned_abstracts_remove_punct{self.date}.pkl", "wb"
        ) as f:
            pickle.dump(new_corpus, f)

        return new_corpus

    @time_decorator(print_args=False)
    def remove_entities_in_tokenized_corpus(self, entity_list):
        """Remove genes in gene_list from tokenized corpus

        # Arguments
            gene_list: genes from GTF
        """
        pbar = ProgressBar()
        new_corpus = []
        for sentence in pbar(self.abstracts):
            new_sentence = [token for token in sentence if token not in entity_list]
            new_corpus.append(new_sentence)

        with open("data/corpus_removed_genes_{}.pkl".format(self.date), "wb") as f:
            pickle.dump(new_corpus, f)

        return new_corpus

    @time_decorator(print_args=False)
    def gram_generator(self, minimum, score):
        """Iterates through prefix list to generate n-grams from 2-8!

        # Arguments
            minimum:
            score:
        """
        maxlen = len(self.GRAMDICT) - 1

        for index in range(0, maxlen + 1):
            if index == 0:
                gram_model = Phrases(
                    self.abstracts_without_entities, min_count=minimum, threshold=score
                )
                gram_model.save(
                    f"models/gram_models/{self.GRAMLIST[index]}_model_{self.date}.pkl"
                )
                gram_model_phraser = Phraser(gram_model)
                gram_sentence = [
                    gram_model_phraser[sentence]
                    for i, sentence in enumerate(self.abstracts_without_entities)
                ]
                gram_main = [
                    gram_model_phraser[sentence]
                    for i, sentence in enumerate(self.abstracts)
                ]
            elif index > 0:
                gram_model = Phrases(
                    self.GRAMDICT[self.GRAMLIST[index - 1]][1],
                    min_count=minimum,
                    threshold=score,
                )
                gram_model.save(
                    f"models/gram_models/{self.GRAMLIST[index]}_model_{self.date}.pkl"
                )
                gram_model_phraser = Phraser(gram_model)
                gram_sentence = [
                    gram_model_phraser[sentence]
                    for i, sentence in enumerate(
                        self.GRAMDICT[self.GRAMLIST[index - 1]][1]
                    )
                ]
                gram_main = [
                    gram_model_phraser[sentence]
                    for i, sentence in enumerate(
                        self.GRAMDICT[self.GRAMLIST[index - 1]][2]
                    )
                ]
            self.GRAMDICT[self.GRAMLIST[index]] = [
                gram_model,
                gram_sentence,
                gram_main,
            ]
            del gram_model, gram_sentence, gram_main
            gc.collect()

        quintgram_main = self.GRAMDICT[self.GRAMLIST[maxlen]][2]

        with open(f"data/gram_applied_dataset_{self.date}.pkl", "wb") as f:
            pickle.dump(quintgram_main, f)
        return quintgram_main

    @time_decorator(print_args=False)
    def normalize_gene_name_to_symbol(self, gene_dict):
        """Looks for grams in corpus that are equivalent to gene names and
        converts them to gene symbols for training.
        """
        pbar = ProgressBar()
        checklist = set(gene_dict.keys())
        gram_corpus_gene_standardized = []
        for sentence in pbar(self.gram_corpus):
            new_sentence = []
            for token in sentence:
                if token in checklist:
                    new_sentence.append(token.replace(token, gene_dict.get(token)))
                else:
                    new_sentence.append(token)
            gram_corpus_gene_standardized.append(new_sentence)
        return gram_corpus_gene_standardized

    @time_decorator(print_args=False)
    def initialize_build_vocab_and_train_word2vec_model(self):
        """Initializes vocab build for corpus, then trains W2v model
        according to parameters set during object init
        """
        # avg_len = averageLen(self.gram_corpus_gene_standardized)
        self.gram_corpus_gene_standardized = self.abstracts

        model = Word2Vec(
            min_count=self.min_count,  # init word2vec class with alpha values from Tshitoyan et al.
            window=20,
            size=self.dimensions,
            workers=self.workers,
            sample=self.sample,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            negative=self.negative,
            sg=self.sg,
            hs=self.hs,
        )

        model.build_vocab(self.gram_corpus_gene_standardized)  # build vocab

        model.train(
            self.gram_corpus_gene_standardized,  # train the w2v model
            total_examples=model.corpus_count,
            epochs=30,
            report_delay=15,
            compute_loss=True,
            callbacks=[EpochSaver()],
        )  # optionally add the callback class

        model.save(
            f"models/w2v_models/word2vec_{self.dimensions}d_{self.date}.model"
        )  # save model!

    def word2vec_processing_pipeline(self, gene_gtf: str) -> None:
        "Runs the entire pipeline for word2vec model training"
        # prepare genes for removal
        genes = normalization_list(gene_gtf, "gene")

        # tokenize abstracts
        # self.processing_and_tokenization(use_gpu=True)
        self.processing_and_tokenization()

        # remove punctuation and standardize numbers with replacement
        self.exclude_punctuation_tokens_replace_standalone_numbers()

        # remove genes so they are not used for gram generation
        self.remove_entities_in_tokenized_corpus(genes)

        # generate ngrams
        self.gram_generator(min_count=50, threshold=30)

        # train model for 30 epochs
        self.initialize_build_vocab_and_train_word2vec_model()


def main(
    abstracts: str,
    gene_gtf: str,
) -> None:
    """Main function"""
    # load classified abstracts
    abstracts = pd.read_pickle(abstracts)
    abstracts = abstracts.loc[abstracts["predictions"] == 1]["abstracts"].to_list()

    # instantiate object
    modelprocessingObj = ProcessWord2VecModel(
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
    modelprocessingObj.word2vec_processing_pipeline(
        gene_gtf=gene_gtf,
    )


if __name__ == "__main__":
    main(
        abstracts="/scratch/remills_root/remills/stevesho/bio_nlp/nlp/classification/abstracts_classified_tfidf_20000.pkl",
        gene_gtf="data/gencode.v43.basic.annotation.gtf",
    )
