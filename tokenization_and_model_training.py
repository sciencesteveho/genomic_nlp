#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Tokenization, token clean-up, and gene removal. Model training for word
embeddings for bio-nlp model!"""

from collections import Counter
from collections import defaultdict
from datetime import date
import logging
import pickle
import re
from typing import Dict, List, Set

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


def gene_symbol_from_gencode(gencode_ref: pybedtools.BedTool) -> List[str]:
    """Returns deduped list of gencode V43 genes"""
    return [
        line[8].split('gene_name "')[1].split('";')[0]
        for line in gencode_ref
        if "gene_name" in line[8]
    ]


def normalization_list(entity_file: str, type: str = "gene") -> Set[str]:
    def handle_ents(entity_file):
        ents = [entity[0].casefold() for entity in entity_file]
        return set(ents)

    def handle_gene(entity_file):
        print("Grabbing genes from GTF")
        gtf = pybedtools.BedTool(entity_file)
        genes = [gene.lower() for gene in gene_symbol_from_gencode(gtf)]
        for key in COPY_GENES:
            genes.remove(key)
            genes.append(COPY_GENES[key])
        return set(genes)

    type_handlers = {
        "ents": handle_ents,
        "gene": handle_gene,
    }

    if type not in type_handlers:
        raise ValueError("type must be either 'gene' or 'ents'")

    return type_handlers[type](entity_file)


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
        nlp = spacy.load("en_core_sci_scibert" if use_gpu else "en_core_sci_sm")
        if use_gpu:
            spacy.prefer_gpu()
            n_process = 1
            batch_size = 16
        else:
            nlp.add_pipe("sentencizer")
            n_process = 8
            batch_size = 512

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

        with open(f"data/tokens_from_cleaned_abstracts_{self.date}.pkl", "wb") as f:
            pickle.dump(dataset_tokens, f)

        return dataset_tokens

    @time_decorator(print_args=False)
    def exclude_punctuation_tokens_replace_standalone_numbers(self, abstracts):
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

        with open(
            f"data/tokens_from_cleaned_abstracts_remove_punct{self.date}.pkl", "wb"
        ) as f:
            pickle.dump(new_corpus, f)

        return new_corpus

    @time_decorator(print_args=False)
    def remove_entities_in_tokenized_corpus(self, entity_list, abstracts):
        """Remove genes in gene_list from tokenized corpus

        # Arguments
            gene_list: genes from GTF
        """
        with open(f"data/corpus_removed_genes_{self.date}.pkl", "wb") as f:
            pickle.dump(
                [
                    [token for token in sentence if token not in entity_list]
                    for sentence in abstracts
                ],
                f,
            )

        return [
            [token for token in sentence if token not in entity_list]
            for sentence in abstracts
        ]

    @time_decorator(print_args=False)
    def gram_generator(self, abstracts_without_entities, abstracts, minimum, score):
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
            f"models/gram_models/{self.GRAMLIST[maxlen]}_model_{self.date}.pkl"
        )

        with open(f"data/gram_applied_dataset_{self.date}.pkl", "wb") as f:
            pickle.dump(quintgram_main, f)
        return quintgram_main

    def normalize_gene_name_to_symbol(self, gene_dict):
        """Looks for grams in corpus that are equivalent to gene names and
        converts them to gene symbols for training.
        """
        pbar = ProgressBar()
        return [
            [gene_dict.get(token, token) for token in sentence]
            for sentence in pbar(self.gram_corpus)
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

        model.save(f"models/w2v_models/word2vec_{self.dimensions}d_{self.date}.model")

    def word2vec_processing_pipeline(self, gene_gtf: str) -> None:
        """Runs the entire pipeline for word2vec model training"""
        # prepare genes for removal
        genes = normalization_list(gene_gtf, "gene")
        genes = set(genes)

        # tokenize abstracts
        # self.processing_and_tokenization(use_gpu=True)
        self.processing_and_tokenization()

        # remove punctuation and standardize numbers with replacement
        abstracts_standard = self.exclude_punctuation_tokens_replace_standalone_numbers(
            abstracts=self.abstracts
        )

        # remove genes so they are not used for gram generation
        abstracts_without_entities = self.remove_entities_in_tokenized_corpus(
            entity_list=genes, abstracts=self.abstracts
        )

        # generate ngrams
        self.gram_generator(
            abstracts_without_entities=abstracts_without_entities,
            abstracts=self.abstracts,
            min_count=50,
            threshold=30,
        )

        # train model for 30 epochs
        self.initialize_build_vocab_and_train_word2vec_model()


def main(
    abstracts: str,
    gene_gtf: str,
) -> None:
    """Main function"""
    # # load classified abstracts
    abstracts = pd.read_pickle(abstracts)
    abstracts = abstracts.loc[abstracts["predictions"] == 1]["abstracts"].to_list()

    # load pretokenized, extend list, and save for later
    tokenized_abs = []
    for i in range(0, 15):
        with open(
            f"data/abstracts_classified_tfidf_20000_chunk_{i}.pkl",
            "rb",
        ) as file:
            tokenized_abs.extend(pickle.load(file))

    with open("data/tokenized_classified_abstracts", "wb") as f:
        pickle.dump(tokenized_abs, f, protocol=4)
    genes = normalization_list(gene_gtf, "gene")
    genes = set(genes)

    with open(
        "data/tokens_from_cleaned_abstracts_remove_punct2023-07-12.pkl", "rb"
    ) as f:
        abstracts = pickle.load(f)

    pbar = ProgressBar()
    new_corpus = []
    for sentence in pbar(abstracts):
        new_sentence = [token for token in sentence if token not in genes]
        new_corpus.append(new_sentence)

    with open("data/corpus_removed_genes.pkl", "wb") as f:
        pickle.dump(new_corpus, f)

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
