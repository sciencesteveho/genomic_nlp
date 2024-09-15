#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup."""


import argparse
import csv
from typing import Any, Iterator, List, Set, Union

import pandas as pd  # type: ignore
from progressbar import ProgressBar  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy.tokens import Token  # type: ignore
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


def gencode_genes(gtf: str) -> Set[str]:
    """Get gene symbols from a gencode gtf file."""
    print("Grabbing genes from GTF")
    gtf = pybedtools.BedTool(gtf)
    genes = list(gene_symbol_from_gencode(gtf))

    for key in COPY_GENES:
        genes.remove(key)
        genes.append(COPY_GENES[key])
    return set(genes)


def hgnc_ncbi_genes(tsv: str, hgnc: bool = False) -> Set[str]:
    """Get gene symbols and names from HGNC file"""
    gene_symbols, gene_names = [], []
    with open(tsv, newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if hgnc:
                gene_symbols.append(row[1])
                gene_names.append(row[2])
            else:
                gene_symbols.append(row[0])
                gene_names.append(row[1])

    gene_names = [
        name.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")
        for name in gene_names
    ]
    return set(gene_symbols + gene_names)


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
        abstracts: pd.DataFrame,
        chunk: int,
        lemmatizer: bool,
        word2vec: bool,
        finetune: bool,
        genes: Set[str],
        max_length: int = 512,
    ):
        """Initialize the class"""
        self.root_dir = root_dir
        self.chunk = chunk
        self.lemmatizer = lemmatizer
        self.word2vec = word2vec
        self.finetune = finetune
        self.genes = genes
        self.word_attr = "lemma_" if lemmatizer else "text"
        self.nlp: Language = None

        self.df = abstracts[["cleaned_abstracts", "year"]]
        self.max_length = max_length

    def _make_directories(self) -> None:
        """Make directories for processing"""
        for dir in [
            "data",
            "models/gram_models",
            "models/sentence_models",
            "models/w2v_models",
        ]:
            dir_check_make(dir)

    def process_batch(
        self, texts: List[str], nlp: Language, batch_size: int = 150
    ) -> Iterator[List[List[str]]]:
        """Use batch processing for efficiency."""
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            docs = list(nlp.pipe(batch))
            yield from (self.process_doc(doc) for doc in docs)

    def setup_pipeline(self, use_gpu: bool = False) -> None:
        """Set up the spaCy pipeline"""
        if use_gpu:
            self.spacy_model = "en_core_sci_scibert"
            spacy.require_gpu()
        else:
            self.spacy_model = "en_core_sci_md"

        print(f"Loading spaCy model: {self.spacy_model}")
        print(f"Using GPU: {use_gpu}")

        self.nlp = spacy.load(self.spacy_model)
        self.nlp.add_pipe("sentencizer")
        disable_pipes = ["parser"]
        if not self.lemmatizer:
            disable_pipes.append("lemmatizer")
        self.nlp.disable_pipes(*disable_pipes)

    def custom_lemmatize(self, token: Token, word_attr: str) -> str:
        """Custom token processing. Only lemmatize tokens that are not
        recognized as entities via NER.
        """
        return token.text if token.ent_type == "ENTITY" else getattr(token, word_attr)

    def process_doc(self, doc: spacy.tokens.Doc) -> List[List[str]]:
        """Process a document. If we are using the scibert model, then sentences
        passing the BERT max_length will need to be split."""
        if self.spacy_model == "en_core_sci_scibert":
            processed_sentences = []
            for sentence in doc.sents:
                if len(sentence) > self.max_length:
                    processed_sentences.append(
                        [
                            self.custom_lemmatize(token, self.word_attr)
                            for token in sentence[: self.max_length]
                        ]
                    )
                    sentence = sentence[self.max_length :]
                processed_sentences.append(
                    [self.custom_lemmatize(token, self.word_attr) for token in sentence]
                )
            return processed_sentences
        return [
            [self.custom_lemmatize(token, self.word_attr) for token in sentence]
            for sentence in doc.sents
        ]

    def process_token(self, token: str) -> Union[str, None]:
        """Replace numbers with a number based symbol, and symbols with None."""
        if token in self.EXTRAS:
            return None
        return "<nUm>" if is_number(token) else token

    def process_sentence(self, sentence: List[str]) -> List[str]:
        """Process a sentence of tokens."""
        processed_tokens = []
        for token in sentence:
            processed_token = self.process_token(token)
            if processed_token is not None:
                processed_tokens.append(processed_token)
        return processed_tokens

    def selective_casefold_token(self, token: str) -> str:
        """Selectively casefold tokens, avoding gene symbols."""
        return token if token in self.genes else token.casefold()

    def casefold_sentence(self, sentence: List[str]) -> List[str]:
        """Casefold a sentence of tokens."""
        return [self.selective_casefold_token(token) for token in sentence]

    def remove_gene(self, token: str) -> Union[str, None]:
        """Remove gene symbols from tokens, for future n-gram generation."""
        return None if token in self.genes else token

    def remove_genes_from_sentence(self, sentence: List[str]) -> List[str]:
        """Remove gene symbols from a sentence of tokens."""
        return [
            token
            for token in (self.remove_gene(t) for t in sentence)
            if token is not None
        ]

    @time_decorator(print_args=False)
    def tokenization_and_ner(self, use_gpu: bool = False) -> None:
        """Tokenize the abstracts using spaCy."""
        self.setup_pipeline(use_gpu=use_gpu)
        tqdm.pandas(desc="SciSpacy pipe")
        self.df["tokenized_abstracts"] = self.df["cleaned_abstracts"].progress_apply(
            lambda x: self.process_doc(self.nlp(x))
        )

    @time_decorator(print_args=False)
    def exclude_punctuation_tokens_replace_standalone_numbers(self) -> None:
        """Exclude punctuation tokens and replace standalone numbers."""
        tqdm.pandas(desc="Cleaning tokens")
        if self.finetune:
            self.df["processed_abstracts"] = self.df[
                "tokenized_abstracts"
            ].progress_apply(lambda x: [self.process_sentence(sent) for sent in x])
        else:
            self.df["processed_abstracts"] = self.df[
                "tokenized_abstracts"
            ].progress_apply(
                lambda x: self.process_sentence([token for sent in x for token in sent])
            )

    @time_decorator(print_args=False)
    def selective_casefold(self) -> None:
        """Selectively casefold the abstracts."""
        tqdm.pandas(desc="Casefolding")
        if self.finetune:
            self.df["casefolded_abstracts"] = self.df[
                "processed_abstracts"
            ].progress_apply(lambda x: [self.casefold_sentence(sent) for sent in x])
        else:
            self.df["casefolded_abstracts"] = self.df[
                "processed_abstracts"
            ].progress_apply(self.casefold_sentence)

    @time_decorator(print_args=False)
    def remove_entities_in_tokenized_corpus(self) -> None:
        """Remove gene symbols from the tokenized corpus for n-gram
        generation.
        """
        tqdm.pandas(desc="Removing entities")
        if self.finetune:
            self.df["final_abstracts"] = self.df["casefolded_abstracts"].progress_apply(
                lambda x: [self.remove_genes_from_sentence(sent) for sent in x]
            )
        else:
            self.df["final_abstracts"] = self.df["casefolded_abstracts"].progress_apply(
                self.remove_genes_from_sentence
            )

    def _save_checkpoints(self, outpref: str) -> None:
        """Save processed abstracts to a pickle file after cleaning and
        lemmatization.
        """
        if self.lemmatizer:
            outpref += "_lemmatized"
        if self.finetune:
            outpref += "_finetune"
        outpref += f"_{self.chunk}.pkl"
        self.df.to_pickle(outpref)

    def processing_pipeline(self, use_gpu: bool = False) -> None:
        """Run the nlp pipeline."""
        # spacy pipeline
        self.tokenization_and_ner(use_gpu=use_gpu)
        self._save_checkpoints(
            outpref=f"{self.root_dir}/data/tokens_ner_cleaned_abstracts"
        )

        # additional processing
        self.exclude_punctuation_tokens_replace_standalone_numbers()
        self._save_checkpoints(
            outpref=f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_punct"
        )
        self.selective_casefold()
        self._save_checkpoints(
            outpref=f"{self.root_dir}/data/tokens_cleaned_abstracts_casefold"
        )

        # additional processing for word2vec
        if self.word2vec:
            self.remove_entities_in_tokenized_corpus()
            self._save_checkpoints(
                outpref=f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_genes"
            )


def main() -> None:
    """Main function"""
    # load classified abstracts
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=str, required=True)
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp",
    )
    parser.add_argument(
        "--gene_gtf",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/reference_files/gencode.v45.basic.annotation.gtf",
    )
    parser.add_argument(
        "--ncbi_genes",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/reference_files/ncbi_genes.tsv",
    )
    parser.add_argument(
        "--hgnc_genes",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/reference_files/hgnc_complete_set.txt",
    )
    parser.add_argument("--lemmatizer", action="store_true")
    parser.add_argument("--prep_word2vec", action="store_true")
    parser.add_argument("--prep_finetune", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    # load abstract df
    abstracts = pd.read_pickle(
        f"{args.root_dir}/data/abstracts_logistic_classified_tfidf_40000_chunk_part_{args.chunk}.pkl"
    )

    # check that we have the required "year" column
    if "year" not in abstracts.columns:
        raise ValueError("Abstracts must have a 'year' column")

    # get genes
    gencode = gencode_genes(
        gtf=args.gene_gtf,
    )

    hgnc = hgnc_ncbi_genes(
        tsv=args.hgnc_genes,
        hgnc=True,
    )

    ncbi = hgnc_ncbi_genes(
        tsv=args.ncbi_genes,
    )

    genes = gencode.union(hgnc).union(ncbi)

    # instantiate document processor
    documentProcessor = ChunkedDocumentProcessor(
        root_dir=args.root_dir,
        abstracts=abstracts,
        chunk=args.chunk,
        lemmatizer=args.lemmatizer,
        word2vec=args.prep_word2vec,
        finetune=args.prep_finetune,
        genes=genes,
    )

    # run processing pipeline
    documentProcessor.processing_pipeline(use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()
