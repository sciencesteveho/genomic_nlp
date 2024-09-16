#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup."""


import argparse
import csv
import logging
import math
from typing import Any, Iterator, List, Optional, Set, Union

import pandas as pd  # type: ignore
from progressbar import ProgressBar  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy.tokens import Doc  # type: ignore
from spacy.tokens import Token  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import COPY_GENES
from utils import dir_check_make
from utils import is_number
from utils import time_decorator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("Grabbing genes from GTF")
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
        root_dir (str): Root directory for the project.
        chunk (int): Chunk identifier for the current set of abstracts.
        lemmatizer (bool): Whether to apply lemmatization.
        word2vec (bool): Whether to apply additional word2vec processing steps.
        finetune (bool): Whether the processing is for fine-tuning.
        genes (Set[str]): Set of gene names to be used in processing.
        word_attr (str): Word attribute to use ('lemma_' if lemmatizer is True,
        else 'text').
        nlp (spacy.language.Language): spaCy language model.
        df (pd.DataFrame): DataFrame containing the abstracts to process.
        max_length (int): Maximum sequence length for the model.
        spacy_model (str): Name of the spaCy model being used.

    Methods
    ----------
    process_doc:
        Process a single document.
    process_doc_scibert:
        Process a document using SciBERT model.
    process_chunk:
        Process a chunk of tokens.
    tokenization_and_ner:
        Perform tokenization and NER on all abstracts.
    setup_pipeline:
        Set up the spaCy pipeline.
    custom_lemmatize:
        Custom lemmatization for tokens.
    processing_pipeline:
        Run the full processing pipeline.

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
        batch_size: int = 64,
    ):
        """Initialize the ChunkedDocumentProcessor object."""
        self.root_dir = root_dir
        self.chunk = chunk
        self.lemmatizer = lemmatizer
        self.word2vec = word2vec
        self.finetune = finetune
        self.genes = genes
        self.word_attr = "lemma_" if lemmatizer else "text"
        self.batch_size = batch_size

        self.df = abstracts[["cleaned_abstracts", "year"]]
        self.max_length = max_length

        self.nlp: Language = None
        self.spacy_model: str = ""

    def _make_directories(self) -> None:
        """Make directories for processing"""
        for dir in [
            "data",
            "models/gram_models",
            "models/sentence_models",
            "models/w2v_models",
        ]:
            dir_check_make(dir)

    def setup_pipeline(self, use_gpu: bool = False) -> None:
        """Set up the spaCy pipeline"""
        if use_gpu:
            self.spacy_model = "en_core_sci_scibert"
            spacy.require_gpu()
        else:
            self.spacy_model = "en_core_sci_md"

        logger.info(f"Loading spaCy model: {self.spacy_model}")
        logger.info(f"Using GPU: {use_gpu}")

        self.nlp = spacy.load(self.spacy_model)

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        # disable unnecessary pipeline components
        disable_pipes = ["parser", "tagger"]
        if not self.lemmatizer:
            disable_pipes.append("lemmatizer")
        self.nlp.disable_pipes(*disable_pipes)

        logger.info(f"Pipeline components: {self.nlp.pipe_names}")

    def custom_lemmatize(self, token: Token) -> str:
        """Custom token processing. Only lemmatize tokens that are not
        recognized as entities via NER.
        """
        return (
            token.text if token.ent_type == "ENTITY" else getattr(token, self.word_attr)
        )

    def process_doc(self, doc: Doc) -> List[List[str]]:
        """Process a document. If we are using the scibert model, then sentences
        passing the BERT max_length will need to be split."""
        if self.spacy_model == "en_core_sci_scibert":
            return self.process_doc_scibert(doc)
        else:
            return self.process_doc_standard(doc)

    def process_doc_standard(self, doc: Doc) -> List[List[str]]:
        """Process a standard document without a core SciSpacy model."""
        return [
            [self.custom_lemmatize(token) for token in sentence]
            for sentence in doc.sents
        ]

    def process_doc_scibert(self, doc: Doc) -> List[List[str]]:
        """Process a document using the SciBERT model, handling long sequences appropriately."""
        processed_sentences: List[List[str]] = []
        current_chunk: List[str] = []
        current_length = 0

        for sent in doc.sents:
            sent_tokens = [self.custom_lemmatize(token) for token in sent]
            sent_length = len(sent_tokens)

            # split long sentences
            if sent_length > self.max_length:
                logger.warning(
                    f"Sentence length {sent_length} exceeds max_length {self.max_length}. Splitting sentence."
                )
                sub_chunks = self._split_into_subchunks(sent_tokens)
                for sub_chunk in sub_chunks:
                    processed_sentences.append(sub_chunk)
                continue

            # accumulate tokens into chunks without exceeding max_length
            if current_length + sent_length > self.max_length:
                if current_chunk:
                    processed_sentences.append(current_chunk)
                current_chunk = sent_tokens
                current_length = sent_length
            else:
                current_chunk.extend(sent_tokens)
                current_length += sent_length

            # start new chunk if current_chunk is full
            if current_length >= self.max_length:
                processed_sentences.append(current_chunk[: self.max_length])
                current_chunk = current_chunk[self.max_length :]
                current_length = len(current_chunk)

        # process remaining tokens
        if current_chunk:
            processed_sentences.append(current_chunk)

        return processed_sentences

    def _split_into_subchunks(self, tokens: List[str]) -> List[List[str]]:
        """Split a list of tokens into subchunks each not exceeding max_length."""
        return [
            tokens[i : i + self.max_length]
            for i in range(0, len(tokens), self.max_length)
        ]

    # def process_chunk(self, chunk: List[str]) -> List[List[str]]:
    #     """Process a chunk of tokens that fits within max_length."""
    #     text = " ".join(chunk)
    #     doc = self.nlp(text)
    #     num_sentences = len(list(doc.sents))
    #     logger.debug(f"Processed chunk into {num_sentences} sentences.")
    #     return [
    #         [self.custom_lemmatize(token) for token in sentence]
    #         for sentence in doc.sents
    #     ]

    def process_token(self, token: str) -> Union[str, None]:
        """Replace numbers with a number based symbol, and symbols with None."""
        if token in self.EXTRAS:
            return None
        return "<nUm>" if is_number(token) else token

    def process_sentence(self, sentence: List[str]) -> List[Optional[str]]:
        """Process a sentence of tokens."""
        return [
            self.process_token(token)
            for token in sentence
            if self.process_token(token) is not None
        ]

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

    def split_on_sentences(self, doc: Doc, max_chars: int = 5000) -> List[Doc]:
        """Split a document into chunks of text based on sentence length."""
        chunks: List[Doc] = []
        current_chunk: List[str] = []
        current_length = 0
        for sent in doc.sents:
            sent_length = len(sent.text)
            if current_length + sent_length > max_chars and current_chunk:
                text = " ".join(current_chunk)
                chunks.append(self.nlp(text))
                current_chunk = []
                current_length = 0
            current_chunk.extend([token.text for token in sent])
            current_length += sent_length
        if current_chunk:
            text = " ".join(current_chunk)
            chunks.append(self.nlp(text))
        return chunks

    @time_decorator(print_args=False)
    def tokenization_and_ner(self) -> None:
        """Tokenize the abstracts using spaCy."""
        cleaned_abstracts = self.df["cleaned_abstracts"].tolist()
        total_docs = len(cleaned_abstracts)
        processed_docs: List[List[List[str]]] = []

        total_batches = math.ceil(total_docs / self.batch_size)
        pbar = tqdm(
            total=total_batches,
            desc="Processing documents",
            unit="Batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        # split abstracts into sentences with smaller model
        sentencizer_nlp = spacy.load(
            "en_core_sci_sm", disable=["parser", "ner", "lemmatizer", "tagger"]
        )
        sentencizer_nlp.add_pipe("sentencizer")

        for abstract in cleaned_abstracts:
            doc = sentencizer_nlp(abstract)
            sentences = [sent.text.strip() for sent in doc.sents]

            processed_sentences = []
            for sent in sentences:
                if not sent:
                    continue
                # split sentences longer than max_length into smaller chunks
                tokens = sent.split()
                if len(tokens) > self.max_length:
                    sub_chunks = self._split_into_subchunks(tokens)
                    for sub_chunk in sub_chunks:
                        sub_sent = " ".join(sub_chunk)
                        sent_doc = self.nlp(sub_sent)
                        processed_sentences.append(
                            [self.custom_lemmatize(token) for token in sent_doc]
                        )
                else:
                    sent_doc = self.nlp(sent)
                    processed_sentences.append(
                        [self.custom_lemmatize(token) for token in sent_doc]
                    )

            processed_docs.append(processed_sentences)
            pbar.update(1)
            pbar.set_postfix({"Processed docs": len(processed_docs)})
        pbar.close()

        self.df["tokenized_abstracts"] = processed_docs

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
        self.setup_pipeline(use_gpu=use_gpu)
        self.tokenization_and_ner()
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
    gencode = gencode_genes(gtf=args.gene_gtf)
    hgnc = hgnc_ncbi_genes(tsv=args.hgnc_genes, hgnc=True)
    ncbi = hgnc_ncbi_genes(tsv=args.ncbi_genes)
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
