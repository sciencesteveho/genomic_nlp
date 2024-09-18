# sourcery skip: lambdas-should-be-short
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup."""


import argparse
from collections import defaultdict
import csv
import logging
from typing import List, Set, Tuple, Union

import pandas as pd  # type: ignore
import pybedtools  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore
from spacy import Span  # type: ignore
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
        genes (Set[str]): Set of gene names to be used in processing.
        batch_size (int): Number of documents to process in each batch.
        df (pd.DataFrame): DataFrame containing the abstracts to process.
        max_length (int): Maximum sequence length for the model.
        nlp (spacy.language.Language): spaCy language model.
        spacy_model (str): Name of the spaCy model being used.

    Methods
    ----------
    tokenize_and_ner:
        Perform tokenization and NER on all abstracts.
    setup_pipeline:
        Set up the spaCy pipeline.
    custom_lemmatize:
        Custom lemmatization for tokens.
    get_canonical_entity:
        Normalize entities using canonical names.
    reconstruct_documents:
        Reconstruct documents from processed sentences.
    exclude_punctuation_replace_standalone_numbers:
        Exclude punctuation and replace standalone numbers.
    selective_casefold:
        Selectively casefold tokens.
    remove_genes:
        Remove gene symbols from tokens.
    _save_checkpoints:
        Save processed abstracts to separate files for Word2Vec and finetune.
    processing_pipeline:
        Run the full processing pipeline.

    # Helpers
        EXTRAS -- set of extra characters to remove

    Examples:
    ----------
    >>> documentProcessor = ChunkedDocumentProcessor(
        root_dir=root_dir,
        abstracts=abstracts,
        chunk=1,
        genes=genes,
    )

    >>> documentProcessor.processing_pipeline(use_gpu=True)
    """

    extras = set(
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
        genes: Set[str],
        max_length: int = 512,
        batch_size: int = 64,
    ):
        """Initialize the ChunkedDocumentProcessor object."""
        self.root_dir = root_dir
        self.chunk = chunk
        self.genes = genes
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

        # add sentencizer if not present
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        # add entity linker if not present
        self.nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": 0.8,
                "max_entities_per_mention": 1,
            },
            last=True,
        )

        # disable unnecessary pipeline components
        disable_pipes = ["parser", "tagger"]
        self.nlp.disable_pipes(*disable_pipes)

        logger.info(f"Pipeline components: {self.nlp.pipe_names}")

    def load_sentencizer_model(self) -> Language:
        """Load a small model that only includes the sentencizer."""
        sentencizer_nlp = spacy.load(
            "en_core_sci_sm", disable=["parser", "ner", "lemmatizer", "tagger"]
        )
        sentencizer_nlp.add_pipe("sentencizer")
        return sentencizer_nlp

    def custom_lemmatize(self, token: Token, lemmatize: bool = True) -> str:
        """Custom token processing. Only lemmatize tokens that are not
        recognized as entities via NER.
        """
        if token.ent_type_ == "ENTITY":
            return token.text
        else:
            return token.lemma_ if lemmatize else token.text

    def _split_into_subchunks(self, tokens: List[str]) -> List[List[str]]:
        """Recursively split a list of tokens into subchunks not exceeding
        max_length.
        """
        subchunks = []
        for i in range(0, len(tokens), self.max_length):
            subchunk = tokens[i : i + self.max_length]
            if len(subchunk) > self.max_length:
                subchunks.extend(self._split_into_subchunks(subchunk))
            else:
                subchunks.append(subchunk)
        return subchunks

    def replace_number_symbol_tokens(self, token: str) -> Union[str, None]:
        """Replace numbers with a number based symbol, and symbols with None."""
        if token in self.extras:
            return None
        return "<nUm>" if is_number(token) else token

    def selective_casefold_token(self, token: str) -> str:
        """Selectively casefold tokens, avoding gene symbols."""
        return token if token in self.genes else token.casefold()

    def collect_sentences(self, doc: Doc) -> Tuple[List[str], List[int], int, int]:
        """Preprocess the abstracts by splitting into sentences. This helps to
        avoid tensor mismatch errors, due to max length limitations of the
        scibert model.
        """
        cleaned_abstracts = self.df["cleaned_abstracts"].tolist()
        sentencizer_nlp = self.load_sentencizer_model()
        total_abstracts = len(cleaned_abstracts)

        sentences, doc_indices = [], []
        total_sentences = 0

        with tqdm(
            total=total_abstracts, desc="Collecting sentences", unit="doc"
        ) as pbar:
            for doc_idx, doc in enumerate(
                sentencizer_nlp.pipe(cleaned_abstracts, batch_size=self.batch_size)
            ):
                current_sentences = [sent.text.strip() for sent in doc.sents]

                for sent in current_sentences:
                    if not sent:
                        continue
                    tokens = sent.split()
                    if len(tokens) > self.max_length:
                        sub_chunks = self._split_into_subchunks(tokens)
                        for subchunk in sub_chunks:
                            text = " ".join(subchunk)
                            sentences.append(text)
                            doc_indices.append(doc_idx)
                            total_sentences += 1
                    else:
                        sentences.append(sent)
                        doc_indices.append(doc_idx)
                        total_sentences += 1
                pbar.update(1)

        return sentences, doc_indices, total_sentences, total_abstracts

    @time_decorator(print_args=False)
    def tokenize_and_ner(self) -> None:
        """Tokenize the abstracts using spaCy with batch processing and standardize entities."""
        sentences, doc_indices, total_sentences, total_abstracts = (
            self.collect_sentences(self.nlp)
        )

        processed_sentences_w2v = []
        processed_sentences_finetune = []
        with tqdm(
            total=total_sentences, desc="Processing sentences", unit="Sent"
        ) as pbar:
            for doc_processed in self.nlp.pipe(sentences, batch_size=self.batch_size):
                new_tokens_w2v, new_tokens_finetune = [], []
                last_index = 0
                for ent in doc_processed.ents:
                    start = ent.start
                    end = ent.end
                    tokens_before = list(doc_processed[last_index:start])

                    # process tokens before the entity
                    for token in tokens_before:
                        new_tokens_w2v.append(
                            self.custom_lemmatize(token, lemmatize=True)
                        )
                        new_tokens_finetune.append(
                            self.custom_lemmatize(token, lemmatize=False)
                        )

                    # normalize the entity
                    canonical_entity_w2v = self.get_canonical_entity(
                        ent=ent, lemmatize=True
                    )
                    canonical_entity_finetune = self.get_canonical_entity(
                        ent=ent, lemmatize=False
                    )
                    new_tokens_w2v.append(canonical_entity_w2v)
                    new_tokens_finetune.append(canonical_entity_finetune)
                    last_index = end

                # add tokens after the last entity
                tokens_after = list(doc_processed[last_index:])
                for token in tokens_after:
                    new_tokens_w2v.append(self.custom_lemmatize(token, lemmatize=True))
                    new_tokens_finetune.append(
                        self.custom_lemmatize(token, lemmatize=False)
                    )

                processed_sentences_w2v.append(new_tokens_w2v)
                processed_sentences_finetune.append(new_tokens_finetune)
                pbar.update(1)

        # reconstruct documents
        processed_docs_w2v, processed_docs_finetune = self.reconstruct_documents(
            processed_sentences_w2v,
            processed_sentences_finetune,
            doc_indices,
            total_abstracts,
        )

        self.df["tokenized_abstracts_w2v"] = processed_docs_w2v
        self.df["tokenized_abstracts_finetune"] = processed_docs_finetune

    def get_canonical_entity(self, ent: Span, lemmatize: bool) -> str:
        """Normalize entities in a document using the UMLS term."""
        if ent._.kb_ents:
            kb_id, score = ent._.kb_ents[0]
            if score >= self.nlp.get_pipe("scispacy_linker").threshold:
                return (
                    self.nlp.get_pipe("scispacy_linker")
                    .kb.cui_to_entity[kb_id]
                    .canonical_name
                )
        if not lemmatize:
            return ent.text
        lemmatized_tokens = [token.lemma_ for token in ent]
        return " ".join(lemmatized_tokens)

    def reconstruct_documents(
        self,
        processed_sentences_w2v: List[List[str]],
        processed_sentences_finetune: List[List[str]],
        doc_indices: List[int],
        total_abstracts: int,
    ) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
        """Reconstruct documents from processed sentences aligning with the
        original document indices.
        """
        docs_w2v = defaultdict(list)
        docs_finetune = defaultdict(list)
        for idx, tokens_w2v, tokens_finetune in zip(
            doc_indices, processed_sentences_w2v, processed_sentences_finetune
        ):
            docs_w2v[idx].append(tokens_w2v)
            docs_finetune[idx].append(tokens_finetune)

        return [docs_w2v[i] for i in range(total_abstracts)], [
            docs_finetune[i] for i in range(total_abstracts)
        ]

    @time_decorator(print_args=False)
    def exclude_punctuation_replace_standalone_numbers(self) -> None:
        """Exclude punctuation tokens and replace standalone numbers."""
        tqdm.pandas(desc="Cleaning tokens")

        # process Word2Vec version
        self.df["processed_abstracts_w2v"] = self.df[
            "tokenized_abstracts_w2v"
        ].progress_apply(
            lambda docs: [
                [
                    replacement
                    for token in sent
                    if (replacement := self.replace_number_symbol_tokens(token))
                    is not None
                ]
                for sent in docs
            ]
        )

        # process finetune version
        self.df["processed_abstracts_finetune"] = self.df[
            "tokenized_abstracts_finetune"
        ].progress_apply(
            lambda docs: [
                replacement
                for sent in docs
                for token in sent
                if (replacement := self.replace_number_symbol_tokens(token)) is not None
            ]
        )

    @time_decorator(print_args=False)
    def selective_casefold(self) -> None:
        """Selectively casefold the abstracts."""
        tqdm.pandas(desc="Casefolding")

        # process Word2Vec version
        self.df["casefolded_abstracts_w2v"] = self.df[
            "processed_abstracts_w2v"
        ].progress_apply(
            lambda docs: [
                [self.selective_casefold_token(token) for token in sent]
                for sent in docs
            ]
        )

        # process finetune version
        self.df["casefolded_abstracts_finetune"] = self.df[
            "processed_abstracts_finetune"
        ].progress_apply(
            lambda tokens: [self.selective_casefold_token(token) for token in tokens]
        )

    @time_decorator(print_args=False)
    def remove_genes(self) -> None:
        """Remove gene symbols from tokens, for future n-gram generation."""
        tqdm.pandas(desc="Removing genes")

        # process Word2Vec version
        self.df["final_abstracts_w2v"] = self.df[
            "casefolded_abstracts_w2v"
        ].progress_apply(
            lambda docs: [
                [token for token in sent if token not in self.genes] for sent in docs
            ]
        )

        # process finetune version - we don't remove genes from this version
        self.df["final_abstracts_finetune"] = self.df["casefolded_abstracts_finetune"]

    def save_data(self, outpref: str) -> None:
        """Save final copies of abstracts with cleaned, processed, and year
        columns.
        """
        columns_to_save = ["cleaned_abstracts", "year"]

        if "final_abstracts_w2v" in self.df.columns:
            w2v_outpref = f"{outpref}_w2v_chunk_{self.chunk}.pkl"
            columns_to_save.append("final_abstracts_w2v")
            self.df[columns_to_save].to_pickle(w2v_outpref)
            logger.info(f"Saved Word2Vec processed abstracts to {w2v_outpref}")

        if "final_abstracts_finetune" in self.df.columns:
            finetune_outpref = f"{outpref}_finetune_chunk_{self.chunk}.pkl"
            columns_to_save.append("final_abstracts_finetune")
            self.df[columns_to_save].to_pickle(finetune_outpref)
            logger.info(f"Saved finetune processed abstracts to {finetune_outpref}")

    @time_decorator(print_args=False)
    def processing_pipeline(self, use_gpu: bool = False) -> None:
        """Run the NLP pipeline for both Word2Vec and BERT fine-tuning."""
        # setup spaCy pipeline
        self.setup_pipeline(use_gpu=use_gpu)

        # tokenization and NER
        self.tokenize_and_ner()

        # exclude punctuation and replace standalone numbers
        self.exclude_punctuation_replace_standalone_numbers()

        # casefolding
        self.selective_casefold()

        # additional processing for Word2Vec and finetune
        self.remove_genes()
        self.save_data(f"{self.root_dir}/data/tokens_cleaned_abstracts_remove_genes")


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
        genes=genes,
    )

    # run processing pipeline
    documentProcessor.processing_pipeline(use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()
