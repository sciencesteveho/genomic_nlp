# sourcery skip: lambdas-should-be-short
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup."""


import argparse
import csv
import logging
import re
from typing import List, Set, Tuple, Union

import pandas as pd  # type: ignore
import pybedtools  # type: ignore
import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy.tokens import Doc  # type: ignore
from spacy.tokens import Span  # type: ignore
from spacy.tokens import Token  # type: ignore
from tqdm import tqdm  # type: ignore

from genomic_nlp.abstracts.gene_entity_normalization import replace_symbols
from genomic_nlp.utils.common import dir_check_make
from genomic_nlp.utils.common import gencode_genes
from genomic_nlp.utils.common import gene_symbol_from_gencode
from genomic_nlp.utils.common import hgnc_ncbi_genes
from genomic_nlp.utils.common import time_decorator
from genomic_nlp.utils.constants import COPY_GENES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("document_processor_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_normalized_disease_names(ctd: str) -> Set[str]:
    """Get casefolded, symbol normalized disease names from CTD."""
    diseases: Set[str] = set()

    with open(ctd, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:

            # skip header, #
            if line[0].startswith("#"):
                continue

            # set disease identifier as key
            diseases.add(replace_symbols(line[1]).casefold())

    return diseases


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
    tokenize:
        Perform tokenization on all abstracts.
    setup_pipeline:
        Set up the spaCy pipeline.
    custom_lemmatize:
        Custom lemmatization for tokens.
    reconstruct_documents:
        Reconstruct documents from processed sentences.
    exclude_punctuation_replace_standalone_numbers:
        Exclude punctuation and replace standalone numbers.
    selective_casefold:
        Selectively casefold tokens.
    _save_checkpoints:
        Save processed abstracts to separate files for w2v and finetune.
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

    extras = {
        ".",
        "\\",
        "~",
        "*",
        "&",
        "#",
        "'",
        '"',
        "^",
        "$",
        "|",
        "(",
        ")",
        "[",
        "]",
        "!",
        "''",
        "+",
        "'s",
        "?",
        "@",
        "**",
        "â",
        "Å",
        ":",
        "©",
        "®",
        ",",
        ">",
        "<",
        "′′",
    }

    finetune_extras = {
        "\\",
        "~",
        "*",
        "&",
        "#",
        "# ",
        '"',
        "^",
        "$",
        "|",
        "!",
        "''",
        "+",
        "'s",
        "?",
        "& ",
        "@",
        "@ ",
        "**",
        "©",
        "®",
        "â",
        "“",
        "”",
        "′′",
    }

    def __init__(
        self,
        root_dir: str,
        abstracts: pd.DataFrame,
        chunk: int,
        genes: Set[str],
        diseases: Set[str],
        batch_size: int = 64,
    ):
        """Initialize the ChunkedDocumentProcessor object."""
        self.root_dir = root_dir
        self.chunk = chunk
        self.batch_size = batch_size
        self.genes = {gene.lower() for gene in genes}
        self.diseases = {disease.lower() for disease in diseases}
        for gene in ["mice", "bad", "insulin", "camp", "plasminogen", "ski"]:
            self.genes.remove(gene)

        self.df = abstracts[["modified_abstracts", "year"]]
        self.nlp: Language = None
        self.spacy_model: str = ""
        self.main_entities = self.genes | self.diseases

    def _make_directories(self) -> None:
        """Make directories for processing"""
        for dir in [
            "data",
            "models/gram_models",
            "models/sentence_models",
            "models/w2v_models",
        ]:
            dir_check_make(dir)

    def setup_pipeline(self) -> None:
        """Set up the spaCy pipeline"""
        self.spacy_model = "en_core_sci_lg"
        logger.info(f"Loading spaCy model: {self.spacy_model}")

        self.nlp = spacy.load(self.spacy_model)

        # add entity ruler for genes
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        gene_patterns = [{"label": "GENE", "pattern": gene} for gene in self.genes]
        ruler.add_patterns(gene_patterns)

        # add sentencizer if not present
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        # disable unnecessary pipeline components
        disable_pipes = ["parser"]
        self.nlp.disable_pipes(*disable_pipes)

        logger.info(f"Pipeline components: {self.nlp.pipe_names}")

    def custom_lemmatize(self, token: Token, lemmatize: bool = True) -> str:
        """Custom token processing. Only lemmatize tokens that are not
        recognized as entities via NER.
        """
        if token.ent_type_ in ["ENTITY", "GENE"]:
            return token.text
        else:
            return token.lemma_ if lemmatize else token.text

    def replace_symbol_tokens(self, token: str, finetune: bool = False) -> List[str]:
        """Replace symbols with an empty string. Replace standalone numbers with
        '<nUm>' only for w2v.
        """
        # remove extras in tokens
        if not finetune:
            # deal with specific cases
            token = re.sub(r"[-/]", " ", token)  # replace '-' and '/' with space
            token = re.sub(r"(\w),(\w)", r"\1 \2", token)  # replace ',' with '_'
            token = re.sub(r"(\w),\s+(\w)", r"\1 \2", token)  # replace ', ' with ' '

            # now remove extras
            sorted_extras = sorted(self.extras, key=len, reverse=True)
            escaped_extras = [re.escape(extra) for extra in sorted_extras]
            pattern = "(" + "|".join(escaped_extras) + ")"
            token = re.sub(pattern, "", token)

            # collapse spaces
            token = re.sub(r"\s+", " ", token).strip()

        # if token is just a symbol, skip it
        if (not finetune and token in self.extras) or (
            finetune and token in self.finetune_extras
        ):
            return []

        # For finetune, keep periods and commas
        if finetune and token in {".", ","}:
            return [token]

        # check if the token contains letters
        if re.search("[a-zA-Z]", token):
            return token.split()

        # check if there are numbers
        if re.search("[0-9]", token):
            return [token] if finetune else ["<nUm>"]

        return []

    def process_document(self, doc: Doc) -> Tuple[List[List[str]], List[str]]:
        """Splits a spaCy doc into sentences."""
        doc_sentences_w2v = []
        doc_tokens_finetune = []

        for sent in doc.sents:
            sent_tokens_w2v = []
            for token in sent:
                token_w2v = self.custom_lemmatize(token, lemmatize=True)
                token_ft = token.text
                sent_tokens_w2v.append(token_w2v)
                doc_tokens_finetune.append(token_ft)
            doc_sentences_w2v.append(sent_tokens_w2v)

        return doc_sentences_w2v, doc_tokens_finetune

    @time_decorator(print_args=False)
    def tokenize(self) -> None:
        """Tokenize the abstracts using spaCy with batch processing and
        standardize entities.
        """
        texts = self.df["modified_abstracts"].tolist()
        doc_indices = self.df.index.tolist()

        processed_sents_w2v, processed_tokens_ft = [], []

        for _, doc in tqdm(
            zip(
                doc_indices,
                self.nlp.pipe(texts, batch_size=self.batch_size, n_process=1),
            ),
            total=len(texts),
            desc="Tokenizing abstracts and NER",
        ):
            sents_w2v, tokens_ft = self.process_document(doc)
            processed_sents_w2v.append(sents_w2v)
            processed_tokens_ft.append(tokens_ft)

        self.df["processed_abstracts_w2v"] = pd.Series(
            processed_sents_w2v, index=doc_indices
        )
        self.df["processed_abstracts_finetune"] = pd.Series(
            processed_tokens_ft, index=doc_indices
        )

    @time_decorator(print_args=False)
    def exclude_symbols(self) -> None:
        """Exclude punctuation tokens, replace standalone numbers, and remove
        double spaces in w2v tokens.
        """
        tqdm.pandas(desc="Cleaning tokens")

        # helper function for w2v
        def clean_sentences(sentences: List[List[str]]) -> List[List[str]]:
            """Clean a list of list of tokens by excluding symbols and removing double spaces."""
            cleaned_sentences = []
            for sent in sentences:
                cleaned_sent = []
                for token in sent:
                    pieces = self.replace_symbol_tokens(token.casefold())
                    if pieces is not None:
                        cleaned_sent.extend(pieces)
                cleaned_sentences.append(cleaned_sent)
            return cleaned_sentences

        # process w2v
        self.df["processed_abstracts_w2v"] = self.df[
            "processed_abstracts_w2v"
        ].progress_apply(clean_sentences)

        # process finetune
        self.df["processed_abstracts_finetune"] = self.df[
            "processed_abstracts_finetune"
        ].progress_apply(
            lambda tokens: [
                replacement
                for token in tokens
                if (replacement := self.replace_symbol_tokens(token, finetune=True))
                is not None
            ]
        )

    @time_decorator(print_args=False)
    def dedupe_gene_disease_tokens(self) -> None:
        """Remove consecutive duplicates of gene or disease tokens that occur
        due to normalization of entities that are also marked with abbreviation
        of that entity.
        E.g. "TP53 TP53" -> "TP53", "VEGFC (VEGFC)" -> "VEGFC"
        """

        def deduplicate_sentences_w2v(
            sentences: List[List[str]],
        ) -> List[List[str]]:
            """Helper to dedupe for w2v, which consists of a list of list of
            tokens.
            """
            new_sentences = []
            for sent in sentences:
                new_sent = []
                prev_token = None
                for token in sent:
                    if token == prev_token and token in self.main_entities:
                        continue
                    new_sent.append(token)
                    prev_token = token
                new_sentences.append(new_sent)
            return new_sentences

        def deduplicate_tokens_finetune(
            tokens: List[Union[str, List[str]]]
        ) -> List[str]:
            """Helper to dedupe for finetune, which consists of a list of
            tokens.
            """
            # flatten the list for downstream
            flat_tokens: List[str] = []
            for token in tokens:
                if isinstance(token, list):
                    flat_tokens.extend(token)
                else:
                    flat_tokens.append(token)

            # dedupe
            new_tokens = []
            prev_token = None
            for token in flat_tokens:
                if token == prev_token and token in self.main_entities:
                    continue
                new_tokens.append(token)
                prev_token = token
            return new_tokens

        tqdm.pandas(desc="Deduplicating repeated gene/disease tokens (w2v)")
        self.df["processed_abstracts_w2v"] = self.df[
            "processed_abstracts_w2v"
        ].progress_apply(deduplicate_sentences_w2v)

        tqdm.pandas(desc="Deduplicating repeated gene/disease tokens (finetune)")
        self.df["processed_abstracts_finetune"] = self.df[
            "processed_abstracts_finetune"
        ].progress_apply(deduplicate_tokens_finetune)

    @time_decorator(print_args=False)
    def remove_genes_and_diseases(self) -> None:
        """Remove gene symbols from tokens, for future n-gram generation. Only
        done for w2v.
        """

        # helper function
        def remove_tokens(
            sentences: List[List[str]],
            tokens_for_removal: Set[str] = self.main_entities,
        ) -> List[List[str]]:
            """Remove gene symbols from a list of list of tokens."""
            return [
                [token for token in sent if token not in tokens_for_removal]
                for sent in sentences
            ]

        # remove genes and diseases
        with tqdm(desc="Removing genes and diseases") as pbar:
            self.df["processed_abstracts_w2v_nogenes"] = [
                remove_tokens(sent) for sent in self.df["processed_abstracts_w2v"]
            ]
            pbar.update()

    def save_data(self, outpref: str) -> None:
        """Save final copies of abstracts with cleaned, processed, and year
        columns.
        """
        if "processed_abstracts_finetune" in self.df.columns:
            finetune_outpref = f"{outpref}_finetune_chunk_{self.chunk}.pkl"
            columns_to_save = [
                "modified_abstracts",
                "year",
                "processed_abstracts_finetune",
            ]

            self.df[columns_to_save].to_pickle(finetune_outpref)

        if "processed_abstracts_w2v" in self.df.columns:
            # remove finetune for next save
            columns_to_save.remove("processed_abstracts_finetune")

            w2v_outpref = f"{outpref}_w2v_chunk_{self.chunk}.pkl"
            columns_to_save.extend(
                ("processed_abstracts_w2v", "processed_abstracts_w2v_nogenes")
            )
            self.df[columns_to_save].to_pickle(w2v_outpref)
            logger.info(f"Saved processed abstracts for chunk {self.chunk}")

    @time_decorator(print_args=False)
    def processing_pipeline(self) -> None:
        """Run the NLP pipeline for finetuning."""
        # setup spaCy pipeline
        logger.info("Setting up spaCy pipeline")
        self.setup_pipeline()

        # tokenization and NER
        logger.info("Tokenizing and NER")
        self.tokenize()

        # exclude punctuation and replace standalone numbers
        logger.info("Excluding punctuation and replacing standalone numbers")
        self.exclude_symbols()

        # deduplicate gene and disease tokens
        logger.info("Deduplicating gene and disease tokens")
        self.dedupe_gene_disease_tokens()

        logger.info("Removing gene symbols and disease names from phraser training")
        self.remove_genes_and_diseases()
        self.save_data(f"{self.root_dir}/data/processed_abstracts")


def main() -> None:
    """Main function"""
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
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="abstracts_with_normalized_entities",
    )
    parser.add_argument(
        "--disease_names",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/CTD_diseases.tsv",
    )
    args = parser.parse_args()

    # load abstract df
    abstracts = pd.read_pickle(
        f"{args.root_dir}/data/{args.file_prefix}_{args.chunk}.pkl"
    )

    # check that we have the required "year" column
    if "year" not in abstracts.columns:
        raise ValueError("Abstracts must have a 'year' column")

    # get genes
    gencode = gencode_genes(gtf=args.gene_gtf)
    hgnc = hgnc_ncbi_genes(tsv=args.hgnc_genes, hgnc=True)
    ncbi = hgnc_ncbi_genes(tsv=args.ncbi_genes)
    genes = gencode.union(hgnc).union(ncbi)
    genes = {gene.casefold() for gene in genes}
    diseases = load_normalized_disease_names(args.disease_names)

    # instantiate document processor
    documentProcessor = ChunkedDocumentProcessor(
        root_dir=args.root_dir,
        abstracts=abstracts,
        chunk=args.chunk,
        genes=genes,
        diseases=diseases,
    )

    # run processing pipeline
    documentProcessor.processing_pipeline()


if __name__ == "__main__":
    main()
