# sourcery skip: lambdas-should-be-short
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Sentence splitting, tokenization, optional lemmatization, and some additional
cleanup."""


import argparse
from collections import defaultdict
import csv
import logging
import re
from typing import List, Set, Tuple, Union

import pandas as pd  # type: ignore
import pybedtools  # type: ignore
from scispacy.linking import EntityLinker  # type: ignore
import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy.tokens import Doc  # type: ignore
from spacy.tokens import Span  # type: ignore
from spacy.tokens import Token  # type: ignore
from tqdm import tqdm  # type: ignore

from utils import COPY_GENES
from utils import dir_check_make
from utils import is_number
from utils import time_decorator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("document_processor_debug.log"),
        logging.StreamHandler(),
    ],
)
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
        "/",
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
        ",",
        "-",
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
        batch_size: int = 64,
    ):
        """Initialize the ChunkedDocumentProcessor object."""
        self.root_dir = root_dir
        self.chunk = chunk
        self.genes = genes
        self.batch_size = batch_size

        self.df = abstracts[["cleaned_abstracts", "year"]]

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

    def setup_pipeline(self) -> None:
        """Set up the spaCy pipeline"""
        self.spacy_model = "en_core_sci_lg"
        logger.info(f"Loading spaCy model: {self.spacy_model}")

        self.nlp = spacy.load(self.spacy_model)

        # add entity ruler for genes
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        gene_patterns = [{"label": "GENE", "pattern": gene} for gene in self.genes]
        ruler.add_patterns(gene_patterns)

        # customize tokenizer to keep hyphens and underscores within tokens
        infix_re = re.compile(r"""[-~]""")
        self.nlp.tokenizer.infix_finditer = infix_re.finditer

        # add sentencizer if not present
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        # add entity linker if not present
        self.nlp.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "threshold": 0.95,
                "max_entities_per_mention": 1,
            },
            last=True,
        )

        # disable unnecessary pipeline components
        disable_pipes = ["parser", "tagger"]
        self.nlp.disable_pipes(*disable_pipes)

        # ensure morphologizer
        if "morphologizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("morphologizer", after="attribute_ruler")
            logger.info("Added 'morphologizer' to the pipeline.")
        else:
            logger.info("'morphologizer' is already active in the pipeline.")

            logger.info(f"Pipeline components: {self.nlp.pipe_names}")

    def custom_lemmatize(self, token: Token, lemmatize: bool = True) -> str:
        """Custom token processing. Only lemmatize tokens that are not
        recognized as entities via NER.
        """
        if token.ent_type_ == "ENTITY":
            return token.text
        else:
            return token.lemma_ if lemmatize else token.text

    def replace_symbol_tokens(
        self, token: str, finetune: bool = False
    ) -> Union[str, None]:
        """Replace numbers with a number based symbol, and symbols with an empty string."""
        if (
            not finetune
            and token in self.extras
            or finetune
            and token in self.finetune_extras
        ):
            return None
        return token

    def selective_casefold_token(self, token: str) -> str:
        """No longer selective casefolding, just normal casefolding. Too many
        instances of genes made it through, so we casefold everything
        to avoid loss of information.
        """
        return token.casefold()

    def process_entity(self, ent: Span) -> Tuple[str, str]:
        """Process an entity span and return tokens for w2v and finetune."""
        if ent.label_ == "GENE":
            casefolded_gene = ent.text.casefold()
            return casefolded_gene, casefolded_gene
        else:
            canonical_entity_w2v = self.get_canonical_entity(ent=ent, lemmatize=True)
            return canonical_entity_w2v.casefold(), ent.text

    def process_token(self, token: Token) -> Tuple[str, str]:
        """Process a single token and return tokens for w2v and finetune."""
        token_processed_w2v = self.custom_lemmatize(token, lemmatize=True)
        token_processed_ft = self.custom_lemmatize(token, lemmatize=False)
        return token_processed_w2v, token_processed_ft

    def process_document(self, doc_processed: Doc) -> Tuple[List[List[str]], List[str]]:
        """Process a single spaCy Doc and return tokens for w2v and finetune."""
        doc_sentences_w2v = []
        doc_tokens_finetune = []

        for sent in doc_processed.sents:
            sent_tokens_w2v = []
            sentence_start = sent.start
            while sentence_start < sent.end:
                token = doc_processed[sentence_start]
                if token.ent_iob_ == "B":
                    # beginning of an ent
                    ent = doc_processed[token.i : token.ent_end]
                    token_w2v, token_ft = self.process_entity(ent)
                    sent_tokens_w2v.append(token_w2v)
                    doc_tokens_finetune.append(token_ft)
                    sentence_start = token.ent_end  # skip to the end of the ent
                elif token.ent_iob_ == "I":
                    # inside an ent, already processed
                    sentence_start += 1
                else:
                    # not part of an ent
                    token_w2v, token_ft = self.process_token(token)
                    sent_tokens_w2v.append(token_w2v)
                    doc_tokens_finetune.append(token_ft)
                    sentence_start += 1
            doc_sentences_w2v.append(sent_tokens_w2v)
        return doc_sentences_w2v, doc_tokens_finetune

    @time_decorator(print_args=False)
    def tokenize_and_ner(self) -> None:
        """Tokenize the abstracts using spaCy with batch processing and
        standardize entities.
        """
        documents = self.df["cleaned_abstracts"].tolist()
        doc_indices = self.df.index.tolist()

        processed_sentences_w2v = []
        processed_tokens_finetune = []
        new_doc_indices = []

        with tqdm(
            total=len(documents), desc="Processing documents", unit="doc"
        ) as pbar:
            for doc_idx, doc_processed in zip(
                doc_indices,
                self.nlp.pipe(documents, batch_size=self.batch_size, n_process=4),
            ):
                try:
                    doc_sentences_w2v, doc_tokens_finetune = self.process_document(
                        doc_processed
                    )
                    processed_sentences_w2v.append(doc_sentences_w2v)
                    processed_tokens_finetune.append(doc_tokens_finetune)
                    new_doc_indices.append(doc_idx)
                except Exception as e:
                    logger.error(f"Error processing document {doc_idx}: {e}")

                pbar.update(1)

        self.df["processed_abstracts_w2v"] = pd.Series(
            processed_sentences_w2v, index=new_doc_indices
        )
        self.df["processed_abstracts_finetune"] = pd.Series(
            processed_tokens_finetune, index=new_doc_indices
        )

    def get_canonical_entity(self, ent: Span, lemmatize: bool) -> str:
        """Normalize entities in a document using the UMLS term."""
        if ent.label_ == "GENE":
            return ent.text.casefold()

        if ent._.kb_ents:
            kb_id, score = ent._.kb_ents[0]
            if score >= self.nlp.get_pipe("scispacy_linker").threshold:
                canonical_name = (
                    self.nlp.get_pipe("scispacy_linker")
                    .kb.cui_to_entity[kb_id]
                    .canonical_name
                )
                return self._remove_substring_from_token(canonical_name)

        if not lemmatize:
            return "<nUm>" if is_number(ent.text) else ent.text

        lemmatized_tokens = [token.lemma_ for token in ent]
        return " ".join(lemmatized_tokens)

    @time_decorator(print_args=False)
    def exclude_symbols(self) -> None:
        """Exclude punctuation tokens, replace standalone numbers, and remove double spaces in w2v tokens."""
        tqdm.pandas(desc="Cleaning tokens")

        # helper function for w2v
        def clean_sentences(sentences: List[List[str]]) -> List[List[str]]:
            """Clean a list of list of tokens by excluding symbols and removing double spaces."""
            cleaned_sentences = []
            for sent in sentences:
                cleaned_sent = []
                for token in sent:
                    replacement = self.replace_symbol_tokens(token)
                    if replacement is not None:
                        # remove double spaces
                        if "  " in replacement:
                            replacement = replacement.replace("  ", " ")
                        cleaned_sent.append(replacement)
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
    def remove_genes(self) -> None:
        """Remove gene symbols from tokens, for future n-gram generation. Only
        done for w2v.
        """
        tqdm.pandas(desc="Removing genes")

        # helper function
        def remove_genes_from_sentences(sentences: List[List[str]]) -> List[List[str]]:
            """Remove gene symbols from a list of list of tokens."""
            cleaned_sentences = []
            for sent in sentences:
                cleaned_sent = [token for token in sent if token not in self.genes]
                cleaned_sentences.append(cleaned_sent)
            return cleaned_sentences

        self.df["processed_abstracts_w2v_nogenes"] = self.df[
            "processed_abstracts_w2v"
        ].progress_apply(remove_genes_from_sentences)

    def save_data(self, outpref: str) -> None:
        """Save final copies of abstracts with cleaned, processed, and year
        columns.
        """
        columns_to_save = ["cleaned_abstracts", "year"]

        if "processed_abstracts_finetune" in self.df.columns:
            finetune_outpref = f"{outpref}_finetune_chunk_{self.chunk}.pkl"
            columns_to_save.append("processed_abstracts_finetune")
            self.df[columns_to_save].to_pickle(finetune_outpref)
            columns_to_save.remove(
                "processed_abstracts_finetune"
            )  # remove finetune for next save

        if "processed_abstracts_w2v" in self.df.columns:
            w2v_outpref = f"{outpref}_w2v_chunk_{self.chunk}.pkl"
            columns_to_save.extend(
                ("processed_abstracts_w2v", "processed_abstracts_w2v_nogenes")
            )
            self.df[columns_to_save].to_pickle(w2v_outpref)
            logger.info(f"Saved processed abstracts for chunk {self.chunk}")

    @time_decorator(print_args=False)
    def processing_pipeline(self) -> None:
        """Run the NLP pipeline for both w2v and BERT fine-tuning."""
        # setup spaCy pipeline
        logger.info("Setting up spaCy pipeline")
        self.setup_pipeline()

        # tokenization and NER
        logger.info("Tokenizing and NER")
        self.tokenize_and_ner()

        # exclude punctuation and replace standalone numbers
        logger.info("Excluding punctuation and replacing standalone numbers")
        self.exclude_symbols()

        # additional processing for w2v and finetune
        logger.info("Removing genes")
        self.remove_genes()
        self.save_data(f"{self.root_dir}/data/processed_abstracts")

    @staticmethod
    def _remove_substring_from_token(token: str) -> str:
        """Remove 'gene' which is added to all ULMS genes."""
        if token.endswith(" gene"):
            token = token.rsplit(" ", 1)[0]
        else:
            token = re.sub(r"\s+[\(\[].*?[\)\]]$", "", token)

        # remove extra characters
        # ner specific issues
        replacements = [
            " -",
            ",",
            "[",
            " ] )",
            " '",
            "( ",
            " )",
        ]

        for replacement in replacements:
            token = token.replace(replacement, " ")
        for extra in ChunkedDocumentProcessor.extras:
            token = token.replace(extra, "")

        return token


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
    genes = {gene.casefold() for gene in genes}

    # instantiate document processor
    documentProcessor = ChunkedDocumentProcessor(
        root_dir=args.root_dir,
        abstracts=abstracts,
        chunk=args.chunk,
        genes=genes,
    )

    # run processing pipeline
    documentProcessor.processing_pipeline()


if __name__ == "__main__":
    main()
