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
from typing import Callable, List, Set, Tuple, Union

import ahocorasick  # type: ignore
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("document_processor_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
        self.batch_size = batch_size
        self.genes = {gene.lower() for gene in genes}
        for gene in ["mice", "bad", "insulin", "camp", "plasminogen", "ski"]:
            self.genes.remove(gene)

        self.df = abstracts[["cleaned_abstracts", "year"]]

        self.nlp: Language = None
        self.spacy_model: str = ""

        # for efficient gene matching
        self.automaton = ahocorasick.Automaton()
        for gene in self.genes:
            self.automaton.add_word(gene, gene)
        self.automaton.make_automaton()

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
        self.nlp.tokenizer = self._custom_tokenizer(self.nlp)

        # add entity ruler for genes
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        gene_patterns = [{"label": "GENE", "pattern": gene} for gene in self.genes]
        ruler.add_patterns(gene_patterns)

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

    def replace_symbol_tokens(
        self, token: str, finetune: bool = False
    ) -> Union[str, None]:
        """Replace symbols with an empty string. Replace standalone numbers with
        '<nUm>' only for w2v.
        """
        if (not finetune and token in self.extras) or (
            finetune and token in self.finetune_extras
        ):
            return None

        # check if the token contains letters
        if re.search("[a-zA-Z]", token):
            return token

        # check if there are numbers
        if re.search("[0-9]", token):
            return token if finetune else "<nUm>"

        return None

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

        # create a mapping from entity start index to the entity Span
        entities = {ent.start: ent for ent in doc_processed.ents}

        for sent in doc_processed.sents:
            sent_tokens_w2v = []
            sentence_start = sent.start
            sentence_end = sent.end

            current = sentence_start
            while current < sentence_end:
                token = doc_processed[current]
                if current in entities:
                    ent = entities[current]
                    token_w2v, token_ft = self.process_entity(ent)
                    current = ent.end  # skip the entire entity span
                else:
                    token_w2v, token_ft = self.process_token(token)
                    current += 1
                doc_tokens_finetune.append(token_ft)
                sent_tokens_w2v.append(token_w2v)
            doc_sentences_w2v.append(sent_tokens_w2v)
        return doc_sentences_w2v, doc_tokens_finetune

    def get_canonical_entity(self, ent: Span, lemmatize: bool) -> str:
        """Normalize entities in a document using the UMLS term."""
        # check if the entity text is a number
        if is_number(ent.text):
            return "<nUm>"

        # don't lemmatize or canonicalize genes
        if ent.label_ == "GENE":
            return ent.text

        if ent._.kb_ents:
            kb_id, score = ent._.kb_ents[0]
            if score >= self.nlp.get_pipe("scispacy_linker").threshold:
                canonical_name = (
                    self.nlp.get_pipe("scispacy_linker")
                    .kb.cui_to_entity[kb_id]
                    .canonical_name
                )
                canonical_name = self._remove_substring_from_token(
                    canonical_name
                ).casefold()

                # iterate over matches
                for end_index, gene in self.automaton.iter(canonical_name):
                    start_index = end_index - len(gene) + 1

                    # boundary check to only match full words
                    if (
                        start_index == 0
                        or not canonical_name[start_index - 1].isalnum()
                    ) and (
                        end_index + 1 == len(canonical_name)
                        or not canonical_name[end_index + 1].isalnum()
                    ):
                        logger.debug(
                            f"Matched gene '{gene}' in canonical name '{canonical_name}'"
                        )
                        return gene

                return canonical_name

        # lemmatize
        if not lemmatize:
            return ent.text

        lemmatized_tokens = [token.lemma_ for token in ent]
        return " ".join(lemmatized_tokens)

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
            "-",
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

    @staticmethod
    def _custom_tokenizer(nlp: Language) -> Callable[[str], Doc]:
        """Add few custom rules to tokenizer."""
        default_tokenizer = nlp.tokenizer

        def custom_tokenizer_func(text: str) -> Doc:
            text = gene_specific_tokenization(text)
            return default_tokenizer(text)

        return custom_tokenizer_func


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


def gene_specific_tokenization(text: str) -> str:
    """Various hand-crafted rules for tokenization of gene names."""
    # Replace various dash-like characters with '-'
    text = re.sub(r"[−–—]", "-", text)

    # Split on hyphens by replacing '-' with space
    text = re.sub(r"-", " ", text)

    # Split on '/' by replacing '/' with space
    text = re.sub(r"([A-Za-z0-9]+)\/([A-Za-z0-9]+)", r"\1 \2", text)

    # Remove patterns like 'p53(+/+)' and 'p53(-/-)' -> 'p53'
    text = re.sub(r"([A-Za-z0-9]+)\([\+\-]\/[\+\-]\)", r"\1", text)

    # Remove patterns like 'p53-/-' and 'trp53+/' -> 'p53', 'trp53'
    text = re.sub(r"([A-Za-z0-9]+)[\+\-]?\/[\+\-]?", r"\1", text)

    # Remove '+' and '-' suffixes from words like 'p53+' or 'p53-'
    text = re.sub(r"([A-Za-z0-9]+)[\+\-]", r"\1", text)

    # Remove standalone '+' or '-' tokens
    text = re.sub(r"\b[\+\-]\b", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


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
