#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Applying HunFlair2 to tag and normalize gene and disease entities.

Our texts are in full abstracts, so we use the flair splitter to split each
abstract into sentences, before applying the flair tagger then normalizing gene
and disease entitites.
"""

import argparse
import pickle
from typing import List, Tuple

import flair  # type: ignore
from flair.data import Sentence  # type: ignore
from flair.models import EntityMentionLinker  # type: ignore
from flair.nn import Classifier  # type: ignore
import pandas as pd
import torch  # type: ignore
from tqdm import tqdm  # type: ignore


class EntityNormalizer:
    """Class to handle entity tagging, linking, and normalization."""

    def __init__(self):
        """Initialize the EntityNormalizer with Flair tools."""
        self.tagger = Classifier.load("hunflair2")
        self.disease_linker = EntityMentionLinker.load("disease-linker")
        self.gene_linker = EntityMentionLinker.load("gene-linker")

    def process_abstracts(
        self, abstracts: pd.DataFrame, batch_size: int = 512, sub_batch_size: int = 128
    ) -> pd.DataFrame:
        """Process abstracts with batched entity tagging and linking."""
        dfs_to_concat = []
        num_abstracts = len(abstracts)
        num_batches = (num_abstracts + batch_size - 1) // batch_size

        with tqdm(total=num_batches, desc="Processing batches") as pbar:
            for start_idx in range(0, num_abstracts, batch_size):
                end_idx = start_idx + batch_size
                batch_df = abstracts.iloc[start_idx:end_idx].copy()

                # create sentences
                all_sents, mapping = self._create_sentences(
                    texts=batch_df["cleaned_abstracts"].tolist()
                )

                # tag and link entities
                self._tag_and_link_entities(
                    sentences=all_sents, mini_batch_size=sub_batch_size
                )

                # normalize entities
                modified_sents = [
                    self._replace_entities_with_links(s) for s in all_sents
                ]

                # rebuild texts
                joined_texts = self._rebuild_texts(
                    texts=modified_sents, mapping=mapping, n_abstracts=len(batch_df)
                )

                batch_df["modified_abstracts"] = joined_texts
                dfs_to_concat.append(batch_df)

                pbar.update(1)

        return pd.concat(dfs_to_concat, ignore_index=True)

    def _create_sentences(
        self,
        texts: List[str],
        max_tokens: int = 512,
        chunk_len: int = 400,
    ) -> Tuple[List[Sentence], List[int]]:
        """Convert each text into a list of Sentence objects. Chunks the
        abstract if it passes token length.
        """
        all_sents: List[Sentence] = []
        mapping: List[int] = []

        for abs_idx, text in enumerate(texts):
            tokens = text.split()
            if len(tokens) <= max_tokens:
                all_sents.append(Sentence(text, use_tokenizer=True))
                mapping.append(abs_idx)
            else:
                chunked_texts = self.chunk_long_abstract(text, max_len=chunk_len)
                for chunk_text in chunked_texts:
                    all_sents.append(Sentence(chunk_text, use_tokenizer=True))
                    mapping.append(abs_idx)

        return all_sents, mapping

    def _tag_and_link_entities(
        self, sentences: List[Sentence], mini_batch_size: int = 128
    ) -> None:
        """Tag and link entities in sentences, using manual mini-batching for
        linkers.

        Args:
            sentences (List[Sentence]): Sentence objects to process.
            linker_batch_size (int): Number of sentences to process per linker
            sub-batch.
        """
        self.tagger.predict(sentences, mini_batch_size)

        for i in range(0, len(sentences), mini_batch_size):
            sub_batch = sentences[i : i + mini_batch_size]

            try:
                self.disease_linker.predict(sub_batch)
                self.gene_linker.predict(sub_batch)

            except UnicodeDecodeError as e:
                print(
                    f"Error linking sub-batch {i // mini_batch_size}: {e}. "
                    "Cleaning text and retrying..."
                )
                cleaned_sub_batch = self.clean_text(sub_batch)
                self.disease_linker.predict(cleaned_sub_batch)
                self.gene_linker.predict(cleaned_sub_batch)
                sentences[i : i + mini_batch_size] = cleaned_sub_batch

    def _replace_entities_with_links(self, sentence: Sentence) -> str:
        """Replace tagged entities in the sentence with their normalized
        names.
        """
        spans = sentence.get_spans("link")
        # sort in reverse so replacements don't affect subsequent offsets
        sorted_spans = sorted(spans, key=lambda span: span.start_position, reverse=True)
        modified_text = sentence.text

        for span in sorted_spans:
            if linked_label := span.get_label("link"):
                if normalized_name := self._extract_normalized_name(str(linked_label)):
                    start, end = span.start_position, span.end_position
                    modified_text = (
                        modified_text[:start] + normalized_name + modified_text[end:]
                    )

        return modified_text

    def _rebuild_texts(
        self,
        texts: List[str],
        mapping: List[int],
        n_abstracts: int,
    ) -> List[str]:
        """Rebuild the list of chunked/modified texts into their corresponding
        abstracts.

        Arguments:
            texts: The modified text pieces.
            mapping: Indices mapping each piece back to its abstract ID.
            n_abstracts: Number of abstracts in this batch.
        """
        abstract_sents: List[List[str]] = [[] for _ in range(n_abstracts)]
        for txt, abs_idx in zip(texts, mapping):
            abstract_sents[abs_idx].append(txt)
        return [" ".join(slist) for slist in abstract_sents]

    @staticmethod
    def _extract_normalized_name(linked_value: str) -> str:
        """Extract the normalized name from the linked identifier.

        Arguments:
            linked_value (str): The linked identifier string (e.g.,
            "MESH:D007239/name=Infections").

        Returns:
            str: The extracted normalized name (e.g., "Infections").
        """
        return linked_value.split("/name=", 1)[1].split(" (")[0]

    @staticmethod
    def chunk_long_abstract(text: str, max_len: int = 400) -> List[str]:
        """Chunk a long abstract into smaller pieces."""
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_len, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(" ".join(chunk_tokens))
            start = end
        return chunks

    @staticmethod
    def clean_text(sentences: List[Sentence]) -> List[Sentence]:
        """Clean text by removing non-utf-8 characters. Get raw text, clean, and
        return as Sentence.
        """
        cleaned_sentences = []
        for sent in sentences:
            raw_text = sent.to_original_text()
            cleaned_text = raw_text.encode("utf-8", errors="ignore").decode("utf-8")
            cleaned_sentences.append(Sentence(cleaned_text, use_tokenizer=True))
        return cleaned_sentences


def load_abstracts(file_path: str) -> pd.DataFrame:
    """Load abstracts from a pickle file."""
    with open(file_path, "rb") as f:
        abstracts = pickle.load(f)
    return abstracts


def main() -> None:
    """Main function to process abstracts and replace entities with normalized
    names.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_idx",
        type=int,
        default=0,
        help="Index of the file chunk to process (0-19)",
    )
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="abstracts_logistic_classified_tfidf_40000_chunk_part",
        help="Prefix for the output filename.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/data",
        help="Path to classified abstract pickles.",
    )
    args = parser.parse_args()

    # ensure gpu
    flair.device = torch.device("cuda:0")

    # define input and output file paths
    input_file = f"{args.path}/{args.filename_prefix}_{args.file_idx}.pkl"
    output_file = f"{args.path}/abstracts_with_normalized_entities_{args.file_idx}.pkl"
    print(f"Processing file {input_file}...")
    print(f"And saving to {output_file}...")

    # process abstracts
    abstracts = load_abstracts(input_file)
    normalizer = EntityNormalizer()
    processed_abstracts = normalizer.process_abstracts(abstracts)
    processed_abstracts.to_pickle(output_file)
    print("Processing complete.")


if __name__ == "__main__":
    main()
