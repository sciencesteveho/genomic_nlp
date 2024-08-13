#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Simple script to format abstracts as PubTator format for use with GNorm2."""


import contextlib
import os
import pickle
from typing import List

import pandas as pd


def _replace_none_abstract(abstract: str) -> str:
    """Replace None abstracts pubtator empty string"""
    return "-no abstract-" if abstract == "None" else abstract


def format_abstracts_as_pubtator(
    abstracts: List[str], fake_id: bool = True
) -> List[str]:
    """Convert a normal abstract string (title and abstract) to PubTator
    format."""
    reformatted_abstracts = []

    for idx, abstract in enumerate(abstracts, start=1):
        if fake_id:
            idx = 36206680
        with contextlib.suppress(ValueError):
            title, content = abstract.split(".", 1)

        # adjust empty abstracts to fit pubtator format
        content = _replace_none_abstract(content)

        reformatted_abstract = f"{idx}|t|{title.strip()}.\n{idx}|a|{content.strip()}"
        reformatted_abstracts.append(reformatted_abstract)

    return reformatted_abstracts


def format_df_as_pubtator(df: pd.DataFrame, outdir: str) -> None:
    """Convert a normal abstract string (title and abstract) to Pubtator
    format."""

    def format_entry(row: pd.Series) -> str:
        """Formats each row of a df"""
        return f"{row['pubmed_id']}|t|{row['title']}\n{row['pubmed_id']}|a|{row['description']}\n"

    # fill missing parts
    df["pubmed_id"] = df["pubmed_id"].fillna(36206680)
    df["description"] = df["description"].fillna("-no abstract-")

    df_filtered = df[["pubmed_id", "title", "description"]]

    # write to a file, entry by entry
    with open(f"{outdir}/output.txt", "w", encoding="utf-8") as f:
        for _, row in df_filtered.iterrows():
            f.write(format_entry(row))
            f.write(os.linesep)


def main() -> None:
    """Main function"""
    # set output dir
    outdir = "/ocean/projects/bio210019p/stevesho/nlp/data/input"

    # load abstracts
    with open("relevant_abstracts.pkl", "rb") as f:
        corpus = pickle.load(f)

    # format as pubtator
    formatted_abstracts = format_abstracts_as_pubtator(corpus)

    # write abstracts out to a file
    with open(f"{outdir}/relevant_abstracts.pubtator", "w") as f:
        for entry in formatted_abstracts:
            f.write(entry)
            f.write(os.linesep)
            f.write(os.linesep)

    # load df
    with open("abstracts_combined.pkl", "rb") as f:
        df = pickle.load(f)

    # format as pubtator and write
    format_df_as_pubtator(df, outdir)


if __name__ == "__main__":
    main()
