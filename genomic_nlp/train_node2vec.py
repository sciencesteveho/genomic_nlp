#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce node embeddings using node2vec on the text-extracted gene
relationships."""


import argparse
import gc
import gzip
from pathlib import Path
import pickle
from typing import Any

from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
import networkx as nx  # type: ignore
from node2vec import Node2Vec  # type: ignore
import numpy as np
import pandas as pd


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, savedir: str):
        self.epoch = 0
        self.savedir = savedir

    def on_epoch_end(self, model: Any) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"{self.savedir}/model_epoch{self.epoch}.pkl")
        self.epoch += 1


def main() -> None:
    """Load the graph and train embeddings."""
    parser = argparse.ArgumentParser(description="Train node2vec embeddings.")
    parser.add_argument(
        "--edge_file",
        type=str,
        required=True,
        help="Path to the text extracted gene edges.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi/text_extracted_gene_edges_syns.tsv",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/n2v",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the embeddings.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings",
    )
    args = parser.parse_args()

    # load edges
    edges = pd.read_csv(
        args.edge_file, sep="\t", header=None, names=["source", "target"]
    )
    edges = edges.drop_duplicates()

    # create graph
    graph = nx.from_pandas_edgelist(edges, create_using=nx.Graph)

    # instantiate model
    node2vec = Node2Vec(
        graph,
        dimensions=128,
        walk_length=40,
        num_walks=10,
        workers=24,
    )

    # train node2vec
    model = node2vec.fit(
        window=10,
        min_count=1,
        batch_words=4,
        callbacks=[EpochSaver(args.model_dir)],
    )

    # try and clear memory before saving
    gc.collect()

    model.save(str(args.output_dir / "node2vec.model"))


if __name__ == "__main__":
    main()
