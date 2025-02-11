#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce node embeddings using node2vec on the text-extracted gene
relationships."""


import argparse
import gc
from pathlib import Path
import pickle
from typing import Any

from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
import grape
from grape import Graph
from grape.embedders import Node2VecSkipGramEnsmallen
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
        "--year",
        type=str,
        required=True,
        help="Edge file to load.",
        default="2003",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to save the model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/n2v",
    )
    args = parser.parse_args()
    edge_file = f"/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi/gene_co_occurence_{args.year}.tsv"

    # make year dir
    model_dir = Path(args.model_dir) / args.year
    model_dir.mkdir(parents=True, exist_ok=True)

    # # load edges
    # edges = pd.read_csv(
    #     edge_file, sep="\t", header=None, names=["source", "destination"]
    # )
    # edges = edges.drop_duplicates()

    graph = grape.Graph.from_csv(
        edge_path=edge_file,
        # sources_column="source",
        # destinations_column="destination",
        directed=False,
    )
    print("Graph loaded.")

    # create graph
    # graph = nx.from_pandas_edgelist(edges, create_using=nx.Graph)

    # instantiate model
    # node2vec = Node2Vec(
    #     graph,
    #     dimensions=128,
    #     walk_length=40,
    #     num_walks=10,
    #     workers=24,
    # )
    embeddings = Node2VecSkipGramEnsmallen().fit_transform(graph)

    # train node2vec
    # model = node2vec.fit(
    #     window=10,
    #     min_count=1,
    #     batch_words=4,
    #     callbacks=[EpochSaver(str(model_dir))],
    # )

    # try and clear memory before saving
    # gc.collect()

    # model.save(str(model_dir / "node2vec.model"))
    # save embeddings
    node_names = graph.get_node_names()
    embedding_vectors = dict(zip(node_names, embeddings))
    with open(model_dir / "embeddings.no", "wb") as f:
        pickle.dump(embedding_vectors, f)


if __name__ == "__main__":
    main()
