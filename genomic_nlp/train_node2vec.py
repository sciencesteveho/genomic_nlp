#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce node embeddings using node2vec on the text-extracted gene
relationships."""


from pathlib import Path

from gensim.models import Word2Vec  # type: ignore
from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
import networkx as nx  # type: ignore
from node2vec import Node2Vec  # type: ignore
import pandas as pd

from utils import EpochSaver


def main() -> None:
    """Load the graph and train embeddings."""
    edge_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi/text_extracted_gene_edges_syns.tsv"
    model_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/n2v"
    output_dir = Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings")

    # load edges
    edges = pd.read_csv(edge_file, sep="\t", header=None, names=["source", "target"])
    edges = edges.drop_duplicates()

    # create graph
    graph = nx.from_pandas_edgelist(edges, create_using=nx.Graph)

    # instantiate model
    node2vec = Node2Vec(
        graph,
        dimensions=128,
        walk_length=80,
        num_walks=10,
        workers=48,
    )

    # train node2vec
    model = node2vec.fit(
        window=10,
        min_count=1,
        batch_words=4,
        callbacks=[EpochSaver(model_dir)],
    )

    model.wv.save_word2vec_format(output_dir / "node2vec_embeddings.txt")


if __name__ == "__main__":
    main()
