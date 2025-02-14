#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce node embeddings using node2vec on the text-extracted gene
relationships."""


import argparse
from pathlib import Path
import pickle
from typing import Any

from grape import Graph  # type: ignore
import grape  # type: ignore
from grape.embedders import Node2VecSkipGramEnsmallen  # type: ignore
import numpy as np


def save_embeddings(embeddings: Any, index: int, model_dir: Path, outname: Any) -> None:
    """Convert ensmallen embeddings to dict and save to file."""
    embedding_vectors = embeddings._node_embeddings[index].to_dict("index")
    embedding_vectors = {
        k: np.array(list(v.values())) for k, v in embedding_vectors.items()
    }
    with open(model_dir / outname, "wb") as f:
        pickle.dump(embedding_vectors, f)


def main() -> None:
    """Load the graph and train embeddings."""
    parser = argparse.ArgumentParser(description="Train node2vec embeddings.")
    parser.add_argument(
        "--year",
        type=str,
        help="Edge file to load.",
        default="2003",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to save the model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/n2v",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        help="Embedding type.",
        default="disease",
        choices=["ppi", "disease"],
    )
    args = parser.parse_args()
    if args.embedding_type == "ppi":
        edge_file = f"/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi/gene_co_occurence_{args.year}.tsv"
    elif args.embedding_type == "disease":
        edge_file = f"/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/gda_co_occurence_{args.year}.tsv"
    else:
        raise ValueError("Invalid embedding type. Must be 'ppi' or 'disease'.")

    # make year dir
    model_dir = Path(f"{args.model_dir}/{args.embedding_type}") / args.year
    model_dir.mkdir(parents=True, exist_ok=True)

    graph = grape.Graph.from_csv(
        edge_path=edge_file,
        # sources_column="source",
        # destinations_column="destination",
        directed=False,
    )
    print("Graph loaded.")

    embeddings = Node2VecSkipGramEnsmallen(embedding_size=128).fit_transform(graph)

    save_embeddings(embeddings, 0, model_dir, "input_embeddings.pkl")
    save_embeddings(embeddings, 1, model_dir, "output_embeddings.pkl")


if __name__ == "__main__":
    main()

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

# # load edges
# edges = pd.read_csv(
#     edge_file, sep="\t", header=None, names=["source", "destination"]
# )
# edges = edges.drop_duplicates()
