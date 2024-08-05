#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to extract positive examples from the experimental datasets.

Experimentally derived gene-gene relationships are used as true positive
examples to evaluate the prediction capabilities of rich semantic vectors. We
extract relationships from three different sources:

    1. Co-essential genes from Wainberg et al., Nature Genetics, 2021.
    2. `HI-union` protein-protein interactions from Luck et al., Nature, 2020.
    3. `OpenCell` protein-protein interactions from Cho et al., Science, 2022.
    
For negative sampling, we generate random negative pairs from the set of all
genes. However, because there's the chance that a negatively paired gene pair is
actually a true positive, we ensure that negatives pairs do not exist across all 3 of
the aformentioned sources as well as the STRING database and Gene Ontology (GO) 
annotation database.

*NOTE - while we tried to opt for a completely programmatic data
download, we found it difficult. The OpenCell file was no longer properly
linked at the time we tried to download it, and the coessential file at 10%
fdr was availabel only as an xlsx. Thus, users will have to manually
download those files."""

from collections import defaultdict
import csv
import gzip
import itertools
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
from typing import Dict, List, Set, Tuple, Union
import zipfile

import mygene  # type: ignore
import pandas as pd
import psutil  # type: ignore


class PrepareTrainingData:
    """Class to handle downloading then processing gene relationship
    datasets.
    """

    FILES_TO_DOWNLOAD = [
        # (
        #     "https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-021-00840-z/MediaObjects/41588_2021_840_MOESM3_ESM.zip",
        #     "coessential_interactions.zip",
        # ),
        # (
        #     "https://opencell.czbiohub.org/data/datasets/opencell-protein-interactions.csv",
        #     "opencell_interactions.csv",
        # ),
        ("http://www.interactome-atlas.org/data/HI-union.tsv", "hi_union.tsv"),
        (
            "https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz",
            "string_protein_links.txt.gz",
        ),
        (
            "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
            "string_protein_aliases.txt.gz",
        ),
        (
            "https://current.geneontology.org/annotations/goa_human.gaf.gz",
            "goa_human.gaf.gz",
        ),
    ]

    FILENAMES = {
        "coessential": "coessential_pairs.txt",
        "opencell": "science.abi6983_table_s4.xlsx",
        "hi_union": "hi_union.tsv",
        "string": "string_protein_links.txt",
        "string_mapper": "string_protein_aliases.txt",
        "goa": "goa_human.gaf",
        "go_mapper": "go_ids_to_gene_symbol.txt",
    }

    CORES = psutil.cpu_count(logical=False) - 1

    def __init__(self, output_dir: str = "./reference_files"):
        """Instantiate the class."""
        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def create_graphs(self) -> None:
        """Make all graphs and save them to the output directory. We first make
        each subgraph, then create some specific combined graphs. We combine the
        coessential, opencell, and hi_union graphs into a single positive graph.
        We also combine the string and go graphs into a `graph for
        filtering`.
        """
        # download files
        self.download_reference_files()

        # Create and save individual graphs
        graphs = {
            "coessential_pos": self.coessential_graph()[0],
            "coessential_neg": self.coessential_graph()[1],
            "opencell": self.opencell_graph(),
            "hi_union": self.hi_union_graph(),
            "string": self.physical_string_graph(),
            "go": self.go_graph(),
        }

        # Save individual graphs
        for name, graph in graphs.items():
            self.save_graph(graph, f"{name}_graph.pkl")

        # Create and save combined graphs
        experimentally_derived_edges = (
            self.gene_only_edges(graphs["coessential_pos"])
            | graphs["opencell"]
            | graphs["hi_union"]
        )
        self.save_graph(
            experimentally_derived_edges, "experimentally_derived_edges.pkl"
        )

        graph_for_filtering = graphs["string"] | graphs["go"]
        self.save_graph(graph_for_filtering, "graph_for_filtering.pkl")

        all_positive_edges = experimentally_derived_edges | graph_for_filtering
        self.save_graph(all_positive_edges, "all_positive_edges.pkl")

        print("All graphs created and saved successfully.")

    def negative_sampler(
        self,
        n_random_edges: int,
    ) -> Set[Tuple[str, ...]]:
        """Generate random negative samples.

        We first create a list of genes from the experimentally derived edges.
        We then initialize the negative samples by adding the experimentally
        derived negative samples. We start to generate negative samples at
        random, only keeping them if they are not in `all_positive_edges`.
        """

        def load_pickle(filename: str) -> Set[Tuple[str, ...]]:
            """Simple loading utility."""
            with open(self.output_dir / filename, "rb") as f:
                return pickle.load(f)

        all_positive_edges = load_pickle("all_positive_edges.pkl")
        exp_derived_edges = load_pickle("experimentally_derived_edges.pkl")
        exp_negative_edges = load_pickle("coessential_neg_graph.pkl")

        # make a list of all genes, used in experimentally derived edges to pull
        # from
        all_genes: List[str] = list(
            {gene for edge in exp_derived_edges for gene in edge}
        )

        # get negative samples from the experimentally derived negative samples
        negative_samples: Set[Tuple[str, ...]] = set(
            self.gene_only_edges(exp_negative_edges)
        )

        # randomly sample negative edges until len matched
        while len(negative_samples) < n_random_edges + len(exp_negative_edges):
            negative_edge = tuple(sorted(random.sample(all_genes, 2)))
            if (
                negative_edge not in all_positive_edges
                and negative_edge not in negative_samples
            ):
                negative_samples.add(negative_edge)

        return set(itertools.islice(negative_samples, n_random_edges))

    def physical_string_graph(self) -> Set[Tuple[str, ...]]:
        """Get the physical subset of StringDB interactions for homo sapiens."""
        db_file = self.output_dir / self.FILENAMES["string"]
        mapfile = self.output_dir / self.FILENAMES["string_mapper"]
        mapping = self._string_to_gene_symbol(mapfile=mapfile)
        edges = set()

        with open(db_file, "r") as file:
            reader = csv.reader(file, delimiter=" ")
            next(reader)  # skip header
            for row in reader:
                protein1 = row[0]
                protein2 = row[1]

                if protein1 in mapping and protein2 in mapping:
                    symbol1 = mapping[protein1]
                    symbol2 = mapping[protein2]
                    edges.add((symbol1, symbol2))
        return self.remove_duplicate_edges(edges)

    def hi_union_graph(self) -> Set[Tuple[str, ...]]:
        """Get edges from the HI-union dataset."""
        file_path = self.output_dir / self.FILENAMES["hi_union"]

        # get all ids for batch rename
        ensembl_ids = set()
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                ensembl_ids.update(row[:2])

        # convert ensembl ids to gene symbols
        id_to_symbol = self.batch_ensembl_to_gene_symbol(ensembl_ids)

        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            edges = {
                (
                    id_to_symbol.get(row[0], "Unknown"),
                    id_to_symbol.get(row[1], "Unknown"),
                )
                for row in reader
                if id_to_symbol.get(row[0], "Unknown") != "Unknown"
                and id_to_symbol.get(row[1], "Unknown") != "Unknown"
            }

        return self.remove_duplicate_edges(edges)

    def opencell_graph(self) -> Set[Tuple[str, ...]]:
        """Get edges from the OpenCell supplementary table S4 (interactome)."""

        def split_semicolon_edges(edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
            """Split edges that contain semicolons into multiple edges, which is
            present in the opencell dataset."""
            new_edges = set()
            for edge in edges:
                try:
                    if any(";" in node for node in edge):
                        split_nodes = [node.split(";") for node in edge]
                        for node1 in split_nodes[0]:
                            for node2 in split_nodes[1]:
                                new_edges.add((node1.strip(), node2.strip()))
                    else:
                        new_edges.add(edge)  # add as is
                except TypeError:
                    print(edge)

            return new_edges

        df = pd.read_excel(self.output_dir / self.FILENAMES["opencell"])
        edges = {
            tuple(row) for row in df.iloc[:, :2].itertuples(index=False, name=None)
        }
        return self.remove_duplicate_edges(split_semicolon_edges(edges))

    def coessential_graph(
        self,
    ) -> Tuple[Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
        """Parse the coessential file to extract positive and negative interactions."""
        with open(self.output_dir / self.FILENAMES["coessential"], "r") as file:
            reader = csv.reader(file, delimiter="\t")
            interactions = [(row[0], row[1], row[2]) for row in reader]
        negative_edges = {
            interaction for interaction in interactions if interaction[2] == "neg"
        }
        positive_edges = {
            interaction for interaction in interactions if interaction[2] == "pos"
        }
        return (
            self.remove_duplicate_edges(positive_edges),
            self.remove_duplicate_edges(negative_edges),
        )

    def go_graph(self) -> Set[Tuple[str, ...]]:
        """Link genes together via shared GO terms"""
        mapper = self._uniprot_to_gene_symbol(
            self.output_dir / self.FILENAMES["go_mapper"]
        )
        annotations = self._get_go_annotations(self.output_dir / self.FILENAMES["goa"])

        go_to_gene: Dict[str, List[str]] = {}
        for gene, go_term in annotations:
            if go_term not in go_to_gene:
                go_to_gene[go_term] = []
            if gene in mapper:
                go_to_gene[go_term].append(mapper[gene])

        def process_go_term(linked_genes: List[str]) -> Set[Tuple[str, ...]]:
            """Process individual terms"""
            return set(itertools.combinations(linked_genes, 2))

        # parallelized edge generation
        with multiprocessing.Pool(processes=self.CORES) as pool:
            all_edges = pool.map(process_go_term, go_to_gene.values())

        # flatten the list of sets and convert to a single set
        all_edges_flat = set().union(*all_edges)

        return self.remove_duplicate_edges(all_edges_flat)

    def _get_go_annotations(self, go_gaf: Path) -> List[Tuple[str, str]]:
        """Create GO ontology graph"""
        with open(go_gaf, newline="", mode="r") as file:
            reader = csv.reader(file, delimiter="\t")
            return [
                (row[1], row[4])
                for row in reader
                if not row[0].startswith("!")
                and row[6] not in ["IEA", "IEP", "IC", "ND"]
            ]

    def _uniprot_to_gene_symbol(self, mapfile: Path) -> Dict[str, str]:
        """Get dictionary for mapping Uniprot IDs to Gencode IDs. We keep the first
        gene if there are multiple genes that uniprot maps to.

        To get this mapfile, we cut uniq values from column 2 of the
        goa_human.gaf file and manually created a mapfile using the uniprot
        mapper tool.
        """
        with open(mapfile, "r") as file:
            return {row[0]: row[1] for row in csv.reader(file, delimiter="\t")}

    def _string_to_gene_symbol(self, mapfile: Path) -> Dict[str, str]:
        """Get a dictionary for mapping StringIDs to HGNC gene symbols."""
        string_mapper = {}
        with open(mapfile, "r") as file:
            for line in file:
                fields = line.strip().split("\t")
                if len(fields) == 3 and fields[2] == "Ensembl_HGNC_symbol":
                    stringid = fields[0]
                    gene_symbol = fields[1]
                    string_mapper[stringid] = gene_symbol
        return string_mapper

    def download_reference_files(self) -> None:
        """Download reference files for the link prediction task. We download the
        coessential, opencell, and hi-union interaction datasets. Additionally,
        we download the STRING and GO homosapiens datasets for using to filter
        the randomly generated negative samples.

        Args:
            output_dir (str): Directory to save downloaded files.
        """
        for url, filename in self.FILES_TO_DOWNLOAD:
            output_path = self.output_dir / filename
            try:
                subprocess.run(["wget", "-O", output_path, url], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["wget", "-O", output_path, url, "--no-check-certificate"],
                    check=True,
                )
            print(f"Downloaded {filename}")

            # decompress files
            if filename.endswith(".gz"):
                self.decompress_gz(output_path)
                print(f"Decompressed {filename}")
            elif filename.endswith(".zip"):
                self.decompress_zip(output_path)
                print(f"Decompressed {filename}")

        print("All files downloaded and processed successfully.")

    def save_graph(self, graph: Set[Tuple[str, ...]], filename: str) -> None:
        """Simple utility to pickle edges"""
        output_path = self.output_dir / filename
        with open(output_path, "wb") as f:
            pickle.dump(graph, f)

    @staticmethod
    def batch_ensembl_to_gene_symbol(ensembl_ids: Set[str]) -> Dict[str, str]:
        """Convert a set of ENSEMBL gene IDs to gene symbols in one batch."""
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            list(ensembl_ids), scopes="ensembl.gene", fields="symbol", species="human"
        )
        return {
            result["query"]: result["symbol"] if "symbol" in result else "Unknown"
            for result in results
        }

    @staticmethod
    def remove_duplicate_edges(
        edges: Union[
            Set[Tuple[str, str]], Set[Tuple[str, str, str]], Set[Tuple[str, ...]]
        ]
    ) -> Set[Union[Tuple[str, str], Tuple[str, str, str], Tuple[str, ...]]]:
        """Remove duplicates from a set of tuples.
        For Tuple[str, str]: Deduplicates based on both elements.
        For Tuple[str, str, str]: Deduplicates based on first two elements,
        preserving the third.
        """
        # get tuple length from first item
        tuple_length = len(next(iter(edges)))

        if tuple_length == 2:
            return {tuple(sorted(edge)) for edge in edges}
        elif tuple_length == 3:
            edges_only = defaultdict(list)
            for edge in edges:
                key = tuple(sorted(edge[:2]))
                edges_only[key].append(edge)
            return {min(group, key=lambda x: x[2]) for group in edges_only.values()}
        else:
            raise ValueError("Input set must contain tuples of length 2 or 3")

    @staticmethod
    def decompress_gz(file_path: Path) -> None:
        """Decompress a .gz file"""
        basename = os.path.basename(file_path)
        with gzip.open(file_path, "rb") as gzipped:
            with open(f"{file_path.parent}/{basename[:-3]}", "wb") as decompressed:
                shutil.copyfileobj(gzipped, decompressed)
        os.remove(file_path)  # remove compressed

    @staticmethod
    def decompress_zip(file_path: Path) -> None:
        """Decompress a .zip file"""
        basename = os.path.basename(file_path)
        with zipfile.ZipFile(file_path, "r") as zipped:
            zipped.extractall(f"{file_path.parent}/{basename[:-4]}")
        os.remove(file_path)  # remove compressed

    @staticmethod
    def gene_only_edges(edges: Set[Tuple[str, ...]]) -> Set[Tuple[str, ...]]:
        """Return a tuple with only the gene symbols, not the third element."""
        return {tuple(sorted(edge[:2])) for edge in edges}
