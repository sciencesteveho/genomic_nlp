#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to extract positive examples from the experimental datasets.

Experimentally derived gene-gene relationships are used as true positive
examples to evaluate the prediction capabilities of rich semantic vectors. We
extract relationships from three different sources:

    1. Co-essential genes from Wainberg et al., Nature Genetics, 2021.
    2. `HI-union` protein-protein interactions from Luck et al., Nature, 2020.
    3. `OpenCell` protein-protein interactions from Cho et al., Science, 2022.
    4. `sc_cop` single-cell gene co-expression pairs from Ribeiro, Zinyani, &
       Delaneau, Communications Biology, 2022
    
For negative sampling, we generate random negative pairs from the set of all
genes. However, because there's the chance that a negatively paired gene pair is
actually a true positive, we ensure that negatives pairs do not exist across all
3 of the aformentioned sources as well as the STRING database and Gene Ontology
(GO) annotation database.

*NOTE - while we tried to opt for a completely programmatic data download, we
found it difficult. The OpenCell file was no longer properly linked at the time
we tried to download it, and the coessential file at 10% fdr was available only
as an xlsx. Users will have to manually download those files."""


from collections import defaultdict
import csv
import gzip
import itertools
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
from typing import Any, Dict, List, Set, Tuple, Union
import zipfile

import mygene  # type: ignore
import pandas as pd
from pybedtools import BedTool  # type: ignore
import pybedtools  # type: ignore


def count_unique_pairs(file_path: str) -> int:
    """Count the number of unique deduplicated pairs from co-occurence TSV."""
    unique_pairs = set()

    reader = csv.reader(open(file_path), delimiter="\t")
    for row in reader:
        gene1, gene2 = row
        pair = tuple(sorted([gene1, gene2]))
        unique_pairs.add(pair)

    return len(unique_pairs)


def get_genes_within_kb(gtf_file: str, kb: int = 100000) -> Set[Tuple[str, str]]:
    """Uses pybedtools and an input gencode GTF to fine all gene pairs within
    100kb of each other on hg38.

    Arguments:
        gtf_file (str): Path to the GTF file.
        kb (int): Distance in base pairs to create a window around each gene.
        This is the distance we AVOID when sampling negative pairs.

    Returns:
        Set[Tuple[str, str]]: Set of gene pairs within 100kb of each other.
    """

    def filter_protein_coding_genes(bedfile: BedTool) -> BedTool:
        """Filter a bedfile to only include protein coding genes."""
        return bedfile.filter(
            lambda x: "gene" in x[2] and "protein_coding" in x[8]
        ).saveas()

    def only_gene_names(feature: Any) -> List[str]:
        """Get gene name from the metadata description. Also in feature 8."""
        meta = feature[8]
        for part in meta.split(";"):
            if part.startswith(" gene_name"):
                feature[8] = part.split('"')[1]
        return feature

    gtf = pybedtools.BedTool(gtf_file)
    protein_coding = filter_protein_coding_genes(gtf)
    genes = protein_coding.each(only_gene_names).saveas()

    # use bedtools to find all gene pairs within 100kb
    genes_within_kb = genes.window(genes, w=kb)

    # collect gene pairs
    gene_pairs = set()
    for record in genes_within_kb:
        gene1 = record[8]
        gene2 = record[17]
        gene_pairs.add((gene1, gene2))
        if gene1 != gene2:
            pair = tuple(sorted((gene1, gene2)))  # sort to avoid duplicates
            gene_pairs.add(pair)

    return gene_pairs


class PrepareTrainingData:
    """Class to handle downloading then processing gene relationship
    datasets.

    Attributes:
        Output_dir (str): Directory to save downloaded files.

    Methods
    -------
    create_graphs()
        Make all graphs and save them to the output directory.
    negative_sampler(n_random_edges: int)
        Generate random negative samples.

    Helpers:
        FILES_TO_DOWNLOAD - list of files to download and their names
        post-download
        FILENAMES - dictionary of file names for each dataset
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
        ("http://www.interactome-atlas.org/data/Lit-BM.tsv", "litbm.tsv"),
        # (
        #     "https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-022-03831-w/MediaObjects/42003_2022_3831_MOESM4_ESM.xlsx",
        #     "sc_cop.xlsx",
        # ),
    ]

    FILENAMES = {
        "coessential": "coessential_pairs.txt",
        "opencell": "science.abi6983_table_s4.xlsx",
        "hi_union": "hi_union.tsv",
        "string": "string_protein_links.txt",
        "string_mapper": "string_protein_aliases.txt",
        "goa": "goa_human.gaf",
        "go_mapper": "go_ids_to_gene_symbol.txt",
        "sc_cop": "sc_cops.txt",
        "litbm": "litbm.tsv",
    }

    def __init__(self, output_dir: str = "./reference_files"):
        """Instantiate the class."""
        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def create_graphs(self) -> int:
        """Make all graphs and save them to the output directory. We first make
        each subgraph, then create some specific combined graphs. We combine the
        coessential, opencell, and hi_union graphs into a single positive graph.
        We also combine the string and go graphs into a `graph for
        filtering`.
        """
        # download files
        self.download_reference_files()

        # create and save individual graphs
        save_graphs = {
            "coessential_pos": self.coessential_graph()[0],
            "coessential_neg": self.coessential_graph()[1],
            "opencell": self.opencell_graph(),
            "sc_cop": self.sc_cop_graph(),
            "hi_union": self.hi_union_graph(),
            "string": self.physical_string_graph(),
            "go": self.go_graph(),
        }

        # save individual graphs
        for name, graph in save_graphs.items():
            self.save_graph(graph, f"{name}_graph.pkl")

        # load individual graphs
        graphs: Dict[str, Any] = {
            "coessential_pos": set(),
            "opencell": set(),
            "hi_union": set(),
            "sc_cop": set(),
            "string": set(),
            "go": set(),
        }
        for name in graphs:
            with open(self.output_dir / f"{name}_graph.pkl", "rb") as f:
                graphs[name] = pickle.load(f)

        # create and save combined graphs
        coessential_pos = self.gene_only_edges(graphs["coessential_pos"])
        experimentally_derived_edges = self.combine_exp_derived_edges(
            (coessential_pos, "coessential"),
            (graphs["opencell"], "opencell"),
            (graphs["hi_union"], "hi_union"),
            (graphs["sc_cop"], "sc_cop"),
        )
        self.save_graph(
            experimentally_derived_edges, "experimentally_derived_edges.pkl"
        )

        graph_for_filtering = self.combine_exp_derived_edges(
            (graphs["string"], "string"), (graphs["go"], "go")
        )
        self.save_graph(graph_for_filtering, "graph_for_filtering.pkl")

        all_positive_edges = experimentally_derived_edges | graph_for_filtering
        self.save_graph(all_positive_edges, "all_positive_edges.pkl")

        print("All graphs created and saved successfully.")
        return len(experimentally_derived_edges)

    def combine_exp_derived_edges(
        self, *edge_sets: Tuple[Set[Tuple[str, ...]], str]
    ) -> Set[Tuple[str, str, Tuple[str, ...]]]:
        """Combine experimentally derived edges and add source information."""
        combined_edges: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for edges, source in edge_sets:
            for edge in edges:
                if len(edge) != 2:
                    raise ValueError(
                        f"Expected edge to be a tuple of 2 elements, got {edge}"
                    )
                key: Tuple[str, str] = (
                    edge[0],
                    edge[1],
                )
                combined_edges[key].add(source)
        return {(*k, tuple(sorted(v))) for k, v in combined_edges.items()}

    def negative_sampler(
        self,
        n_random_edges: int,
        genes_within_kb: Set[Tuple[str, str]],
    ) -> Set[Tuple[str, ...]]:
        """Generate random negative samples.

        We first create a list of genes from the experimentally derived edges.
        We then initialize the negative samples by adding the experimentally
        derived negative samples. We start to generate negative samples at
        random, only keeping them if they are not in `all_positive_edges` and
        not in `genes_within_kb` to ensure that gene pairs within 100kb linear
        distance are not sampled.
        """

        def load_pickle(filename: str) -> Set[Tuple[str, ...]]:
            """Simple loading utility."""
            with open(self.output_dir / filename, "rb") as f:
                return pickle.load(f)

        all_positive_edges = load_pickle("all_positive_edges.pkl")
        all_positive_edges = self.gene_only_edges(all_positive_edges)
        exp_derived_edges = load_pickle("experimentally_derived_edges.pkl")
        exp_negative_edges = load_pickle("coessential_neg_graph.pkl")

        # add genes within 100kb to the positive set
        all_positive_edges |= genes_within_kb

        # make a list of all genes, used in experimentally derived edges to pull
        # from
        exp_derived_edges = {(edge[0], edge[1]) for edge in exp_derived_edges}
        all_genes: List[str] = list(
            {gene for edge in exp_derived_edges for gene in edge}
        )

        # get negative samples from the experimentally derived negative samples
        # and filter out any negative samples that might be in the positive set
        negative_samples: Set[Tuple[str, ...]] = {
            edge
            for edge in self.gene_only_edges(exp_negative_edges)
            if edge not in all_positive_edges
        }

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
        hi_union_path = self.output_dir / self.FILENAMES["hi_union"]
        litbm_path = self.output_dir / self.FILENAMES["litbm"]

        # get all ids for batch rename
        ensembl_ids = set()
        with open(hi_union_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                ensembl_ids.update(row[:2])

        with open(litbm_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                ensembl_ids.update(row[:2])

        # convert ensembl ids to gene symbols
        id_to_symbol = self.batch_ensembl_to_gene_symbol(ensembl_ids)

        with open(hi_union_path, "r") as file:
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

        # get lit_bm edges
        with open(litbm_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            litbm_edges = {
                (
                    id_to_symbol.get(row[0], "Unknown"),
                    id_to_symbol.get(row[1], "Unknown"),
                )
                for row in reader
                if id_to_symbol.get(row[0], "Unknown") != "Unknown"
                and id_to_symbol.get(row[1], "Unknown") != "Unknown"
            }

        # add litbm to hi_union
        # ensure no duplicates
        edges |= litbm_edges

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

    def sc_cop_graph(self) -> Set[Tuple[str, ...]]:
        """Get edges from the sc_cop dataset. Because the sc_cop file is an
        xlsx, we manually removed the interaction columns and saved as a
        tab-delimited file.
        """
        edges = set()
        file_path = self.output_dir / self.FILENAMES["sc_cop"]
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                edges.add((row[0], row[1]))
                edges.add((row[2], row[3]))
        return self.remove_duplicate_edges(edges)

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

        all_edges = set()
        go_to_gene: Dict[str, List[str]] = {}
        for gene, go_term in annotations:
            if go_term not in go_to_gene:
                go_to_gene[go_term] = []
            if gene in mapper:
                go_to_gene[go_term].append(mapper[gene])
        for linked_genes in go_to_gene.values():
            for gene_pair in itertools.combinations(linked_genes, 2):
                all_edges.add(gene_pair)
        return self.remove_duplicate_edges(all_edges)

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

    def save_graph(self, graph: Any, filename: str) -> None:
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


def main() -> None:
    """Main function"""

    # get genes_within_kb, 100kb
    gtf_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/reference_files/gencode.v45.basic.annotation.gtf"
    genes_within_kb = get_genes_within_kb(gtf_file)

    # make graphs!
    data_prep_obect = PrepareTrainingData(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data"
    )
    # len_edges = data_prep_obect.create_graphs()

    # use negative sampler for test set
    # make negative samples, with n = positive samples
    # negative_samples = data_prep_obect.negative_sampler(
    #     n_random_edges=len_edges, genes_within_kb=genes_within_kb
    # )

    # with open(
    #     "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/negative_edges.pkl",
    #     "wb",
    # ) as f:
    #     pickle.dump(negative_samples, f)

    # run negative sampler for each year
    data_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi"
    for year in range(2003, 2024):
        co_occurence = f"{data_dir}/gene_co_occurrence_{year}.tsv"
        unique_pairs = count_unique_pairs(co_occurence)
        print(f"Unique pairs for {year}: {unique_pairs}")

        # get negative samples
        negative_samples = data_prep_obect.negative_sampler(
            n_random_edges=unique_pairs, genes_within_kb=genes_within_kb
        )

        with open(
            f"{data_dir}/negative_edges_{year}.pkl",
            "wb",
        ) as f:
            pickle.dump(negative_samples, f)

    # # check if a negative sample is in the positive set
    # matching = next((e1 for e1 in edges for e2 in neg if e1 == e2), None)
    # print(f"Match found: {matching}" if matching else "No match")


if __name__ == "__main__":
    main()
