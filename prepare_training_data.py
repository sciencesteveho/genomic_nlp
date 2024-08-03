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
actually a true positive, we first filter out any negative pairs across all 3 of
the aformentioned sources. Then we remove any pairs present in the STRING
database and Gene Ontology (GO) annotation database.
    

# parse each (gene name) to a deduplicated edge list (txt)
    hi_union needs to be converted
    opencell needs to be converted
    string protein needs to be converted
    coessential needs to be parsed
# write a loader to load the edge lists, either as individual, or from all
# create an all positive_examples set
# create a set from GO + STRING (full network)
# create negative sampler with n_random samples
#    the sampler makes negative samples, but only keeps them if they are not in the positive_examples + GO + STRING
#    stops when n_random samples are made
    
"""

import csv
import gzip
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Tuple
import zipfile


class PrepareTrainingData:
    """Class to handle downloading then processing gene relationship
    datasets."""

    FILES_TO_DOWNLOAD = [
        (
            "https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-021-00840-z/MediaObjects/41588_2021_840_MOESM3_ESM.zip",
            "coessential_interactions.zip",
        ),
        (
            "https://opencell.czbiohub.org/data/datasets/opencell-protein-interactions.csv",
            "opencell_interactions.csv",
        ),
        ("http://www.interactome-atlas.org/data/HI-union.tsv", "hi_union.tsv"),
        (
            "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
            "string_protein_links.txt.gz",
        ),
        (
            "https://current.geneontology.org/annotations/goa_human.gaf.gz",
            "goa_human.gaf.gz",
        ),
    ]

    def __init__(self, output_dir: str = "./reference_files"):
        """Instantiate the class."""
        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.coessential_interactions = self.FILES_TO_DOWNLOAD[0][1]
        self.opencell_interactions = self.FILES_TO_DOWNLOAD[1][1]
        self.hi_union_interactions = self.FILES_TO_DOWNLOAD[2][1]
        self.string_protein_links = self.FILES_TO_DOWNLOAD[3][1]
        self.goa_human = self.FILES_TO_DOWNLOAD[4][1]

    def _create_go_graph(self, go_gaf: Path) -> List[Tuple[str, str]]:
        """Create GO ontology graph"""
        with open(go_gaf, newline="", mode="r") as file:
            reader = csv.reader(file, delimiter="\t")
            return [(row[1], row[4]) for row in reader if not row[0].startswith("!")]

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
            subprocess.run(["wget", "-O", output_path, url], check=True)
            print(f"Downloaded {filename}")

            # decompress gz files
            if filename.endswith(".gz"):
                self.decompress_gz(os.path.basename(output_path))
                print(f"Decompressed {filename}")

        print("All files downloaded and processed successfully.")

    @staticmethod
    def decompress_gz(file_path: str) -> None:
        """Decompress a .gz file"""
        with gzip.open(file_path, "rb") as f_in:
            with open(file_path[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)  # remove compressed

    @staticmethod
    def decompress_zip(file_path: str) -> None:
        """Decompress a .zip file"""
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(file_path[:-4])
        os.remove(file_path)


PrepareTrainingData("training_data").download_reference_files()
