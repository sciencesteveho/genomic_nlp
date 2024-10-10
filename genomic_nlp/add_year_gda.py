#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Add publication year to manually curated gene-disease associations from
Ehrhart et al., Scientific Data, 2021.

The curated resource contains GDAs for rare monogenic diseases with known causal
genes. Each GDA has associated historical provenance but not publication year,
so we use the PubMed API to fetch the publication year for each PMID, which is
column 3.

We download the provenance file with:
    wget --content-disposition https://figshare.com/ndownloader/files/25769330
"""


import csv
import time
from typing import Dict, List

from Bio import Entrez
import requests  # type: ignore


class PubMedYearFetcher:
    """Simple class to fetch publication years for PMIDs using biopython.

    Methods
    ----------
    get_publication_years(pmids: list) -> dict[str, str]
        Fetch the publication year for a given PMID.
    """

    def __init__(self, email: str, api_key: str) -> None:
        """Initialize the PubMedYearFetcher with an email address."""
        Entrez.email = email  # type: ignore
        if api_key:
            Entrez.api_key = api_key  # type: ignore

    def get_publication_year(self, pmid: str) -> str:
        """Fetch the publication year for a given PMID with delay and retry."""
        retries = 3
        for i in range(retries):
            try:
                time.sleep(0.2)
                handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
                records = Entrez.read(handle)
                return records["PubmedArticle"][0]["MedlineCitation"]["Article"][
                    "Journal"
                ]["JournalIssue"]["PubDate"]["Year"]
            except Exception as e:
                print(f"Error fetching year for PMID {pmid}: {str(e)}")
                wait_time = (i + 1) * 2
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        return "N/A"


class GeneDataProcessor:
    """Simple class to fetch publication years for PMIDs using biopython.

    Methods
    ----------
    process_file(input_file: str, output_file: str, pmid_column: int) -> None
        Process the input file and add publication year to each row.
    """

    def __init__(self, year_fetcher: PubMedYearFetcher) -> None:
        """Initialize the GeneDataProcessor."""
        self.year_fetcher = year_fetcher

    def process_file(
        self, input_file: str, output_file: str, pmid_column: int = 2
    ) -> None:
        """Add publication year to each row in the input provenance file using
        batching.
        """
        with (
            open(input_file, "r") as infile,
            open(output_file, "w", newline="") as outfile,
        ):
            reader = csv.reader(infile, delimiter="\t")
            writer = csv.writer(outfile, delimiter="\t")

            header = next(reader)
            header.append("Publication Year")
            writer.writerow(header)

            for row in reader:
                if pmid := row[pmid_column].strip():
                    year = self.year_fetcher.get_publication_year(pmid)
                    print(f"PMID {pmid}: {year}")
                    row.append(year)
                    writer.writerow(row)
                else:
                    print(f"No PMID found for row: {row}")


def main() -> None:
    """Main function to add publication year to gene-disease associations."""
    input_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/Gene-RD-Provenance_V2.1.txt"
    output_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/Gene-RD-Provenance_V2.1_with_year.txt"
    email = "stevesho@umich.edu"
    api_key = "9283b8c0bc0a8b00999a65428df421ec5708"

    year_fetcher = PubMedYearFetcher(email=email, api_key=api_key)
    processor = GeneDataProcessor(year_fetcher)
    processor.process_file(input_file, output_file)
    print("Years added!")


if __name__ == "__main__":
    main()
