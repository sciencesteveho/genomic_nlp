#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract gene-disease associates from the DisGeNET API. Specifically, we are
looking for all possible GDAs along with their year_initial columns."""


import json
import time
from typing import Dict, List

import pandas as pd
import requests  # type: ignore

API_KEY = "1ff75c7a-9933-4cb8-aea0-626fe03cf1fa"
BASE_URL = "https://api.disgenet.com/api/v1/gda/summary"

CURATED_SOURCES = ["PSYGENET", "ORPHANET", "CLINGEN", "CLINVAR", "UNIPROT", "RGD_HUMAN"]
HEADERS = {"Authorization": API_KEY, "accept": "application/json"}

# pagination
PAGE_SIZE = 1
MAX_PAGES = 5

OUTPUT_FILE = "disgenet_curated_gdas_with_evidences.csv"


def fetch_gdas_for_source(source: str) -> List[Dict[str, str]]:
    """Fetch all GDAs for a given curated source."""
    all_results = []
    page_number = 0

    print(f"\nFetching data for source: {source}")

    while page_number < MAX_PAGES:
        params = {
            "source": source,
            "page_number": str(page_number),
            "page_size": str(PAGE_SIZE),
        }

        try:
            response = requests.get(
                BASE_URL, params=params, headers=HEADERS, verify=True
            )
        except requests.exceptions.RequestException as e:
            print(f"Request failed for source {source}, page {page_number}: {e}")
            break

        # handle rate limiting
        if response.status_code == 429:
            retry_after = int(
                response.headers.get("x-rate-limit-retry-after-seconds", "60")
            )
            print(
                f"Rate limit exceeded for source {source}, "
                f"page {page_number}. Retrying after {retry_after} seconds..."
            )
            time.sleep(retry_after)
            continue

        # handle other errors
        if not response.ok:
            print(
                "Failed to fetch data for source "
                f"{source}, page {page_number}: {response.status_code} - {response.text}"
            )
            break

        # parse the response
        try:
            response_parsed = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed for source {source}, page {page_number}: {e}")
            break

        # check for status
        if response_parsed.get("status") != "OK":
            error_message = response_parsed.get("error", "Unknown error")
            print(
                f"Error in response for source {source}, page {page_number}: {error_message}"
            )
            break

        # get the results from the payload
        results = response_parsed.get("payload", [])

        if not results:
            print(f"No more data to fetch for source {source} at page {page_number}.")
            break

        # append results to the list
        all_results.extend(results)
        print(
            f"Fetched page {page_number} with {len(results)} records for source {source}."
        )

        # check if we've fetched all available results
        paging_info = response_parsed.get("paging", {})
        total_elements = paging_info.get("totalElements", 0)
        total_pages = (total_elements // PAGE_SIZE) + (
            1 if total_elements % PAGE_SIZE != 0 else 0
        )

        if page_number >= total_pages - 1:
            print(f"All available data fetched for source {source}.")
            break

        page_number += 1

        # sleep to relax the API
        time.sleep(0.2)

    print(f"Total GDAs fetched for source {source}: {len(all_results)}")
    return all_results


def extract_year_initial(evidences):
    """
    Extract the earliest 'year_initial' from the evidences list.

    Parameters:
        evidences (list): List of evidence dictionaries.

    Returns:
        int or None: The earliest year, or None if not available.
    """
    if isinstance(evidences, list):
        years = [
            e.get("year_initial")
            for e in evidences
            if isinstance(e, dict) and "year_initial" in e
        ]
        years = [year for year in years if isinstance(year, int)]
        return min(years) if years else None
    return None


def main():
    all_gdas = []

    for source in CURATED_SOURCES:
        gdas = fetch_gdas_for_source(source)
        all_gdas.extend(gdas)

    print(f"\nTotal GDAs fetched across all curated sources: {len(all_gdas)}")

    if not all_gdas:
        print("No GDAs were fetched. Exiting.")
        return

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(all_gdas)

    # Check if 'evidences' and 'genesymbol' columns exist
    if "evidences" not in df.columns:
        print("Warning: 'evidences' column is missing in the response data.")
        df["evidences"] = None  # Assign None if missing

    if "genesymbol" not in df.columns:
        print("Warning: 'genesymbol' column is missing in the response data.")
        df["genesymbol"] = None  # Assign None if missing

    # Extract 'year_initial' from 'evidences'
    df["year_initial"] = df["evidences"].apply(extract_year_initial)

    # Select relevant columns (adjust column names as per actual response)
    # Common columns based on DisGeNET documentation
    relevant_columns = [
        "diseaseid",
        "diseasename",
        "geneid",
        "genesymbol",
        "score",
        "evidences",
        "year_initial",
    ]

    # Verify available columns and filter accordingly
    available_columns = [col for col in relevant_columns if col in df.columns]
    df_relevant = df[available_columns]

    # Save the DataFrame to a CSV file
    try:
        df_relevant.to_csv(OUTPUT_FILE, index=False)
        print(f"\nData successfully saved to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Failed to save data to CSV: {e}")


if __name__ == "__main__":
    main()
