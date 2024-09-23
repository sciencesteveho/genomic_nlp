import csv

import requests

# set up the URL and query parameters
url = "https://www.disgenet.org/api/gda/gene"

params = {
    "gene": "list_of_hg38_genes",  # list of genes
    "evidence_level": "definitive,strong,moderate",
}

headers = {"Authorization": "Bearer YOUR_API_KEY", "Accept": "application/json"}

# make the API request
response = requests.get(url, headers=headers, params=params)

# check if the request was successful
if response.status_code == 200:
    data = response.json()

    # define the output file
    output_file = "gda_results.csv"

    # open the file and set up CSV writer
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # write the header
        writer.writerow(["Gene", "Disease", "Year Initial", "Evidence Level"])

        # write the data
        for gda in data:
            gene = gda.get("geneSymbol")
            disease = gda.get("diseaseName")
            year_initial = gda.get("year_initial")
            evidence_level = gda.get("evidence_level")
            writer.writerow([gene, disease, year_initial, evidence_level])

    print(f"Results have been written to {output_file}")
else:
    print(f"Error: {response.status_code}")
