import pickle

from utils import (
    _random_subset_abstract_printer,
    FILTER_TERMS,
    filter_abstract_by_terms,
    SPECIFIC_TERMS,
)


def map_abs_to_titles(abstracts, lines):
    full_abs = []
    for line in lines:
        if line in abstracts.keys():
            full_abs.append(abstracts[line] + '. ' + line)


def main() -> None:
    # abstracts = filter_abstract_by_terms(
    #     pickle.load(open("cleaned_abstracts.pkl", "rb")), set(FILTER_TERMS), 9
    # )

    # with open(f"gene_filtered_abstracts_full.pkl", "wb") as output:
    #     pickle.dump(abstracts, output)

    abstracts = filter_abstract_by_terms(
        pickle.load(open("cleaned_abstracts.pkl", "rb")), set(SPECIFIC_TERMS), 7
    )

    with open(f"specific_filtered_abstracts_full.pkl", "wb") as output:
        pickle.dump(abstracts, output)



    with open(f"gene_filtered_abstracts_full.pkl", "rb") as file:
        abstracts = pickle.load(file)

    _random_subset_abstract_printer(25, abstracts)

    abstracts = filter_abstract_by_terms(
        pickle.load(open("cleaned_abstracts.pkl", "rb")), 
        set(FILTER_TERMS), 
        2, 
        keep="Remove",
    )

    with open(f"rev_filtered_abstracts_full.pkl", "wb") as output:
        pickle.dump(abstracts, output)



    
# with open('abstract_dicts/all_abstracts.dict', 'rb') as f:
#     abstracts = pickle.load(f)

# with open("relevant") as file:
#     lines = [line.strip() for line in file.readlines()]

# full_abstracts = map_abs_to_titles(abstracts, lines)


# import os
# import pickle
# import pandas as pd

# def swap_dictionary(dict):
#     return {v: k for k, v in dict.items()}

# abstract_dir = '/gpfs/accounts/remills_root/remills/stevesho/bio_nlp/nlp/abstract_dicts'

# all_abstracts = {}
# for filename in os.listdir(abstract_dir):
#     old_dict = swap_dictionary(pickle.load(open(filename, 'rb')))
#     all_abstracts.update(old_dict)

# with open('all_abstracts.dict', 'wb') as f:
#     pickle.dump(all_abstracts, f)

# all_abstracts_combined = []
# for file in os.listdir():
#     all_abstracts_combined += list(pd.read_pickle(file))

# with open('abstracts_combined.pkl', 'wb') as f:
#     pickle.dump(all_abstracts_combined , f)
