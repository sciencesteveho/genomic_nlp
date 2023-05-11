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
    abstracts = filter_abstract_by_terms(
        string=pickle.load(open("cleaned_abstracts.pkl", "rb")),
        substr=set(FILTER_TERMS),
        matches=15,
        remove=set(['cancer', 'plant', 'carcinoma']),
        keep='match'
    )

    with open(f"gene_filtered_abstracts_full.pkl", "wb") as output:
        pickle.dump(abstracts, output)

    FILTER_TERMS = [
        'racial',
        'race',
        'battery',
        'batteries',
        'socioeconomic',
        'sociology',
        'counseling',
        'electricity',
        'nanoparticle',
        'nanoparticles',
        'nanomaterial',
        'physics',
        'NASA',
        'infared',
        'market',
        'economic',
        'electroreduction',
        'lunar',
        'quantum',
        'ferromagnetism',
        'urban',
        'urbanization',
        'car',
        'classroom',
        'classrooms',
        'rainforest',
        'copper',
        'atmosphere',
        'political',
        'industrial',
        'capitalism',
        'capitalist',
        'communism',
        'voltage',
        'financial',
        'management',
        'counseling',
        'counselor',
        'stakeholder',
        'investor',
        'economic',
        'economics',
        'investing',
        'outreach',
        'shipping',
        'spacecraft',
        'government',
        'governments',
        'governmental',
        'power',
        'windmill',
        'forecasting',
        'forecast',
        'demographic',
        'demographics',
        'demography',
        'toxicology',
        'toxic',
        'toxicity',
        'toxicities',
        'public health',
        'community',
        'center'
    ]

    # other_abs = filter_abstract_by_terms(
    #     pickle.load(open("cleaned_abstracts.pkl", "rb")), 
    #     set(FILTER_TERMS), 
    #     2, 
    #     keep="match"
    # )

    # other_abs = filter_abstract_by_terms(
    #     pickle.load(open("cleaned_abstracts.pkl", "rb")), 
    #     set(['structural variation', 'structural variations', 'structural variant', 'structural variants', 'genome', 'genomics']), 
    #     matches=2, 
    #     keep="match",
    #     remove=''
    # )

    # abstracts = filter_abstract_by_terms(
    #     pickle.load(open("cleaned_abstracts.pkl", "rb")), 
    #     set(FILTER_TERMS), 
    #     2, 
    #     keep="remove",
    # )

    # with open(f"rev_filtered_abstracts_full.pkl", "wb") as output:
    #     pickle.dump(abstracts, output)

    # with open(f"other_rev_filtered_abstracts_full.pkl", "wb") as output:
    #     pickle.dump(other_abs, output)

    with open('other_rev_filtered_abstracts_full.pkl', 'rb') as f:
        other_abs  = pickle.load(f)

    _random_subset_abstract_printer(50, other_abs)
    _random_subset_abstract_printer(50, abstracts)


    
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
