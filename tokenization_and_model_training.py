#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Tokenization, token clean-up, and gene removal. Model training for word
embeddings for bio-nlp model!"""

import datetime
import gc
import string
import pickle
import logging
import re
from tqdm import tqdm
from typing import List

import en_core_sci_sm
from fse import IndexedList
from fse.models import uSIF
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
from progressbar import ProgressBar


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def is_number(entry):
    """

    # Arguments
        entry: the string to be checked
    # Returns
        True for float, false for other
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False

def averageLen(lst: List[str]) -> int:
    """Takes the average length of elements in a list

    Args:
        lst (_type_): _description_

    Returns:
        int
    """
    print("Getting average length of sentences within corpus for W2V training")
    total_lengths = [len(i) for i in lst]
    if len(total_lengths) == 0:
        return 0
    else:
        return int(sum(total_lengths)) / len(total_lengths)

def normalization_list(entity_file, type):
    """Uses gtf_parser to parse a GTF to a dataframe. Grabs a list
    of gene_names in the GTf, removes duplicates, and adds fixers
    for the weirdly named genes.

    # Arguments
        entity_file: either a list of tuples or GTF file
        type: ents(scispaCy entities), gene(GTF)
    # Returns
        gene_names_list: list of unique genes from GTF
    """
    if type == "gene":
        print("Grabbing genes from GTF")
        gene_names = entity_file['gene_name']
        gene_names_list = sorted(list(set([name.lower() for index, name in enumerate(gene_names)])))
        for key in copy_genes:
            gene_names_list.remove(key)
            gene_names_list.append(copy_genes[key])
        return set(gene_names_list)
    elif type == "ents":
        ent_list = [entity[0].casefold() for entity in entity_file]
        return set(ent_list)


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()