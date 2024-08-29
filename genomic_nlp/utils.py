#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for bio-genetics-NLP"""


from collections import Counter
import contextlib
from datetime import timedelta
import functools
import glob
import inspect
import os
from pathlib import Path
import pickle
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

from gensim.models.callbacks import CallbackAny2Vec  # type: ignore
import numpy as np
import pandas as pd
import pybedtools  # type: ignore
from tqdm import tqdm  # type: ignore

from constants import COPY_GENES


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after every epoch."""

    def __init__(self, savedir: str):
        self.epoch = 0
        self.savedir = savedir

    def on_epoch_end(self, model: Any) -> None:
        """Save model after every epoch."""
        print(f"Save model number {self.epoch}.")
        model.save(f"{self.savedir}/model_epoch{self.epoch}.pkl")
        self.epoch += 1


def casefold_genes(genes: Set[str]) -> Set[str]:
    """Casefold all genes."""
    return {gene.casefold() for gene in genes}


def filter_zero_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Filter out key: value pairs where the value (embedding) consists of all
    zeroes.
    """
    return {key: value for key, value in embeddings.items() if np.any(value != 0)}


def gencode_genes(gtf: str) -> Set[str]:
    """_summary_

    Args:
        entity_file (str): _description_
        genes (Set[str]): _description_
        type (str, optional): _description_. Defaults to "gene".

    Returns:
        Set[str]: _description_
    """

    def gene_symbol_from_gencode(gencode_ref: pybedtools.BedTool) -> Set[str]:
        """Returns deduped set of genes from a gencode gtf. Written for the gencode
        45 and avoids header"""
        return {
            line[8].split('gene_name "')[1].split('";')[0]
            for line in gencode_ref
            if not line[0].startswith("#") and "gene_name" in line[8]
        }

    print("Grabbing genes from GTF")
    gtf = pybedtools.BedTool(gtf)
    genes = list(gene_symbol_from_gencode(gtf))

    for key in COPY_GENES:
        genes.remove(key)
        genes.append(COPY_GENES[key])
    return set(genes)


def _concat_chunks(filenames: List[str]) -> List[List[str]]:
    """Concatenates chunks of abstracts"""
    combined = []
    combined += [pickle.load(open(file, "rb")) for file in filenames]
    return combined


def _chunk_locator(path: str, prefix: str) -> List[str]:
    """Returns abstract chunks matching a specific prefix"""
    pattern = f"{path}/{prefix}_*.pkl"
    return glob.glob(pattern)


def _combine_chunks(path: str, prefix: str) -> List[List[str]]:
    """Combines chunks of abstracts"""
    filenames = _chunk_locator(path, prefix)
    print(f"Combining chunks of abstracts: {filenames}")
    return _concat_chunks(filenames)


def dir_check_make(dir_path: Union[str, Path]) -> None:
    """Check if a directory exists, if not, create it."""
    Path(dir_path).mkdir(exist_ok=True)


def time_decorator(print_args: bool = False, display_arg: str = "") -> Callable:
    """Decorator to time functions.

    Args:
        print_args (bool, optional): Whether to print the function arguments.
        Defaults to False. display_arg (str, optional): The argument to display
        in the print statement. Defaults to "".

    Returns:
        Callable: The decorated function.
    """

    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            result = function(*args, **kwargs)
            end_time = time.monotonic()
            args_to_print = list(fxn_args.values()) if print_args else display_arg
            print(
                f"Finished {function.__name__} {args_to_print} - Time: {timedelta(seconds=end_time - start_time)}"
            )
            return result

        return _execute

    return _time_decorator_func


def filter_abstract_by_terms(
    string: str,
    substr: Set[str],
    matches: int,
    remove: Set[str] = set(),
    keep: str = "match",
) -> List[str]:
    """Filter abstracts by the number of matches of substrings"""
    filtered = []
    for s in tqdm(string):
        if keep == "match":
            if len(substr.intersection(s.split())) >= matches:
                if remove:
                    if not remove.intersection(s.split()):
                        filtered.append(s)
                else:
                    filtered.append(s)
        elif keep == "remove ":
            if len(substr.intersection(s.split())) <= matches:
                filtered.append(s)
        else:
            raise ValueError("keep must be either 'match' or 'remove'")
    return filtered


def _abstract_retrieval_concat(
    data_path: Union[Path, str], save: bool = True
) -> pd.DataFrame:
    """Take abstract outputs and combine into a single pd.series. Only needs to
    be done initially after downloading abstracts"""
    frames = [
        pd.read_pickle(file, compression=None)
        for file in glob.glob(f"{data_path}/*.pkl")
    ]
    df = pd.concat(frames, ignore_index=True)
    if save:
        with open(f"{data_path}/abstracts_combined.pkl", "wb") as f:
            df.to_pickle(f)
    return df


def _random_subset_abstract_printer(n: int, abstracts: List) -> None:
    """Prints N random abstracts"""
    for num in random.sample(range(len(abstracts)), n):
        print(abstracts[num])


def _listdir_isfile_wrapper(dir_path: str) -> List[str]:
    return [file.name for file in Path(dir_path).iterdir() if file.is_file()]


def is_number(entry: str) -> bool:
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


def avg_len(lst: List[str]) -> float:
    """Takes the average length of elements in a list

    Args:
        lst (_type_): _description_

    Returns:
        int
    """
    total_lengths = [len(i) for i in lst]
    return int(sum(total_lengths)) / len(total_lengths) if total_lengths else 0


def dict_from_gene_symbol_and_name_list(gene_file_path):
    """Takes a tab delimited file organized as 'symbol'\t''name' and
    parses as a dictionary, removing entries with values in remove_words,
    which includes duplicates."

    # Arguments
        gene_file_path: filepath for gene tab file

    # Returns
        dictionary of values
    """
    remove_words = {"novel transcript", ""}
    namedict = {}
    with open(gene_file_path) as f:
        for line in f:
            symbol, name = line.strip().split("\t")
            name = re.sub(r"[^\w\s]|_", "", name).replace("  ", " ").strip().lower()
            namedict[name] = symbol.lower()

    # Find duplicates
    duplicates = {
        symbol for symbol, count in Counter(namedict.values()).items() if count > 1
    }
    remove_words.update(duplicates)

    # Filter out remove_words and duplicates
    return {
        name: symbol for name, symbol in namedict.items() if symbol not in remove_words
    }
    # remove_words = ["novel transcript", ""]
    # namedict = {}
    # with open(gene_file_path) as f:
    #     for line in f:
    #         line = line.strip("\n")
    #         a, b = line.split("\t")
    #         b = "".join(e for e in b if e.isalnum() or e in string.whitespace)
    #         b = re.sub("  ", " ", b)
    #         b = b.rstrip()
    #         b = re.sub(" ", "_", b)
    #         namedict.update({b.lower(): a.lower()})
    # dup_list = list(namedict.values())
    # val_dupes = set([item for item in dup_list if dup_list.count(item) > 1])
    # for dupe in val_dupes:
    #     remove_words.append(dupe)
    # set(remove_words)
    # return {key: value for key, value in namedict.items() if value not in remove_words}
