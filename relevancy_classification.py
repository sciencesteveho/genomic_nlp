#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Classify relevancy of abstracts based on term frequency and inverse document
frequency. Implements a logistic classifier and a simple multi-layer perceptron,
validated by 10-fold cross validation."""

import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from cleaning import AbstractCollection
from utils import _abstract_retrieval_concat


RANDOM_SEED = 42


def prepare_annotated_classification_set(
    abstracts: str,
    encoding: int,
) -> pd.DataFrame:
    """Reads in text file of abstracts and returns a dataframe with the
    abstracts, shuffled randomly

    Args:
        abstracts (str): path/to/abstracts.txt
        encoding (int): 1 for relevant, 0 for irrelevant

    Returns:
        pd.DataFrame
    """
    data = [line for line in csv.reader(open(abstracts, "r"), delimiter="\n")]
    df = pd.DataFrame(data, columns=["abstracts"])
    df["encoding"] = encoding
    return df.sample(frac=1).reset_index(drop=True)


def vectorize_and_train_logistic_classifier(
    trainset: pd.DataFrame,
    k: int,
) -> Tuple[TfidfVectorizer, SelectKBest, LogisticRegression]:
    """Vectorizes abstracts and trains a logistic classifier with k features

    Args:
        df (pd.DataFrame): dataframe with abstracts and encodings
        k (int): number of features to use

    Returns:
        A tuple containing the vectorizer, selector, and classifier
    """
    vectorizer_kwargs = {
        "ngram_range": (1, 2),  # Use 1-grams + 2-grams
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": "word",  # Split text into word tokens
    }
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    selector = SelectKBest(
        score_func=f_classif,
        k=k,
    )
    classifier = LogisticRegression(C=20.0, max_iter=500, random_state=RANDOM_SEED)

    y = trainset["encoding"].values
    y = y.astype(int)
    x_vectorized = vectorizer.fit_transform(
        [abstract for abstract in trainset["abstracts"]]
    )
    x_train = selector.fit_transform(x_vectorized, y)

    classifier.fit(x_train, y)
    cv_accuracy = cross_val_score(
        classifier,
        x_train,
        y,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    print(f"Mean F1 for K = {k} features: {np.mean(cv_accuracy)}")
    return vectorizer, selector, classifier


def classify_corpus(
    corpus: pd.DataFrame,
    k: int,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: LogisticRegression,
) -> pd.DataFrame:
    full = corpus["cleaned_combined"].values
    ex = vectorizer.transform(full)
    ex2 = selector.transform(ex)
    predictions = [label for label in classifier.predict(ex2)]
    corpus[k] = "None"
    corpus[k] = predictions
    return corpus


def _get_positive_testset(
    data_path: str,
) -> pd.DataFrame:
    """Get positive testset from directory

    Args:
        data_path (str): path/to/abstracts
        save (bool): whether to save dataframe

    Returns:
        pd.DataFrame
    """
    df = _abstract_retrieval_concat(data_path=data_path, save=False)

    df.sample(n=20000, random_state=RANDOM_SEED).reset_index(
        drop=True
    )  # get random 20k
    PositiveTestCorpus = AbstractCollection(
        df["title"].astype(str) + ". " + df["description"].astype(str)
    )
    PositiveTestCorpus.process_abstracts()
    return PositiveTestCorpus.cleaned_abstracts


def _get_negative_testset(
    data_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    NegativeTestCorpus = AbstractCollection(
        df["Title"].astype(str) + ". " + df["Abstract"].astype(str)
    )
    NegativeTestCorpus.process_abstracts()
    return NegativeTestCorpus.cleaned_abstracts


def main(
    corpus: str,
    relevant_abstracts: str,
    negative_abstracts: str,
    pos_set_path: str,
    negative_set_file: str,
    model_save_dir: str,
) -> None:
    """Main function"""
    # get training data and set-up annotated abstracts
    abstract_corpus = pd.read_pickle(corpus)
    
    classification_trainset = pd.concat(
        [
            prepare_annotated_classification_set(
                abstracts=relevant_abstracts, encoding=1
            ),
            prepare_annotated_classification_set(
                abstracts=negative_abstracts, encoding=0
            ),
        ],
        ignore_index=True,
    )
    
    # get positive test set data
    positive_test_data = _get_positive_testset(
        data_path=pos_set_path,
    )

    # get negative test set data
    negative_test_data = _get_negative_testset(
        data_path=negative_set_file,
    )

    # train logistic classifier with different k values
    for num in (20000, 75000, 125000):
        (
            vectorizer,
            selector,
            classifier,
        ) = vectorize_and_train_logistic_classifier(
            trainset=classification_trainset, k=num
        )
        joblib.dump(classifier, f"{model_save_dir}/logistic_classifier_{num}.pkl")

        abstracts_classified = classify_corpus(
            corpus=abstract_corpus,
            k=num,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
        )
        with open(f'{model_save_dir}/abstracts_classified_tfidf_{num}.pkl', 'wb') as f:
            pickle.dump(abstracts_classified, f)
        
        positive_classified = classify_corpus(
            corpus=positive_test_data,
            k=num,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
        )
        with open(f'{model_save_dir}/positive_classified_tfidf_{num}.pkl', 'wb') as f:
            pickle.dump(positive_classified, f)
        
        negative_classified = classify_corpus(
            corpus=negative_test_data,
            k=num,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
        )
        with open(f'{model_save_dir}/negative_classified_tfidf_{num}.pkl', 'wb') as f:
            pickle.dump(negative_classified, f)


if __name__ == "__main__":
    main(
        corpus="abstracts/cleaned_abstracts.pkl",
        relevant_abstracts="classification/relevant_sorted.txt",
        negative_abstracts="classification/negative_sorted.txt",
        pos_set_path = 'abstracts/test',
        negative_set_file = 'classification/irrelevant_texts.csv',
        model_save_dir="classification"
    )
