#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Classify relevancy of abstracts based on term frequency and inverse document
frequency. Implements a logistic classifier and a simple multi-layer perceptron,
validated by 10-fold cross validation."""

from typing import Type, Tuple

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


RANDOM_SEED = 42


def prepare_annotated_classification_set(
    abstracts: str,
    encoding: int,
) -> pd.DataFrame:
    """Reads in text file of abstracts and returns a dataframe with the
    abstracts, shuffled randomly

    Args:
        abstracts (str): path/to/abstracts.txt encoding (int): 1 for relevant, 0
        for irrelevant

    Returns:
        pd.DataFrame
    """
    data = pd.read_csv(abstracts, sep="\n", header=None)
    data.columns = ["abstracts"]
    data["encoding"] = encoding
    return data.sample(frac=1).reset_index(drop=True)


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


def main(corpus: str, relevant_abstracts: str, negative_abstracts: str) -> None:
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

    # train logistic classifier with different k values
    for num in (20000, 75000, 125000):
        (
            vectorizer,
            selector,
            classifier,
        ) = vectorize_and_train_logistic_classifier(
            trainset=classification_trainset, k=num
        )
        joblib.dump(classifier, f"logistic_classifier_{num}.pkl")
        
        abstracts_classified = classify_corpus(
            corpus=abstract_corpus,
            k=num,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
        )


if __name__ == "__main__":
    main(
        corpus='cleaned_abstracts.pkl',
        relevant_abstracts='relevant_sorted.txt',
        negative_abstracts='negative_sorted.txt',
    )
