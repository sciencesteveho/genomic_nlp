#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Classify relevancy of abstracts based on term frequency and inverse document
frequency. Implements a logistic classifier and a simple multi-layer perceptron,
validated by 10-fold cross validation."""

import argparse
import csv
from pathlib import Path
import pickle
from typing import Set, Tuple, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_selection import f_classif  # type: ignore
from sklearn.feature_selection import SelectKBest  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore

from cleaning import AbstractCollection
from utils import _abstract_retrieval_concat

RANDOM_SEED = 42


def _prepare_annotated_classification_set(
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
    with open(abstracts, "r") as f:
        lines = f.readlines()
    df = pd.DataFrame(lines, columns=["abstracts"])
    df = df.assign(encoding=encoding)
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

    y_train = trainset["encoding"].astype(int).values
    x_vectorized = vectorizer.fit_transform(trainset["abstracts"])
    x_train = selector.fit_transform(x_vectorized, y_train)

    classifier.fit(x_train, y_train)
    cv_accuracy = cross_val_score(
        classifier,
        x_train,
        y_train,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    print(f"Mean F1 for K = {k} features: {np.mean(cv_accuracy)}")
    return vectorizer, selector, classifier


def _classify_test_corpus(corpus, vectorizer, selector, classifier):
    """Classify a test corpus using the provided vectorizer, selector, and classifier.

    Args:
        corpus (pd.DataFrame): The test corpus containing abstracts and encodings.
        vectorizer: The vectorizer used to transform the text data.
        selector: The feature selector for transforming the vectorized data.
        classifier: The classification model for predicting labels.

    Yields:
        tuple: A generator yielding tuples of abstracts and their corresponding predictions.
    """
    corpora = corpus["abstracts"].values
    y_test = corpus["encoding"].values
    predictions = _classify_full_corpus(vectorizer, corpora, selector, classifier)
    accuracy = accuracy_score(y_test, predictions)
    yield from zip(corpora, predictions)


def _classify_full_corpus(vectorizer, corpora, selector, classifier):
    """Classify a full corpus using the provided vectorizer, selector, and classifier.

    Args:
        vectorizer: The vectorizer used to transform the text data.
        corpora: The corpus to be classified.
        selector: The feature selector for transforming the vectorized data.
        classifier: The classification model for predicting labels.

    Returns:
        array-like: Predicted labels for the input corpus.
    """
    ex = vectorizer.transform(corpora)
    ex2 = selector.transform(ex)
    return classifier.predict(ex2)


def _classify_single_abstract(vectorizer, abstract, selector, classifier):
    """Classify a single abstract using the provided vectorizer, selector, and classifier.

    Args:
        vectorizer: The vectorizer used to transform the text data.
        abstract: The single abstract to be classified.
        selector: The feature selector for transforming the vectorized data.
        classifier: The classification model for predicting labels.

    Returns:
        array-like: Predicted label for the input abstract.
    """
    ex = vectorizer.transform([abstract])
    ex2 = selector.transform(ex)
    return classifier.predict(ex2)[0]


def classify_corpus(
    corpus: Union[Set[str], pd.DataFrame],
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: LogisticRegression,
    test: bool = False,
) -> pd.DataFrame:
    """Classifies a corpus of abstracts using the provided vectorizer, feature selector, and classifier.

    Args:
        corpus (Union[Set[str], pd.DataFrame]): The corpus of abstracts to classify.
        vectorizer (TfidfVectorizer): The vectorizer used to transform the abstracts into feature vectors.
        selector (SelectKBest): The feature selector used to select the most informative features.
        classifier (LogisticRegression): The classifier used to predict the class labels.
        test (bool, optional): Flag indicating whether the corpus is a test set. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the classified abstracts.
    """
    if test:
        generator = _classify_test_corpus(corpus, vectorizer, selector, classifier)
    else:
        generator = _classify_full_corpus(corpus, vectorizer, selector, classifier)

    results = list(generator)
    abstracts, predictions = zip(*results[:-1])
    accuracy = results[-1]

    df = pd.DataFrame({"abstracts": abstracts, "predictions": predictions})
    df["accuracy"] = accuracy

    return df


# def classify_corpus(
#     corpus: Union[Set[str], pd.DataFrame],
#     vectorizer: TfidfVectorizer,
#     selector: SelectKBest,
#     classifier: LogisticRegression,
#     test: bool = False,
# ) -> Generator[Tuple[str, int], None, None]:
#     """Classifies a corpus of abstracts using the provided vectorizer, feature selector, and classifier.

#     Args:
#         corpus (Union[Set[str], pd.DataFrame]): The corpus of abstracts to classify.
#         vectorizer (TfidfVectorizer): The vectorizer used to transform the abstracts into feature vectors.
#         selector (SelectKBest): The feature selector used to select the most informative features.
#         classifier (LogisticRegression): The classifier used to predict the class labels.
#         test (bool, optional): Flag indicating whether the corpus is a test set. Defaults to False.

#     Yields:
#         Tuple[str, int]: A tuple containing the classified abstract and its prediction label.
#     """
#     if test:
#         yield from _classify_test_corpus(corpus, vectorizer, selector, classifier)
#     else:
#         for abstract in corpus:
#             prediction = _classify_single_abstract(vectorizer, abstract, selector, classifier)
#             yield abstract, prediction


def _get_testset(
    data_path: str,
    positive: bool,
) -> pd.DataFrame:
    """Reads in test set data and returns a dataframe with the abstracts and
    encoding

    Args:
        data_path (str): _description_
        positive (bool): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if positive:
        df = (
            _abstract_retrieval_concat(data_path=data_path, save=False)
            .sample(n=20000, random_state=RANDOM_SEED)
            .reset_index(drop=True)
        )
    else:
        df = pd.read_csv(data_path)

    testCorpus = AbstractCollection(
        df[df.columns[0]].astype(str) + ". " + df[df.columns[1]].astype(str)
    )
    testCorpus.clean_abstracts()
    newdf = pd.DataFrame(testCorpus.cleaned_abstracts, columns=["abstracts"])
    newdf["encoding"] = 1 if positive else 0
    return newdf


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--k",
        help="number of features for tf-idf",
        type=int,
    )
    parser.add_argument("--corpus", help="Path to the corpus file")
    parser.add_argument(
        "--relevant_abstracts", help="Path to the relevant abstracts file"
    )
    parser.add_argument(
        "--negative_abstracts", help="Path to the negative abstracts file"
    )
    parser.add_argument("--pos_set_path", help="Path to the positive set directory")
    parser.add_argument("--negative_set_file", help="Path to the negative set file")
    parser.add_argument("--model_save_dir", help="Directory to save the model")
    return parser.parse_args()


def main() -> None:
    """Main function to classify relevancy of abstracts based on term
    frequency"""
    args = _parse_args()
    savepath = Path(args.model_save_dir)

    # get training data and set-up annotated abstracts
    abstract_corpus = pd.read_pickle(args.corpus)

    classification_trainset = pd.concat(
        [
            _prepare_annotated_classification_set(
                abstracts=args.relevant_abstracts, encoding=1
            ),
            _prepare_annotated_classification_set(
                abstracts=args.negative_abstracts, encoding=0
            ),
        ],
        ignore_index=True,
    )

    # get positive test set data
    positive_test_data = _get_testset(
        data_path=args.pos_set_path,
        positive=True,
    )

    # get negative test set data
    negative_test_data = _get_testset(
        data_path=args.negative_set_file,
        positive=False,
    )

    # combine
    testset = pd.concat([positive_test_data, negative_test_data], ignore_index=True)

    # train logistic classifier
    num = args.k
    (
        vectorizer,
        selector,
        classifier,
    ) = vectorize_and_train_logistic_classifier(trainset=classification_trainset, k=num)
    joblib.dump(classifier, savepath / f"logistic_classifier_{num}.pkl")

    testset_classified = classify_corpus(
        corpus=testset,
        vectorizer=vectorizer,
        selector=selector,
        classifier=classifier,
        test=True,
    )
    with open(savepath / f"testset_classified_tfidf_{num}.pkl", "wb") as f:
        pickle.dump(testset_classified, f)

    abstracts_classified = classify_corpus(
        corpus=abstract_corpus,
        vectorizer=vectorizer,
        selector=selector,
        classifier=classifier,
    )
    with open(savepath / f"abstracts_classified_tfidf_{num}.pkl", "wb") as f:
        pickle.dump(abstracts_classified, f)


if __name__ == "__main__":
    main()
    # main(
    #     corpus="abstracts/cleaned_abstracts.pkl",
    #     relevant_abstracts="classification/relevant_sorted.txt",
    #     negative_abstracts="classification/negative_sorted.txt",
    #     pos_set_path="abstracts/test",
    #     negative_set_file="classification/irrelevant_texts.csv",
    #     model_save_dir="classification",
    # )
