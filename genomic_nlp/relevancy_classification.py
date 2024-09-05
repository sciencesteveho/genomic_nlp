#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implements logistic regression or XGBoost to classify relevancy of abstracts
based on term frequency and inverse document frequency. Optionally performs a
grid search, but does not save the model and instead outputs the model params to
a text file.

Due to a difference in the size of the manually annotated train set, the testing
set, and the corpus, we opt to use the entiety the annotated set for training,
then apply the model to the testing set and full corpus. Five-fold
cross-validation to used to evaluate the robustness of the model, but not as a
means to model training.
"""


import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_selection import f_classif  # type: ignore
from sklearn.feature_selection import SelectKBest  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from xgboost import XGBClassifier

from abstract_cleaner import AbstractCleaner
from utils import _abstract_retrieval_concat
from utils import get_physical_cores

RANDOM_SEED = 42


def get_training_data(corpus_path: str) -> pd.DataFrame:
    """Prepares and cleans an abstract collection object."""
    # if corpus file exists, load it
    if os.path.exists(corpus_path):
        return pd.read_pickle(corpus_path)

    # if corpus file does not exist, create, clean, and save
    with contextlib.suppress(FileExistsError):
        _abstract_retrieval_concat(data_path=corpus_path, save=True)
    abstractcollectionObj = AbstractCleaner(pd.read_pickle(corpus_path))
    abstractcollectionObj.clean_abstracts()
    with open(corpus_path, "wb") as f:
        pickle.dump(abstractcollectionObj.cleaned_abstracts, f)
    return abstractcollectionObj.cleaned_abstracts


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


def perform_grid_search(
    features: Any,
    labels: Any,
    classifier: Union[LogisticRegression, XGBClassifier],
    param_grid: Dict[str, List[Any]],
    cores: int,
    savepath: Path,
    classifier_name: str,
) -> None:
    """Perform grid search to find optimal hyperparameters."""
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=cores,
        return_train_score=True,
    )
    grid_search.fit(features, labels)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

    # save grid search results
    results = pd.DataFrame(grid_search.cv_results_)

    # save best params
    with open(savepath / f"{classifier_name}_best_params.json", "w") as f:
        json.dump(grid_search.best_params_, f)


def vectorize_and_train_classifier(
    trainset: pd.DataFrame,
    classifier: Union[LogisticRegression, XGBClassifier],
    k: int,
    savepath: Path,
    grid_search: bool = False,
) -> Tuple[
    TfidfVectorizer, SelectKBest, Union[LogisticRegression, XGBClassifier, None]
]:
    """Vectorizes abstracts and trains a logistic classifier with k features

    Args:
        df (pd.DataFrame): dataframe with abstracts and encodings
        classifier: classification model
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

    y_train = trainset["encoding"].astype(int).values
    x_vectorized = vectorizer.fit_transform(trainset["abstracts"])
    x_train = selector.fit_transform(x_vectorized, y_train)

    param_grid: Dict[str, List[Any]] = {}
    if grid_search:
        if isinstance(classifier, LogisticRegression):
            param_grid = {"C": [0.1, 1, 10, 20, 50], "max_iter": [100, 200, 500, 1000]}
        elif isinstance(classifier, XGBClassifier):
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "n_estimators": [100, 200, 300],
            }
        perform_grid_search(
            features=x_train,
            labels=y_train,
            classifier=classifier,
            param_grid=param_grid,
            cores=get_physical_cores(),
            savepath=savepath,
            classifier_name=type(classifier).__name__,
        )
        return vectorizer, selector, None
    else:
        classifier.fit(x_train, y_train)

    cv_accuracy = cross_val_score(
        classifier,
        x_train,
        y_train,
        scoring="f1",
        cv=5,
        n_jobs=get_physical_cores(),
    )
    print(f"Mean F1 for K = {k} features: {np.mean(cv_accuracy)}")
    return vectorizer, selector, classifier


def _classify_test_corpus(
    corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: Union[LogisticRegression, XGBClassifier],
    savepath: Path,
    k: int,
) -> Tuple[zip, float]:
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

    # get probabilities for ROC curve
    tfidf_feats = vectorizer.transform(corpora)
    selected_feats = selector.transform(tfidf_feats)
    label_probabilities = classifier.predict_proba(selected_feats)[:, 1]

    # save roc
    get_roc_auc(
        true_labels=y_test,
        predicted_labels=label_probabilities,
        classifier_name=f"{type(classifier).__name__,}_tfidf_{k}",
        savepath=savepath,
    )

    predictions = _classify_full_corpus(vectorizer, corpora, selector, classifier)
    accuracy = accuracy_score(y_test, predictions)
    return zip(corpora, predictions), accuracy


def get_roc_auc(
    true_labels: Any,
    predicted_labels: Any,
    classifier_name: str,
    savepath: Path,
) -> None:
    """Calculate and save ROC curve and AUC score."""
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)

    roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    roc_data.to_csv(savepath / f"{classifier_name}_roc_data.csv", index=False)

    with open(savepath / f"{classifier_name}_auc_score.json", "w") as f:
        json.dump({"auc": auc}, f)


def _classify_full_corpus(
    vectorizer: TfidfVectorizer,
    corpora: Any,
    selector: SelectKBest,
    classifier: Union[LogisticRegression, XGBClassifier],
):
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


def classify_corpus(
    corpus: Any,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: LogisticRegression,
    savepath: Path,
    k: int,
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
        results, accuracy = _classify_test_corpus(
            corpus=corpus,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
            savepath=savepath,
            k=k,
        )
        abstracts, predictions = zip(*results)
        df = pd.DataFrame(
            {"abstracts": abstracts, "predictions": predictions, "accuracy": accuracy}
        )
    else:
        predictions = _classify_full_corpus(vectorizer, corpus, selector, classifier)
        accuracy = None
        df = pd.DataFrame({"abstracts": list(corpus), "predictions": predictions})

    if accuracy is not None:
        df["accuracy"] = accuracy

    return df


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

    testCorpus = AbstractCleaner(
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
        default=10000,
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
    parser.add_argument(
        "--classifier",
        help="Classifier to use",
        default="logistic",
        choices=["logistic", "xgboost"],
    )
    parser.add_argument(
        "--classify_only",
        help="Run only classification without creating and cleaning abstract collection",
        action="store_true",
    )
    parser.add_argument(
        "--grid_search",
        help="Run grid search to find optimal hyperparameters",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to classify relevancy of abstracts based on term
    frequency"""
    args = _parse_args()
    savepath = Path(args.model_save_dir)
    num = args.k

    # get training data and set-up annotated abstracts
    abstract_corpus = get_training_data(args.corpus)

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
    if args.classifier == "logistic":
        classifier = LogisticRegression(C=50, max_iter=100, random_state=RANDOM_SEED)
    elif args.classifier == "xgboost":
        classifier = XGBClassifier(
            learning_rate=0.3, max_depth=3, n_estimators=100, random_state=RANDOM_SEED
        )
    (
        vectorizer,
        selector,
        classifier,
    ) = vectorize_and_train_classifier(
        trainset=classification_trainset,
        classifier=classifier,
        k=num,
        grid_search=args.grid_search,
        savepath=savepath,
    )

    if not args.grid_search:
        joblib.dump(
            classifier,
            savepath / f"{args.classifier}_relevancy_classifier_tfidf_{num}.pkl",
        )  # save stuff

        testset_classified = classify_corpus(
            corpus=testset,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
            test=True,
            savepath=savepath,
            k=num,
        )
        with open(
            savepath / f"testset_{args.classifier}_classified_tfidf_{num}.pkl", "wb"
        ) as f:
            pickle.dump(testset_classified, f)

        abstracts_classified = classify_corpus(
            corpus=abstract_corpus,
            vectorizer=vectorizer,
            selector=selector,
            classifier=classifier,
            savepath=savepath,
            k=num,
        )
        with open(
            savepath / f"abstracts_{args.classifier}_classified_tfidf_{num}.pkl", "wb"
        ) as f:
            pickle.dump(abstracts_classified, f)


if __name__ == "__main__":
    main()
