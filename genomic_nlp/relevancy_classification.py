#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implements logistic regression or XGBoost to classify relevancy of abstracts
based on term frequency and inverse document frequency. Optionally performs a
grid search, but does not save the model and instead outputs the model params to
a text file.

Manually annotating abstracts is labor and resource intensive so we opt for a
pre-training approach. We first train the model on a large set of abstracts
stratified by journal type as a proxy for relevancy. We then fine-tune the model
on a smaller set of manually annotated abstracts with a lowered learning rate to
utilize transfer learning.

We run the script in two steps. First, with --grid_search flag across a variety
of features in separate jobs (10K, 20K, and 40K tf-idf feats). Based on
performance, we manually updated the script with the best hyperparameters and
ran the script without the --grid_search flag to train the final model.
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
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_selection import f_classif  # type: ignore
from sklearn.feature_selection import SelectKBest  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.utils import resample  # type: ignore

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
    cleaned_abstracts = abstractcollectionObj.clean_abstracts()
    with open(corpus_path, "wb") as f:
        pickle.dump(cleaned_abstracts, f)
    return cleaned_abstracts


def _prepare_annotated_classification_set(
    abstracts: str,
    encoding: int,
) -> pd.DataFrame:
    """Reads in text file of abstracts and returns a dataframe with the
    abstracts, shuffled randomly
    """
    with open(abstracts, "r") as f:
        lines = f.readlines()
    df = pd.DataFrame(lines, columns=["abstracts"])
    df = df.assign(encoding=encoding)
    return df.sample(frac=1).reset_index(drop=True)


def perform_grid_search(
    features: Any,
    labels: Any,
    classifier: Union[LogisticRegression, MLPClassifier],
    param_grid: Dict[str, List[Any]],
    cores: int,
    savepath: Path,
    k: int,
    pretrain: bool = False,
    return_model: bool = False,
) -> Union[None, LogisticRegression, MLPClassifier]:
    """Perform grid search to find optimal hyperparameters."""
    classifier_name = (
        f"{type(classifier).__name__}_pretrain"
        if pretrain
        else type(classifier).__name__
    )

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
    print(f"Run with {k} features")

    # save grid search results
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_pickle(savepath / f"{classifier_name}_grid_search_results.pkl")

    # save best params
    with open(savepath / f"{classifier_name}_best_params.json", "w") as f:
        json.dump(grid_search.best_params_, f)

    return grid_search.best_estimator_ if return_model else None


def get_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    dtype: str = "int32",
    strip_accents: str = "unicode",
    decode_error: str = "replace",
    analyzer: str = "word",
) -> TfidfVectorizer:
    """Create a TfidfVectorizer with the provided parameters."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        dtype=dtype,
        strip_accents=strip_accents,
        decode_error=decode_error,
        analyzer=analyzer,
    )


def get_selector(
    score_func: Any = f_classif,
    k: int = 10000,
) -> SelectKBest:
    """Create a SelectKBest feature selector with the provided parameters."""
    return SelectKBest(score_func=score_func, k=k)


def fit_vectorizer_selector(
    abstracts: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
) -> Tuple[TfidfVectorizer, SelectKBest]:
    """Fit the vectorizer and selector on the given abstracts."""
    x_vectorized = vectorizer.fit_transform(abstracts["abstracts"])
    selector.fit(x_vectorized, abstracts["encoding"])
    return vectorizer, selector


def vectorize_abstracts(
    abstracts: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
) -> Tuple[Any, Any]:
    """Vectorize abstracts using the provided vectorizer and feature
    selector.
    """
    x_vectorized = vectorizer.transform(abstracts["abstracts"])
    x_train = selector.transform(x_vectorized)
    return x_train, abstracts["encoding"].astype(int).values


def get_param_grid(
    classifier: Union[LogisticRegression, MLPClassifier],
) -> Dict[str, List[Any]]:
    """Get grid search parameters for the provided classifier."""
    if isinstance(classifier, LogisticRegression):
        return {"C": [0.1, 1, 10, 20, 50], "max_iter": [100, 200, 500, 1000]}
    elif isinstance(classifier, MLPClassifier):
        return {
            "hidden_layer_sizes": [(32,), (64,), (128,)],
            "max_iter": [10, 50, 100],
            "alpha": [0.001, 0.01],
        }
    else:
        raise ValueError(
            "Invalid classifier type. Must be `LogisticRegression` or `MLPClassifier`."
        )


def share_model_params(
    pretrained_classifier: Union[LogisticRegression, MLPClassifier],
    finetune_classifier: Union[LogisticRegression, MLPClassifier],
) -> Union[LogisticRegression, MLPClassifier]:
    """Share model parameters from a pre-trained classifier to a fine-tuned
    classifier.
    """
    if isinstance(pretrained_classifier, LogisticRegression):
        finetune_classifier.coef_ = pretrained_classifier.coef_
        finetune_classifier.intercept_ = pretrained_classifier.intercept_
    elif isinstance(pretrained_classifier, MLPClassifier):
        finetune_classifier.coefs_ = pretrained_classifier.coefs_
        finetune_classifier.intercepts_ = pretrained_classifier.intercepts_
    return finetune_classifier


def pretrain_model(
    pretrain_abstracts: pd.DataFrame,
    classifier: Union[LogisticRegression, MLPClassifier],
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    k: int,
    savepath: Path,
    grid_search: bool = False,
) -> Tuple[
    Union[LogisticRegression, MLPClassifier, None], TfidfVectorizer, SelectKBest
]:
    """Pre-trains a classifier on abstracts stratified by journal type."""
    vectorizer, selector = fit_vectorizer_selector(
        abstracts=pretrain_abstracts, vectorizer=vectorizer, selector=selector
    )

    x_pretrain, y_pretrain = vectorize_abstracts(
        abstracts=pretrain_abstracts, vectorizer=vectorizer, selector=selector
    )
    print(f"Pre-train x shape: {x_pretrain.shape}")
    print(f"Pre-train y shape: {y_pretrain.shape}")

    if grid_search:
        param_grid = get_param_grid(classifier)
        best_model = perform_grid_search(
            features=x_pretrain,
            labels=y_pretrain,
            classifier=clone(classifier),
            param_grid=param_grid,
            cores=get_physical_cores(),
            savepath=savepath,
            k=k,
            pretrain=True,
            return_model=True,
        )
    else:
        best_model = clone(classifier)
        best_model.fit(x_pretrain, y_pretrain)

    return best_model, vectorizer, selector


def finetune_model(
    finetune_abstracts: pd.DataFrame,
    pretrained_model: Union[LogisticRegression, MLPClassifier],
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    k: int,
    savepath: Path,
    grid_search: bool = False,
) -> Union[LogisticRegression, MLPClassifier, None]:
    """Fine-tunes a model on manually annotated abstracts."""
    x_finetune, y_finetune = vectorize_abstracts(
        abstracts=finetune_abstracts, vectorizer=vectorizer, selector=selector
    )
    # finetuned_model = LogisticRegression(C=50, max_iter=100, random_state=RANDOM_SEED)
    finetuned_model = MLPClassifier(
        solver="adam",
        alpha=0.001,
        max_iter=100,
        hidden_layer_sizes=(32,),
        random_state=RANDOM_SEED,
        early_stopping=True,
    )

    if grid_search:
        param_grid = get_param_grid(pretrained_model)
        return perform_grid_search(
            features=x_finetune,
            labels=y_finetune,
            classifier=pretrained_model,
            param_grid=param_grid,
            cores=get_physical_cores(),
            savepath=savepath,
            k=k,
            pretrain=False,
            return_model=True,
        )
    else:
        finetuned_model = share_model_params(
            pretrained_classifier=pretrained_model, finetune_classifier=finetuned_model
        )  # share params from pre-trained model

        # fine-tune with smaller learning rate
        if hasattr(finetuned_model, "learning_rate_init"):
            finetuned_model.learning_rate_init *= 0.1
        finetuned_model.fit(x_finetune, y_finetune)
        return finetuned_model


def pretrain_and_finetune_classifier(
    pretrain_abstracts: pd.DataFrame,
    finetune_abstracts: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    k: int,
    savepath: Path,
    grid_search: bool = False,
) -> Tuple[
    Union[LogisticRegression, MLPClassifier, None], TfidfVectorizer, SelectKBest
]:
    """Pre-trains a classifier on abstracts stratified by journal type as a
    proxy for relevancy before fine-tuning on manually annotated abstracts.
    """
    # pretrain_classifier = LogisticRegression(
    #     C=0.1, max_iter=100, random_state=RANDOM_SEED
    # )
    pretrain_classifier = MLPClassifier(
        solver="adam",
        alpha=0.001,
        max_iter=200,
        hidden_layer_sizes=(32,),
        random_state=RANDOM_SEED,
        early_stopping=True,
    )
    pretrained_model, fitted_vectorizer, fitted_selector = pretrain_model(
        pretrain_abstracts=pretrain_abstracts,
        classifier=pretrain_classifier,
        vectorizer=vectorizer,
        selector=selector,
        k=k,
        savepath=savepath,
        grid_search=grid_search,
    )

    if pretrained_model is None:
        return None, fitted_vectorizer, fitted_selector

    finetuned_model = finetune_model(
        finetune_abstracts=finetune_abstracts,
        pretrained_model=pretrained_model,
        vectorizer=fitted_vectorizer,
        selector=fitted_selector,
        k=k,
        savepath=savepath,
        grid_search=grid_search,
    )

    return finetuned_model, fitted_vectorizer, fitted_selector


def evaluate_model(
    model: BaseEstimator,
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """Perform cross-validation and bootstrap sampling to evaluate the model."""
    cv_scores = cross_val_score(model, features, labels, cv=5, scoring="f1")

    # bootstrap sampling
    bootstrap_scores: List[float] = []
    for _ in range(n_bootstrap):
        features_boot, labels_boot = resample(features, labels, n_samples=len(labels))
        model_boot = clone(model).fit(features_boot, labels_boot)
        predictions = model_boot.predict(features)
        bootstrap_scores.append(f1_score(labels, predictions))

    # calculate confidence intervals
    ci_lower = np.percentile(bootstrap_scores, 5)
    ci_upper = np.percentile(bootstrap_scores, 95)

    return {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "bootstrap_mean": float(np.mean(bootstrap_scores)),
        "bootstrap_ci": (float(ci_lower), float(ci_upper)),
    }


def _classify_test_corpus(
    corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: Union[LogisticRegression, MLPClassifier],
    savepath: Path,
    k: int,
) -> Tuple[zip, float]:
    """Classify a test corpus using the provided vectorizer, selector, and
    classifier.

    Args:
        corpus (pd.DataFrame): The test corpus containing abstracts and
        encodings. vectorizer: The vectorizer used to transform the text data.
        selector: The feature selector for transforming the vectorized data.
        classifier: The classification model for predicting labels.

    Yields:
        tuple: A generator yielding tuples of abstracts and their corresponding
        predictions.
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
        classifier_name=f"str({type(classifier).__name__,})_tfidf_{k}",
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
    classifier: Union[LogisticRegression, MLPClassifier],
):
    """Classify a full corpus using the provided vectorizer, selector, and
    classifier.

    Args:
        vectorizer: The vectorizer used to transform the text data. corpora: The
        corpus to be classified. selector: The feature selector for transforming
        the vectorized data. classifier: The classification model for predicting
        labels.

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
    """Classifies a corpus of abstracts using the provided vectorizer, feature
    selector, and classifier.

    Args:
        corpus (Union[Set[str], pd.DataFrame]): The corpus of abstracts to
        classify. vectorizer (TfidfVectorizer): The vectorizer used to transform
        the abstracts into feature vectors. selector (SelectKBest): The feature
        selector used to select the most informative features. classifier
        (LogisticRegression): The classifier used to predict the class labels.
        test (bool, optional): Flag indicating whether the corpus is a test set.
        Defaults to False.

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
    cleaned = testCorpus.clean_abstracts()
    newdf = pd.DataFrame(cleaned, columns=["abstracts"])
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
        choices=["logistic", "mlp"],
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

    finetune_abstracts = pd.concat(
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

    # get pretrain data
    positive_pretrain = _get_testset(
        data_path=args.pos_set_path,
        positive=True,
    )
    negative_pretrain = _get_testset(
        data_path=args.negative_set_file,
        positive=False,
    )

    # combine pretrain data
    pretrain_abstracts = pd.concat(
        [positive_pretrain, negative_pretrain], ignore_index=True
    )

    # get classifier
    if args.classifier == "logistic":
        classifier = LogisticRegression(C=20, max_iter=100, random_state=RANDOM_SEED)
    if args.classifier == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=1000, random_state=RANDOM_SEED
        )

    # get vectorizer and selector
    vectorizer = get_vectorizer()
    selector = get_selector(k=num)

    # perform grid search
    if args.grid_search:
        print("Performing grid search...")
        pretrain_and_finetune_classifier(
            pretrain_abstracts=pretrain_abstracts,
            finetune_abstracts=finetune_abstracts,
            # classifier=classifier,
            vectorizer=vectorizer,
            selector=selector,
            k=num,
            grid_search=True,
            savepath=savepath,
        )
        print(
            "Grid search completed. Params saved to disk - use them to train the final model."
        )
        return

    # final model training
    print("Training final model...")
    final_model, fitted_vectorizer, fitted_selector = pretrain_and_finetune_classifier(
        pretrain_abstracts=pretrain_abstracts,
        finetune_abstracts=finetune_abstracts,
        vectorizer=vectorizer,
        selector=selector,
        k=num,
        grid_search=False,
        savepath=savepath,
    )

    if final_model is not None:
        # evaluate final model
        x_finetune, y_finetune = vectorize_abstracts(
            abstracts=finetune_abstracts,
            vectorizer=fitted_vectorizer,
            selector=fitted_selector,
        )
        evaluation_results = evaluate_model(final_model, x_finetune, y_finetune)

        # save evaluation results
        with open(
            savepath / f"{args.classifier}_tfidf_{num}_evaluation_results.json", "w"
        ) as f:
            json.dump(evaluation_results, f)

        # get and save ROC/AUC
        y_pred_proba = final_model.predict_proba(x_finetune)[:, 1]
        get_roc_auc(
            true_labels=y_finetune,
            predicted_labels=y_pred_proba,
            classifier_name=f"{args.classifier}_final_model_tfidf_{num}",
            savepath=savepath,
        )

        # save final model
        joblib.dump(
            final_model,
            savepath / f"{args.classifier}_final_model_tfidf_{num}.pkl",
        )

        # classify full corpus
        abstracts_classified = classify_corpus(
            corpus=abstract_corpus,
            vectorizer=fitted_vectorizer,
            selector=fitted_selector,
            classifier=final_model,
            savepath=savepath,
            k=num,
        )
        with open(
            savepath / f"abstracts_{args.classifier}_classified_tfidf_{num}.pkl", "wb"
        ) as f:
            pickle.dump(abstracts_classified, f)

        print("Final model training, evaluation, and classification completed.")
    else:
        print("Error: Final model training failed.")


if __name__ == "__main__":
    main()
