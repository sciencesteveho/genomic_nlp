#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Classes to handle plotting of model performances."""


from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors  # type: ignore
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import roc_curve

from baseline_models import BaselineModel


class BaselineModelVisualizer:
    """Class to handle plotting of model performances."""

    def __init__(
        self,
        output_path: str,
    ) -> None:
        """Instantiate a PerformanceVisualizer object."""
        self.output_path = output_path
        self._set_matplotlib_publication_parameters()

    def plot_model_performances(
        self,
        train_results: Dict[str, float],
        test_results: Dict[str, float],
    ) -> None:
        """Plot model performances on train and test sets."""
        model_names = list(train_results.keys())

        _, axis = self._setup_plot(
            model_names, "Model Performance on Train and Test Sets"
        )

        bar_width = 0.35
        group_spacing = 0.8
        x_positions = np.arange(len(model_names)) * (2 * bar_width + group_spacing)
        train_bar_positions = [x - bar_width / 2 for x in x_positions]
        test_bar_positions = [x + bar_width / 2 for x in x_positions]

        muted_colors = self._get_muted_colors(2)

        axis.bar(
            train_bar_positions,
            [train_results[model] for model in model_names],
            bar_width,
            label="Train",
            color=muted_colors[0],
        )
        axis.bar(
            test_bar_positions,
            [test_results[model] for model in model_names],
            bar_width,
            label="Test",
            color=muted_colors[1],
        )

        axis.set_xticks(x_positions)
        self._finalize_plot(axis=axis, model_names=model_names, savename="train_test")

    def plot_stratified_performance(
        self, stratified_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Plot model performances on the test set, but stratified by source."""
        model_names = list(stratified_results.keys())
        sources = sorted(
            {
                source
                for model_results in stratified_results.values()
                for source in model_results
            }
        )

        _, axis = self._setup_plot(
            model_names, "Model Performance on Test Set by Source", figsize=(12, 6)
        )

        bar_width = 0.15
        group_spacing = 0.8
        x_positions = np.arange(len(model_names)) * (
            len(sources) * bar_width + group_spacing
        )

        muted_colors = self._get_muted_colors(len(sources))

        for i, source in enumerate(sources):
            source_scores = [
                stratified_results[model].get(source, 0) for model in model_names
            ]
            source_bar_positions = x_positions + i * bar_width
            axis.bar(
                source_bar_positions,
                source_scores,
                bar_width,
                label=source,
                color=muted_colors[i],
            )

        axis.set_xticks(x_positions + (len(sources) - 1) * bar_width / 2)
        self._finalize_plot(
            axis,
            model_names,
            savename="stratified",
            legend_title="Source",
            legend_loc="upper left",
        )

    def plot_roc_curve(
        self,
        models: Dict[str, BaselineModel],
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        """Plot ROC curves for all models."""
        for model_name, model in models.items():
            y_pred = model.predict_probability(test_features)
            fpr, tpr, _ = roc_curve(test_labels, y_pred)
            auc = roc_auc_score(test_labels, y_pred)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

        plt.plot([0, 1], [0, 1], color="black", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        self.plot_layout_and_save(plt=plt, savename="roc_curves")

    def plot_bootstrap_results(
        self,
        bootstrap_stats: Dict[str, Dict[str, Dict[str, float]]],
    ) -> None:
        """Plot results of the bootstrapped evaluation."""
        models = list(bootstrap_stats.keys())
        metrics = list(bootstrap_stats[models[0]].keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 6), sharey=True)

        for i, metric in enumerate(metrics):
            ax = axes[i]
            data: List[Tuple[str, float, float, float]] = [
                (
                    m,
                    bootstrap_stats[m][metric]["mean"],
                    bootstrap_stats[m][metric]["ci_lower"],
                    bootstrap_stats[m][metric]["ci_upper"],
                )
                for m in models
            ]

            y_pos = range(len(models))
            ax.errorbar(
                [d[1] for d in data],
                y_pos,
                xerr=[[d[1] - d[2] for d in data], [d[3] - d[1] for d in data]],
                fmt="o",
                capsize=5,
                capthick=2,
                ecolor="black",
                markersize=8,
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_title(metric.upper())
            ax.set_xlabel("Score")

            if i == 0:
                ax.set_ylabel("Models")

        self.plot_layout_and_save(plt=plt, savename="bootstrap_results")

    def _setup_plot(
        self, model_names: List[str], title: str, figsize: Tuple[int, int] = (8, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plot with common parameters."""
        fig, axis = plt.subplots(figsize=figsize)
        axis.set_ylabel("AUC Score")
        axis.set_title(title)
        axis.set_xticks(range(len(model_names)))
        return fig, axis

    def _finalize_plot(
        self,
        axis: plt.Axes,
        model_names: List[str],
        savename: str,
        legend_loc: str = "best",
        legend_title: Optional[str] = None,
    ) -> None:
        """Finalize the plot with common parameters and save it."""
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        if legend_title:
            axis.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc=legend_loc)
        else:
            axis.legend()
        self.plot_layout_and_save(plt=plt, savename=savename)

    def plot_layout_and_save(
        self,
        plt: plt.Figure,
        savename: str,
    ) -> None:
        """Adjust the layout and save the plot."""
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{savename}.png")
        plt.close()

    @staticmethod
    def _set_matplotlib_publication_parameters() -> None:
        plt.rcParams.update(
            {
                "font.size": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.dpi": 300,
                "figure.figsize": (1.5, 2),
                "font.sans-serif": "Nimbus Sans",
            }
        )

    @staticmethod
    def _get_muted_colors(num_colors) -> List:
        """Generate a list of muted colors."""
        base_colors = plt.cm.Set2(np.linspace(0, 1, num_colors))
        return mcolors.LinearSegmentedColormap.from_list("muted", base_colors)(
            np.linspace(0, 1, num_colors)
        )
