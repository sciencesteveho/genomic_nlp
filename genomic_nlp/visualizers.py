#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Classes to handle plotting of model performances."""


from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore


class BaselineModelVisualizer:
    """Class to handle plotting of model performances."""

    def __init__(
        self,
        output_path: str,
    ) -> None:
        """Instantiate a PerformanceVisualizer object."""
        self.output_path = output_path

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
        x_positions = range(len(model_names))
        train_bar_positions = [x - bar_width / 2 for x in x_positions]
        test_bar_positions = [x + bar_width / 2 for x in x_positions]

        axis.bar(
            train_bar_positions,
            [train_results[model] for model in model_names],
            bar_width,
            label="Train",
            color="skyblue",
        )
        axis.bar(
            test_bar_positions,
            [test_results[model] for model in model_names],
            bar_width,
            label="Test",
            color="lightcoral",
        )

        self._finalize_plot(axis, model_names)

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

        bar_width = 0.8 / len(sources)
        x_positions = range(len(model_names))

        for i, source in enumerate(sources):
            source_scores = [
                stratified_results[model].get(source, 0) for model in model_names
            ]
            source_bar_positions = [
                x + i * bar_width - 0.4 + bar_width / 2 for x in x_positions
            ]
            axis.bar(source_bar_positions, source_scores, bar_width, label=source)

        self._finalize_plot(
            axis, model_names, legend_title="Source", legend_loc="upper left"
        )

    def _setup_plot(
        self, model_names: List[str], title: str, figsize: Tuple[int, int] = (8, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the plot with common parameters."""
        self._set_matplotlib_publication_parameters()
        fig, axis = plt.subplots(figsize=figsize)
        axis.set_ylabel("AUC Score")
        axis.set_title(title)
        axis.set_xticks(range(len(model_names)))
        return fig, axis

    def _finalize_plot(
        self,
        axis: plt.Axes,
        model_names: List[str],
        legend_loc: str = "best",
        legend_title: Optional[str] = None,
    ) -> None:
        """Finalize the plot with common parameters and save it."""
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        if legend_title:
            axis.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc=legend_loc)
        else:
            axis.legend()
        self.plot_layout_and_save(plt=plt)

    def plot_bootstrap_results(
        self,
        bootstrap_stats: Dict[str, Dict[str, Dict[str, float]]],
    ) -> None:
        """Plot results of the bootstrapped evaluation."""
        self._set_matplotlib_publication_parameters()
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

        self.plot_layout_and_save(plt=plt)

    def plot_layout_and_save(
        self,
        plt: plt.Pyplot,
    ) -> None:
        """Adjust the layout and save the plot."""
        plt.tight_layout()
        plt.savefig(self.output_path)
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
                "figure.figsize": (8, 6),
                "font.sans-serif": "Nimbus Sans",
            }
        )
