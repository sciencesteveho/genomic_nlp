import json

import numpy as np


def calculate_stats(
    cv_mean, cv_std, bootstrap_mean, lower_ci, upper_ci, decimal_places=4
):
    """
    Calculate and format statistics for CV mean ± std dev and bootstrap mean with CI and error.
    """

    def round_and_format(value):
        return np.round(value, decimal_places)

    cv_lower = cv_mean - cv_std
    cv_upper = cv_mean + cv_std

    bootstrap_error_lower = bootstrap_mean - lower_ci
    bootstrap_error_upper = upper_ci - bootstrap_mean
    bootstrap_error = max(
        bootstrap_error_lower, bootstrap_error_upper
    )  # Use the larger error

    rounded_cv_mean = round_and_format(cv_mean)
    rounded_cv_std = round_and_format(cv_std)
    rounded_cv_lower = round_and_format(cv_lower)
    rounded_cv_upper = round_and_format(cv_upper)
    rounded_bootstrap_mean = round_and_format(bootstrap_mean)
    rounded_lower_ci = round_and_format(lower_ci)
    rounded_upper_ci = round_and_format(upper_ci)
    rounded_bootstrap_error = round_and_format(bootstrap_error)

    return {
        "cv_mean_std": f"{rounded_cv_mean} ± {rounded_cv_std}",
        "cv_range": f"({rounded_cv_lower} to {rounded_cv_upper})",
        "bootstrap_mean_error": f"{rounded_bootstrap_mean} ± {rounded_bootstrap_error}",
        "bootstrap_mean_ci": f"{rounded_bootstrap_mean} ({rounded_lower_ci} to {rounded_upper_ci})",
        "bootstrap_error": f"-{round_and_format(bootstrap_error_lower)}/+{round_and_format(bootstrap_error_upper)}",
    }


# List of JSON files
json_files = [
    "logistic_tfidf_10000_evaluation_results.json",
    "logistic_tfidf_20000_evaluation_results.json",
    "logistic_tfidf_40000_evaluation_results.json",
    "mlp_tfidf_10000_evaluation_results.json",
    "mlp_tfidf_20000_evaluation_results.json",
    "mlp_tfidf_30000_evaluation_results.json",
]

# Process each file
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)

    result = calculate_stats(
        data["cv_mean"],
        data["cv_std"],
        data["bootstrap_mean"],
        data["bootstrap_ci"][0],
        data["bootstrap_ci"][1],
    )

    print(f"\nResults for {file}:")
    print(f"CV Mean ± Std Dev: {result['cv_mean_std']}")
    print(f"CV Range: {result['cv_range']}")
    print(f"Bootstrap Mean ± Error: {result['bootstrap_mean_error']}")
    print(f"Bootstrap Mean (CI): {result['bootstrap_mean_ci']}")
    print(f"Bootstrap Error: {result['bootstrap_error']}")
