import os
import random
import tempfile
import warnings

import numpy as np
from ste.empirical_performance import DEFAULT_ADRS, get_and_draw_disagreement_rates_and_runtimes, get_bayesian_bad_ss, get_bayesian_flat_ss, get_bayesian_perfect_ss, get_bayesian_ss, get_minimax_ss
from ste.optimization import get_optimal_stopping_strategy, show_stopping_strategy
from ste.utils.data import get_names_and_datasets
from ste.utils.figures import save_drawing
from tests.utils import assert_directory_of_images_matches_reference


REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


def test_ss_visualization_sanity():
    reference_dir = os.path.join(REFERENCES_DIR, "ss_visualization_11_submodels_1e-2_adr")

    n = 11
    adr = 1e-2

    oss = get_optimal_stopping_strategy(n, adr)

    with tempfile.TemporaryDirectory() as output_dir:
        show_stopping_strategy(oss, output_dir)
        assert_directory_of_images_matches_reference(reference_dir, output_dir)


def test_ss_performance_comparison_visualization():
    reference_dir = os.path.join(REFERENCES_DIR, "empirical_comparison_30_forests_of_101_trees")

    n_forests = 30
    n_trees = 101
    random_seed = 1234
    dataset_names, datasets = get_names_and_datasets()
    adrs = DEFAULT_ADRS

    random.seed(random_seed)
    np.random.seed(random_seed)

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        drawing = get_and_draw_disagreement_rates_and_runtimes(
            n_forests,
            n_trees,
            datasets,
            dataset_names,
            adrs,
            {
                "Minimax": get_minimax_ss,
                "Minimean (Cal)": get_bayesian_ss,
                "Minimean (Test)": get_bayesian_perfect_ss,
                "Minimean (Train)": get_bayesian_bad_ss,
                "Minimean (Flat)": get_bayesian_flat_ss
            },
            combine_plots=False
        )

    with tempfile.TemporaryDirectory() as output_dir:
        save_drawing(drawing, output_dir)
        assert_directory_of_images_matches_reference(reference_dir, output_dir)
