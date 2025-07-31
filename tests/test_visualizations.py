import os
import random
import tempfile
import warnings

import numpy as np
import pytest
from ste.empirical_performance import (
    DEFAULT_ADRS, get_and_draw_disagreement_rates_and_runtimes,
    get_minimean_flat_ss, get_minimean_ss, get_minimax_ss, get_minimixed_flat_ss, get_minimixed_ss
)
from ste.optimization import get_optimal_stopping_strategy, plot_stopping_strategy_state_graphs, show_stopping_strategy_state_graphs
from ste.utils.data import get_datasets_with_names
from ste.utils.figures import save_drawing
from ste.utils.misc import unzip
from tests.utils import assert_directory_of_images_matches_reference


REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


@pytest.mark.skip()
def test_ss_state_graph_visualization_sanity():
    reference_dir = os.path.join(REFERENCES_DIR, "ss_state_graph_visualization_11_submodels_1e-2_adr")

    n = 11
    adr = 1e-2

    oss = get_optimal_stopping_strategy(n, adr)

    with tempfile.TemporaryDirectory() as output_dir:
        fig = plot_stopping_strategy_state_graphs(oss)
        save_drawing(fig, output_dir)
        assert_directory_of_images_matches_reference(reference_dir, output_dir)


@pytest.mark.skip()
def test_ss_performance_comparison_visualization():
    reference_dir = os.path.join(REFERENCES_DIR, "empirical_comparison_10_forests_of_51_trees")

    n_forests = 10
    n_trees = 51
    random_seed = 1234
    dataset_names, datasets = unzip(get_datasets_with_names())
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
                "Minimean (Cal)": get_minimean_ss,
                "Minimean (Flat)": get_minimean_flat_ss,
                "Minimixed (Cal)": get_minimixed_ss,
                "Minimixed (Flat)": get_minimixed_flat_ss,
            },
            combine_plots=False
        )

    with tempfile.TemporaryDirectory() as output_dir:
        save_drawing(drawing, output_dir)
        assert_directory_of_images_matches_reference(reference_dir, output_dir)
