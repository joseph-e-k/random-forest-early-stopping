import argparse
from collections.abc import Sequence
import functools
import os
import random
import warnings

import matplotlib
import numpy as np
import pandas as pd

from ste.theoretical_performance import DEFAULT_ADRS, PLOT_DETAILS_BY_SS_GETTER, ReproducibilityError, StoppingStrategyGetter, _getitem, analyse_stopping_strategies, get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss, get_schwing_ss
from ste.utils.caching import memoize
from ste.utils.data import Dataset
from ste.utils.figures import DISTINCT_DASH_STYLES, create_independent_plots_grid, create_subplot_grid, plot_functions, save_drawing
from ste.utils.logging import configure_logging
from ste.utils.misc import get_output_path
from ste.utils.multiprocessing import parallelize_to_array


@memoize()
def get_worst_case_metrics(adrs: Sequence[float], n_trees: int, stopping_strategy_getters: list[StoppingStrategyGetter]):
    """Compute the metrics for a given collection of kinds of stopping strategies on the given dataset.

    Args:
        adrs (Sequence[float]): Allowable disagreement rates to compute the stopping strategies with.
        n_trees (int): Number of trees in the forest.
        stopping_strategy_getters (list[StoppingStrategyGetter]): List of functions to compute stopping strategies.
    
    Returns:
        np.ndarray: 3D array of estimated metrics, with axes corresponding to:
            0. Stopping strategy (length = len(stopping_strategy_getters))
            1. Allowable disagreement rate (length = len(adrs))
            2. Metric kind: base error rate, disagreement rate, expected runtime, and error rate (length = 4).
            
    """
    smopdis = np.zeros(shape=(n_trees + 1))
    smopdis[n_trees // 2] = 1
    smopdis[n_trees // 2 + 1] = 1
    smopdis_estimates_for_evaluation = np.vstack([smopdis, smopdis])
    smopdis_estimate_for_ss = smopdis

    stopping_strategies = parallelize_to_array(
        stopping_strategy_getters,
        argses_to_iter=[
            (adr, n_trees, smopdis_estimate_for_ss)
            for adr in adrs
        ],
        job_name="get_ss"
    )

    return analyse_stopping_strategies(stopping_strategies, smopdis_estimates_for_evaluation)


def get_and_draw_worst_case_metrics(ss_getters, allowable_disagreement_rates, n_trees, combine_plots=False):
    """Draw precomputed performance metrics for a collection of datasets and stopping strategies.

    Args:
        ss_getters (Sequence[StoppingStrategyGetter]): Functions used to compute the stopping strategies.
        allowable_disagreement_rates (Sequence[float]): Allowable disagreement rates used to compute the stopping strategies.
        n_trees (int): Number of trees in the forest.
        combine_plots (bool, optional): Whether to combine all plots into a single figure. Defaults to False.
    
    Returns:
        matplotlib.Figure or MultiFigure: Plots of the metrics.
    """
    metrics = get_worst_case_metrics(allowable_disagreement_rates, n_trees, ss_getters)

    metrics = metrics[..., :-2]
    n_ss_kinds, n_adrs, n_metrics = metrics.shape

    assert len(allowable_disagreement_rates) == n_adrs
    assert n_metrics == 2


    if combine_plots:
        fig, axs = create_subplot_grid(n_metrics, n_rows=1, tight_layout=False, figsize=(10, 3))
        fig.subplots_adjust(hspace=10)
    else:
        fig, axs = create_independent_plots_grid(n_metrics, n_rows=1, figsize=(4, 4))

    metric_names = ["Disagreement Rate", "Expected Runtime"]
    metric_maxima = np.max(metrics, axis=(0, 1))

    plot_labels = []
    plot_kwargses = []

    for ss_getter in ss_getters:
        plot_details = PLOT_DETAILS_BY_SS_GETTER[ss_getter]
        plot_labels.append(plot_details.label)
        plot_kwargses.append(dict(marker=plot_details.marker, fillstyle="none", color=plot_details.color))

    
    for i_metric in range(n_metrics):
        metric = metrics[:, :, i_metric]
        metric_name = metric_names[i_metric]
        ax = axs[0, i_metric]

        if metric_name == "Disagreement Rate":
            ax.set_yscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
            ax.set_ylim((0, 1))
            ax.plot([0, 1], [0, 1], color="black", label="ADR", linestyle='dashed')
            ax.legend()
        elif metric_name == "Expected Runtime":
            ax.set_yticks(list(range(0, int(metric_maxima[i_metric]), 5)), minor=True)
            ax.set_ylim((0, metric_maxima[i_metric]))
        else:
            raise ValueError(f"Unrecognized metric name: {metric_name!r}")
        
        ax.grid(visible=True, axis='y', which='major')
        ax.set_xscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
        ax.set_xlim((min(allowable_disagreement_rates), max(allowable_disagreement_rates)))

        metric_value_getters = [
            functools.partial(_getitem, metric[i_ss])
            for i_ss in range(n_ss_kinds)
        ]
        
        lines = plot_functions(
            ax=ax,
            x_axis_arg_name="index",
            functions=metric_value_getters,
            function_kwargs=dict(
                index=range(n_adrs)[::-1],
            ),
            concurrently=False,
            labels=plot_labels,
            x_axis_values_transform=lambda i_adrs: [allowable_disagreement_rates[i_adr] for i_adr in i_adrs],
            plot_kwargses=plot_kwargses
        )

        for (line, dash_pattern) in zip(lines, DISTINCT_DASH_STYLES):
            line.set_dashes(dash_pattern)

        ax.legend(framealpha=0.5)
        ax.set_title(metric_name)
        ax.set_xlabel("Allowable disagreement rate")
        ax.figure.canvas.draw()

        xtick_offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, ax.figure.dpi_scale_trans)
        for tick in ax.xaxis.get_major_ticks():
            if tick.get_loc() >= ax.get_xlim()[1]:
                tick.label1.set_transform(tick.label1.get_transform() + xtick_offset)
    
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.set_defaults(action_name="detailed_empirical_comparison")
    parser.add_argument("--n-trees", "--number-of-trees", "-N", type=int, default=101)
    parser.add_argument("--alphas", "--adrs", "-a", type=float, nargs="+", default=DEFAULT_ADRS)
    parser.add_argument("--output-path", "-o", type=str, default=None)
    parser.add_argument("--random-seed", "-s", type=int, default=1234)
    parser.add_argument("--combine-plots", "-c", action="store_true")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    configure_logging()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not os.getenv("PYTHONHASHSEED"):
        raise ReproducibilityError("Please set the PYTHONHASHSEED environment variable for reproducible results")

    pd.options.mode.chained_assignment = None

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        drawing = get_and_draw_worst_case_metrics(
            [
                get_minimax_ss,
                get_minimean_flat_ss,
                get_minimixed_flat_ss,
                get_schwing_ss
            ],
            args.alphas,
            args.n_trees,
            args.combine_plots
        )

    output_path = args.output_path or get_output_path(f"worst_case_performance_{args.n_trees}_trees")
    save_drawing(drawing, output_path)


if __name__ == "__main__":
    main()
