import argparse
import os

from ste.empirical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss
from ste.optimization import get_optimal_stopping_strategy, plot_stopping_strategy_state_graphs
from ste.utils.figures import create_subplot_grid, label_subplots, plot_stopping_strategies_as_envelopes, save_drawing
from ste.utils.misc import get_output_path


def generate_figure_1(output_dir):
    oss = get_optimal_stopping_strategy(N=3, alpha=0.1)
    fig = plot_stopping_strategy_state_graphs(oss, font_size=24)
    output_path = f"{output_dir}/Figure 1"
    save_drawing(fig, output_path)


def generate_figure_2(output_dir):
    adrs = [1e-2, 1e-4, 0]
    ss_getters = get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss
    sses = [[get_ss(adr, 101, None) for adr in adrs] for get_ss in ss_getters]

    fig, axs = create_subplot_grid(len(ss_getters), n_rows=1, tight_layout=False, figsize=(10, 3))
    fig.subplots_adjust(hspace=10)

    for i_ss_kind, ss_getter in enumerate(ss_getters):
        ax = axs[0, i_ss_kind]
        sses = [ss_getter(adr, 101, None) for adr in adrs]
        plot_stopping_strategies_as_envelopes(sses, ax, [f"ADR = {adr}" for adr in adrs])
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_xlim((0, 101))
        ax.set_ylim((0, 52))
        ax.grid(which="major")

    label_subplots(axs, from_top=0, from_left=-0.205, fontsize=14, bbox=None)

    output_path = f"{output_dir}/Figure 2"
    save_drawing(fig, output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or get_output_path("all_figs_and_tables", file_name_suffix="")
    os.mkdir(output_dir)

    generate_figure_1(output_dir)
    generate_figure_2(output_dir)


if __name__ == "__main__":
    main()
