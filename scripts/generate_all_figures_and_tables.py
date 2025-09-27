import argparse
import os

from ste.optimization import get_optimal_stopping_strategy, plot_stopping_strategy_state_graphs
from ste.utils.figures import save_drawing
from ste.utils.misc import get_output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or get_output_path("all_figs_and_tables", file_name_suffix="")
    os.mkdir(output_dir)

    oss = get_optimal_stopping_strategy(N=3, alpha=0.1)
    fig = plot_stopping_strategy_state_graphs(oss, font_size=24)
    output_path = f"{output_dir}/Figure 1"
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
