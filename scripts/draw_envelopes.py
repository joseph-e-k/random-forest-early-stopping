import matplotlib.pyplot as plt
from ste.empirical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss
from ste.utils.figures import create_subplot_grid, label_subplots, plot_stopping_strategies_as_envelopes, save_drawing
from ste.utils.misc import get_output_path


def main():
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

    output_path = get_output_path(f"ss_visualization_combined")
    save_drawing(fig, output_path)


if __name__ == "__main__":
    main()
