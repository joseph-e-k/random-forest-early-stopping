import numpy as np
from ste.timing import get_os_solution_times, aggregate_and_plot_times
from ste.utils.figures import create_subplot_grid, save_drawing
from ste.utils.misc import get_output_path


def main(output_path=None):
    low_ensemble_sizes = list(range(1, 16))
    low_combined_times = get_os_solution_times(
        ensemble_sizes=low_ensemble_sizes,
        adrs=[0.01],
        n_reps=3,
        nonce=1234,
        problem_kinds=("lp", "qp")
    )

    high_ensemble_sizes = list(range(50, 201, 5))
    high_lp_times = get_os_solution_times(
        ensemble_sizes=high_ensemble_sizes,
        adrs=[0.01],
        n_reps=3,
        nonce=20250731,
        problem_kinds=("lp",)
    )

    fig, axs = create_subplot_grid(2, 1, figsize=(10, 4))

    aggregate_and_plot_times(
        ensemble_sizes=low_ensemble_sizes,
        times=low_combined_times,
        adrs=[0.01],
        problem_kinds=("lp", "qp"),
        log=True,
        fig_and_axs=(None, np.array([[axs[0, 0]]]))
    )

    aggregate_and_plot_times(
        ensemble_sizes=high_ensemble_sizes,
        times=high_lp_times,
        adrs=[0.01],
        problem_kinds=("lp",),
        log=True,
        fig_and_axs=(None, np.array([[axs[0, 1]]]))
    )

    axs[0, 1].set_ylim(axs[0, 0].get_ylim())
    axs[0, 1].set_ylabel(None)
    axs[0, 1].set_yticklabels([])

    save_drawing(fig, output_path or get_output_path(f"timing_combined_combined"))


if __name__ == "__main__":
    main()
