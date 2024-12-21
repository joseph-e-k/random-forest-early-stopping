# Presets for making nice LaTeX-compatible figures.
#
# Use as follows:
#
# with matplotlib.rc_context(rc = RCPARAMS_LATEX_DOUBLE_COLUMN):
#     fig, ax = plt.subplots()
#     ax.plot(...)
#     ...
#     save_figure(fig, 'very-plot-such-amazing-wow')
#
#
# Amit Moscovich, Tel Aviv University, 2023.
import functools
import os

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy
import networkx as nx

from ste.utils.logging import get_module_logger
from ste.utils.multiprocessing import parallelize
from ste.utils.misc import get_name, stringify_kwargs


_logger = get_module_logger()


FIGURES_PATH = r"C:\Users\Josep\Dropbox\Joseph's Dropbox\School\Thesis\Graphs"
DPI = 600

_RCPARAMS_LATEX_SINGLE_COLUMN = {
    'font.family': 'serif',
    'text.usetex': True,

    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,

    # 'axes.prop_cycle': matplotlib.pyplot.cycler('color', ['#006496', '#ff816b', '#fbca60', '#6d904f', '#8b8b8b']) + matplotlib.pyplot.cycler('marker', ['o', 'd', 's', '*', '>']),
    'axes.prop_cycle': matplotlib.pyplot.cycler('color', ['#ff7d66', '#ffdc30', '#40a0cc', '#529915',
                                                          '#8b8b8b']) + matplotlib.pyplot.cycler('marker',
                                                                                                 ['d', 's', 'o',
                                                                                                  r'$\clubsuit$', '>']),

    'lines.markersize': 9,
    'lines.markeredgewidth': 0.75,
    'lines.markeredgecolor': 'k',

    'grid.color': '#C0C0C0',  # 25% black

    'legend.fancybox': True,  # Rounded legend box
    'legend.framealpha': 0.8,

    'axes.linewidth': 1,
}

# This is the right width (in inches) for a 'letter' page LaTeX document that imports the geometry package with default parameters. For an A4 page you will need to adjust the width.
_PAGE_WIDTH_INCHES = 6.775
_GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
_WIDTH = _PAGE_WIDTH_INCHES
_HEIGHT = _GOLDEN_RATIO * _WIDTH
RCPARAMS_LATEX_DOUBLE_COLUMN = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH / 2, _HEIGHT / 2)}
RCPARAMS_LATEX_SINGLE_COLUMN_WIDE = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH, _HEIGHT / 2)}
RCPARAMS_LATEX_SINGLE_COLUMN_LARGE = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH, _HEIGHT)}
RCPARAMS_LATEX_SINGLE_COLUMN_LARGE_SHORT = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH, _HEIGHT * 0.75)}
RCPARAMS_LATEX_SINGLE_COLUMN_LARGE_TALL = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH, 1.2 * _WIDTH)}
RCPARAMS_ONE_TIME_THING = {**_RCPARAMS_LATEX_SINGLE_COLUMN, 'figure.figsize': (_WIDTH / 2, _WIDTH * 540 / 960)}

def save_figure(fig, name):
    os.makedirs(FIGURES_PATH, exist_ok=True)
    filename = os.path.join(FIGURES_PATH, name).replace('.', '_') + '.pdf'
    _logger.info(f'Saving figure to "{os.path.realpath(filename)}"')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')


def plot_function(ax, x_axis_arg_name, function, function_kwargs=None, plot_kwargs=None, results_transform=lambda y: y,
                  x_axis_values_transform=lambda x: x, concurrently=True):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    x_axis_values = function_kwargs.pop(x_axis_arg_name)

    ax.set_xlabel(x_axis_arg_name)

    title = get_name(function)
    if function_kwargs:
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    y_axis_values = np.zeros(len(x_axis_values))

    tasks = parallelize(
        functools.partial(function, **function_kwargs),
        argses_to_iter=({x_axis_arg_name: x} for x in x_axis_values),
        dummy=(not concurrently)
    )

    for task in tasks:
        x = task.args_or_kwargs[x_axis_arg_name]
        _logger.debug(f"Computed {get_name(function)} value at {x!r} in {task.duration:.1f}s")
        y_axis_values[task.index] = task.result

    y_axis_values = results_transform(y_axis_values)
    x_axis_values = x_axis_values_transform(x_axis_values)

    ax.plot(x_axis_values, y_axis_values, **plot_kwargs)


def plot_functions(ax, x_axis_arg_name, functions, function_kwargs=None, plot_kwargs=None, results_transform=lambda y: y,
                   x_axis_values_transform=lambda x: x, concurrently=True, labels=None):
    if plot_kwargs is None:
        plot_kwargs = {}

    if labels is None:
        labels = [get_name(function) for function in functions]

    for label, function in zip(labels, functions):
        _logger.info(f"Plotting {get_name(function)}")
        plot_function(ax, x_axis_arg_name, function, dict(function_kwargs), plot_kwargs | dict(label=label),
                      results_transform, x_axis_values_transform, concurrently)

    title = ", ".join(get_name(function) for function in functions)
    if function_kwargs:
        function_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    ax.legend()


def plot_function_many_curves(ax, x_axis_arg_name, distinct_curves_arg_name, function,
                              function_kwargs=None, plot_kwargs=None, concurrently=True):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    distinct_curves_arg_values = function_kwargs.pop(distinct_curves_arg_name)

    for distinct_curves_arg_value in distinct_curves_arg_values:
        plot_function(
            ax,
            x_axis_arg_name,
            function,
            function_kwargs | {distinct_curves_arg_name: distinct_curves_arg_value},
            plot_kwargs | dict(label=f"{distinct_curves_arg_name}={distinct_curves_arg_value}"),
            concurrently=concurrently
        )

    title = get_name(function)
    if function_kwargs:
        title_kwargs = dict(function_kwargs)
        title_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(title_kwargs)})"
    ax.set_title(title)

    ax.legend()


def create_subplot_grid(n_subplots, n_rows=None, n_columns=None, tight_layout=True, figsize=None):
    n_rows = n_rows or 1
    n_columns = n_columns or n_subplots // n_rows
    if n_columns * n_rows != n_subplots:
        raise ValueError("Number of subplots does not fit evenly into given number of rows and columns")

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_columns, tight_layout=tight_layout, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    return fig, axs



def plot_stopping_strategy(ss, ax, ytick_gap=None):
    n_submodels = ss.shape[0] - 1
    triviality_boundary = n_submodels // 2 + 1
    n_difference_values = 2 * triviality_boundary + 1
    i_center_row = triviality_boundary
    out = np.empty((n_difference_values, n_submodels + 1), dtype=float)
    out[:, :] = np.nan

    for n_seen in range(n_submodels + 1):
        for n_seen_good in range(n_seen + 1):
            n_seen_bad = n_seen - n_seen_good

            if n_seen_good > triviality_boundary or n_seen_bad > triviality_boundary:
                continue

            difference = n_seen_bad - n_seen_good
            out[difference + i_center_row, n_seen] = ss[n_seen, n_seen_good]

    cmap = matplotlib.colormaps.get_cmap('viridis')
    cmap.set_bad(color="lightgray")
    cax = ax.matshow(out, cmap=cmap)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

    if ytick_gap is None:
        ytick_gap = out.shape[0] // min(11, out.shape[0])
    n_yticks = out.shape[0] // ytick_gap

    yticks = [i_center_row]
    for i_ytick in range(1, n_yticks // 2 + 1):
        yticks.append(i_center_row + i_ytick * ytick_gap)
        yticks.append(i_center_row - i_ytick * ytick_gap)

    ytick_labels = [tick_row - i_center_row for tick_row in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    plt.colorbar(cax, cax=cbar_ax)
    return ax


def draw_smooth_curve(points, **kwargs):
    # Separate the points into x and y
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Generate more points along the curve
    x_smooth = np.linspace(x.min(), x.max(), 500)  # Dense x for smooth curve
    # Interpolate with spline
    spline = scipy.interpolate.make_interp_spline(x, y, k=3)  # k=3 gives cubic spline
    y_smooth = spline(x_smooth)
    
    # Plot the original points and the smooth curve
    plt.plot(x_smooth, y_smooth, **kwargs)


def compute_node_size_in_square_points(ax: Axes, r, axis="x"):
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())

    match axis:
        case "x" | 0:
            ax_extent = bbox.width
            data_limits = ax.get_xlim()
        case "y" | 1:
            ax_extent = bbox.height
            data_limits = ax.get_ylim()
        case _:
            raise ValueError(f"'axis' must be 'x', 0, 'y', or 1; got {axis!r}")

    data_per_inch = (data_limits[1] - data_limits[0]) / ax_extent
    r_inches = r / data_per_inch
    r_points = r_inches * 72
    return np.pi * (r_points ** 2)


def plot_fwss(fwss, ax: Axes, node_radius=0.2, cmap="viridis_r"):
    ss = np.asarray(fwss.get_prob_stop(), dtype=float)
    n_base_models = ss.shape[0] - 1
    G = nx.DiGraph()

    # Add nodes and positions
    positions = {}
    for i in range(n_base_models + 1):
        for j in range(i + 1):
            G.add_node((i, j))
            positions[(i, j)] = (i, 2 * j - i)

    # Add edges
    for i in range(n_base_models):
        for j in range(i + 1):
            G.add_edge((i, j), (i + 1, j))
            G.add_edge((i, j), (i + 1, j + 1))

    node_probs = np.exp([fwss.get_log_state_probability(*node) for node in G.nodes])

    # Draw the graph
    x_positions, y_positions = zip(*positions.values())
    ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
    ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)

    # Draw edges
    transition_probs = []

    for ((i_src, j_src), (_, j_dest)) in G.edges:
        prob_reach = np.exp(fwss.get_log_state_probability(i_src, j_src))
        prob_continue_if_reached = 1 - ss[i_src, j_src]
        prob_transition_if_continue = fwss.prob_see_bad[i_src, j_src] if j_dest == j_src else fwss.prob_see_good[i_src, j_src]
        transition_probs.append(prob_reach * prob_continue_if_reached * prob_transition_if_continue)

    node_size = compute_node_size_in_square_points(ax, node_radius)
    base_arrow_size = np.sqrt(node_size) / 4
    arrow_sizes = list(base_arrow_size * np.array(transition_probs))
    nx.draw_networkx_edges(G, positions, width=transition_probs, ax=ax, arrowsize=arrow_sizes, edge_color="black", node_size=node_size)

    # Draw nodes
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    node_body_color = "skyblue"
    node_border_weights = 2 * np.array([ss[i, j] if fwss.get_log_state_probability(i, j) > -np.inf else 1 for (i, j) in G.nodes])
    node_border_colors = [node_body_color if w == 0 else "black" for w in node_border_weights]
    nx.draw_networkx_nodes(G, positions, ax=ax, node_color=node_body_color, edgecolors=node_border_colors, alpha=node_probs, node_size=node_size, linewidths=node_border_weights)

    # Add labels
    nx.draw_networkx_labels(G, positions, {(i, j): j for (i, j) in G.nodes}, ax=ax, font_size=10)

    # Add vertical dashed lines for columns
    for i in range(1, n_base_models + 1):
        ax.axvline(x=i-0.5, color="gray", linestyle="dashed", linewidth=1)

    # Add column headers
    for i in range(n_base_models + 1):
        ax.text(i, n_base_models + 1, f"i = {i}", fontsize=12, ha="center")

    ax.axis("off")
