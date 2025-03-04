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
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import scipy
import networkx as nx

from ste.utils.logging import get_module_logger
from ste.utils.multiprocessing import parallelize
from ste.utils.misc import Dummy, extend_array, get_name, stringify_kwargs


_logger = get_module_logger()

DPI = 600
DISTINCT_DASH_STYLES = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1)]

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


STYLE_PATH = os.path.join(os.path.dirname(__file__), "latex-paper.mplstyle")
mplstyle.use(STYLE_PATH)


def save_drawing(fig, filename):
    _logger.info(f'Saving figure to "{os.path.realpath(filename)}"')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')


def interpolate_color(color1, color2, t):
    rgba1 = colors.to_rgba(color1)
    rgba2 = colors.to_rgba(color2)
    return tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(rgba1, rgba2))


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

    return ax.plot(x_axis_values, y_axis_values, **plot_kwargs)


def plot_functions(ax, x_axis_arg_name, functions, function_kwargs=None, plot_kwargs=None, results_transform=lambda y: y,
                   x_axis_values_transform=lambda x: x, concurrently=True, labels=None):
    if plot_kwargs is None:
        plot_kwargs = {}

    if labels is None:
        labels = [get_name(function) for function in functions]

    lines = []

    for label, function in zip(labels, functions):
        _logger.info(f"Plotting {get_name(function)}")
        lines += plot_function(ax, x_axis_arg_name, function, dict(function_kwargs), plot_kwargs | dict(label=label),
                               results_transform, x_axis_values_transform, concurrently)

    title = ", ".join(get_name(function) for function in functions)
    if function_kwargs:
        function_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    ax.legend()

    return lines


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


def resolve_grid_shape(n_total=None, n_rows=None, n_columns=None):
    if sum(arg is not None for arg in (n_total, n_rows, n_columns)) < 2:
        raise ValueError("At least 2 of n_total, n_rows and n_columns must be specified")
    if n_total is None:
        return n_rows, n_columns
    
    if n_rows is None:
        n_rows = n_total // n_columns
    
    n_rows = n_rows or (n_total // n_columns)
    n_columns = n_columns or (n_total // n_rows)
    if n_columns * n_rows != n_total:
        raise ValueError("Number of items does not fit evenly into given number of rows and columns")
    
    return n_rows, n_columns


def create_subplot_grid(n_subplots=None, n_rows=None, n_columns=None, tight_layout=True, figsize=None):
    n_rows, n_columns = resolve_grid_shape(n_subplots, n_rows, n_columns)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_columns, tight_layout=tight_layout, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.reshape((n_rows, n_columns))
    
    return fig, axs


class MultiFigure(Dummy):
    def __init__(self, axs):
        super().__init__("<dummy figure>")
        self.axs = axs


def create_independent_plots_grid(n_plots=None, n_rows=None, n_columns=None, axs_kw=None, **fig_kw):
    axs_kw = axs_kw or {}

    grid_shape = resolve_grid_shape(n_plots, n_rows, n_columns)

    axs = np.empty(shape=grid_shape, dtype=object)
    
    for (i, j) in np.ndindex(grid_shape):
        fig = plt.figure(**fig_kw)
        axs[i, j] = fig.add_subplot(**axs_kw)
    
    return MultiFigure(axs), axs


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


def plot_fwss(fwss, ax: Axes, node_radius=0.2):
    ss = np.asarray(fwss.get_prob_stop(), dtype=float)
    n_base_models = ss.shape[0] - 1
    ss = extend_array(ss, new_shape=(n_base_models + 1, n_base_models + 1), fill_value=1)
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
        if i_src > fwss.n_total or j_src > fwss.n_total_positive:
            transition_probs.append(0)
        else:
            prob_reach = np.exp(fwss.get_log_state_probability(i_src, j_src))
            prob_continue_if_reached = 1 - ss[i_src, j_src]
            prob_transition_if_continue = fwss.prob_see_bad[i_src, j_src] if j_dest == j_src else fwss.prob_see_good[i_src, j_src]
            transition_probs.append(prob_reach * prob_continue_if_reached * prob_transition_if_continue)

    node_size = compute_node_size_in_square_points(ax, node_radius)
    arrow_size = np.sqrt(node_size) / 4
    nx.draw_networkx_edges(G, positions, ax=ax, arrowsize=arrow_size, edge_color="black", node_size=node_size, alpha=transition_probs)

    # Draw nodes
    node_body_color = "skyblue"
    node_border_colors = [interpolate_color("black", "red", ss[i, j]) for (i, j) in G.nodes]
    nx.draw_networkx_nodes(G, positions, ax=ax, node_color=node_body_color, alpha=node_probs, node_size=node_size, edgecolors=node_border_colors, linewidths=2)

    # Add labels
    nx.draw_networkx_labels(G, positions, {(i, j): j for (i, j) in G.nodes}, ax=ax, font_size=10)

    # Add vertical dashed lines for columns
    for i in range(1, n_base_models + 1):
        ax.axvline(x=i-0.5, color="gray", linestyle="dashed", linewidth=1)

    # Add column headers
    for i in range(n_base_models + 1):
        ax.text(i, n_base_models + 1, f"i = {i}", fontsize=12, ha="center")

    ax.axis("off")


def save_drawing(drawing, path, file_name_suffix=".pdf"):
    if isinstance(drawing, Axes):
        drawing = drawing.figure

    if isinstance(drawing, Figure):
        drawing.savefig(path + file_name_suffix)
        return

    if isinstance(drawing, MultiFigure):
        path = os.path.splitext(path)[0]
        if not os.path.exists(path):
            os.mkdir(path)
        
        for index in np.ndindex(drawing.axs.shape):
            ax = drawing.axs[index]
            name = "_".join(str(i) for i in index)
            
            save_drawing(ax, os.path.join(path, name), file_name_suffix)
        
        return

    raise TypeError("First argument to save_drawing must be an Axes, Figure, or MultiFigure")
