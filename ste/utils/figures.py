import functools
import itertools
import os

import matplotlib
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import networkx as nx

from .logging import get_module_logger
from .multiprocessing import parallelize_to_array
from .misc import Dummy, extend_array, get_name, retain_central_nonzeros, stringify_kwargs, no_change


_logger = get_module_logger()

DPI = 600
DISTINCT_DASH_STYLES = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 1)]
MARKERS = ['o', 's', '^', 'D', 'x', '*', 'P', 'v', 'H', '+']

STYLE_PATH = os.path.join(os.path.dirname(__file__), "latex-paper.mplstyle")
mplstyle.use(STYLE_PATH)


def interpolate_color(color1, color2, t):
    """Return a color somewhere between the two given colors

    Args:
        color1 (Matplotlib color): First color
        color2 (Matplotlib color): Second color
        t (float): Where the new color should fall on a scale between 0 (color1) and 1 (color2)

    Returns:
        Matplotlib color
    """
    rgba1 = colors.to_rgba(color1)
    rgba2 = colors.to_rgba(color2)
    return tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(rgba1, rgba2))


def plot_function(ax, x_axis_arg_name, function, function_kwargs, plot_kwargs=None, results_transform=no_change,
                  x_axis_values_transform=no_change, concurrently=True):
    """Plot values of given function.

    Args:
        ax (Axes): Axes on which to plot the function.
        x_axis_arg_name (str): Argument to use as the horizontal axis. Must be the name of an argument of the function.
        function (Callable): Function to plot.
        function_kwargs (dict, optional): Dictionary of keyword arguments to pass to the function. Must have a key equal to x_axis_arg_name, which should be a Sequence of values to use for that argument. All other keys should map to single values.
        plot_kwargs (dict, optional): Dictionary of keyword arguments to pass to ax.plot(). Defaults to None.
        results_transform (Callable, optional): Transformation to apply to the function's return values before plotting. Defaults to no_change.
        x_axis_values_transform (Callable, optional): Transformation to apply to the x-axis values before plotting. Defaults to no_change.
        concurrently (bool, optional): If True (the default), compute the different values of the function concurrently using parallelize().

    Returns:
        list[Line2D]: List of lines drawn by ax.plot()
    """
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    x_axis_values = function_kwargs.pop(x_axis_arg_name)

    ax.set_xlabel(x_axis_arg_name)

    title = get_name(function)
    if function_kwargs:
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    if concurrently:
        y_axis_values = parallelize_to_array(
            functools.partial(function, **function_kwargs),
            argses_to_iter=[{x_axis_arg_name: x} for x in x_axis_values]
        )
    else:
        y_axis_values = np.array([function(**{x_axis_arg_name: x}, **function_kwargs) for x in x_axis_values])

    y_axis_values = results_transform(y_axis_values)
    x_axis_values = x_axis_values_transform(x_axis_values)

    return ax.plot(x_axis_values, y_axis_values, **plot_kwargs)


def plot_functions(ax, x_axis_arg_name, functions, function_kwargs, plot_kwargses=None, results_transform=no_change,
                   x_axis_values_transform=no_change, concurrently=True, labels=None):
    """Plot multiple functions on the same set of axes.

    Args:
        ax (Axes): Axes on which to plot the functions.
        x_axis_arg_name (str): Argument to use as the horizontal axis. Must be the name of an argument of all functions.
        functions (Iterable[Callable]): Functions to plot.
        function_kwargs (dict, optional): Dictionary of keyword arguments to pass to the functions. Must have a key equal to x_axis_arg_name, which should be a Sequence of values to use for that argument. All other keys should map to single values.
        plot_kwargses (Iterable[dict], optional): Iterable of dictionaries of keyword arguments to pass to ax.plot(). If provided, should be the same length as `functions`. Defaults to repeated empty dictionaries.
        results_transform (Callable, optional): Transformation to apply to the functions' return values before plotting. Defaults to no_change.
        x_axis_values_transform (Callable, optional): Transformation to apply to the x-axis values before plotting. Defaults to no_change.
        concurrently (bool, optional): If True (the default), compute the different values of the functions concurrently using parallelize().
        labels (Iterable[str], optional): Label to use for each function in the plot. Defaults to function names.

    Returns:
        list[Line2D]: List of lines drawn by ax.plot()
    """
    if plot_kwargses is None:
        plot_kwargses = itertools.repeat({})

    if labels is None:
        labels = [get_name(function) for function in functions]

    lines = []

    for label, function, plot_kwargs in zip(labels, functions, plot_kwargses):
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


def resolve_grid_shape(n_total: int = None, n_rows: int = None, n_columns: int = None) -> tuple[int, int]:
    """Compute correct grid dimensions given the total number of items and a known number of rows or columns."""
    
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
    """Create a Figure object and a 2D array of Axes objects as returned from plt.subplots(). Unlike plt.subplots(), however:
        - One missing dimension of the grid may be inferred from the total number of subplots required; and
        - The second member of the returned pair is always a 2D array, instead of sometimes being a 1D array or a bare Axes object.

    Args:
        n_subplots (int, optional): Total number of subplots required. May be omitted if n_rows and n_columns are specified.
        n_rows (int, optional): Number of rows in the grid of subplots. May be omitted if n_subplots and n_columns are specified.
        n_columns (int, optional): Number of columns in the grid of subplots. May be omitted if n_subplots and n_rows are specified.
        tight_layout (bool, optional): See the tight_layout argument to plt.subplots(). Defaults to True.
        figsize (tuple[int, int], optional): See the figsize argument to plt.subplots(). Defaults to None.

    Returns:
        tuple[Figure, np.ndarray]: a Figure and a 2D array of Axes contained therein, arranged on a grid.
    """
    n_rows, n_columns = resolve_grid_shape(n_subplots, n_rows, n_columns)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_columns, tight_layout=tight_layout, figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.reshape((n_rows, n_columns))
    
    return fig, axs


class MultiFigure(Dummy):
    """Dummy object to stand in for the Figure returned by plt.subplots(), when we want to create multiple Axes at once with *separate* Figures instead."""
    def __init__(self, axs):
        super().__init__("<dummy figure>")
        self.axs = axs


def create_independent_plots_grid(n_plots=None, n_rows=None, n_columns=None, axs_kw=None, **fig_kw):
    """Create a 2D array of Axes objects, each owned by a separate Figure.
    Designed to be usable interchangably with create_subplot_grid() and therefore also returns a dummy MultiFigure object.

    Args:
        n_plots (int, optional): Total number of desired Axes. May be omitted if n_rows and n_columns are specified.
        n_rows (int, optional): Number of rows in the array of Axes. May be omitted if n_plots and n_columns are specified.
        n_columns (int, optional): Number of columns in the array of Axes. May be omitted if n_plots and n_rows are specified.
        axs_kw (dict[str, Any], optional): Additional keywords to be passed to fig.add_subplot() call used to create each Axes.
    
    All other keyword arguments will be passed directly to the plt.figure() call used to create each Figure.

    Returns:
        tuple[MultiFigure, np.ndarray]
    """
    axs_kw = axs_kw or {}

    grid_shape = resolve_grid_shape(n_plots, n_rows, n_columns)

    axs = np.empty(shape=grid_shape, dtype=object)
    
    for (i, j) in np.ndindex(grid_shape):
        fig = plt.figure(**fig_kw)
        axs[i, j] = fig.add_subplot(**axs_kw)
    
    return MultiFigure(axs), axs


def _compute_node_size_in_square_points(ax: Axes, r, axis="x"):
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


def plot_evwss(evwss, ax: Axes, node_radius=0.2):
    """Plot a state transition chart of the given EnsembleVoteWithStoppingStrategy on the given Axes

    Args:
        evwss (EnsembleVoteWithStoppingStrategy): EVWSS whose transition chart is to be generated.
        ax (Axes): Axes to draw the chart on.
        node_radius (float, optional): Radius of the circles used to represent states. Defaults to 0.2.
    """
    ss = evwss.stopping_strategy
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

    node_probs = np.exp([evwss.get_log_state_probability(*node) for node in G.nodes])

    # Draw the graph
    x_positions, y_positions = zip(*positions.values())
    ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
    ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)

    # Draw edges
    transition_probs = []

    for ((i_src, j_src), (_, j_dest)) in G.edges:
        if i_src > evwss.n_total or j_src > evwss.n_yes:
            transition_probs.append(0)
        else:
            prob_reach = np.exp(evwss.get_log_state_probability(i_src, j_src))
            prob_continue_if_reached = 1 - ss[i_src, j_src]
            prob_transition_if_continue = evwss.prob_see_no[i_src, j_src] if j_dest == j_src else evwss.prob_see_yes[i_src, j_src]
            transition_probs.append(prob_reach * prob_continue_if_reached * prob_transition_if_continue)

    node_size = _compute_node_size_in_square_points(ax, node_radius)
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
    """Save a given Axes, Figure or Multifigure to disk. MultiFigures will be saved as a directory containing multiple image files.

    Args:
        drawing (Axes | Figure | MultiFigure): Image to be saved to disk.
        path (str): Path to save at, without the file name suffix.
        file_name_suffix (str, optional): File name suffix, which also determines file type. Defaults to ".pdf".

    Raises:
        TypeError: _description_
    """
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


def enforce_character_limit(x, pos, max_characters):
    """Stringify the given floating-point number, but return an empty string if over the given number of characters.

    Args:
        x (float): Number to represent.
        pos (Any): Dummy argument to conform to the ticker.FuncFormatter() interface.
        max_characters (int): Maximum number of allowable characters.

    Returns:
        str
    """
    label = f"{x:.10g}"
    if len(label) > max_characters:
        return ""
    return label


def plot_stopping_strategy(ss, ax, ytick_gap=None):
    n_submodels = ss.shape[0] - 1
    triviality_boundary = n_submodels // 2 + 1
    n_difference_values = 2 * triviality_boundary + 1
    i_center_row = triviality_boundary
    out = np.zeros(shape=(n_difference_values, n_submodels + 1), dtype=float)

    for n_seen in range(n_submodels + 1):
        for n_seen_yes in range(n_seen + 1):
            n_seen_no = n_seen - n_seen_yes

            if n_seen_yes > triviality_boundary or n_seen_no > triviality_boundary:
                continue

            difference = n_seen_no - n_seen_yes
            out[difference + i_center_row, n_seen] = ss[n_seen, n_seen_yes]

    for j in range(1, out.shape[1]):
        out[:, j] = retain_central_nonzeros(out[:, j])

    ax.imshow(out, cmap="Greys")

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
    return ax
