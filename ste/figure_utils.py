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
import numpy as np
from matplotlib import pyplot as plt

from ste.multiprocessing_utils import parallelize
from ste.utils import stringify_kwargs

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
    print(f'Saving figure to "{os.path.realpath(filename)}"')
    fig.savefig(filename, dpi=DPI, bbox_inches='tight')


def plot_function(ax, x_axis_arg_name, function, function_kwargs=None, plot_kwargs=None, results_transform=lambda y: y,
                  x_axis_values_transform=lambda x: x, verbose=False):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    x_axis_values = function_kwargs.pop(x_axis_arg_name)

    ax.set_xlabel(x_axis_arg_name)

    title = function.__name__
    if function_kwargs:
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    y_axis_values = np.zeros(len(x_axis_values))

    results = parallelize(
        functools.partial(function, **function_kwargs),
        ({x_axis_arg_name: x} for x in x_axis_values)
    )

    for i, (kwargs, success, outcome, duration) in enumerate(results):
        if not success:
            raise outcome
        x = kwargs[x_axis_arg_name]
        if verbose:
            print(f"Computed {function.__name__} value at {x!r} in {duration:.1f}s")
        y_axis_values[i] = outcome

    y_axis_values = results_transform(y_axis_values)
    x_axis_values = x_axis_values_transform(x_axis_values)

    ax.plot(x_axis_values, y_axis_values, **plot_kwargs)


def plot_functions(ax, x_axis_arg_name, functions, function_kwargs=None, plot_kwargs=None,
                   results_transform=lambda y: y,
                   x_axis_values_transform=lambda x: x,
                   verbose=False):
    if plot_kwargs is None:
        plot_kwargs = {}

    for function in functions:
        if verbose:
            print(f"Plotting {function.__name__}")
        plot_function(ax, x_axis_arg_name, function, dict(function_kwargs), plot_kwargs | dict(label=function.__name__),
                      results_transform, x_axis_values_transform, verbose)

    title = ", ".join(function.__name__ for function in functions)
    if function_kwargs:
        function_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(function_kwargs)})"
    ax.set_title(title)

    ax.legend()


def plot_function_many_curves(ax, x_axis_arg_name, distinct_curves_arg_name, function,
                              function_kwargs=None, plot_kwargs=None):
    function_kwargs = function_kwargs or {}
    plot_kwargs = plot_kwargs or {}

    distinct_curves_arg_values = function_kwargs.pop(distinct_curves_arg_name)

    for distinct_curves_arg_value in distinct_curves_arg_values:
        plot_function(
            ax,
            x_axis_arg_name,
            function,
            function_kwargs | {distinct_curves_arg_name: distinct_curves_arg_value},
            plot_kwargs | dict(label=f"{distinct_curves_arg_name}={distinct_curves_arg_value}")
        )

    title = function.__name__
    if function_kwargs:
        title_kwargs = dict(function_kwargs)
        title_kwargs.pop(x_axis_arg_name)
        title += f" ({stringify_kwargs(title_kwargs)})"
    ax.set_title(title)

    ax.legend()


def create_subplot_grid(n_subplots, n_rows=None, n_columns=None):
    n_rows = n_rows or 1
    n_columns = n_columns or n_subplots // n_rows
    if n_columns * n_rows != n_subplots:
        raise ValueError("Number of subplots does not fit evenly into given number of rows and columns")

    return plt.subplots(nrows=n_rows, ncols=n_columns, tight_layout=True)
