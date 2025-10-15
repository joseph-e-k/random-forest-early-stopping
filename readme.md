To install all packages necessary to run this code, we recommend using an Anaconda environment. You can create one with the command:

```conda create --name ste python=3.13 numpy matplotlib scikit-learn diskcache networkx tblib pandas openml ucimlrepo mpir conda-forge::soplex=8.0.0 gurobi::gurobi```

After the environment is created, activate it with the command:

```conda activate ste```
 
Define the following environment variables:

* `SOPLEX_PATH`: The path to the SoPlex executable, which can now be found inside the binaries directory of the new Conda environment.

* `PYTHONHASHSEED`: An arbitrary number to make your runs reproducible. We used 1234 in the paper.

Finally, with your working directory set to the root of this project, run any of our scripts, e.g.,

```python -m scripts.generate_figures_and_tables```

Much of the code in this repository is heavily multiprocessed. If this causes problems, consider tweaking the following environment variables:

* `STE_N_WORKER_PROCESSES`: Number of worker processes to use. Defaults to 32.

* `STE_DUMMY_MULTIPROCESS`: Set to anything nonempty to turn off multiprocessing entirely.