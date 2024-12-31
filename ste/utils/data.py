import logging
import os
from typing import Iterable, Sequence

from diskcache import Cache
import openml
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo as _fetch_uci_repo

from ste.utils.caching  import memoize
from ste.utils.logging import logged
from ste.utils.misc import unzip


BENCHMARK_SUITE_IDS = [298, 300]
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../data")


dataset_cache = Cache(os.path.join(DATA_DIRECTORY, ".cache"))


@memoize(cache=dataset_cache)
@logged(message_level=logging.INFO)
def fetch_uci_repo(repo_id):
    return _fetch_uci_repo(id=repo_id)


class LazyDataset:
    def __init__(self):
        self._loaded = False
        self._features = None
        self._target = None
    
    def _load(self):
        raise NotImplementedError()
    

    def __iter__(self):
        if not self._loaded:
            self._load()
        self._loaded = True
        yield from [self._features, self._target]


class UCIDataset(LazyDataset):
    def __init__(self, id):
        super().__init__()
        self._id = id
    
    def _load(self):
        uci_dataset = fetch_uci_repo(self._id)
        self._features = uci_dataset.data.features
        self._target = uci_dataset.data.targets.iloc[:, 0]


class OpenMLDataset(LazyDataset):
    def __init__(self, id):
            super().__init__()
            self._id = id
    
    def _load(self):
        openml_dataset = openml.datasets.get_dataset(self._id)
        data = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
        self._features, self._target, _, _ = data


type Dataset = tuple[pd.DataFrame, np.ndarray] | LazyDataset


def covariates_response_split(dataframe: pd.DataFrame, response_column=-1) -> Dataset:
    if isinstance(response_column, int):
        response_column = dataframe.columns[response_column]
    
    return dataframe.drop([response_column], axis=1), dataframe[response_column]


def to_binary_classifications(classifications, seed=0):
    classes = set(classifications)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Cannot make binary classification from fewer than 2 classes")

    randomizer = np.random.RandomState(seed=seed)
    positive_classes = randomizer.choice(np.array(list(classes)), size=n_classes // 2, replace=False)

    return np.isin(classifications, positive_classes)


def coerce_nonnumeric_columns_to_numeric(df: pd.DataFrame):
    object_columns = df.select_dtypes(["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    category_columns = df.select_dtypes(["category"]).columns
    df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)
    return df


def enforce_nice_dataset(dataset: Dataset, coercion_seed=0) -> Dataset:
    X, y = dataset
    X = coerce_nonnumeric_columns_to_numeric(X)
    y = to_binary_classifications(y, coercion_seed)
    return X, y


def get_benchmark_datasets():
    suite = openml.study.get_suite(298)
    datasets_by_name = {}
    for suite_id in BENCHMARK_SUITE_IDS:
        suite = openml.study.get_suite(suite_id)
        for task_id in suite.tasks:
            task = openml.tasks.get_task(task_id)
            dataset = openml.datasets.get_dataset(task.dataset_id)
            datasets_by_name[dataset.name] = OpenMLDataset(task.dataset_id)

    return datasets_by_name


@logged(message_level=logging.DEBUG)
def load_datasets(coercion_seed=0, full_benchmark=False):
    if full_benchmark:
        named_raw_datasets = get_benchmark_datasets()
    else:
        named_raw_datasets = {
            "Ground Cover": UCIDataset(id=31),
            "Income": UCIDataset(id=117),
            "Diabetes": UCIDataset(id=891),
            "Skin": UCIDataset(id=229)
        }

    return unzip([
        (name, enforce_nice_dataset(dataset, coercion_seed))
        for name, dataset in named_raw_datasets.items()
    ])


def split_dataset(dataset: Dataset, relative_proportions: Sequence[float | int]) -> Iterable[Dataset]:
    X, y = dataset
    n_rows = len(X)

    cumulative_proportion = 0
    partition_indices = []
    for relative_proportion in relative_proportions[:-1]:
        proportion = relative_proportion / sum(relative_proportions)
        cumulative_proportion += proportion
        partition_indices.append(int(np.round(cumulative_proportion * n_rows)))

    row_indices = np.arange(n_rows)
    np.random.shuffle(row_indices)
    part_indiceses = np.split(row_indices, partition_indices)

    parts = []
    for part_indices in part_indiceses:
        parts.append((X.iloc[part_indices, :], y[part_indices]))
    
    return parts
