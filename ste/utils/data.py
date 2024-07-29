import logging
import os
from typing import Callable, Iterable, Sequence

from diskcache import Cache
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from ste.utils.caching  import memoize
from ste.utils.logging import logged
from ste.utils.misc import unzip


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../data")

dataset_cache = Cache(os.path.join(DATA_DIRECTORY, ".cache"))

type Dataset = tuple[pd.DataFrame, np.ndarray]


def covariates_response_split(dataframe: pd.DataFrame, response_column=-1) -> Dataset:
    if isinstance(response_column, int):
        response_column = dataframe.columns[response_column]
    
    return dataframe.drop([response_column], axis=1), dataframe[response_column]


def load_local_dataset(file_name: str, reader: Callable[[str], pd.DataFrame] = pd.read_csv, response_column: str | int = -1) -> Dataset:
    return covariates_response_split(reader(os.path.join(DATA_DIRECTORY, file_name)), response_column)


def load_uci_dataset(id: int) -> Dataset:
    uci_dataset = fetch_ucirepo(id=id)
    return uci_dataset.data.features, uci_dataset.data.targets.iloc[:, 0]


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

@logged(message_level=logging.DEBUG)
@memoize(cache=dataset_cache)
def load_datasets(coercion_seed=0):
    named_raw_datasets = {
        "Salaries": load_local_dataset("adult.data"),
        "Dry Beans": load_local_dataset("dry_beans.xlsx", reader=pd.read_excel),
        "Phishing": load_uci_dataset(id=327),
        "Diabetes": load_uci_dataset(id=891),
        "IoT": load_uci_dataset(id=942),
        "Android Permissions": load_uci_dataset(id=722)
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
