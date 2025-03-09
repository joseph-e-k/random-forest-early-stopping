from dataclasses import dataclass
import logging
import os
from typing import Iterable, Sequence
from weakref import WeakKeyDictionary

from diskcache import Cache
import openml
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo as _fetch_uci_repo

from ste.utils.caching  import memoize
from ste.utils.logging import logged
from ste.utils.misc import unzip


BENCHMARK_DATASET_IDS = [44089, 44090, 44091, 44120, 44121, 44122, 44123, 44124, 44125, 44126, 44127, 44128, 44129, 44130, 44131, 44156, 44157, 44158, 44159, 44160, 44161, 44162]
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../data")


dataset_cache = Cache(os.path.join(DATA_DIRECTORY, ".cache"))


@memoize(cache=dataset_cache)
@logged(message_level=logging.INFO)
def fetch_uci_repo(repo_id):
    return _fetch_uci_repo(id=repo_id)


class LazyDataset:
    _wkd = WeakKeyDictionary()
    
    def _load(self):
        raise NotImplementedError()

    def __iter__(self):
        try:
            data = self._wkd[self]
        except KeyError:
            data = self._clean_data(*self._load())
            self._wkd[self] = data
        yield from data

    @staticmethod
    def _clean_data(features, target):
        features = coerce_nonnumeric_columns_to_numeric(features)
        target = to_binary_classifications(target)
        return features, target


@dataclass(frozen=True)
class UCIDataset(LazyDataset):
    id: int
    
    def _load(self):
        uci_dataset = fetch_uci_repo(self.id)
        return uci_dataset.data.features, uci_dataset.data.targets.iloc[:, 0]


@dataclass(frozen=True)
class OpenMLDataset(LazyDataset):
    id: int
    
    def _load(self):
        openml_dataset = openml.datasets.get_dataset(self.id)
        data = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
        features, target, _, _ = data
        return features, target


type Dataset = tuple[pd.DataFrame, np.ndarray] | LazyDataset


def covariates_response_split(dataframe: pd.DataFrame, response_column=-1) -> Dataset:
    if isinstance(response_column, int):
        response_column = dataframe.columns[response_column]
    
    return dataframe.drop([response_column], axis=1), dataframe[response_column]


def to_binary_classifications(classifications):
    classes = classifications.unique()
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Cannot make binary classification from fewer than 2 classes")
    
    classes, class_counts = np.unique(classifications, return_counts=True)
    most_common_class = classes[np.argmax(class_counts)]
    return np.array(classifications == most_common_class, dtype=int)


def coerce_nonnumeric_columns_to_numeric(df: pd.DataFrame):
    object_columns = df.select_dtypes(["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    category_columns = df.select_dtypes(["category"]).columns
    df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)
    return df


def get_benchmark_datasets():
    datasets_by_name = {}
    for dataset_id in BENCHMARK_DATASET_IDS:
        dataset = openml.datasets.get_dataset(dataset_id)
        datasets_by_name[dataset.name] = OpenMLDataset(dataset_id)

    return datasets_by_name


@logged(message_level=logging.DEBUG)
def get_names_and_datasets(full_benchmark=False):
    if full_benchmark:
        named_datasets = get_benchmark_datasets()
    else:
        named_datasets = {
            "Ground Cover": UCIDataset(id=31),
            "Income": UCIDataset(id=117),
            "Diabetes": UCIDataset(id=891),
            "Skin": UCIDataset(id=229),
            "Sepsis": UCIDataset(id=827),
            "Dota2": UCIDataset(id=367),
            "Hospitalization": UCIDataset(id=296),
            "Shuttle": UCIDataset(id=148)
        }

    return unzip(named_datasets.items())


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
