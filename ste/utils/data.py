from dataclasses import dataclass
import logging
import os
from typing import Iterable, Iterator, Sequence
from weakref import WeakKeyDictionary

from diskcache import Cache
import openml
import pandas as pd
import numpy as np

import ucimlrepo

from .caching  import memoize
from .logging import logged


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../../data")

# Since we will often want to delete the cache of computations without deleting the cache of datasets, we store them separately
dataset_cache = Cache(os.path.join(DATA_DIRECTORY, ".cache"))


@memoize(cache=dataset_cache)
@logged(message_level=logging.INFO)
def fetch_uci_repo(repo_id):
    """Logged and cached wrapper to ucimlrepo.fetch_ucirepo"""
    return ucimlrepo.fetch_ucirepo(id=repo_id)


# For our purposes, a "concrete dataset" is a DataFrame of features and a 1-D array of target categories.
type ConcreteDataset = tuple[pd.DataFrame, np.ndarray]

type Dataset = ConcreteDataset | LazyDataset

class LazyDataset:
    """Dataset that actually loads its data only when unpacked to features and target. Useful for multiprocessing."""
    _wkd = WeakKeyDictionary()
    
    def load_raw(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[ConcreteDataset]:
        try:
            data = self._wkd[self]
        except KeyError:
            data = self._clean_data(*self.load_raw())
            self._wkd[self] = data
        yield from data

    @staticmethod
    def _clean_data(features, target):
        features = coerce_nonnumeric_columns_to_numeric(features)
        target = dichotomize_classifications(target)
        return features, target


@dataclass(frozen=True)
class UCIDataset(LazyDataset):
    """A dataset stored in the UCI Machine Learning Repository"""
    id: int
    
    def load_raw(self):
        uci_dataset = fetch_uci_repo(self.id)
        return uci_dataset.data.features, uci_dataset.data.targets.iloc[:, 0]


@dataclass(frozen=True)
class OpenMLDataset(LazyDataset):
    """A dataset stored in the OpenML repository"""
    id: int
    
    def load_raw(self):
        openml_dataset = openml.datasets.get_dataset(self.id)
        data = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
        features, target, _, _ = data
        return features, target


SHORT_BENCHMARK_DATASETS = {
    "Ground Cover": UCIDataset(id=31),
    "Income": UCIDataset(id=117),
    "Diabetes": UCIDataset(id=891),
    "Skin": UCIDataset(id=229),
    "Sepsis": UCIDataset(id=827),
    "Dota2": UCIDataset(id=367),
    "Hospitalization": UCIDataset(id=296),
    "Shuttle": UCIDataset(id=148)
}


# This benchmark is that of Grinsztajn, Grinjsztajn, and Varoquax (2022). The exact list can be found in Appendix A.1 of
# https://arxiv.org/pdf/2207.08815. Since we are only interested in classification, only tables A.1.1 and A.1.3 were used.
GRINSZTAJN_DATASETS = {
    "credit": OpenMLDataset(id=44089),
    "california": OpenMLDataset(id=44090),
    "wine": OpenMLDataset(id=44091),
    "electricity": OpenMLDataset(id=44156),
    "covertype": OpenMLDataset(id=44159),
    "pol": OpenMLDataset(id=44122),
    "house_16H": OpenMLDataset(id=44123),
    "kdd_ipums_la_97-small": OpenMLDataset(id=44124),
    "MagicTelescope": OpenMLDataset(id=44125),
    "bank-marketing": OpenMLDataset(id=44126),
    "phoneme": OpenMLDataset(id=44127),
    "MiniBooNE": OpenMLDataset(id=44128),
    "Higgs": OpenMLDataset(id=44129),
    "eye_movements": OpenMLDataset(id=44157),
    "jannis": OpenMLDataset(id=44131),
    "KDDCup09_upselling": OpenMLDataset(id=44158),
    "rl": OpenMLDataset(id=44160),
    "road-safety": OpenMLDataset(id=44161),
    "compass": OpenMLDataset(id=44162)
}


ALL_BENCHMARK_DATASETS = {**SHORT_BENCHMARK_DATASETS, **GRINSZTAJN_DATASETS}


def dichotomize_classifications(classifications):
    """Dichotomize an array of classes.

    Args:
        classifications (np.ndarray): original classifications

    Raises:
        ValueError: if there are fewer than 2 classes to begin with

    Returns:
        np.ndarray: new array of the same length, with only 2 classes: the original most common class, and an "everything else" class
    """
    classes = classifications.unique()
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Cannot make binary classification from fewer than 2 classes")
    
    classes, class_member_counts = np.unique(classifications, return_counts=True)
    most_common_class = classes[np.argmax(class_member_counts)]
    return np.array(classifications == most_common_class, dtype=int)


def coerce_nonnumeric_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Transform all 'object' and 'category' columns in a dataframe to numeric codes using Series.cat.codes.
    Note that this mutates the original dataframe.

    Args:
        df (pd.DataFrame): dataframe to be adjusted

    Returns:
        pd.DataFrame: the original dataframe, after coercion
    """
    object_columns = df.select_dtypes(["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    category_columns = df.select_dtypes(["category"]).columns
    df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)
    return df


def split_dataset(dataset: Dataset, relative_proportions: Sequence[float | int]) -> Iterable[Dataset]:
    """Randomly split a given dataset into chunks with specified relative sizes."""
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
