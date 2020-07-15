"""Bias detection in datasets"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from ....src.famly.bias import *
from ....src.famly.bias.metrics import *


class FacetColumn:
    def __init__(self, name):
        self.name = name


class FacetCategoricalColumn(FacetColumn):
    def __init__(self, name, protected_values: Optional[List[Any]] = None):
        """
        :param name: Name of the column
        :param protected_values: list of protected values.
        """
        super().__init__(name)
        self.protected_values = protected_values


class FacetContinuousColumn(FacetColumn):
    def __init__(self, name, interval_indices: pd.IntervalIndex, intervals: List[pd.Interval]):
        """
        :param name: Name of the column
            thresholds for binning.
        FIXME
        """
        super().__init__(name)
        self.interval_indices = interval_indices
        self.intervals = intervals


class ProblemType(Enum):
    """Type of problem deduced from the label values"""

    BINARY = 0
    REGRESSION = 1
    MULTICLASS = 2
    OTHER = 3


def problem_type(labels: pd.Series) -> ProblemType:
    """
    :returns: problem type according to heuristics on the labels. So far only binary classification is supported.
    """
    # TODO: add other problem types
    labels = labels.dropna()
    n_rows = len(labels)
    n_unique = labels.unique().size
    if n_unique == 2:
        return ProblemType.BINARY
    return ProblemType.OTHER


def column_list_to_str(xs: List[Any]) -> str:
    """
    Format a metric name from multiple aggregated columns
    :returns: joint string separated by commas.
    """
    metricname = ", ".join([str(x) for x in xs])
    return metricname


def call_metric_facet_values(metric: Callable, col: pd.Series, facet_values: Optional[List[Any]] = None) -> Dict:
    """
    Calculate CI from a list of values or 1 vs all
    """

    def index_key(col, facet_values: List[Any]) -> pd.Series:
        """
        :returns: a boolean series where facet_values are present in col
        """
        index_key_series: pd.Series = (col == facet_values[0])
        for val in facet_values[1:]:
            index_key_series = index_key_series | (col == val)
        return index_key_series

    if facet_values:
        # A list of protected values
        # Build index series selecting protected values
        # create indexing series with boolean OR of values

        metric_result = metric(col, index_key(col, facet_values))
        ci_all = {metric.__name__: metric_result}
    else:
        # FIXME
        assert False
        # Do one vs all for every value
        # ci_all = famly.bias.metrics.metric_one_vs_all(col)
    return ci_all


def bias_report(df: pd.DataFrame, facet_column: FacetColumn, label_column: str) -> Dict:
    """
    Run Full bias report on a dataset.

    :param df: Dataset as a pandas.DataFrame
    :param facet_column: marks which column to consider for Bias analysis
    :param label_column: column name which has the labels.
    :return:
    """
    if facet_column:
        assert facet_column.name in df.columns, "Facet column {} is not present in the dataset".format(
            facet_column.name
        )

    if problem_type(df[label_column]) != ProblemType.BINARY:
        raise RuntimeError("Only binary classification problems are supported")

    col: pd.Series = df[facet_column.name].dropna()
    col_cat: pd.Series  # Category series
    result = dict()
    if issubclass(facet_column.__class__, FacetCategoricalColumn):
        facet_column: FacetCategoricalColumn
        col_cat = col.astype("category")
        for metric in famly.bias.metrics.PRETRAINING_METRICS:
            result[metric.__name__] = call_metric_facet_values(metric, col_cat, facet_column.protected_values)
        return result

    elif issubclass(facet_column.__class__, FacetContinuousColumn):
        facet_column: FacetContinuousColumn
        col_cat = pd.cut(col, facet_column.interval_indices)
        # TODO: finish impl
        # In [44]: df=pd.DataFrame({'age': [5,25,10,80]})
        # In [50]: df
        # Out[50]:
        #   age
        # 0    5
        # 1   25
        # 2   10
        # 3   80

        # In [51]: pd.cut(df['age'], pd.IntervalIndex.from_tuples([(0,21),(22,100)]))
        # Out[51]:
        # 0      (0, 21]
        # 1    (22, 100]
        # 2      (0, 21]
        # 3    (22, 100]
        # Name: age, dtype: category
        # Categories (2, interval[int64]): [(0, 21] < (22, 100]]

        # In [52]: pd.cut(df['age'], pd.IntervalIndex.from_tuples([(0,21)]))
        # Out[52]:
        # 0    (0.0, 21.0]
        # 1            NaN
        # 2    (0.0, 21.0]
        # 3            NaN
        # Name: age, dtype: category
        raise RuntimeError("Continous case to be finished")
    else:
        raise RuntimeError("facet_column should be an instance of FacetCategoricalColumn or FacetContinuousColumn")
