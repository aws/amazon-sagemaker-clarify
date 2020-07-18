"""Bias detection in datasets"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import pandas as pd


import famly


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


class LabelColumn:
    def __init__(self, name, positive_label_value: Optional[Any]):
        self.name = name
        self.positive_label_value = positive_label_value


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


def call_metric_facet_values(
    metric: Callable, col: pd.Series, facet_values: Optional[List[Any]], positive_label_index: pd.Series
) -> Dict:
    """
    Calculate CI from a list of values or 1 vs all
    """

    def index_key(col, _facet_values: List[Any]) -> pd.Series:
        """
        :returns: a boolean series where facet_values are present in col
        """
        index_key_series: pd.Series = (col == _facet_values[0])
        for val in _facet_values[1:]:
            index_key_series = index_key_series | (col == val)
        return index_key_series

    if facet_values:
        # A list of protected values
        # Build index series selecting protected values
        # create indexing series with boolean OR of values
        metric_result = metric(col, index_key(col, facet_values))
        metric_values = {metric.__name__: metric_result}
    else:
        # Do one vs all for every value
        metric_values = famly.bias.metrics.metric_one_vs_all(metric, col, positive_label_index)
    return metric_values


def bias_report(df: pd.DataFrame, facet_column: FacetColumn, label_column: LabelColumn) -> Dict:
    """
    Run Full bias report on a dataset.

    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :return:
    """
    if facet_column:
        assert facet_column.name in df.columns, "Facet column {} is not present in the dataset".format(
            facet_column.name
        )

    if problem_type(df[label_column.name]) != ProblemType.BINARY:
        raise RuntimeError("Only binary classification problems are supported")

    data_series: pd.Series = df[facet_column.name]
    label_series: pd.Series = df[label_column.name]
    positive_label_index: pd.Series = df[label_column.name] == label_column.positive_label_value
    data_series_cat: pd.Series  # Category series
    result = dict()
    if issubclass(facet_column.__class__, FacetCategoricalColumn):
        facet_column: FacetCategoricalColumn
        data_series_cat = data_series.astype("category")
        for metric in famly.bias.metrics.PRETRAINING_METRICS:
            if (
                metric == famly.bias.CDD
                or metric == famly.bias.JS
                or metric == famly.bias.KL
                or metric == famly.bias.KS
            ):
                continue
            result[metric.__name__] = call_metric_facet_values(
                metric, data_series_cat, facet_column.protected_values, positive_label_index
            )
        return result

    elif issubclass(facet_column.__class__, FacetContinuousColumn):
        facet_column: FacetContinuousColumn
        data_series_cat = pd.cut(data_series, facet_column.interval_indices)
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
