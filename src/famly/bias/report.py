"""Bias detection in datasets"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import pandas as pd


import famly
import famly.bias.metrics


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


def _column_list_to_str(xs: List[Any]) -> str:
    """
    Format a metric name from multiple aggregated columns
    :returns: joint string separated by commas.
    """
    metricname = ", ".join([str(x) for x in xs])
    return metricname


def _metric_call_wrapper(
    metric: Callable,
    x: pd.Series,
    facet_values: Optional[List[Any]],
    label: pd.Series,
    positive_label_index: pd.Series,
    predicted_label: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> Dict:
    """
    Dispatch calling of different metric functions with the correct arguments

    Calculate CI from a list of values or 1 vs all
    """

    def facet_idx(col: pd.Series, _facet_values: List[Any]) -> pd.Series:
        """
        :returns: a boolean series where facet_values are present in col
        """
        # create indexing series with boolean OR of facet values
        index_key_series: pd.Series = (col == _facet_values[0])
        for val in _facet_values[1:]:
            index_key_series = index_key_series | (col == val)
        return index_key_series

    if facet_values:
        # Build index series from facet
        facet = facet_idx(x, facet_values)
        f = famly.bias.metrics.metric_partial_nullary(
            metric, x, facet, label, positive_label_index, predicted_label, positive_predicted_label_index
        )
        metric_values = {metric.__name__: f()}
    else:
        # Do one vs all for every value
        metric_values = famly.bias.metrics.metric_one_vs_all(
            metric, x, label, positive_label_index, predicted_label, positive_predicted_label_index
        )
    return metric_values


def bias_report(
    df: pd.DataFrame, facet_column: FacetColumn, label_column: LabelColumn, predicted_label_column: LabelColumn = None
) -> Dict:
    """
    Run Full bias report on a dataset.

    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :param predicted_label_column: description of column with predicted labels
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

    metrics_to_run = []
    if predicted_label_column:
        metrics_to_run.extend(famly.bias.metrics.POSTTRAINING_METRICS)
        positive_predicted_label_index = df[predicted_label_column.name] == predicted_label_column.positive_label_value
        predicted_label_series = df[predicted_label_column.name]
    else:
        positive_predicted_label_index = None
        predicted_label_series = None
    metrics_to_run.extend(famly.bias.metrics.PRETRAINING_METRICS)

    result = dict()
    data_series_cat: pd.Series  # Category series
    if issubclass(facet_column.__class__, FacetCategoricalColumn):
        facet_column: FacetCategoricalColumn
        data_series_cat = data_series.astype("category")
        for metric in metrics_to_run:
            result[metric.__name__] = _metric_call_wrapper(
                metric,
                data_series_cat,
                facet_column.protected_values,
                label_series,
                positive_label_index,
                predicted_label_series,
                positive_predicted_label_index,
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
