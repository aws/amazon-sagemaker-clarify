"""Bias detection in datasets"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

import famly
import famly.bias.metrics

logger = logging.getLogger(__name__)


class FacetColumn:
    def __init__(self, name, protected_values: Optional[List[Any]] = None):
        self.name = name
        self.protected_values = protected_values


class FacetContinuousColumn(FacetColumn):
    def __init__(self, name, interval_indices: pd.IntervalIndex):
        """
        :param name: Name of the column
            thresholds for binning.
        """
        super().__init__(name)
        self.interval_indices = interval_indices


class LabelColumn:
    def __init__(self, name, positive_label_value: Optional[Any] = 1):
        self.name = name
        self.positive_label_value = positive_label_value


class ProblemType(Enum):
    """Type of problem deduced from the label values"""

    BINARY = 0
    REGRESSION = 1
    MULTICLASS = 2
    OTHER = 3


class StageType(Enum):
    """Stage Types in which bias metrics is calculated"""

    PRE_TRAINING = "pre_training"
    POST_TRAINING = "post_training"


class DataType(Enum):
    """
    Type of facet data series distribution
    """

    CATEGORICAL = 0
    CONTINUOUS = 1


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


def _interval_index(facet: pd.Series, thresholds: List[Any]) -> pd.IntervalIndex:
    """
    Creates a Interval Index from list of threshold values
    :param facet: input data series
    :param thresholds: list of int or float values
    :return: pd.IntervalIndex
    """
    # Fix: use mean as threshold when no input is provided.
    thresholds = [int(facet.mean())] if not thresholds else thresholds
    facet_max, facet_min = facet.max(), facet.min()
    # add  max value if not exists in threshold limits
    thresholds.append(facet_max) if facet_max not in thresholds else thresholds
    return pd.IntervalIndex.from_breaks(thresholds)


def _facet_datatype(facet: pd.Series) -> DataType:
    """
    deterimine given facet data is categorical or continous using set of rules
    :return: Enum {CATEGORICAL|CONTINUOUS}
    """
    # if datatype is boolean or categorical we return data as categorical
    data_type = DataType.CATEGORICAL
    data_uniqueness_fraction = facet.nunique() / facet.count()
    logger.info(f"facet uniqueness fraction: {data_uniqueness_fraction}")
    if facet.dtype.name == "category":
        return data_type
    if facet.dtype.name in ["str", "string", "object"]:
        # cast the dtype to int, if exception is raised data is categorical
        casted_facet = facet.astype("int64", copy=True, errors="ignore")
        if np.issubdtype(casted_facet.dtype, np.integer) and data_uniqueness_fraction >= 0.05:
            data_type = DataType.CONTINOUS
    elif np.issubdtype(facet.dtype, np.floating):
        data_type = DataType.CONTINUOUS
    elif np.issubdtype(facet.dtype, np.integer):
        # If data is more than 10% if unique values then it is continuous
        # Todo: Needs to be enhanced, This rule doesn't always determine the datatype correctly
        if data_uniqueness_fraction >= 0.05:
            data_type = DataType.CONTINUOUS
    return data_type


def _categorical_metric_call_wrapper(
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
        metric_values = {",".join(map(str, facet_values)): f()}
    else:
        # Do one vs all for every value
        metric_values = famly.bias.metrics.metric_one_vs_all(
            metric, x, label, positive_label_index, predicted_label, positive_predicted_label_index
        )
    return metric_values


def _continous_metric_call_wrapper(
    metric: Callable,
    x: pd.Series,
    facet_threshold_index: pd.IntervalIndex,
    label: pd.Series,
    positive_label_index: pd.Series,
    predicted_label: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> Dict:
    """
    Dispatch calling of different metric functions with the correct arguments and bool facet data
    """

    def facet_from_thresholds(x: pd.Series, _facet_threshold_index: pd.IntervalIndex) -> pd.Series:
        """
        returns bool Series after checking threshold index for each value from input
        :param x:
        :param _facet_threshold_index:
        :return: boolean Series for facet
        """
        return x.map(lambda y: any(facet_threshold_index.contains(y)))

    facet = facet_from_thresholds(x, facet_threshold_index)
    f = famly.bias.metrics.metric_partial_nullary(
        metric, x, facet, label, positive_label_index, predicted_label, positive_predicted_label_index
    )
    metric_values = {",".join(map(str, facet_threshold_index)): f()}
    return metric_values


def bias_report(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    predicted_label_column: LabelColumn = None,
) -> Dict:
    """
    Run Full bias report on a dataset.:
    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :param stage_type: pre_training or post_training for which bias metrics is computed
    :param predicted_label_column: description of column with predicted labels
    :return:
    """
    if facet_column:
        assert facet_column.name in df.columns, "Facet column {} is not present in the dataset".format(
            facet_column.name
        )
    if not predicted_label_column and stage_type == StageType.POST_TRAINING:
        raise ValueError("predicted_label_column has to be provided for Post training metrics")

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

    facet_dtype = _facet_datatype(data_series)
    result = dict()
    data_series_cat: pd.Series  # Category series
    if facet_dtype == DataType.CATEGORICAL:
        data_series_cat = data_series.astype("category")
        for metric in metrics_to_run:
            result[metric.__name__] = _categorical_metric_call_wrapper(
                metric,
                data_series_cat,
                facet_column.protected_values,
                label_series,
                positive_label_index,
                predicted_label_series,
                positive_predicted_label_index,
            )
        return result

    elif facet_dtype == DataType.CONTINUOUS:
        facet_interval_indices = _interval_index(data_series, facet_column.protected_values)
        facet_column = FacetContinuousColumn(facet_column.name, facet_interval_indices)
        logger.info(f"Threshold Interval indices: {facet_interval_indices}")
        for metric in metrics_to_run:
            result[metric.__name__] = _continous_metric_call_wrapper(
                metric,
                data_series,
                facet_column.interval_indices,
                label_series,
                positive_label_index,
                predicted_label_series,
                positive_predicted_label_index,
            )
        return result
    else:
        raise RuntimeError("facet_column data is invalid or can't be classified")
