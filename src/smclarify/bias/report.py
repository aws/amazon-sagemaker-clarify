# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

"""Bias detection in datasets"""
import logging
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple

import pandas as pd

import smclarify
import smclarify.bias.metrics
from smclarify.bias.metrics import common

logger = logging.getLogger(__name__)


class Column:
    def __init__(self, name: str):
        self.name = name


class FacetColumn(Column):
    def __init__(self, name: str, sensitive_values: Optional[List[Any]] = None):
        """
        initialize facet column name and  facet_values if present
        :param name: str
        :param sensitive_values: list of values indicating categories or threshold
        """
        super().__init__(name)
        self.sensitive_values = sensitive_values


class FacetContinuousColumn(Column):
    def __init__(self, name, interval_indices: pd.IntervalIndex):
        """
        :param name: Name of the column
        :param interval_indices: thresholds for binning.
        """
        super().__init__(name)
        self.interval_indices = interval_indices


class LabelColumn(Column):
    def __init__(self, name: str, data: pd.Series, positive_label_values: Optional[Any] = None):
        """
        initialize the label column with name, data  and positive values
        :param data: data series for the label column
        :param positive_label_values: positive label values for target column
        """
        super().__init__(name)
        self.data = data
        self.positive_label_values = positive_label_values


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


class MetricResult:
    """Metric Result with name, description and computed metric values"""

    def __init__(self, name: str, description: str, value: Optional[float]):
        self.name = name
        self.description = description
        self.value = value


class MetricError(MetricResult):
    """Metric Result with name, description and computed metric value and error"""

    def __init__(self, name: str, description: str, value: Optional[float] = None, error: Exception = None):
        super().__init__(name, description, value)
        self.error = str(error)


class FacetReport:
    """Facet Report with facet value_or_threshold and list MetricResult objects"""

    def __init__(self, facet_value_or_threshold: str, metrics: List[MetricResult]):
        self.value_or_threshold = facet_value_or_threshold
        self.metrics = metrics

    def toJson(self):
        return json.loads(json.dumps(self, default=lambda o: o.__dict__), object_hook=inf_as_str)


def problem_type(labels: pd.Series) -> ProblemType:
    """
    :returns: problem type according to heuristics on the labels. So far only binary classification is supported.
    """
    # TODO: add other problem types
    labels = labels.dropna()
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


def inf_as_str(obj):
    """Checks each dict passed to this function if it contains the key "value" with infinity float value assigned
    Args:
        obj (dict): The object to decode

    Returns:
        dict: The new dictionary with change in value from float('inf') to "Infinity"
    """
    if "value" in obj and obj["value"] in [float("inf"), float("-inf")]:
        obj["value"] = str(obj["value"]).replace("inf", "Infinity")
    return obj


def fetch_metrics_to_run(full_metrics: List[Callable[..., Any]], metric_names: List[str]):
    """
    Validates the list of metric names passed and returns the callable methods for them
    :param full_metrics:
    :param metric_names:
    :return: List[Callable..] methods
    """
    full_metrics_names = [f.__name__ for f in full_metrics]
    if not (set(metric_names).issubset(set(full_metrics_names))):
        raise ValueError("Invalid metric_name: metrics should be one of the registered metrics" f"{full_metrics_names}")
    metrics_to_run = [metric for metric in full_metrics if metric.__name__ in metric_names]
    return metrics_to_run


def _interval_index(facet: pd.Series, thresholds: Optional[List[Any]]) -> pd.IntervalIndex:
    """
    Creates a Interval Index from list of threshold values. See pd.IntervalIndex.from_breaks
    Ex. [0,1,2] -> [(0, 1], (1,2]]
    :param facet: input data series
    :param thresholds: list of int or float values defining the threshold splits
    :return: pd.IntervalIndex
    """
    if not thresholds:
        raise ValueError("Threshold values must be provided for continuous features")
    facet_max, facet_min = facet.max(), facet.min()
    threshold_intervals = thresholds.copy()
    # add  max value if not exists in threshold limits
    if abs(facet_max) not in thresholds:
        threshold_intervals.append(facet_max)
    return pd.IntervalIndex.from_breaks(threshold_intervals)


def _positive_predicted_index(
    predicted_label_data: pd.Series, label_data: pd.Series, positive_label_values: List[Any]
) -> pd.Series:
    """
    creates a list of bool series for positive predicted label index based on the input data type,
    list of positive label values or intervals

    :param predicted_label_data: input data for predicted label column
    :param label_datatype:  input data for the label column
    :param positive_label_values: list of positive label values
    :return: list of positive predicted label index series
    """
    predicted_label_datatype = common.series_datatype(predicted_label_data, positive_label_values)
    label_datatype = common.series_datatype(label_data, positive_label_values)
    if predicted_label_datatype != label_datatype:
        raise AssertionError("Predicted Label Column series datatype is not the same as Label Column series")
    try:
        predicted_label_data = predicted_label_data.astype(label_data.dtype)
    except ValueError as e:
        raise ValueError(
            "Labels and predicted labels cannot have different types (%s, %s)."
            % (label_data.dtype, predicted_label_data.dtype)
        )
    if predicted_label_datatype == common.DataType.CONTINUOUS:
        data_interval_indices = _interval_index(label_data.append(predicted_label_data), positive_label_values)
        positive_predicted_index = _continuous_data_idx(predicted_label_data, data_interval_indices)
    elif predicted_label_datatype == common.DataType.CATEGORICAL and positive_label_values:
        positive_predicted_index = _categorical_data_idx(predicted_label_data, positive_label_values)
    else:
        raise RuntimeError("Predicted Label_column data is invalid or can't be classified")
    # check if positive index boolean series has all False values
    if (~positive_predicted_index).all():
        raise ValueError(
            "No Label values are present in the predicted Label Column,"
            "Positive Predicted Index Series contains all False values"
        )
    return positive_predicted_index


def _positive_label_index(data: pd.Series, positive_values: List[Any]) -> Tuple[pd.Series, str]:
    """
    creates a list of bool series for positive label index based on the input data type, list of positive
    label values or intervals

    :param data: input data for label column
    :param positive_values: list of positive label values
    :return: list of positive label index series, positive_label_values or intervals
    """
    data_type = common.series_datatype(data, positive_values)
    if data_type == common.DataType.CONTINUOUS:
        data_interval_indices = _interval_index(data, positive_values)
        positive_index = _continuous_data_idx(data, data_interval_indices)
        label_values_or_intervals = ",".join(map(str, data_interval_indices))
    elif data_type == common.DataType.CATEGORICAL and positive_values:
        positive_index = _categorical_data_idx(data, positive_values)
        label_values_or_intervals = ",".join(map(str, positive_values))
    else:
        raise RuntimeError("Label_column data is invalid or can't be classified")
    logger.debug(f"positive index: {positive_index}")
    logger.debug(f"label values or intervals: {label_values_or_intervals}")
    return positive_index, label_values_or_intervals


def label_value_or_threshold(label_series: pd.Series, positive_values: List[str]) -> str:
    """
    Fetch label values or threshold intervals for the input label data and label values
    :param label_series: label column data
    :param positive_values: list of positive label values
    :return: string with category values or threshold indices seperated with ','
    """
    if not positive_values:
        raise ValueError("Positive label values or thresholds are empty for Label column")
    _, value_or_threshold = _positive_label_index(data=label_series, positive_values=positive_values)
    return value_or_threshold


def _categorical_data_idx(col: pd.Series, data_values: List[Any]) -> pd.Series:
    """
    :param col: input data series
    :param data_values: list of category values to generate boolean index
    :returns: a boolean series where data_values are present in col as True
    """
    # create indexing series with boolean OR of facet values
    index_key_series: pd.Series = col == data_values[0]
    for val in data_values[1:]:
        index_key_series = index_key_series | (col == val)
    return index_key_series


def _continuous_data_idx(x: pd.Series, data_threshold_index: pd.IntervalIndex) -> pd.Series:
    """
    returns bool Series after checking threshold index for each value from input
    :param x:
    :param data_threshold_index: group of threshold intervals
    :return: boolean Series of data against threshold interval index
    """
    return x.map(lambda y: any(data_threshold_index.contains(y)))


def _categorical_metric_call_wrapper(
    metric: Callable,
    df: pd.DataFrame,
    feature: pd.Series,
    facet_values: Optional[List[Any]],
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
    group_variable: pd.Series,
) -> MetricResult:
    """
    Dispatch calling of different metric functions with the correct arguments
    Calculate CI from a list of values or 1 vs all
    """
    if facet_values:
        try:
            # Build index series from facet
            sensitive_facet_index = _categorical_data_idx(feature, facet_values)
            metric_description = common.metric_description(metric)
            metric_value = smclarify.bias.metrics.call_metric(
                metric,
                df=df,
                feature=feature,
                sensitive_facet_index=sensitive_facet_index,
                label=positive_label_index,
                positive_label_index=positive_label_index,
                predicted_label=positive_predicted_label_index,
                positive_predicted_label_index=positive_predicted_label_index,
                group_variable=group_variable,
            )
        except Exception as exc:
            logger.exception(f"{metric.__name__} metrics failed")
            return MetricError(metric.__name__, metric_description, error=exc)
    else:
        raise ValueError("Facet values must be provided to compute the bias metrics")
    return MetricResult(metric.__name__, metric_description, metric_value)


def _continuous_metric_call_wrapper(
    metric: Callable,
    df: pd.DataFrame,
    feature: pd.Series,
    facet_threshold_index: pd.IntervalIndex,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
    group_variable: pd.Series,
) -> MetricResult:
    """
    Dispatch calling of different metric functions with the correct arguments and bool facet data
    """
    try:
        sensitive_facet_index = _continuous_data_idx(feature, facet_threshold_index)
        metric_description = common.metric_description(metric)
        metric_value = smclarify.bias.metrics.call_metric(
            metric,
            df=df,
            feature=feature,
            sensitive_facet_index=sensitive_facet_index,
            label=positive_label_index,
            positive_label_index=positive_label_index,
            predicted_label=positive_predicted_label_index,
            positive_predicted_label_index=positive_predicted_label_index,
            group_variable=group_variable,
        )
    except Exception as exc:
        logger.exception(f"{metric.__name__} metrics failed")
        return MetricError(metric.__name__, metric_description, error=exc)
    return MetricResult(metric.__name__, metric_description, metric_value)


def _metric_name_comparator(e):
    return e.__name__


def bias_report(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    predicted_label_column: LabelColumn = None,
    metrics: List[Any] = ["all"],
    group_variable: Optional[pd.Series] = None,
) -> List[Dict]:
    """
    Run full bias report on a dataset.

    The report computes the bias metric for multi-facet, and multi-class inputs by
    computing the sensitive_facet_index, positive_label_index, and positive_predicted_label_index by collapsing the
    multiple categories into two, as indicated by the facet_column, label_column, and predicted_label_column respectively.

    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :param stage_type: pre_training or post_training for which bias metrics is computed
    :param predicted_label_column: description of column with predicted labels
    :param metrics: list of metrics names to provide bias metrics
    :param group_variable: data series for the group variable
    :return: list of dictionaries with metrics for different label values
    """
    if facet_column:
        assert facet_column.name in df.columns, "Facet column {} is not present in the dataset".format(
            facet_column.name
        )
    if not label_column.positive_label_values:
        raise ValueError("Positive label values or thresholds are empty for Label column")
    if isinstance(predicted_label_column, LabelColumn) and predicted_label_column.positive_label_values:
        if predicted_label_column.positive_label_values != label_column.positive_label_values:
            raise ValueError(
                "Positive predicted label values or threshold should be empty or same as label values or thresholds"
            )
    if not isinstance(stage_type, StageType):
        raise ValueError("stage_type should be a Enum value of StageType")
    if not predicted_label_column and stage_type == StageType.POST_TRAINING:
        raise ValueError("predicted_label_column has to be provided for Post training metrics")
    data_series: pd.Series = df[facet_column.name]
    df = df.drop(facet_column.name, 1)
    label_series: pd.Series = label_column.data
    positive_label_index, label_values = _positive_label_index(
        data=label_series, positive_values=label_column.positive_label_values
    )
    if label_column.name in df.columns:
        df = df.drop(label_column.name, 1)

    metrics_to_run = []
    if predicted_label_column and stage_type == StageType.POST_TRAINING:
        post_training_metrics = (
            smclarify.bias.metrics.POSTTRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(smclarify.bias.metrics.POSTTRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(post_training_metrics)
        predicted_label_series = predicted_label_column.data
        positive_predicted_label_index = _positive_predicted_index(
            predicted_label_data=predicted_label_series,
            label_data=label_series,
            positive_label_values=label_column.positive_label_values,
        )
        if predicted_label_column.name in df.columns:
            df = df.drop(predicted_label_column.name, 1)
    else:
        positive_predicted_label_index = [None]
        pre_training_metrics = (
            smclarify.bias.metrics.PRETRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(smclarify.bias.metrics.PRETRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(pre_training_metrics)
    metrics_to_run.sort(key=_metric_name_comparator)

    facet_dtype = common.series_datatype(data_series, facet_column.sensitive_values)
    data_series_cat: pd.Series  # Category series
    # result values can be str for label_values or dict for metrics
    result: MetricResult
    facet_metric: FacetReport
    metrics_result = []
    if facet_dtype == common.DataType.CATEGORICAL:
        data_series_cat = data_series.astype("category")
        # pass the values for metric one vs all case
        facet_values_list = (
            [[val] for val in list(data_series.unique())]
            if not facet_column.sensitive_values
            else [facet_column.sensitive_values]
        )
        for facet_values in facet_values_list:
            # list of metrics with values
            metrics_list = []
            for metric in metrics_to_run:
                result = _categorical_metric_call_wrapper(
                    metric,
                    df,
                    data_series_cat,
                    facet_values,
                    positive_label_index,
                    positive_predicted_label_index,
                    group_variable,
                )
                metrics_list.append(result)
            facet_metric = FacetReport(facet_value_or_threshold=",".join(map(str, facet_values)), metrics=metrics_list)
            metrics_result.append(facet_metric.toJson())
        logger.debug("metric_result: %s", str(metrics_result))
        return metrics_result

    elif facet_dtype == common.DataType.CONTINUOUS:
        facet_interval_indices = _interval_index(data_series, facet_column.sensitive_values)
        facet_continuous_column = FacetContinuousColumn(facet_column.name, facet_interval_indices)
        logger.info(f"Threshold Interval indices: {facet_interval_indices}")
        # list of metrics with values
        metrics_list = []
        for metric in metrics_to_run:
            result = _continuous_metric_call_wrapper(
                metric,
                df,
                data_series,
                facet_continuous_column.interval_indices,
                positive_label_index,
                positive_predicted_label_index,
                group_variable,
            )
            metrics_list.append(result)
        facet_metric = FacetReport(
            facet_value_or_threshold=",".join(map(str, facet_interval_indices)), metrics=metrics_list
        )
        metrics_result.append(facet_metric.toJson())
        logger.debug("metric_result:", metrics_result)
        return metrics_result
    else:
        raise RuntimeError("facet_column data is invalid or can't be classified")
