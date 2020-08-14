"""Bias detection in datasets"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd

import famly
import famly.bias.metrics

logger = logging.getLogger(__name__)


class FacetColumn:
    def __init__(self, name, protected_values: Optional[List[Any]] = None):
        """
        initialize facet column name and  facet_values if present
        :param name: str
        :param protected_values: list of values indicating categories or threshold
        """
        self.name = name
        self.protected_values = protected_values


class FacetContinuousColumn(FacetColumn):
    def __init__(self, name, interval_indices: pd.IntervalIndex):
        """
        :param name: Name of the column
        :param interval_indices: thresholds for binning.
        """
        super().__init__(name)
        self.interval_indices = interval_indices


class LabelColumn:
    def __init__(self, data: pd.Series, positive_label_values: Optional[Any] = None):
        """
        initialize the label column with data  and positive values
        :param data: data series for the label column
        :param positive_label_values: positive label values for target column
        """
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


def fetch_metrics_to_run(full_metrics: Callable, metric_names: List[Any]):
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


def _metric_description(metric: Callable, metric_values: dict) -> dict:
    """
    returns a dict with metric description and computed metric values
    :param metric: metric function name
     :param metric_values: dict with facet value (key) and its results
    :return: metric result dict
    {"description": "Class Imbalance (CI)",
    "value": {1: -0.9288888888888889, 0: 0.9288888888888889}
    }
    """
    if not metric.description:
        raise KeyError(f"Description is not found for the registered metric: {metric}")
        result_metrics[metric] = {"description": metric_description, "value": result_metrics[metric]}
    return {"description": metric.description, "value": metric_values}


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


def _series_datatype(data: pd.Series) -> DataType:
    """
    determine given data series is categorical or continuous using set of rules
    :return: Enum {CATEGORICAL|CONTINUOUS}
    """
    # if datatype is boolean or categorical we return data as categorical
    data_type = DataType.CATEGORICAL
    data_uniqueness_fraction = data.nunique() / data.count()
    logger.info(f"data uniqueness fraction: {data_uniqueness_fraction}")
    if data.dtype.name == "category":
        return data_type
    if data.dtype.name in ["str", "string", "object"]:
        # cast the dtype to int, if exception is raised data is categorical
        casted_data = data.astype("int64", copy=True, errors="ignore")
        if np.issubdtype(casted_data.dtype, np.integer) and data_uniqueness_fraction >= 0.05:
            data_type = DataType.CONTINOUS
    elif np.issubdtype(data.dtype, np.floating):
        data_type = DataType.CONTINUOUS
    elif np.issubdtype(data.dtype, np.integer):
        # Current rule: If data has more than 5% if unique values then it is continuous
        # Todo: Needs to be enhanced, This rule doesn't always determine the datatype correctly
        if data_uniqueness_fraction >= 0.05:
            data_type = DataType.CONTINUOUS
    return data_type


def _positive_index(data: pd.Series, positive_values: List[Any]) -> Tuple[List[pd.Series], List[str]]:
    """
    creates a list of bool series for positive label index|positive predicted label index
    based on the type of input data values,
    list of positive label values or intervals for which the positive label index is created

    :param data: input data for label or predicted label columns
    :param positive_values: list of {positive label values|predicted positive label values}
    :return: list of positive label index series, positive_label_values or intervals
    """
    data_type = _series_datatype(data)
    if data_type == DataType.CONTINUOUS:
        data_interval_indices = _interval_index(data, positive_values)
        positive_index = [_continuous_data_idx(data, data_interval_indices)]
        label_values_or_intervals = [",".join(map(str, data_interval_indices))]
    elif data_type == DataType.CATEGORICAL:
        if positive_values:
            positive_index = [_categorical_data_idx(data, positive_values)]
            label_values_or_intervals = [",".join(map(str, positive_values))]
        else:
            positive_index = [data == val for val in data.unique()]
            label_values_or_intervals = list(map(str, data.unique()))
    else:
        raise RuntimeError("Label_column data is invalid or can't be classified")
    if isinstance(positive_index, list) and not list:
        raise RuntimeError("positive label index can't be derived from the label data")
    return positive_index, label_values_or_intervals


def _categorical_data_idx(col: pd.Series, data_values: List[Any]) -> pd.Series:
    """
    :param col: input data series
    :param data_values: list of category values to generate boolean index
    :returns: a boolean series where data_values are present in col as True
   """
    # create indexing series with boolean OR of facet values
    index_key_series: pd.Series = (col == data_values[0])
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
    feature: pd.Series,
    facet_values: Optional[List[Any]],
    label: pd.Series,
    positive_label_index: pd.Series,
    predicted_label: pd.Series,
    positive_predicted_label_index: pd.Series,
    group_variable: pd.Series,
) -> Dict:
    """
    Dispatch calling of different metric functions with the correct arguments
    Calculate CI from a list of values or 1 vs all
    """
    if facet_values:
        # Build index series from facet
        facet = _categorical_data_idx(feature, facet_values)
        result = famly.bias.metrics.call_metric(
            metric,
            feature=feature,
            facet=facet,
            label=label,
            positive_label_index=positive_label_index,
            predicted_label=predicted_label,
            positive_predicted_label_index=positive_predicted_label_index,
            group_variable=group_variable,
        )
        metric_values = {",".join(map(str, facet_values)): result}
    else:
        # Do one vs all for every value
        metric_values = famly.bias.metrics.metric_one_vs_all(
            metric,
            feature,
            label=label,
            positive_label_index=positive_label_index,
            predicted_label=predicted_label,
            positive_predicted_label_index=positive_predicted_label_index,
            group_variable=group_variable,
        )
    metric_result = _metric_description(metric, metric_values)
    return metric_result


def _continuous_metric_call_wrapper(
    metric: Callable,
    feature: pd.Series,
    facet_threshold_index: pd.IntervalIndex,
    label: pd.Series,
    positive_label_index: pd.Series,
    predicted_label: pd.Series,
    positive_predicted_label_index: pd.Series,
    group_variable: pd.Series,
) -> Dict:
    """
    Dispatch calling of different metric functions with the correct arguments and bool facet data
    """

    facet = _continuous_data_idx(feature, facet_threshold_index)
    result = famly.bias.metrics.call_metric(
        metric,
        feature=feature,
        facet=facet,
        label=label,
        positive_label_index=positive_label_index,
        predicted_label=predicted_label,
        positive_predicted_label_index=positive_predicted_label_index,
        group_variable=group_variable,
    )
    metric_values = {",".join(map(str, facet_threshold_index)): result}
    metric_result = _metric_description(metric, metric_values)
    return metric_result


def bias_report(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    predicted_label_column: LabelColumn = None,
    metrics: List[Any] = ["all"],
    group_variable: Optional[pd.Series] = None,
) -> Dict:
    """
    Run Full bias report on a dataset.:
    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :param stage_type: pre_training or post_training for which bias metrics is computed
    :param predicted_label_column: description of column with predicted labels
    :param metrics: list of metrics names to provide bias metrics
    :param group_variable: data series for the group variable
    :return:
    """
    if facet_column:
        assert facet_column.name in df.columns, "Facet column {} is not present in the dataset".format(
            facet_column.name
        )
    if not predicted_label_column and stage_type == StageType.POST_TRAINING:
        raise ValueError("predicted_label_column has to be provided for Post training metrics")

    data_series: pd.Series = df[facet_column.name]
    label_series: pd.Series = label_column.data
    positive_label_index, label_values = _positive_index(
        data=label_series, positive_values=label_column.positive_label_values
    )

    metrics_to_run = []
    if predicted_label_column and stage_type == StageType.POST_TRAINING:
        post_training_metrics = (
            famly.bias.metrics.POSTTRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(famly.bias.metrics.POSTTRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(post_training_metrics)
        predicted_label_series = df[predicted_label_column.name]
        positive_predicted_label_index, predicted_label_values = _positive_index(
            data=predicted_label_series, positive_values=predicted_label_column.positive_label_values
        )
    else:
        positive_predicted_label_index = None
        predicted_label_series = None
    pre_training_metrics = (
        famly.bias.metrics.PRETRAINING_METRICS
        if metrics == ["all"]
        else fetch_metrics_to_run(famly.bias.metrics.PRETRAINING_METRICS, metrics)
    )
    metrics_to_run.extend(pre_training_metrics)

    facet_dtype = _series_datatype(data_series)
    metric_result = []
    data_series_cat: pd.Series  # Category series
    if facet_dtype == DataType.CATEGORICAL:
        data_series_cat = data_series.astype("category")
        for val, index in zip(label_values, positive_label_index):
            result = dict()
            for metric in metrics_to_run:
                result[metric.__name__] = _categorical_metric_call_wrapper(
                    metric,
                    data_series_cat,
                    facet_column.protected_values,
                    label_series,
                    index,
                    predicted_label_series,
                    positive_predicted_label_index,
                    group_variable,
                )
            metric_result.append(result)
            result["label_value"] = val
        logger.debug("metric_result:", metric_result)
        return metric_result

    elif facet_dtype == DataType.CONTINUOUS:
        facet_interval_indices = _interval_index(data_series, facet_column.protected_values)
        facet_column = FacetContinuousColumn(facet_column.name, facet_interval_indices)
        logger.info(f"Threshold Interval indices: {facet_interval_indices}")
        for val, index in zip(label_values, positive_label_index):
            result = dict()
            for metric in metrics_to_run:
                result[metric.__name__] = _continuous_metric_call_wrapper(
                    metric,
                    data_series,
                    facet_column.interval_indices,
                    label_series,
                    index,
                    predicted_label_series,
                    positive_predicted_label_index,
                    group_variable,
                )
            metric_result.append(result)
            result["label_value"] = val
        logger.debug("metric_result:", metric_result)
        return metric_result
    else:
        raise RuntimeError("facet_column data is invalid or can't be classified")
