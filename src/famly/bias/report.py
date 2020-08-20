"""Bias detection in datasets"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple

import pandas as pd

import famly
import famly.bias.metrics
from famly.bias.metrics import common

logger = logging.getLogger(__name__)


class Column:
    def __init__(self, name: str):
        self.name = name


class FacetColumn(Column):
    def __init__(self, name: str, protected_values: Optional[List[Any]] = None):
        """
        initialize facet column name and  facet_values if present
        :param name: str
        :param protected_values: list of values indicating categories or threshold
        """
        super().__init__(name)
        self.protected_values = protected_values


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
    if not metric.description:  # type: ignore
        raise KeyError(f"Description is not found for the registered metric: {metric}")
    return {"description": metric.description, "value": metric_values}  # type: ignore


def _interval_index(facet: pd.Series, thresholds: Optional[List[Any]]) -> pd.IntervalIndex:
    """
    Creates a Interval Index from list of threshold values. See pd.IntervalIndex.from_breaks
    Ex. [0,1,2] -> [(0, 1], (1,2]]
    :param facet: input data series
    :param thresholds: list of int or float values defining the threshold splits
    :return: pd.IntervalIndex
    """
    # Fix: use mean as threshold when no input is provided.
    thresholds = [int(facet.mean())] if not thresholds else thresholds
    facet_max, facet_min = facet.max(), facet.min()
    # add  max value if not exists in threshold limits
    thresholds.append(facet_max) if facet_max not in thresholds else thresholds
    return pd.IntervalIndex.from_breaks(thresholds)


def _positive_predicted_index(
    predicted_label_data: pd.Series, label_data: pd.Series, positive_label_values: List[Any]
) -> List[pd.Series]:
    """
    creates a list of bool series for positive predicted label index based on the input data type,
    list of positive label values or intervals

    :param predicted_label_data: input data for predicted label column
    :param label_datatype:  input data for the label column
    :param positive_label_values: list of positive label values
    :return: list of positive predicted label index series
    """
    predicted_label_datatype = common.series_datatype(predicted_label_data)
    label_datatype = common.series_datatype(label_data)
    if predicted_label_datatype != label_datatype:
        raise AssertionError("Predicted Label Column series datatype is not the same as Label Column series")
    if predicted_label_datatype == common.DataType.CONTINUOUS:
        data_interval_indices = _interval_index(label_data, positive_label_values)
        positive_predicted_index = [_continuous_data_idx(predicted_label_data, data_interval_indices)]
    elif predicted_label_datatype == common.DataType.CATEGORICAL and positive_label_values:
        positive_predicted_index = [_categorical_data_idx(predicted_label_data, positive_label_values)]
    else:
        raise RuntimeError("Predicted Label_column data is invalid or can't be classified")
    # check if positive index boolean series has all False values
    for index in positive_predicted_index:
        if (~index).all():
            raise ValueError(
                "No Label values are present in the predicted Label Column,"
                "Positive Predicted Index Series contains all False values"
            )
    return positive_predicted_index


def _positive_label_index(data: pd.Series, positive_values: List[Any]) -> Tuple[List[pd.Series], List[str]]:
    """
    creates a list of bool series for positive label index based on the input data type, list of positive
    label values or intervals

    :param data: input data for label column
    :param positive_values: list of positive label values
    :return: list of positive label index series, positive_label_values or intervals
    """
    data_type = common.series_datatype(data)
    if data_type == common.DataType.CONTINUOUS:
        data_interval_indices = _interval_index(data, positive_values)
        positive_index = [_continuous_data_idx(data, data_interval_indices)]
        label_values_or_intervals = [",".join(map(str, data_interval_indices))]
    elif data_type == common.DataType.CATEGORICAL and positive_values:
        positive_index = [_categorical_data_idx(data, positive_values)]
        label_values_or_intervals = [",".join(map(str, positive_values))]
    else:
        raise RuntimeError("Label_column data is invalid or can't be classified")
    if isinstance(positive_index, list) and not list:
        raise RuntimeError("positive label index can't be derived from the label data")
    logger.debug(f"positive index: {positive_index}")
    logger.debug(f"label values or Intervals: {label_values_or_intervals}",)
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
) -> List[Dict]:
    """
    Run Full bias report on a dataset.:
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
    if not predicted_label_column and stage_type == StageType.POST_TRAINING:
        raise ValueError("predicted_label_column has to be provided for Post training metrics")

    data_series: pd.Series = df[facet_column.name]
    label_series: pd.Series = label_column.data
    positive_label_index, label_values = _positive_label_index(
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
        predicted_label_series = predicted_label_column.data
        positive_predicted_label_index = _positive_predicted_index(
            predicted_label_data=predicted_label_series,
            label_data=label_series,
            positive_label_values=label_column.positive_label_values,
        )
    else:
        positive_predicted_label_index = [None]
        predicted_label_values = [None]
        predicted_label_series = None
        pre_training_metrics = (
            famly.bias.metrics.PRETRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(famly.bias.metrics.PRETRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(pre_training_metrics)

    facet_dtype = common.series_datatype(data_series)
    metric_result = []
    data_series_cat: pd.Series  # Category series
    # result values can be str for label_values or dict for metrics
    result: Dict[str, Any]
    if facet_dtype == common.DataType.CATEGORICAL:
        data_series_cat = data_series.astype("category")
        for val, label_index in zip(label_values, positive_label_index):
            for predicted_label_index in positive_predicted_label_index:
                result = dict()
                for metric in metrics_to_run:
                    result[metric.__name__] = _categorical_metric_call_wrapper(
                        metric,
                        data_series_cat,
                        facet_column.protected_values,
                        label_series,
                        label_index,
                        predicted_label_series,
                        predicted_label_index,
                        group_variable,
                    )
                result["label_value_or_threshold"] = val
                metric_result.append(result)
        logger.debug("metric_result:", metric_result)
        return metric_result

    elif facet_dtype == common.DataType.CONTINUOUS:
        facet_interval_indices = _interval_index(data_series, facet_column.protected_values)
        facet_continuous_column = FacetContinuousColumn(facet_column.name, facet_interval_indices)
        logger.info(f"Threshold Interval indices: {facet_interval_indices}")
        for val, label_index in zip(label_values, positive_label_index):
            for predicted_label_index in positive_predicted_label_index:
                result = dict()
                for metric in metrics_to_run:
                    result[metric.__name__] = _continuous_metric_call_wrapper(
                        metric,
                        data_series,
                        facet_continuous_column.interval_indices,
                        label_series,
                        label_index,
                        predicted_label_series,
                        predicted_label_index,
                        group_variable,
                    )
                result["label_value_or_threshold"] = val
                metric_result.append(result)
        logger.debug("metric_result:", metric_result)
        return metric_result
    else:
        raise RuntimeError("facet_column data is invalid or can't be classified")
