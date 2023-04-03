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
from smclarify.bias.metrics import common, basic_stats

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
    def __init__(self, name: str, series: pd.Series, positive_label_values: Optional[Any] = None):
        """
        initialize the label column with name, data  and positive values
        :param data: data series for the label column
        :param positive_label_values: positive label values for target column
        """
        super().__init__(name)
        self.series = series
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

    def __init__(self, name: str, description: str, value: Optional[float] = None, error: Exception = None):  # type: ignore
        super().__init__(name, description, value)
        self.error = str(error)


class FacetReport:
    """Facet Report with facet value_or_threshold and list MetricResult objects"""

    def __init__(self, facet_value_or_threshold: str, metrics: List[MetricResult]):
        self.value_or_threshold = facet_value_or_threshold
        self.metrics = metrics

    def to_json(self):
        return json.loads(json.dumps(self, default=lambda o: o.__dict__), object_hook=inf_as_str)


class ModelPerformanceReport:
    """Model Performance Report with label name, list of MetricResult objects,
    (multi-class) confusion_matrix, binary_confusion_matrix"""

    def __init__(
        self,
        label_name: str,
        metrics: List[MetricResult],
        binary_confusion_matrix: List[float],
        multicategory_confusion_matrix: Optional[Dict] = None,
    ):
        self.label = label_name
        self.model_performance_metrics = metrics
        self.binary_confusion_matrix = binary_confusion_matrix
        if multicategory_confusion_matrix:
            self.confusion_matrix = multicategory_confusion_matrix

    def to_json(self):
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


def inf_as_str(obj: Dict) -> Dict:
    """Checks each dict passed to this function if it contains the key "value" with infinity float value assigned

    :param obj: The object to decode
    :return: The new dictionary with change in value from float('inf') to "Infinity"
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


def _interval_index(data: pd.Series, thresholds: Optional[List[Any]]) -> pd.IntervalIndex:
    """
    Creates a Interval Index from list of threshold values. See pd.IntervalIndex.from_breaks
    Ex. [0,1,2] -> [(0, 1], (1,2]]
    :param data: input data series
    :param thresholds: list of int or float values defining the threshold splits
    :return: pd.IntervalIndex
    """
    if not thresholds:
        raise ValueError("Threshold values must be provided for continuous features")
    max_value, min_value = data.max(), data.min()
    threshold_intervals = thresholds.copy()
    # add  max value if not exists in threshold limits
    if abs(max_value) not in thresholds:
        threshold_intervals.append(max_value)
    sorted_threshold_intervals = sorted(threshold_intervals)
    return pd.IntervalIndex.from_breaks(sorted_threshold_intervals)


def _positive_predicted_index(
    predicted_label_data: pd.Series,
    predicted_label_datatype: common.DataType,
    label_data: pd.Series,
    label_datatype: common.DataType,
    positive_label_values: List[Any],
) -> pd.Series:
    """
    creates a list of bool series for positive predicted label index based on the input data type,
    list of positive label values or intervals

    :param predicted_label_data: input data for predicted label column
    :param predicted_label_datatype: data type of the predicted label data
    :param label_data: input data for label column
    :param label_datatype:  input data for the label column
    :param positive_label_values: list of positive label values
    :return: list of positive predicted label index series
    """
    if predicted_label_datatype != label_datatype:
        raise ValueError("Predicted Label Column series datatype is not the same as Label Column series")
    if predicted_label_datatype == common.DataType.CONTINUOUS:
        predicted_label_data = predicted_label_data.astype(label_data.dtype)
        data_interval_indices = _interval_index(pd.concat([label_data, predicted_label_data]), positive_label_values)
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


def _positive_label_index(
    data: pd.Series, data_type: common.DataType, positive_values: List[Any]
) -> Tuple[pd.Series, str]:
    """
    creates a list of bool series for positive label index based on the input data type, list of positive
    label values or intervals

    :param data: input data for label column
    :param data_type: DataType of the label series data
    :param positive_values: list of positive label values
    :return: list of positive label index series, positive_label_values or intervals
    """
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


def label_value_or_threshold(label_series: pd.Series, positive_values: List[Any]) -> str:
    """
    Fetch label values or threshold intervals for the input label data and label values
    :param label_series: label column data
    :param positive_values: list of positive label values
    :return: string with category values or threshold indices seperated with ','
    """
    if not positive_values:
        raise ValueError("Positive label values or thresholds are empty for Label column")

    label_data_type, label_series = common.ensure_series_data_type(label_series, positive_values)
    _, value_or_threshold = _positive_label_index(
        data=label_series, data_type=label_data_type, positive_values=positive_values
    )
    return value_or_threshold


def _categorical_data_idx(col: pd.Series, positive_values: List[Any]) -> pd.Series:
    """
    Converts `col` series to True / False based on the `positive_values`.

    If no True values found, it tries converting elements of the `positive_value`
    to the data type of the series' elements.

    :param col: input data series
    :param positive_values: list of category values to generate boolean index
    :returns: a boolean series where data_values are present in col as True
    """

    def __categorical_data_idx(col: pd.Series, data_values: List[Any]) -> pd.Series:
        # create indexing series with boolean OR of facet values
        index_key_series: pd.Series = col == data_values[0]
        for val in data_values[1:]:
            index_key_series = index_key_series | (col == val)
        return index_key_series

    index_key_series = __categorical_data_idx(col, positive_values)
    if any(index_key_series):
        return index_key_series

    positive_values = common.convert_positive_label_values(col, positive_values)
    return __categorical_data_idx(col, positive_values)


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


def _model_performance_metric_call_wrapper(
    feature: pd.DataFrame, positive_label_index: pd.Series, positive_predicted_label_index: pd.Series
) -> List[MetricResult]:
    """
    Wrapper function to invoke model performance metric methods and collect results as MetricResult objects
    :param feature: input dataframe
    :param positive_label_index:
    :param positive_predicted_label_index:
    :return:
    """
    TP, TN, FP, FN = common.calc_confusion_matrix_quadrants(
        feature, positive_label_index, positive_predicted_label_index
    )

    metric_functions: List[Callable] = [
        basic_stats.accuracy,
        basic_stats.PPL,
        basic_stats.PNL,
        basic_stats.recall,
        basic_stats.specificity,
        basic_stats.precision,
        basic_stats.rejection_rate,
        basic_stats.conditional_acceptance,
        basic_stats.conditional_rejection,
        basic_stats.f1_score,
    ]

    metric_names: List[str] = [
        "Accuracy",
        "Proportion of Positive Predictions in Labels",
        "Proportion of Negative Predictions in Labels",
        "True Positive Rate / Recall",
        "True Negative Rate / Specificity",
        "Acceptance Rate / Precision",
        "Rejection Rate",
        "Conditional Acceptance",
        "Conditional Rejection",
        "F1 Score",
    ]

    metrics_list = []
    for (metric, name) in zip(metric_functions, metric_names):
        description = common.metric_description(metric)
        value = smclarify.bias.metrics.call_metric(metric, TP=TP, FP=FP, FN=FN, TN=TN)
        metrics_list.append(MetricResult(name, description, value))
    return metrics_list


def model_performance_report(df: pd.DataFrame, label_column: LabelColumn, predicted_label_column: LabelColumn) -> Dict:
    """
    Generate model performance report on a dataset.
    :param df: Dataset as a pandas.DataFrame
    :param label_column: description of column which has the labels.
    :param predicted_label_column: description of column with predicted labels
    :return: a dictionary with metrics for different label values
    """
    assert label_column.positive_label_values

    positive_label_values: List[Any] = label_column.positive_label_values
    label_data_type, label_data_series = common.ensure_series_data_type(label_column.series, positive_label_values)

    positive_label_index, _ = _positive_label_index(
        data=label_data_series, data_type=label_data_type, positive_values=positive_label_values
    )
    if label_column.name in df.columns:
        df = df.drop(labels=label_column.name, axis=1)

    predicted_label_data_type, predicted_label_data_series = common.ensure_series_data_type(
        predicted_label_column.series, positive_label_values
    )
    positive_predicted_label_index = _positive_predicted_index(
        predicted_label_data=predicted_label_data_series,
        predicted_label_datatype=predicted_label_data_type,
        label_data=label_data_series,
        label_datatype=label_data_type,
        positive_label_values=positive_label_values,
    )

    perf_metrics: List[MetricResult] = _model_performance_metric_call_wrapper(
        df, positive_label_index, positive_predicted_label_index
    )
    binary_confusion_matrix = common.binary_confusion_matrix(df, positive_label_index, positive_predicted_label_index)
    if label_data_type == common.DataType.CATEGORICAL:
        try:
            multicategory_confusion_matrix = basic_stats.multicategory_confusion_matrix(
                label_data_series, predicted_label_data_series
            )
        except Exception as e:
            multicategory_confusion_matrix = {"error": {str(e): 0.0}}
            logger.warning("Multicategory Confusion Matrix failed to compute due to: %s", e)

        return ModelPerformanceReport(
            label_column.name, perf_metrics, binary_confusion_matrix, multicategory_confusion_matrix
        ).to_json()

    return ModelPerformanceReport(label_column.name, perf_metrics, binary_confusion_matrix).to_json()


def _metric_name_comparator(e):
    return e.__name__


def bias_report(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    predicted_label_column: Optional[LabelColumn] = None,
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
    metrics_to_run = []
    if predicted_label_column and stage_type == StageType.POST_TRAINING:
        post_training_metrics = (
            smclarify.bias.metrics.POSTTRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(smclarify.bias.metrics.POSTTRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(post_training_metrics)
        predicted_label_series = predicted_label_column.series
        if predicted_label_column.name in df.columns:
            df = df.drop(labels=predicted_label_column.name, axis=1)
    else:
        pre_training_metrics = (
            smclarify.bias.metrics.PRETRAINING_METRICS
            if metrics == ["all"]
            else fetch_metrics_to_run(smclarify.bias.metrics.PRETRAINING_METRICS, metrics)
        )
        metrics_to_run.extend(pre_training_metrics)
    metrics_to_run.sort(key=_metric_name_comparator)
    return _report(df, facet_column, label_column, stage_type, metrics_to_run, predicted_label_column, group_variable)


def bias_basic_stats(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    predicted_label_column: Optional[LabelColumn] = None,
) -> List[Dict]:
    """Computes size and confusion matrix.

    :param df: Dataset as a pandas.DataFrame
    :param facet_column: description of column to consider for Bias analysis
    :param label_column: description of column which has the labels.
    :param stage_type: pre_training or post_training for which bias metrics is computed
    :param predicted_label_column: description of column with predicted labels
    :param metrics: list of metrics names to provide bias metrics
    :param group_variable: data series for the group variable

    :return: list of dictionaries with stats (size and confusion matrix) for each label value."""
    methods = [
        smclarify.bias.metrics.basic_stats.proportion,
        smclarify.bias.metrics.basic_stats.observed_label_distribution,
    ]
    if predicted_label_column and stage_type == StageType.POST_TRAINING:
        methods.append(smclarify.bias.metrics.basic_stats.confusion_matrix)
    return _report(df, facet_column, label_column, stage_type, methods, predicted_label_column)


def _report(
    df: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    stage_type: StageType,
    methods: List[Callable],
    predicted_label_column: Optional[LabelColumn] = None,
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
    :param methods: list of methods to provide metrics.
    :param predicted_label_column: description of column with predicted labels
    :param group_variable: data series for the group variable
    :return: list of dictionaries with metrics for different label values
    """
    if facet_column and facet_column.name not in df.columns:
        raise ValueError("Facet column {} is not present in the dataset".format(facet_column.name))
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

    sensitive_facet_values = facet_column.sensitive_values
    facet_data_type, facet_data_series = common.ensure_series_data_type(df[facet_column.name], sensitive_facet_values)
    df = df.drop(labels=facet_column.name, axis=1)

    positive_label_values = label_column.positive_label_values
    label_data_type, label_data_series = common.ensure_series_data_type(label_column.series, positive_label_values)
    positive_label_index, _ = _positive_label_index(
        data=label_data_series, data_type=label_data_type, positive_values=positive_label_values
    )
    if label_column.name in df.columns:
        df = df.drop(labels=label_column.name, axis=1)

    positive_predicted_label_index = [None]
    if predicted_label_column:
        if stage_type == StageType.POST_TRAINING:
            predicted_label_data_type, predicted_label_data_series = common.ensure_series_data_type(
                predicted_label_column.series, positive_label_values
            )
            positive_predicted_label_index = _positive_predicted_index(
                predicted_label_data=predicted_label_data_series,
                predicted_label_datatype=predicted_label_data_type,
                label_data=label_data_series,
                label_datatype=label_data_type,
                positive_label_values=positive_label_values,
            )
        if predicted_label_column.name in df.columns:
            df = df.drop(labels=predicted_label_column.name, axis=1)

    # Above are validations and preprocessing, the real reporting logic is moved to a new method for clarity and
    # to avoid using wrong data by chance (e.g., label_data_series should be used, instead of label_column.data).
    return _do_report(
        methods=methods,
        df=df,
        facet_data_type=facet_data_type,
        facet_data_series=facet_data_series,
        sensitive_facet_values=sensitive_facet_values,
        positive_label_index=positive_label_index,
        positive_predicted_label_index=positive_predicted_label_index,
        group_variable=group_variable,
    )


def _do_report(
    methods: List[Callable],
    df: pd.DataFrame,
    facet_data_type: common.DataType,
    facet_data_series: pd.Series,
    sensitive_facet_values: Optional[List[Any]],
    positive_label_index: pd.Series,
    positive_predicted_label_index: Optional[pd.Series] = None,
    group_variable: Optional[pd.Series] = None,
) -> List[Dict]:
    """
    Run full bias report on a dataset for real.

    :param methods: list of methods to provide metrics.
    :param df: Dataset of features (no facet column, label column or predicted label column).
    :param facet_data_series: facet data series
    :param facet_data_type: data type of the facet data series.
    :param sensitive_facet_values: list of values indicating categories or threshold
    :param positive_label_index: positive label index series
    :param positive_predicted_label_index: positive predicted label index series
    :param group_variable: data series for the group variable
    :return: list of dictionaries with metrics for different label values
    """
    # result values can be str for label_values or dict for metrics
    result: MetricResult
    facet_metric: FacetReport
    metrics_result = []
    if facet_data_type == common.DataType.CATEGORICAL:
        # pass the values for metric one vs all case
        facet_values_list = (
            [[val] for val in list(facet_data_series.unique())]
            if not sensitive_facet_values
            else [sensitive_facet_values]
        )
        for facet_values in facet_values_list:
            # list of metrics with values
            metrics_list = []
            for metric in methods:
                result = _categorical_metric_call_wrapper(
                    metric,
                    df,
                    facet_data_series,
                    facet_values,
                    positive_label_index,
                    positive_predicted_label_index,
                    group_variable,
                )
                metrics_list.append(result)
            facet_metric = FacetReport(facet_value_or_threshold=",".join(map(str, facet_values)), metrics=metrics_list)
            metrics_result.append(facet_metric.to_json())
        logger.debug("metric_result: %s", str(metrics_result))
        return metrics_result

    elif facet_data_type == common.DataType.CONTINUOUS:
        facet_interval_indices = _interval_index(facet_data_series, sensitive_facet_values)
        logger.info(f"Threshold Interval indices: {facet_interval_indices}")
        # list of metrics with values
        metrics_list = []
        for metric in methods:
            result = _continuous_metric_call_wrapper(
                metric,
                df,
                facet_data_series,
                facet_interval_indices,
                positive_label_index,
                positive_predicted_label_index,
                group_variable,
            )
            metrics_list.append(result)
        facet_metric = FacetReport(
            facet_value_or_threshold=",".join(map(str, facet_interval_indices)), metrics=metrics_list
        )
        metrics_result.append(facet_metric.to_json())
        logger.debug("metric_result:", metrics_result)
        return metrics_result
    else:
        raise RuntimeError("facet_column data is invalid or can't be classified")
