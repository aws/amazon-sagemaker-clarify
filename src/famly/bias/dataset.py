"""Bias detection in datasets"""
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from . import metrics


class RestrictedColumn:
    def __init__(self, name):
        self.name = name


class RestrictedCategoricalColumn(RestrictedColumn):
    def __init__(self, name, protected_values: Optional[List[Any]] = None):
        """
        :param name: Name of the column
        :param protected_values: list of protected values. For example, for ethnicity it could be Black, Asian, Latino,
            White, Native American, etc.
        """
        super().__init__(name)
        self.protected_values = protected_values


class RestrictedContinuousColumn(RestrictedColumn):
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
    """:returns: problem type according to heuristics on the labels. So far only binary classification is supported."""
    labels = labels.dropna()
    n_rows = len(labels)
    n_unique = labels.unique()
    if n_unique == 2:
        return ProblemType.BINARY
    return ProblemType.OTHER


def _metricname_fmt(xs: List[Any]) -> str:
    """Format a metric name from multiple aggregated columns
    :returns: joint string separated by commas."""
    metricname = ", ".join([str(x) for x in xs])
    return metricname


def class_imbalance_series(col: pd.Series, protected: Optional[List[Any]] = None) -> Dict:
    if protected:
        # A list of protected values
        # Build index series selecting protected values
        # create indexing series with boolean OR of values
        key: pd.Series = (col == protected[0])
        for val in protected[1:]:
            key = key | (col == val)
        ci = metrics.class_imbalance(col, key)
        metricname = _metricname_fmt(protected)
        ci_all = {metricname: ci}
    else:
        # Do one vs all for every value
        ci_all = metrics.class_imbalance_one_vs_all(col)
    return ci_all


def bias_report(df: pd.DataFrame, restricted_column: RestrictedColumn, label_column: str) -> Dict:
    """
    :param df: Dataset as a pandas.DataFrame
    :param restricted_column: marks which column to consider for Bias analysis
    :param label_column: column name which has the labels.
    :return:
    """
    if restricted_column:
        assert restricted_column.name in df.columns, "Restricted column {} is not present in the dataset".format(
            restricted_column.name
        )

    if problem_type(df[label_column]) != ProblemType.BINARY:
        raise RuntimeError("Only binary classification problems are supported")

    col: pd.Series = df[restricted_column.name].dropna()
    col_cat: pd.Series  # Category series
    label_values = df[label_column].dropna().unique()
    result = dict()
    if issubclass(restricted_column.__class__, RestrictedCategoricalColumn):
        restricted_column: RestrictedCategoricalColumn
        col_cat = col_series.astype("category")
        result["CI"] = class_imbalance_series(col_cat, restricted_column.protected_values)
        return result

    elif issubclass(restricted_column.__class__, RestrictedContinuousColumn):
        restricted_column: RestrictedContinuousColumn
        col_cat = pd.cut(col, restricted_column.interval_indices)
        raise RuntimeError("TODO")
    else:
        raise RuntimeError(
            "restricted_column should be an instance of RestrictedCategoricalColumn or " "RestrictedContinuousColumn"
        )
