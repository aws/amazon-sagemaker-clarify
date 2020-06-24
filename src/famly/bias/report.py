"""Bias detection in datasets"""
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from . import metrics


class SensitiveColumn:
    def __init__(self, name):
        self.name = name


class SensitiveCategoricalColumn(SensitiveColumn):
    def __init__(self, name, protected_values: Optional[List[Any]] = None):
        """
        :param name: Name of the column
        :param protected_values: list of protected values.
        """
        super().__init__(name)
        self.protected_values = protected_values


class SensitiveContinuousColumn(SensitiveColumn):
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


def class_imbalance_values(col: pd.Series, sensitive_values: Optional[List[Any]] = None) -> Dict:
    """
    Calculate CI from a list of values or 1 vs all
    """

    def index_key(sensitive_values: List[Any]) -> pd.Series:
        index_key_series: pd.Series = (col == sensitive_values[0])
        for val in sensitive_values[1:]:
            index_key_series = index_key_series | (col == val)
        return index_key_series

    if sensitive_values:
        # A list of protected values
        # Build index series selecting protected values
        # create indexing series with boolean OR of values

        ci = metrics.class_imbalance(col, index_key(sensitive_values))
        metric_name = column_list_to_str(sensitive_values)
        ci_all = {metric_name: ci}
    else:
        # Do one vs all for every value
        ci_all = metrics.class_imbalance_one_vs_all(col)
    return ci_all


def bias_report(df: pd.DataFrame, sensitive_column: SensitiveColumn, label_column: str) -> Dict:
    """
    Run Full bias report on a dataset.

    :param df: Dataset as a pandas.DataFrame
    :param sensitive_column: marks which column to consider for Bias analysis
    :param label_column: column name which has the labels.
    :return:
    """
    if sensitive_column:
        assert sensitive_column.name in df.columns, "Sensitive column {} is not present in the dataset".format(
            sensitive_column.name
        )

    if problem_type(df[label_column]) != ProblemType.BINARY:
        raise RuntimeError("Only binary classification problems are supported")

    col: pd.Series = df[sensitive_column.name].dropna()
    col_cat: pd.Series  # Category series
    result = dict()
    if issubclass(sensitive_column.__class__, SensitiveCategoricalColumn):
        sensitive_column: SensitiveCategoricalColumn
        col_cat = col.astype("category")
        result["CI"] = class_imbalance_values(col_cat, sensitive_column.protected_values)
        return result

    elif issubclass(sensitive_column.__class__, SensitiveContinuousColumn):
        sensitive_column: SensitiveContinuousColumn
        col_cat = pd.cut(col, sensitive_column.interval_indices)
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
        raise RuntimeError(
            "sensitive_column should be an instance of SensitiveCategoricalColumn or SensitiveContinuousColumn"
        )
