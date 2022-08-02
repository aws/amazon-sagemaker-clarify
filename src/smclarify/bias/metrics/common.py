# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import logging
from enum import Enum
from typing import List, Optional, Tuple, Callable, Any, Union
import pandas as pd
import numpy as np
from smclarify.bias.metrics.constants import INFINITY

from smclarify.bias.metrics.constants import UNIQUENESS_THRESHOLD

logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Type of facet data series distribution
    """

    CATEGORICAL = 0
    CONTINUOUS = 1


def divide(a, b):
    if b == 0 and a == 0:
        return 0.0
    if b == 0:
        if a < 0:
            return -INFINITY
        return INFINITY
    return a / b


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metric_description(metric: Callable[..., float]) -> str:
    """
    fetch metric description from doc strings
    :param metric: metric callable function
    :return: short description of metric
    """
    if not metric.__doc__:
        logger.exception(f"Description is not found for the registered metric: {metric}")
    return metric.__doc__.lstrip().split("\n")[0]  # type: ignore


def DPL(feature: pd.Series, sensitive_facet_index: pd.Series, positive_label_index: pd.Series) -> float:
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "label_index must be of type bool")
    na = len(feature[~sensitive_facet_index])
    nd = len(feature[sensitive_facet_index])
    na_pos = len(feature[~sensitive_facet_index & positive_label_index])
    nd_pos = len(feature[sensitive_facet_index & positive_label_index])
    if na == 0:
        raise ValueError("Negative facet set is empty.")
    if nd == 0:
        raise ValueError("Facet set is empty.")
    qa = na_pos / na
    qd = nd_pos / nd
    dpl = qa - qd
    return dpl


def CDD(
    feature: pd.Series, sensitive_facet_index: pd.Series, label_index: pd.Series, group_variable: pd.Series
) -> float:
    """
    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label_index: boolean column indicating positive labels or predicted labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    if group_variable is None or group_variable.empty:
        raise ValueError("Group variable is empty or not provided")
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(label_index.dtype == bool, "label_index must be of type bool")
    unique_groups = np.unique(group_variable)

    # Global demographic disparity (DD)]
    denomA = len(feature[label_index])

    if denomA == 0:
        raise ValueError("No positive labels in set")
    denomD = len(feature[~label_index])

    if denomD == 0:
        raise ValueError("No negative labels in set")

    # Conditional demographic disparity (CDD)
    # FIXME: appending to numpy arrays is inefficient
    CDD: np.typing.NDArray = np.array([])
    counts: np.typing.NDArray = np.array([])
    for subgroup_variable in unique_groups:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(feature[label_index & sensitive_facet_index & (group_variable == subgroup_variable)])
        denomA = len(feature[label_index & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(feature[(~label_index) & sensitive_facet_index & (group_variable == subgroup_variable)])
        denomD = len(feature[(~label_index) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)

    wtd_mean_CDD = divide(np.sum(counts * CDD), np.sum(counts))

    return wtd_mean_CDD


def series_datatype(series: pd.Series, values: Optional[List[Any]] = None) -> DataType:
    """
    Determine given data series is categorical or continuous using set of rules.
    WARNING: The deduced data type can be different from real data type of the data series. Please
    use the function `ensure_series_data_type` instead if you'd like ensure the series data type.

    :param series: data for facet/label/predicted_label columns
    :param values: list of facet or label values provided by user
    :return: Enum {CATEGORICAL|CONTINUOUS}
    """
    # if datatype is boolean or categorical we return data as categorical
    data_type = DataType.CATEGORICAL
    data_uniqueness_fraction = divide(series.nunique(), series.count())
    # Assumption: user will give single value for threshold currently
    # Todo: fix me if multiple thresholds for facet or label are supported
    if series.dtype.name == "category" or (isinstance(values, list) and len(values) > 1):
        logger.info(
            f"Column {series.name} with data uniqueness fraction {data_uniqueness_fraction} is classifed as a "
            f"{data_type.name} column"
        )
        return data_type
    if series.dtype.name in ["str", "string", "object"]:
        # cast the dtype to int, if exception is raised data is categorical
        casted_data = series.astype("int64", copy=True, errors="ignore")
        if np.issubdtype(casted_data.dtype, np.integer) and data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINUOUS  # type: ignore
    elif np.issubdtype(series.dtype, np.floating):
        data_type = DataType.CONTINUOUS
    elif np.issubdtype(series.dtype, np.integer):
        # Current rule: If data has more than 5% if unique values then it is continuous
        # Todo: Needs to be enhanced, This rule doesn't always determine the datatype correctly
        if data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINUOUS
    logger.info(
        f"Column {series.name} with data uniqueness fraction {data_uniqueness_fraction} is classifed as a "
        f"{data_type.name} column"
    )
    return data_type


def ensure_series_data_type(series: pd.Series, values: Optional[List[Any]] = None) -> Tuple[DataType, pd.Series]:
    """
    Determine the type of the given data series using set of rules, and then do necessary type conversion
    to ensure the series data type.
    :param series: data for facet/label/predicted_label columns
    :param values: list of facet or label values provided by user
    :return: A tuple of DataType and the converted data series
    """
    data_type = series_datatype(series, values)
    if data_type == DataType.CATEGORICAL:
        return data_type, series.astype("category")
    elif data_type == DataType.CONTINUOUS:
        if values:
            if not (isinstance(values[0], int) or isinstance(values[0], float)):
                try:
                    values[0] = float(values[0])
                except ValueError:
                    raise ValueError(
                        "Facet/label value provided must be a single numeric threshold for continuous data"
                    )
        return data_type, pd.to_numeric(series)
    raise ValueError("Data series is invalid or can't be classified as neither categorical nor continous.")


def convert_positive_label_values(series: pd.Series, positive_label_values: List[Union[str, int, float]]) -> List:
    """
    Determines the type of the given data series and then do necessary type conversion to ensure the positive_lable_values
    are of the same type as those in series.


    Example problem when it helps:
    The problem is that the `label_values_or_threshold` and the actual `label` values are not the same -
    i.e. do not have the same type. This leads to customer facing errors when they pass numerical values
    to `label_values_or_threshold` (for instance `[1, 2, 3]`) but having string values in the label column
    of the dataset (for instance, `['1', '2', '3', '4', '5']`).

    :param series: data for facet/label/predicted_label columns
    :param positive_label_values: list of label values provided by user
    :return: list of label values provided after the conversion (if any)
    """
    def _convert(items: List, _type: Callable) -> List:
        try:
            return [_type(item) for item in items]
        except ValueError as e:
            # int('1.0') raises a ValueError
            if "invalid literal for int() with base 10" in str(e):
                return [float(item) for item in items]
            raise Exception(f"'label' has not positive elements. Double-check if 'label' and 'positive_label_values'"
                            f"have correct data-types or values.")

    if isinstance(positive_label_values[0], type(series[0])):
        return positive_label_values

    # if the types are different, convert positive_label_values
    converted_values: List[Any]
    if isinstance(series[0], bool) and isinstance(positive_label_values, str) and positive_label_values[0].isalpha():
        # when values = ['True', 'False'] and series = [False, True, ...]
        converted_values = [True if label.lower() == 'true' else False for label in positive_label_values]
        # else when values = [1, 1.0, 0, 0.0] and series = [False, True, ...], _convert(positive_label_values, bool)
        # see else below
    else:
        converted_values = _convert(positive_label_values, type(series[0]))
    logger.warning(f"Data type of the elements in `positive_label_values` and in `label` must match. "
                   f"Converted positive_label_values from {positive_label_values} to {converted_values}")
    return converted_values


# Todo: Fix the function to avoid redundant calls for DCA and DCR
def DCO(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> Tuple[float, float]:
    """
    Difference in Conditional Outcomes (DCO)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Difference in Conditional Outcomes (Acceptance and Rejection) between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("DCO: Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("DCO: Negated Facet set is empty")

    na0 = len(feature[~positive_label_index & ~sensitive_facet_index])
    na0hat = len(feature[~positive_predicted_label_index & ~sensitive_facet_index])
    nd0 = len(feature[~positive_label_index & sensitive_facet_index])
    nd0hat = len(feature[~positive_predicted_label_index & sensitive_facet_index])

    na1 = len(feature[positive_label_index & ~sensitive_facet_index])
    na1hat = len(feature[positive_predicted_label_index & ~sensitive_facet_index])
    nd1 = len(feature[positive_label_index & sensitive_facet_index])
    nd1hat = len(feature[positive_predicted_label_index & sensitive_facet_index])

    rr_a = divide(na0, na0hat)
    rr_d = divide(nd0, nd0hat)

    ca = divide(na1, na1hat)
    cd = divide(nd1, nd1hat)

    dca = ca - cd
    dcr = rr_d - rr_a

    if ca == cd and ca == INFINITY:
        dca = 0
    if rr_a == rr_d and rr_a == INFINITY:
        dcr = 0

    return dca, dcr


# Todo: Fix the function to avoid redundant calls for DAR and DRR
def DLR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> Tuple[float, float]:
    """
    Difference in Label Rates (DLR)

    For cases where both the nominator and the denominator are 0 we use 0 as result.

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Difference in Label Rates (aka Difference in Acceptance Rates AND Difference in Rejected Rates)
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("DLR: Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("DLR: Negated Facet set is empty")

    TP_a = len(feature[positive_label_index & positive_predicted_label_index & (~sensitive_facet_index)])
    na1hat = len(feature[positive_predicted_label_index & (~sensitive_facet_index)])
    TP_d = len(feature[positive_label_index & positive_predicted_label_index & sensitive_facet_index])
    nd1hat = len(feature[positive_predicted_label_index & sensitive_facet_index])

    TN_a = len(feature[(~positive_label_index) & (~positive_predicted_label_index) & (~sensitive_facet_index)])
    na0hat = len(feature[(~positive_predicted_label_index) & (~sensitive_facet_index)])
    TN_d = len(feature[(~positive_label_index) & (~positive_predicted_label_index) & sensitive_facet_index])
    nd0hat = len(feature[(~positive_predicted_label_index) & sensitive_facet_index])

    ar_a = divide(TP_a, na1hat)
    ar_d = divide(TP_d, nd1hat)

    rr_a = divide(TN_a, na0hat)
    rr_d = divide(TN_d, nd0hat)

    dar = ar_a - ar_d
    drr = rr_d - rr_a

    if ar_a == ar_d and ar_a == INFINITY:
        dar = 0
    if rr_a == rr_d and rr_a == INFINITY:
        drr = 0

    return dar, drr
