# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import logging
from enum import Enum
from typing import List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from smclarify.bias.metrics.constants import INFINITY

from smclarify.bias.metrics.constants import UNIQUENESS_THRESHOLD

logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    test commit
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
    CDD = np.array([])
    counts = np.array([])
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


def series_datatype(data: pd.Series, values: Optional[List[str]] = None) -> DataType:
    """
    determine given data series is categorical or continuous using set of rules

    :param data: data for facet/label/predicted_label columns
    :param values: list of facet or label values provided by user
    :return: Enum {CATEGORICAL|CONTINUOUS}
    """
    # if datatype is boolean or categorical we return data as categorical
    data_type = DataType.CATEGORICAL
    data_uniqueness_fraction = divide(data.nunique(), data.count())
    logger.info(f"data uniqueness fraction: {data_uniqueness_fraction}")
    # Assumption: user will give single value for threshold currently
    # Todo: fix me if multiple thresholds for facet or label are supported
    if data.dtype.name == "category" or (isinstance(values, list) and len(values) > 1):
        return data_type
    if data.dtype.name in ["str", "string", "object"]:
        # cast the dtype to int, if exception is raised data is categorical
        casted_data = data.astype("int64", copy=True, errors="ignore")
        if np.issubdtype(casted_data.dtype, np.integer) and data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINUOUS  # type: ignore
    elif np.issubdtype(data.dtype, np.floating):
        data_type = DataType.CONTINUOUS
    elif np.issubdtype(data.dtype, np.integer):
        # Current rule: If data has more than 5% if unique values then it is continuous
        # Todo: Needs to be enhanced, This rule doesn't always determine the datatype correctly
        if data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINUOUS
    return data_type


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
