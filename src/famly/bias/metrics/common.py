import logging
from enum import Enum
from typing import List, Optional
import pandas as pd
import numpy as np

from famly.bias.metrics.constants import UNIQUENESS_THRESHOLD

logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Type of facet data series distribution
    """

    CATEGORICAL = 0
    CONTINUOUS = 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def DPL(feature: pd.Series, facet: pd.Series, label: pd.Series, positive_label_index: pd.Series) -> float:
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    na = len(feature[~facet])
    nd = len(feature[facet])
    na_pos = len(label[~facet & positive_label_index])
    nd_pos = len(label[facet & positive_label_index])
    if na == 0:
        raise ValueError("DPL: negative facet set is empty.")
    if nd == 0:
        raise ValueError("DPL: facet set is empty.")
    qa = na_pos / na
    qd = nd_pos / nd
    dpl = qa - qd
    return dpl


def CDD(feature: pd.Series, facet: pd.Series, label_index: pd.Series, group_variable: pd.Series) -> float:
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label_index: boolean column indicating positive labels or predicted labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    if group_variable is None or group_variable.empty:
        return float("NAN")
    facet = facet.astype(bool)
    label_index = label_index.astype(bool)
    unique_groups = np.unique(group_variable)

    # Global demographic disparity (DD)]
    denomA = len(feature[label_index])

    if denomA == 0:
        raise ValueError("CDD: No positive labels in set")
    denomD = len(feature[~label_index])

    if denomD == 0:
        raise ValueError("CDD: No negative labels in set")

    # Conditional demographic disparity (CDD)
    # FIXME: appending to numpy arrays is inefficient
    CDD = np.array([])
    counts = np.array([])
    for subgroup_variable in unique_groups:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(label_index[label_index & facet & (group_variable == subgroup_variable)])
        denomA = len(feature[label_index & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(label_index[(~label_index) & facet & (group_variable == subgroup_variable)])
        denomD = len(feature[(~label_index) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)

    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)

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
    data_uniqueness_fraction = data.nunique() / data.count()
    logger.info(f"data uniqueness fraction: {data_uniqueness_fraction}")
    # Assumption: user will give single value for threshold currently
    # Todo: fix me if multiple thresholds for facet or label are supported
    if data.dtype.name == "category" or (isinstance(values, list) and len(values) >= 2):
        return data_type
    if data.dtype.name in ["str", "string", "object"]:
        # cast the dtype to int, if exception is raised data is categorical
        casted_data = data.astype("int64", copy=True, errors="ignore")
        if np.issubdtype(casted_data.dtype, np.integer) and data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINOUS  # type: ignore
    elif np.issubdtype(data.dtype, np.floating):
        data_type = DataType.CONTINUOUS
    elif np.issubdtype(data.dtype, np.integer):
        # Current rule: If data has more than 5% if unique values then it is continuous
        # Todo: Needs to be enhanced, This rule doesn't always determine the datatype correctly
        if data_uniqueness_fraction >= UNIQUENESS_THRESHOLD:
            data_type = DataType.CONTINUOUS
    return data_type
