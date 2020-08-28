import logging
from enum import Enum
from typing import List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from famly.bias.metrics.constants import INFINITY

from famly.bias.metrics.constants import UNIQUENESS_THRESHOLD
from famly.util import compute_aligned_pdfs

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
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    na = len(feature[~sensitive_facet_index])
    nd = len(feature[sensitive_facet_index])
    na_pos = len(feature[~sensitive_facet_index & positive_label_index])
    nd_pos = len(feature[sensitive_facet_index & positive_label_index])
    if na == 0:
        raise ValueError("DPL: negative facet set is empty.")
    if nd == 0:
        raise ValueError("DPL: facet set is empty.")
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
    sensitive_facet_index = sensitive_facet_index.astype(bool)
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
        numA = len(label_index[label_index & sensitive_facet_index & (group_variable == subgroup_variable)])
        denomA = len(feature[label_index & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(label_index[(~label_index) & sensitive_facet_index & (group_variable == subgroup_variable)])
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
    if data.dtype.name == "category" or (isinstance(values, list) and len(values) > 1):
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
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

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

    if na0hat != 0:
        rr_a = na0 / na0hat
    else:
        rr_a = INFINITY

    if nd0hat != 0:
        rr_d = nd0 / nd0hat
    else:
        rr_d = INFINITY

    if na1hat != 0:
        ca = na1 / na1hat
    else:
        ca = INFINITY

    if nd1hat != 0:
        cd = nd1 / nd1hat
    else:
        cd = INFINITY

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

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Difference in Label Rates (aka Difference in Acceptance Rates AND Difference in Rejected Rates)
    """
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

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

    if na1hat != 0:
        ar_a = TP_a / na1hat
    else:
        ar_a = INFINITY

    if nd1hat != 0:
        ar_d = TP_d / nd1hat
    else:
        ar_d = INFINITY

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = INFINITY

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = INFINITY

    dar = ar_a - ar_d
    drr = rr_d - rr_a

    if ar_a == ar_d and ar_a == INFINITY:
        dar = 0
    if rr_a == rr_d and rr_a == INFINITY:
        drr = 0

    return dar, drr


def KL(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kullback-Liebler Divergence (KL)

    .. math::
        KL(Pa, Pd) = \sum_{x}{Pa(x) \ log2 \frac{Pa(x)}{Pd(x)}}
    Use this function for multi-category labels.

    :param label: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kullback and Leibler (KL) divergence metric
    """
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd) = compute_aligned_pdfs(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    kl = np.sum(Pa * np.log(Pa / Pd))
    return kl


def JS(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Jensen-Shannon Divergence (JS)

    .. math::
        JS(Pa, Pd, P) = 0.5 [KL(Pa,P) + KL(Pd,P)] \geq 0
    Use this function for multi-category labels.

    :param label: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Jensen-Shannon (JS) divergence metric
    """
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd, P) = compute_aligned_pdfs(xs_a, xs_d, label)
    if len(Pa) == 0 or len(Pd) == 0 or len(P) == 0:
        return np.nan
    res = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    return res


def LP(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    L-p Norm (LP)

    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{x} |Pa(x)-Pd(x)|^p]^{1/p}
    Use this function for multi-category labels.

    :param label: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Returns the LP norm of the difference between class distributions
    """
    return LP_norm(label, sensitive_facet_index, 2)


def LP_norm(label: pd.Series, sensitive_facet_index: pd.Series, norm_order) -> float:
    sensitive_facet_index = sensitive_facet_index.astype(bool)
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd) = compute_aligned_pdfs(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    res = np.linalg.norm(Pa - Pd, norm_order)
    return res


def TVD(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Total Variation Distance (TVD)

    .. math::
        TVD = 0.5 * L1(Pa, Pd) \geq 0
    Use this function for multi-category labels.

    :param label: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: total variation distance metric
    """
    Lp_res = LP_norm(label, sensitive_facet_index, 1)
    tvd = 0.5 * Lp_res
    return tvd


def KS(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kolmogorov-Smirnov Distance (KS)

    .. math::
        KS = max(\left | Pa-Pd \right |) \geq 0
    Use this function for multi-category labels.

    :param label: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kolmogorov-Smirnov metric
    """
    return LP_norm(label, sensitive_facet_index, 1)
