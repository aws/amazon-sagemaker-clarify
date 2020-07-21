"""
Pre training metrics
"""
import logging
from famly.util import PDF
import pandas as pd
import numpy as np
from typing import Any

log = logging.getLogger(__name__)


def CI(x: pd.Series, facet: pd.Series) -> float:
    """
    Class imbalance (CI)
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the protected class.

    Bias is often generated from an under-representation of
    the protected class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (np − p)/(np + p). Where np is the number of instances in the not protected group
    and p is number of instances in the sensitive group.
    """
    facet = facet.astype(bool)
    pos = len(x[facet])
    neg = len(x[~facet])
    q = pos + neg
    if neg == 0:
        raise ValueError("CI: negated facet set is empty. Check that x[~facet] has non-zero length.")
    if pos == 0:
        raise ValueError("CI: facet set is empty. Check that x[facet] has non-zero length.")
    assert q != 0
    ci = float(neg - pos) / q
    return ci


def DPL(x: pd.Series, facet: pd.Series, label: pd.Series, positive_label: Any) -> float:
    """
    Difference in positive proportions in labels
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param label: pandas series of labels (binary, multicategory, or continuous)
    :param positive_label_index: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    positive_label_index = label == positive_label
    facet = facet.astype(bool)
    positive_label_index_neg_facet = positive_label_index & ~facet
    positive_label_index_facet = positive_label_index & facet
    na = len(x[~facet])
    nd = len(x[facet])
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


def KL(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Kullback and Leibler (KL) divergence metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    facet = np.array(facet)
    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]
    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        kl = np.sum(Pa * np.log(Pa / Pd))  # note log is base e, measured in nats
    else:
        raise ValueError("KL: Either facet set or negated facet set is empty")
    return kl


def JS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Jenson-Shannon (JS) divergence metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        P = PDF(positive_label_index)
        js_divergence = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    else:
        raise ValueError("JS: Either facet set or negated facet set is empty")

    return js_divergence


def LP(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, norm_order: int = 2) -> float:
    r"""
    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{y} |Pa(y)-Pd(y)|^p]^{1/p}

    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param norm_order: the order of norm desired (2 by default).
    :return: Returns the LP norm of the difference between class distributions
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        lp_norm = np.linalg.norm(Pa - Pd, norm_order)
    else:
        raise ValueError("LP: Either facet set or negated facet set is empty")

    return lp_norm


def TVD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
   :param x: input feature
   :param facet: boolean column indicating sensitive group
   :param positive_label_index: boolean column indicating positive labels
   :return: 1/2 * L-1 norm
   """

    Lp_res = LP(x, facet, positive_label_index, p=1)

    tvd = 0.5 * Lp_res

    return tvd


def KS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Kolmogorov-Smirnov metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        max_distance = np.max(np.abs(Pa - Pd))
    else:
        raise ValueError("KS: Either facet set or negated facet set is empty")

    return max_distance


def CDD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, group_variable: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    unique_groups = np.unique(group_variable)
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    # Global demographic disparity (DD)
    numA = len(positive_label_index[(positive_label_index) & (facet)])
    denomA = len(facet[positive_label_index])

    if denomA == 0:
        raise ValueError("CDD: No positive labels in set")

    A = numA / denomA
    numD = len(positive_label_index[(~positive_label_index) & (facet)])
    denomD = len(facet[~positive_label_index])

    if denomD == 0:
        raise ValueError("CDD: No negative labels in set")

    D = numD / denomD
    DD = D - A

    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup_variable in unique_groups:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(positive_label_index[(positive_label_index) & (facet) & (group_variable == subgroup_variable)])
        denomA = len(facet[(positive_label_index) & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(positive_label_index[(~positive_label_index) & (facet) & (group_variable == subgroup_variable)])
        denomD = len(facet[(~positive_label_index) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)

    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)

    return wtd_mean_CDD
