"""
Pre training metrics
"""
import logging
from famly.util import pdfs_aligned_nonzero
from . import registry
import pandas as pd
import numpy as np
from typing import Any

log = logging.getLogger(__name__)


@registry.pretraining
def CI(x: pd.Series, facet: pd.Series) -> float:
    r"""
    Class imbalance (CI)
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the protected class.

    .. math::
        CI = \frac{na-nd}{na+nd}

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


@registry.pretraining
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


@registry.pretraining
def KL(x: pd.Series, facet: pd.Series) -> float:
    r"""
    Kullback and Leibler divergence or relative entropy in bits.

    .. math::
        KL(Pa, Pd) = \sum_{x}{Pa(x) \ log2 \frac{Pa(x)}{Pd(x)}}

    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :return: Kullback and Leibler (KL) divergence metric
    """
    facet = facet.astype(bool)
    xs_a = x[facet]
    xs_d = x[~facet]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    kl = np.sum(Pa * np.log2(Pa / Pd))
    return kl


def JS(x: pd.Series, facet: pd.Series) -> float:
    r"""
    Jensen-Shannon divergence

    .. math::
        JS(Pa, Pd, P) = 0.5 [KL(Pa,P) + KL(Pd,P)] \geq 0

@registry.pretraining
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Jensen-Shannon (JS) divergence metric
    """
    facet = facet.astype(bool)
    xs_a = x[facet]
    xs_d = x[~facet]
    (Pa, Pd, P) = pdfs_aligned_nonzero(xs_a, xs_d, x)
    if len(Pa) == 0 or len(Pd) == 0 or len(P) == 0:
        return np.nan
    res = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    return res


@registry.pretraining
def LP(x: pd.Series, facet: pd.Series, norm_order: int = 2) -> float:
    r"""
    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{x} |Pa(x)-Pd(x)|^p]^{1/p}

    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param norm_order: the order of norm desired (2 by default).
    :return: Returns the LP norm of the difference between class distributions
    """
    facet = facet.astype(bool)
    xs_a = x[facet]
    xs_d = x[~facet]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    res = np.linalg.norm(Pa - Pd, norm_order)
    return res


@registry.pretraining
def TVD(x: pd.Series, facet: pd.Series) -> float:
    r"""
    Total Variation Distance

    .. math::
        TVD = 0.5 * L1(Pa, Pd) \geq 0

    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :return: total variation distance metric
    """
    Lp_res = LP(x, facet, 1)
    tvd = 0.5 * Lp_res
    return tvd


@registry.pretraining
def KS(x: pd.Series, facet: pd.Series) -> float:
    r"""
    Kolmogorov-Smirnov

    .. math::
        KS = max(\left | Pa-Pd \right |) \geq 0

    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Kolmogorov-Smirnov metric
    """
    return LP(x, facet, 1)


@registry.pretraining
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
