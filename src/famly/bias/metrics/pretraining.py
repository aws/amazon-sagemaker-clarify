"""
Pre training metrics
"""
import logging
from famly.util import pdfs_aligned_nonzero
from . import registry, common
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@registry.pretraining
def CI(feature: pd.Series, facet: pd.Series) -> float:
    r"""
    Class Imbalance (CI)

    :param feature: input feature
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
    pos = len(feature[facet])
    neg = len(feature[~facet])
    q = pos + neg
    if neg == 0:
        raise ValueError("CI: negated facet set is empty. Check that x[~facet] has non-zero length.")
    if pos == 0:
        raise ValueError("CI: facet set is empty. Check that x[facet] has non-zero length.")
    assert q != 0
    ci = float(neg - pos) / q
    return ci


@registry.pretraining
def DPL(feature: pd.Series, facet: pd.Series, label: pd.Series, positive_label_index: pd.Series) -> float:
    """
    Difference in Positive proportions in Labels (DPL)

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: pandas series of labels (binary, multicategory, or continuous)
    :param positive_label_index: boolean column indicating positive labels
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    return common.DPL(feature, facet, label, positive_label_index)


@registry.pretraining
def KL(label: pd.Series, facet: pd.Series) -> float:
    r"""
    Kullback - Liebler divergence (KL)

    .. math::
        KL(Pa, Pd) = \sum_{x}{Pa(x) \ log2 \frac{Pa(x)}{Pd(x)}}

    :param label: input feature
    :param facet: boolean column indicating sensitive group
    :return: Kullback and Leibler (KL) divergence metric
    """
    facet = facet.astype(bool)
    xs_a = label[facet]
    xs_d = label[~facet]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    kl = np.sum(Pa * np.log2(Pa / Pd))
    return kl


@registry.pretraining
def JS(label: pd.Series, facet: pd.Series) -> float:
    r"""
    Jensen-Shannon divergence (JS)

    .. math::
        JS(Pa, Pd, P) = 0.5 [KL(Pa,P) + KL(Pd,P)] \geq 0

    :param label: input feature
    :param facet: boolean column indicating sensitive group
    :return: Jensen-Shannon (JS) divergence metric
    """
    facet = facet.astype(bool)
    xs_a = label[facet]
    xs_d = label[~facet]
    (Pa, Pd, P) = pdfs_aligned_nonzero(xs_a, xs_d, label)
    if len(Pa) == 0 or len(Pd) == 0 or len(P) == 0:
        return np.nan
    res = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    return res


@registry.pretraining
def LP(label: pd.Series, facet: pd.Series) -> float:
    r"""
    L-p norm (LP)

    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{x} |Pa(x)-Pd(x)|^p]^{1/p}

    :param label: input feature
    :param facet: boolean column indicating sensitive group
    :return: Returns the LP norm of the difference between class distributions
    """
    return LP_norm(label, facet, 2)


def LP_norm(label: pd.Series, facet: pd.Series, norm_order) -> float:
    facet = facet.astype(bool)
    xs_a = label[facet]
    xs_d = label[~facet]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        return np.nan
    res = np.linalg.norm(Pa - Pd, norm_order)
    return res


@registry.pretraining
def TVD(label: pd.Series, facet: pd.Series) -> float:
    r"""
    Total variation distance (TVD)

    .. math::
        TVD = 0.5 * L1(Pa, Pd) \geq 0

    :param label: input feature
    :param facet: boolean column indicating sensitive group
    :return: total variation distance metric
    """
    Lp_res = LP_norm(label, facet, 1)
    tvd = 0.5 * Lp_res
    return tvd


@registry.pretraining
def KS(label: pd.Series, facet: pd.Series) -> float:
    r"""
    Kolmogorov-Smirnov distance (KS)

    .. math::
        KS = max(\left | Pa-Pd \right |) \geq 0

    :param label: input feature
    :param facet: boolean column indicating sensitive group
    :return: Kolmogorov-Smirnov metric
    """
    return LP_norm(label, facet, 1)


@registry.pretraining
def CDDL(feature: pd.Series, facet: pd.Series, positive_label_index: pd.Series, group_variable: pd.Series) -> float:
    r"""
    Conditional Demographic Disparity in labels (CDDL)

    .. math::
        CDD = \frac{1}{n}\sum_i n_i * DD_i \\\quad\:where \: DD_i = \frac{Number\:of\:rejected\:applicants\:protected\:facet}{Total\:number\:of\:rejected\:applicants} -
        \frac{Number\:of\:rejected\:applicants\:protected\:facet}{Total\:number\:of\:rejected\:applicants} \\\quad\:\quad\:\quad\:\quad\:\quad\:\quad\:for\:each\:group\:variable\: i

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index : boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    return common.CDD(feature, facet, positive_label_index, group_variable)
