# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

"""
Pre-training metrics
The metrics defined in this file can be computed before training the model.

Note:
    Some metrics signatures include label as an argument, while others include positive_label_index. positive_label_index
    is a boolean column that indicates positive labels. In case of multi-class labels, positive_label_index can be computed
    by collapsing the multiple classes of labels into two categories.
    The metrics that consume positive_label_index require it to be of type 'bool'. The metrics that consume label can work with other data types as well.
"""
import logging
from smclarify.util import pdfs_aligned_nonzero
from . import registry, common
from .common import require
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


@registry.pretraining
def CI(feature: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Class Imbalance (CI)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
        of the sensitive class.

    .. math::
        CI = \frac{na-nd}{na+nd}

    Bias is often generated from an under-representation of
    the sensitive class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (np − p)/(np + p). Where np is the number of instances in the not sensitive group
    and p is number of instances in the sensitive group.
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    pos = len(feature[sensitive_facet_index])
    neg = len(feature[~sensitive_facet_index])
    q = pos + neg
    if neg == 0:
        raise ValueError("CI: negated facet set is empty. Check that x[~facet] has non-zero length.")
    if pos == 0:
        raise ValueError("CI: facet set is empty. Check that x[facet] has non-zero length.")
    assert q != 0
    ci = float(neg - pos) / q
    return ci


@registry.pretraining
def DPL(feature: pd.Series, sensitive_facet_index: pd.Series, positive_label_index: pd.Series) -> float:
    r"""
    Difference in Positive Proportions in Labels (DPL)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    return common.DPL(feature, sensitive_facet_index, positive_label_index)


@registry.pretraining
def KL(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kullback-Liebler Divergence (KL)

    .. math::
        KL(Pa, Pd) = \sum_{x}{Pa(x) \ log \frac{Pa(x)}{Pd(x)}}

    :param label: column of labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kullback and Leibler (KL) divergence metric
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        raise ValueError("No instance of common facet found, dataset may be too small")
    kl = np.sum(Pa * np.log(Pa / Pd))
    return kl


@registry.pretraining
def JS(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Jensen-Shannon Divergence (JS)

    .. math::
        JS(Pa, Pd, P) = 0.5 [KL(Pa,P) + KL(Pd,P)] \geq 0

    :param label: column of labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Jensen-Shannon (JS) divergence metric
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd, P) = pdfs_aligned_nonzero(xs_a, xs_d, label)
    if len(Pa) == 0 or len(Pd) == 0 or len(P) == 0:
        raise ValueError("No instance of common facet found, dataset may be too small")
    res = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    return res


@registry.pretraining
def LP(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    L-p Norm (LP)

    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{x} |Pa(x)-Pd(x)|^p]^{1/p}

    :param label: column of labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Returns the LP norm of the difference between class distributions
    """
    return LP_norm(label, sensitive_facet_index, 2)


def LP_norm(label: pd.Series, sensitive_facet_index: pd.Series, norm_order) -> float:
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    xs_a = label[~sensitive_facet_index]
    xs_d = label[sensitive_facet_index]
    (Pa, Pd) = pdfs_aligned_nonzero(xs_a, xs_d)
    if len(Pa) == 0 or len(Pd) == 0:
        raise ValueError("No instance of common facet found, dataset may be too small")
    res = np.linalg.norm(Pa - Pd, norm_order)
    return res


@registry.pretraining
def TVD(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Total Variation Distance (TVD)

    .. math::
        TVD = 0.5 * L1(Pa, Pd) \geq 0

    :param label: column of labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: total variation distance metric
    """
    Lp_res = LP_norm(label, sensitive_facet_index, 1)
    tvd = 0.5 * Lp_res
    return tvd


@registry.pretraining
def KS(label: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kolmogorov-Smirnov Distance (KS)

    .. math::
        KS = max(\left | Pa-Pd \right |) \geq 0

    :param label: column of labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kolmogorov-Smirnov metric
    """
    return LP_norm(label, sensitive_facet_index, 1)


@registry.pretraining
def CDDL(
    feature: pd.Series, sensitive_facet_index: pd.Series, positive_label_index: pd.Series, group_variable: pd.Series
) -> float:
    r"""
    Conditional Demographic Disparity in Labels (CDDL)

    .. math::
        CDD = \frac{1}{n}\sum_i n_i * DD_i \\\quad\:where \: DD_i = \frac{Number\:of\:rejected\:applicants\:sensitive\:facet}{Total\:number\:of\:rejected\:applicants} -
        \frac{Number\:of\:rejected\:applicants\:sensitive\:facet}{Total\:number\:of\:rejected\:applicants} \\\quad\:\quad\:\quad\:\quad\:\quad\:\quad\:for\:each\:group\:variable\: i

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    return common.CDD(feature, sensitive_facet_index, positive_label_index, group_variable)
