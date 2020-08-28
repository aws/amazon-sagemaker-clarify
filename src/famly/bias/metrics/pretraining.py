"""
Pre training metrics
"""
import logging
from . import registry, common
import pandas as pd

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
    sensitive_facet_index = sensitive_facet_index.astype(bool)
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
    """
    Difference in Positive Proportions in Labels (DPL)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    return common.DPL(feature, sensitive_facet_index, positive_label_index)


@registry.pretraining
def KL(positive_label_index: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kullback-Liebler Divergence (KL)

    Use this function for binary label categories.

    .. math::
        KL(Pa, Pd) = \sum_{x}{Pa(x) \ log2 \frac{Pa(x)}{Pd(x)}}

    :param positive_label_index: boolean column indicating positive labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kullback and Leibler (KL) divergence metric
    """
    return common.KL(positive_label_index.astype(bool), sensitive_facet_index.astype(bool))


@registry.pretraining
def JS(positive_label_index: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Jensen-Shannon Divergence (JS)
    .. math::
        JS(Pa, Pd, P) = 0.5 [KL(Pa,P) + KL(Pd,P)] \geq 0
    Use this function for binary label categories.

    :param positive_label_index: boolean column indicating positive labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Jensen-Shannon (JS) divergence metric
    """
    return common.JS(positive_label_index.astype(bool), sensitive_facet_index.astype(bool))


@registry.pretraining
def LP(positive_label_index: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    L-p Norm (LP)

    Difference of norms of the distributions defined by the facet selection and its complement.

    .. math::
        Lp(Pa, Pd) = [\sum_{x} |Pa(x)-Pd(x)|^p]^{1/p}
    Use this function for binary label categories.

    :param positive_label_index: boolean column indicating positive labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Returns the LP norm of the difference between class distributions
    """
    return common.LP_norm(positive_label_index.astype(bool), sensitive_facet_index.astype(bool), 2)


@registry.pretraining
def TVD(positive_label_index: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Total Variation Distance (TVD)

    .. math::
        TVD = 0.5 * L1(Pa, Pd) \geq 0
    Use this function for binary label categories.

    :param positive_label_index: boolean column indicating positive labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: total variation distance metric
    """
    Lp_res = common.LP_norm(positive_label_index.astype(bool), sensitive_facet_index.astype(bool), 1)
    tvd = 0.5 * Lp_res
    return tvd


@registry.pretraining
def KS(positive_label_index: pd.Series, sensitive_facet_index: pd.Series) -> float:
    r"""
    Kolmogorov-Smirnov Distance (KS)

    .. math::
        KS = max(\left | Pa-Pd \right |) \geq 0
    Use this function for binary label categories.

    :param positive_label_index: boolean column indicating positive labels
    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: Kolmogorov-Smirnov metric
    """
    return common.LP_norm(positive_label_index.astype(bool), sensitive_facet_index.astype(bool), 1)


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
