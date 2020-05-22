import logging
from typing import Dict

import pandas as pd

log = logging.getLogger(__name__)


def class_imbalance_one_vs_all(x: pd.Series) -> Dict:
    """
    Calculate class imbalance for a categorical series doing 1 vs all
    :param x: pandas series
    :return:
    """
    categories = x.unique()
    res = dict()
    for cat in categories:
        res[cat] = class_imbalance(x, x == cat)
    return res


def class_imbalance(x: pd.Series, facet_index: pd.Series) -> float:
    """
    Class imbalance (CI)
    :param x: pandas series
    :param facet_index: boolean index series selecting disadvantaged instances
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the disadvantaged class.

    Bias is often generated from an under-representation of
    the disadvantaged class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (na − nd)/(na+nd). Where na is the number of instances in the advantaged group
    and nd is number of instances in the disadvantaged group.
    """
    na = len(x[~facet_index])
    nd = len(x[facet_index])
    q = na + nd
    if na == 0:
        raise ValueError("class_imbalance: negated facet set is empty. Check that x[~facet_index] has non-zero length.")
    if nd == 0:
        raise ValueError("class_imbalance: facet set is empty. Check that x[facet_index] has non-zero length.")
    q = na + nd
    assert q != 0
    ci = float(na - nd) / q
    return ci


def diff_positive_labels(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> float:
    """
    Difference in positive proportions in predicted labels
    :param x: pandas series of the target column
    :param label: pandas series of labels
    :param facet_index:
    :param positive_label_index: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    positive_label_index_neg_facet = (positive_label_index) & ~facet_index
    positive_label_index_facet = (positive_label_index) & facet_index
    n_neg_facet = len(x[~facet_index])
    n_facet = len(x[facet_index])
    n_pos_label_neg_facet = len(x[positive_label_index_neg_facet])
    n_pos_label_facet = len(x[positive_label_index_facet])
    if n_neg_facet == 0:
        raise ValueError("diff_positive_labels: negative facet set is empty.")
    if n_facet == 0:
        raise ValueError("diff_positive_labels: facet set is empty.")
    q_neg = n_pos_label_neg_facet / n_neg_facet
    q_pos = n_pos_label_facet / n_facet
    if (q_neg + q_pos) == 0:
        raise ValueError("diff_positive_labels: label facet is empty.")
    res = (q_neg - q_pos) / (q_neg + q_pos)
    return res
