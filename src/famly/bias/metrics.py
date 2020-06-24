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


def class_imbalance(x: pd.Series, sensitive_index: pd.Series) -> float:
    """
    Class imbalance (CI)
    :param x: pandas series
    :param sensitive_index: boolean index series selecting sensitive instances on the series given as x argument.
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the sensitive class.

    Bias is often generated from an under-representation of
    the sensitive class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (ns − s)/(ns + s). Where ns is the number of instances in the non sensitive group
    and s is number of instances in the sensitive group.
    """
    ns = len(x[~sensitive_index])
    s = len(x[sensitive_index])
    sum = ns + s
    if ns == 0:
        raise ValueError(
            "class_imbalance: negated sensitive set is empty. Check that x[~sensitive_index] has non-zero length."
        )
    if s == 0:
        raise ValueError("class_imbalance: sensitive set is empty. Check that x[sensitive_index] has non-zero length.")
    assert sum != 0
    ci = float(ns - s) / sum
    return ci


def diff_positive_labels(x: pd.Series, sensitive_index: pd.Series, positive_label_index: pd.Series) -> float:
    """
    Difference in positive proportions in predicted labels
    :param x: pandas series of the target column
    :param label: pandas series of labels
    :param sensitive_index:
    :param positive_label_index: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    positive_label_index_not_sensitive = (positive_label_index) & ~sensitive_index
    positive_label_index_sensitive = (positive_label_index) & sensitive_index
    ns = len(x[~sensitive_index])
    s = len(x[sensitive_index])
    n_pos_label_not_sensitive = len(x[positive_label_index_not_sensitive])
    n_pos_label_sensitive = len(x[positive_label_index_sensitive])
    if ns == 0:
        raise ValueError("diff_positive_labels: negative sensitive set is empty.")
    if s == 0:
        raise ValueError("diff_positive_labels: sensitive set is empty.")
    q_neg = n_pos_label_not_sensitive / ns
    q_pos = n_pos_label_sensitive / s
    if (q_neg + q_pos) == 0:
        raise ValueError("diff_positive_labels: label sensitive is empty.")
    res = (q_neg - q_pos) / (q_neg + q_pos)
    return res
