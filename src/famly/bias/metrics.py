import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def class_imbalance_one_vs_all(x: pd.Series) -> Dict:
    categories = x.unique()
    res = dict()
    for cat in categories:
        res[cat] = class_imbalance(x, x == cat)
    return res


def class_imbalance(x: pd.Series, disadvantaged_index: pd.Series) -> float:
    """
    Class imbalance (CI)
    :param x: pandas series
    :param disadvantaged_index: boolean index series selecting disadvantaged instances
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the disadvantaged class.

    Bias is often generated from an under-representation of
    the disadvantaged class in the dataset, especially if the desired “golden truth”
    is equality across classes. As an example, algorithms for granting small business
    loans are often biased against women because the historical record of loan approvals contains very few women,
    because women did not usually apply for loans
    to start small businesses. This imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (na − nd)/(na+nd). Where na is the number of instances in the advantaged group
    and nd is number of instances in the disadvantaged group.
    """
    na = len(x[~disadvantaged_index])
    nd = len(x[disadvantaged_index])
    q = na + nd
    if na == 0:
        log.warning("class_imbalance: advantaged set is empty.")
        return np.nan
    if nd == 0:
        log.warning("class_imbalance: disadvantaged set is empty.")
        return np.nan
    if q == 0:
        log.warning("class_imbalance: set is empty.")
        return np.nan
    ci = float(na - nd) / q
    return ci


def diff_positive_labels(
    x: pd.Series, label: pd.Series, disadvantaged_index: pd.Series, positive_label_value: Any = 1
) -> float:
    """
    Difference in positive proportions in predicted labels
    :param x: pandas series of the target column
    :param label: pandas series of labels
    :param disadvantaged_index:
    :param positive_label_value: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    advantaged_positive_label_index = (label == positive_label_value) & ~disadvantaged_index
    disadvantaged_positive_label_index = (label == positive_label_value) & disadvantaged_index
    na = len(x[~disadvantaged_index])
    nd = len(x[disadvantaged_index])
    nap = len(x[advantaged_positive_label_index])
    ndp = len(x[disadvantaged_positive_label_index])
    if na == 0:
        log.warning("diff_positive_labels: advantaged set is empty.")
        return np.nan
    if nd == 0:
        log.warning("diff_positive_labels: disadvantaged set is empty.")
        return np.nan
    qa = nap / na
    qd = ndp / nd
    if (qa + qd) == 0:
        log.warning("diff_positive_labels: set is empty.")
        return np.nan
    res = (qa - qd) / (qa + qd)
    return res
