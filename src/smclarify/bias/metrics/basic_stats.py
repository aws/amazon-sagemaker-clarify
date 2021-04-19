import logging
from typing import List

import pandas as pd
from .common import divide

log = logging.getLogger(__name__)


def confusion_matrix(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> List[float]:
    r"""
    Fractions of TP, FP, FN, TN.

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return list of fractions of true positives, false positives, false negatives, true negatives
    """
    TP_d = len(feature[positive_label_index & positive_predicted_label_index & sensitive_facet_index])
    FN_d = len(feature[positive_label_index & (~positive_predicted_label_index) & sensitive_facet_index])

    TN_d = len(feature[(~positive_label_index) & (~positive_predicted_label_index) & sensitive_facet_index])
    FP_d = len(feature[(~positive_label_index) & positive_predicted_label_index & sensitive_facet_index])
    size = len(feature[sensitive_facet_index])
    return [divide(TP_d, size), divide(FP_d, size), divide(FN_d, size), divide(TN_d, size)]


def proportion(sensitive_facet_index: pd.Series) -> float:
    r"""
    Proportion of examples in sensitive facet.

    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: the fraction of examples in the sensitive facet.
    """
    return sum(sensitive_facet_index) / len(sensitive_facet_index)
