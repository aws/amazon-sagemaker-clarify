import logging
from typing import List, Optional, Dict
from smclarify.bias.metrics.common import divide, binary_confusion_matrix
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from pandas.api.types import CategoricalDtype
import pandas as pd
from functional import seq

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
    :return fractions of true positives, false positives, true negatives, false negatives
    """
    return binary_confusion_matrix(feature[sensitive_facet_index], positive_label_index, positive_predicted_label_index)


def proportion(sensitive_facet_index: pd.Series) -> float:
    r"""
    Proportion of examples in sensitive facet.

    :param sensitive_facet_index: boolean column indicating sensitive group
    :return: the fraction of examples in the sensitive facet.
    """
    return sum(sensitive_facet_index) / len(sensitive_facet_index)


def observed_label_distribution(
    feature: pd.DataFrame, sensitive_facet_index: pd.Series, positive_label_index: pd.Series
) -> List[float]:
    r"""
    Distribution of observed label outcomes for sensitive facet

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: List of Proportion of positive and negative label outcomes
    """
    pos = len(feature[sensitive_facet_index & positive_label_index])
    n = len(feature[sensitive_facet_index])
    proportion_pos = divide(pos, n)
    return [proportion_pos, 1 - proportion_pos]


# Model Performance Metrics
def accuracy(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""
    Proportion of inputs assigned the correct predicted label by the model.

    :param: TP Counts of labels which were correctly predicted positive
    :param: FP Counts of labels which were incorrectly predicted positive
    :param: TN Counts of labels which were correctly predicted negative
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Proportion of inputs assigned the correct predicted label by the model.
    """
    return divide(TN + TP, TN + FP + FN + TP)


def PPL(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""
    Proportion of input assigned in positive predicted label.

    :param: TP: Counts of labels which were correctly predicted positive
    :param: FP: Counts of labels which were incorrectly predicted positive
    :param: TN: Counts of labels which were correctly predicted negative
    :param: FN: Counts of labels which were incorrectly predicted negative
    :return: Proportion of inputs assigned the positive predicted label.
    """
    return divide(TP + FP, TN + FP + FN + TP)


def PNL(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""
    Proportion of input assigned the negative predicted label.

    :param: TP: Counts of labels which were correctly predicted positive
    :param: FP: Counts of labels which were incorrectly predicted positive
    :param: TN: Counts of labels which were correctly predicted negative
    :param: FN: Counts of labels which were incorrectly predicted negative
    :return: Proportion of inputs assigned the negative predicted label.
    """
    return divide(TN + FN, TN + FP + FN + TP)


def recall(TP: int, FN: int) -> float:
    r"""
    Proportion of inputs with positive observed label correctly assigned the positive predicted label.

    :param: TP Counts of labels which were correctly predicted positive
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Proportion of inputs with positive observed label correctly assigned the positive predicted label.
    """
    return divide(TP, TP + FN)


def specificity(TN: int, FP: int) -> float:
    r"""
    Proportion of inputs with negative observed label correctly assigned the negative predicted label.

    :param: FP Counts of labels which were incorrectly predicted positive
    :param: TN Counts of labels which were correctly predicted negative
    :return: Proportion of inputs with negative observed label correctly assigned the negative predicted label.
    """
    return divide(TN, TN + FP)


def precision(TP: int, FP: int) -> float:
    r"""
    Proportion of inputs with positive predicted label that actually have a positive observed label.

    :param: TP Counts of labels which were correctly predicted positive
    :param: FP Counts of labels which were incorrectly predicted positive
    :return: Proportion of inputs with positive predicted label that actually have a positive observed label.
    """
    return divide(TP, FP + TP)


def rejection_rate(TN: int, FN: int) -> float:
    r"""
    Proportion of inputs with negative predicted label that actually have a negative observed label.

    :param: TN Counts of labels which were correctly predicted negative
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Proportion of inputs with negative predicted label that actually have a negative observed label.
    """
    return divide(TN, TN + FN)


def conditional_acceptance(TP: int, FP: int, FN: int) -> float:
    r"""
    Ratio between the positive observed labels and positive predicted labels.

    :param: TP Counts of labels which were correctly predicted positive
    :param: FP Counts of labels which were incorrectly predicted positive
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Ratio between the positive observed labels and positive predicted labels.
    """
    return divide(FN + TP, FP + TP)


def conditional_rejection(FP: int, TN: int, FN: int) -> float:
    r"""
    Ratio between the negative observed labels and negative predicted labels.

    :param: FP Counts of labels which were incorrectly predicted positive
    :param: TN Counts of labels which were correctly predicted negative
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Ratio between the negative observed labels and negative predicted labels.
    """
    return divide(TN + FP, TN + FN)


def f1_score(TP: int, FP: int, FN: int) -> float:
    r"""
    Harmonic mean of precision and recall.

    :param: TP Counts of labels which were correctly predicted positive
    :param: FP Counts of labels which were incorrectly predicted positive
    :param: FN Counts of labels which were incorrectly predicted negative
    :return: Harmonic mean of precision and recall.
    """
    precision_score = precision(TP, FP)
    recall_score = recall(TP, FN)
    return 2 * divide(precision_score * recall_score, precision_score + recall_score)


# Model Performance Metrics
def multicategory_confusion_matrix(
    label_series: pd.Series, predicted_label_series: pd.Series
) -> Optional[Dict[str, Dict]]:
    """
    Confusion Matrix for categorical label cases.
    :param label_series: Label Data Series
    :param predicted_label_series: Predicted Label Data Series
    :param unique_label_values: List of unique label values computed from the set of true and predicted labels
    :return: Matrix JSON where rows refer to true labels, and columns refer to predicted labels
    """
    # Handle differing pd.Series dtypes
    unique_label_values = set(label_series.unique())
    unique_label_values.update(predicted_label_series.unique())
    if label_series.dtype.name != predicted_label_series.dtype.name:
        try:
            predicted_label_series = predicted_label_series.astype(label_series.dtype)
        except:
            log.warning(
                "Predicted Label Series could not be cast as Label Series type. Multicategory Confusion Matrix won't be computed"
            )
            return None
    # Handle CategoricalDtype difference (see test/integration/test_bias_metrics)
    if label_series.dtype == "category" and label_series.dtype != predicted_label_series.dtype:
        try:
            pred_label_category = predicted_label_series.dtype.categories.astype(label_series.dtype.categories.dtype)
            category_obj = CategoricalDtype(pred_label_category, label_series.dtype.ordered)
            predicted_label_series = predicted_label_series.astype(category_obj)
        except:
            log.warning(
                "Predicted Label Series could not be cast as Label Series type. Multicategory Confusion Matrix won't be computed"
            )
            return None
    confusion_matrix_array = sklearn_confusion_matrix(
        label_series, predicted_label_series, labels=list(unique_label_values)
    )
    assert confusion_matrix_array.shape == (
        len(unique_label_values),
        len(unique_label_values),
    )
    matrix_json = {}
    unique_label_strings = [str(val) for val in unique_label_values]
    for index, val in enumerate(unique_label_strings):
        confusion_matrix_floats = [float(cfn_val) for cfn_val in confusion_matrix_array[index]]
        matrix_json[val] = seq(unique_label_strings).zip(confusion_matrix_floats).dict()

    return matrix_json
