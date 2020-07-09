from typing import Callable, Dict, Any, Optional
import pandas as pd
import numpy as np

from .posttraining import *
from .pretraining import *
import inspect

from . import pretraining
from . import posttraining


def public_functions(module):
    return [x[1] for x in inspect.getmembers(module) if inspect.isfunction(x[1]) and not x[0].startswith("_")]


PRETRAINING_METRICS = public_functions(pretraining)
POSTTRAINING_METRICS = public_functions(posttraining)


__all__ = [x.__name__ for x in PRETRAINING_METRICS + POSTTRAINING_METRICS]


def metric_one_vs_all(
    metric: Callable[..., float],
    x: pd.Series,
    positive_label_index: Optional[pd.Series] = None,
    predicted_labels: Optional[pd.Series] = None,
    labels: Optional[pd.Series] = None,
    group_variable: Optional[pd.Series] = None,
    dataset: Optional[pd.DataFrame] = None,
) -> Dict[Any, float]:
    """
    Calculate any metric for a categorical facet and/or label using 1 vs all
    :param metric: a callable for a bias metric
    :param x: pandas series containing categorical values
    :param positive_label_index: series of boolean values indicating positive target labels (optional)
    :param predicted_labels: series of model predictions of target column (optional)
    :param labels: series of true labels (optional)
    :param group_variable: series indicating strata each point belongs to (used for CDD metric) (optional)
    :param dataset: full dataset (used only for FlipTest metric) (optional)
    :return: A dictionary in which each key is one of the unique values in x and each value is
            its corresponding metric according to the requested metric
    """
    categories = x.unique()
    res = {}
    for cat in categories:
        if labels is None or len(np.unique(labels)) <= 2:
            if metric in PRETRAINING_METRICS:
                if metric != pretraining.CDD:
                    res[cat] = metric(x, x == cat, positive_label_index)
                else:
                    res[cat] = metric(x, x == cat, positive_label_index, group_variable)
            else:
                if metric == posttraining.FT:
                    res[cat] = metric(dataset, x == cat, labels, predicted_labels)
                else:
                    res[cat] = metric(x, x == cat, labels, predicted_labels)
        else:
            res[cat] = label_one_vs_all(
                metric, x, x == cat, predicted_labels=predicted_labels, labels=labels, group_variable=group_variable
            )

    return res


def label_one_vs_all(
    metric: Callable[..., float],
    x: pd.Series,
    facet: pd.Series,
    labels: pd.Series,
    predicted_labels: pd.Series = None,
    group_variable: pd.Series = None,
) -> Dict:
    """
    :param metric: one of the bias measures defined in this file
    :param x: data from the feature of interest
    :param facet: boolean column with true values indicate a sensitive value
    :param predicted_labels: predictions for labels made by model
    :param labels: True values of the target column
    :param group_variable: column of values indicating the subgroup each data point belongs to (used for calculating CDD metric only)
    :return:
    """

    values = {}
    label_unique = np.unique(labels)

    for label in label_unique:
        if metric in PRETRAINING_METRICS:
            if metric != CDD:
                values[label] = metric(x, facet, labels == label)
            else:
                values[label] = metric(x, facet, labels == label, group_variable)
        else:
            values[label] = metric(x, facet, labels == label, predicted_labels == label)

    return values
