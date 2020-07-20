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


# FIXME
# Use a decorator or a more robust mechanism to register the metrics.
# See https://github.com/aws/famly/pull/16
PRETRAINING_METRICS = public_functions(pretraining)
POSTTRAINING_METRICS = public_functions(posttraining)


__all__ = [x.__name__ for x in PRETRAINING_METRICS + POSTTRAINING_METRICS]


def metric_partial_nullary(
    metric: Callable,
    x: pd.Series,
    facet: pd.Series,
    labels: pd.Series,
    positive_label_index: pd.Series,
    predicted_labels: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    if metric == pretraining.CI:
        return lambda: pretraining.CI(x, facet)
    elif metric == pretraining.DPL:
        return lambda: pretraining.DPL(x, facet, positive_label_index)
    elif metric == pretraining.KL:
        # return lambda: pretraining.KL(x, facet, positive_label_index)
        # FIXME
        return lambda: 0
    elif metric == pretraining.JS:
        # return lambda: pretraining.JS(x, facet, positive_label_index)
        # FIXME
        return lambda: 0
    elif metric == pretraining.LP:
        # return lambda: pretraining.LP(x, facet, positive_label_index)
        # FIXME
        return lambda: 0
    elif metric == pretraining.TVD:
        # return lambda: pretraining.TVD(x, facet, positive_label_index)
        # FIXME
        return lambda: 0
    elif metric == pretraining.KS:
        return lambda: 0
    elif metric == pretraining.CDD:
        # FIXME
        # return pretraining.CDD(x, facet, positive_label_index)
        return lambda: 0
    else:
        # raise RuntimeError("wrap_metric_partial_nullary: Unregistered metric")
        log.warning("unregistered metric: %s, FIXME", metric.__name__)
        return lambda: 0


def metric_partial_binary_x_facet(
    metric: Callable,
    labels: pd.Series,
    positive_label_index: pd.Series,
    predicted_labels: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    if metric == pretraining.CI:
        return lambda x, facet: pretraining.CI(x, facet)
    elif metric == pretraining.DPL:
        return lambda x, facet: pretraining.DPL(x, facet, positive_label_index)
    elif metric == pretraining.KL:
        return lambda x, facet: pretraining.KL(x, facet, positive_label_index)
    elif metric == pretraining.JS:
        return lambda x, facet: pretraining.JS(x, facet, positive_label_index)
    elif metric == pretraining.JS:
        return lambda x, facet: pretraining.JS(x, facet, positive_label_index)
    elif metric == pretraining.LP:
        return lambda x, facet: pretraining.LP(x, facet, positive_label_index)
    elif metric == pretraining.TVD:
        return lambda x, facet: pretraining.TVD(x, facet, positive_label_index)
    elif metric == pretraining.CDD:
        # FIXME
        # return pretraining.CDD(x, facet, positive_label_index)
        return lambda x, facet: 0
    elif metric == posttraining.DPPL:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.DI:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.DCO:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.RD:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.DLR:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.AD:
        # FIXME
        return lambda x, facet: 0
    elif metric == posttraining.FT:
        # FIXME
        return lambda x, facet: 0
    else:
        raise RuntimeError("metric_partial_binary_x_facet: Unregistered metric")


def metric_one_vs_all(metric: Callable[..., float], x: pd.Series, *args, **kwargs) -> Dict[Any, float]:
    """
    Calculate any metric for a categorical facet and/or label using 1 vs all
    :param metric: a callable for a bias metric
    :param x: pandas series containing categorical values
    :param args: additional argument list
    :param kwargs: additional keyword argument list
    :return: A dictionary in which each key is one of the unique values in x and each value is
            its corresponding metric according to the requested metric
    """
    categories = x.unique()
    res = {}
    for cat in categories:
        f = metric_partial_nullary(metric, x, (x == cat), *args, **kwargs)
        res[cat] = f()
    return res
