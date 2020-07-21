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

PRETRAINING_METRICS_BINARY = set([CI])
PRETRAINING_METRICS_TERNARY = set([DPPL, KL, JS, LP, TVD, KS])
POSTTRAINING_METRICS_QUATERNARY = set([AD, DPPL, DI, DCO, RD, DLR, AD, TE, FT])


def metric_partial_nullary(
    metric: Callable,
    x: pd.Series,
    facet: pd.Series,
    labels: pd.Series,
    positive_label: Any,
    predicted_labels: pd.Series,
    positive_predicted_label: Any,
) -> float:
    if metric == PRETRAINING_METRICS_BINARY:
        return lambda: metric(x, facet)
    elif metric == PRETRAINING_METRICS_TERNARY:
        return lambda: metric(x, facet, positive_label_index)
    elif metric in POSTTRAINING_METRICS_QUATERNARY:
        return lambda: metric(x, facet, labels, positive_label, predicted_labels, positive_predicted_label)
    else:
        # raise RuntimeError("wrap_metric_partial_nullary: Unregistered metric")
        log.warning("unregistered metric: %s, FIXME", metric.__name__)
        return lambda: 0


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
