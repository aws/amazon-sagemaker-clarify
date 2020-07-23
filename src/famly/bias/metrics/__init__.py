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


METRICS_ARITY_DYADIC = set([CI])
METRICS_ARITY_TETRADIC = set([DPL])
METRICS_ARITY_HEXADIC = set([AD, DPPL, DI, DCO, RD, DLR, AD, TE, FT, DPPL, LP, TVD, KS])


def metric_partial_nullary(
    metric: Callable,
    x: pd.Series,
    facet: pd.Series,
    label: pd.Series = None,
    positive_label: Any = None,
    predicted_label: pd.Series = None,
    positive_predicted_label: Any = None,
) -> float:
    if metric in METRICS_ARITY_DYADIC:
        return lambda: metric(x, facet)
    elif metric in METRICS_ARITY_TETRADIC:
        return lambda: metric(x, facet, label, positive_label)
    elif metric in METRICS_ARITY_HEXADIC:
        return lambda: metric(x, facet, label, positive_label, predicted_label, positive_predicted_label)
    elif metric in set([KL, JS]):
        return lambda: metric(label, facet)
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
