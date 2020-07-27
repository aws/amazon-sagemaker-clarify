from typing import Callable, Dict

from .posttraining import *
from .pretraining import *

from . import pretraining
from . import posttraining
from . import registry

PRETRAINING_METRICS = registry.PRETRAINING_METRIC_FUNCTIONS
POSTTRAINING_METRICS = registry.POSTTRAINING_METRIC_FUNCTIONS

__all__ = registry.all_metrics()


METRICS_ARITY_DYADIC = set([CI])
METRICS_ARITY_TETRADIC = set([DPL])
METRICS_ARITY_HEXADIC = set([AD, DPPL, DI, DCO, RD, DLR, AD, TE, FT])


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
    elif metric in {KL, JS, LP, TVD, KS}:
        return lambda: metric(label, facet)
    else:
        # raise RuntimeError("wrap_metric_partial_nullary: Unregistered metric")
        log.error("unregistered metric: %s, FIXME", metric.__name__)
        raise RuntimeError("Unregistered metric {}".format(metric.__name__))


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
