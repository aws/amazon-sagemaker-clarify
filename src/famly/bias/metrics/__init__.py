from typing import Any, Callable, Dict
import inspect

from .posttraining import *
from .pretraining import *

from . import pretraining
from . import posttraining
from . import registry

PRETRAINING_METRICS = registry.PRETRAINING_METRIC_FUNCTIONS
POSTTRAINING_METRICS = registry.POSTTRAINING_METRIC_FUNCTIONS
PRETRAINING_METRIC_DESCRIPTIONS = {
    "CI": "Class imbalance (CI)",
    "CDDL": "Conditional Demographic Disparity in labels (CDDL)",
    "DPL": "Difference in positive proportions in true labels (DPL)",
    "JS": "Jensen-Shannon divergence in true labels (JS)",
    "KL": "Kullback - Liebler divergence in true labels(KL)",
    "KS": "Kolmogorov-Smirnov distance (KS)",
    "LP": "L-p norm (LP)",
    "TVD": "Total variation distance (TVD)",
}

__all__ = registry.all_metrics()


def call_metric(metric: Callable[..., float], **kwargs) -> float:
    """
    Call metric function with keyword arguments. The excess arguments will be ignored.
    :param metric: a callable for a bias metric
    :param kwargs: keyword argument list
    :return: Return value of the callable
    :raise: KeyError: if a mandatory argument is missing from kwargs.
    """
    return metric(**{key: kwargs[key] for key in inspect.signature(metric).parameters.keys()})


def metric_one_vs_all(metric: Callable[..., float], feature: pd.Series, **kwargs) -> Dict[Any, float]:
    """
    Calculate any metric for a categorical facet and/or label using 1 vs all
    :param metric: a callable for a bias metric
    :param feature: pandas series containing categorical values
    :param kwargs: additional keyword argument list
    :return: A dictionary in which each key is one of the unique values in x and each value is
            its corresponding metric according to the requested metric
    """
    categories = feature.unique()
    results = {}
    for category in categories:
        results[category] = call_metric(metric, feature=feature, facet=(feature == category), **kwargs)
    return results
