"""
A module to register metric functions by decorators. Not supposed to be used outside of the package.
"""
import types

PRETRAINING_METRIC_FUNCTIONS = []
POSTTRAINING_METRIC_FUNCTIONS = []


def pretraining(function):
    """
    Decorator to register a pretraining function.
    :return: The function
    :raise: TypeError: if the actual parameter is not a function.
    :raise: AssertError: if function name is duplicate with an registered one.
    """
    __register(PRETRAINING_METRIC_FUNCTIONS, function)
    return function


def posttraining(function):
    """
    Decorator to register a posttraining function.
    :return: The function
    :raise: Exception: if the actual parameter is not a function.
    :raise: AssertError: if function name is duplicate with an registered one.
    """
    __register(POSTTRAINING_METRIC_FUNCTIONS, function)
    return function


def __register(metrics, function):
    if not isinstance(function, types.FunctionType):
        raise TypeError("{} is not a function".format(function))

    assert not any(x == function.__name__ for x in all_metrics()), "{} is already registered".format(function.__name__)
    # assign first line of docstring as description
    if not function.__doc__:
        raise ValueError("Metric function doesn't have a docstring")
    function.description = function.__doc__.lstrip().split("\n")[0]
    metrics.append(function)


def all_metrics():
    """
    Collect a list of names of all registered metric functions.
    :return: The name list.
    """
    pretraining_metrics = [x.__name__ for x in PRETRAINING_METRIC_FUNCTIONS]
    posttraining_metrics = [x.__name__ for x in POSTTRAINING_METRIC_FUNCTIONS]
    return pretraining_metrics + posttraining_metrics
