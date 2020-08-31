"""
A module to register metric functions by decorators. Not supposed to be used outside of the package.
"""
import types
from enum import Enum
from typing import Callable, List, Any

PRETRAINING_BINARY_METRIC_FUNCTIONS: List[Callable[..., Any]] = []
PRETRAINING_MULTI_CLASS_METRIC_FUNCTIONS: List[Callable[..., Any]] = []
POSTTRAINING_BINARY_METRIC_FUNCTIONS: List[Callable[..., Any]] = []
POSTTRAINING_MULTI_CLASS_METRIC_FUNCTIONS: List[Callable[..., Any]] = []


class ProblemType(Enum):
    """Type of problem deduced from the label values"""

    BINARY = 0
    REGRESSION = 1
    MULTICLASS = 2
    OTHER = 3


def pretraining(problem_type: ProblemType = ProblemType.BINARY):
    """
    Decorator to register a pretraining function.
    :return: The function
    :raise: TypeError: if the actual parameter is not a function.
    :raise: AssertError: if function name is duplicate with an registered one.
    """

    def pretraining_function(function):
        assert not any(x == function.__name__ for x in all_metrics()), "{} is already registered".format(
            function.__name__
        )
        __register(PRETRAINING_BINARY_METRIC_FUNCTIONS, function)
        if problem_type == ProblemType.MULTICLASS:
            __register(PRETRAINING_MULTI_CLASS_METRIC_FUNCTIONS, function)
        return function

    return pretraining_function


def posttraining(problem_type: ProblemType = ProblemType.BINARY):
    """
    Decorator to register a posttraining function.
    :return: The function
    :raise: Exception: if the actual parameter is not a function.
    :raise: AssertError: if function name is duplicate with an registered one.
    """

    def posttraining_function(function):
        assert not any(x == function.__name__ for x in all_metrics()), "{} is already registered".format(
            function.__name__
        )
        __register(POSTTRAINING_BINARY_METRIC_FUNCTIONS, function)
        if problem_type == ProblemType.MULTICLASS:
            __register(POSTTRAINING_MULTI_CLASS_METRIC_FUNCTIONS, function)
        return function

    return posttraining_function


def __register(metrics, function):
    if not isinstance(function, types.FunctionType):
        raise TypeError("{} is not a function".format(function))
    metrics.append(function)


def all_metrics():
    """
    Collect a list of names of all registered metric functions.
    :return: The name list.
    """
    pretraining_binary_metrics = [x.__name__ for x in PRETRAINING_BINARY_METRIC_FUNCTIONS]
    pretraining_multi_class_metrics = [x.__name__ for x in PRETRAINING_MULTI_CLASS_METRIC_FUNCTIONS]
    posttraining_binary_metrics = [x.__name__ for x in POSTTRAINING_BINARY_METRIC_FUNCTIONS]
    posttraining_multi_class_metrics = [x.__name__ for x in POSTTRAINING_MULTI_CLASS_METRIC_FUNCTIONS]
    return (
        pretraining_binary_metrics
        + pretraining_multi_class_metrics
        + posttraining_binary_metrics
        + posttraining_multi_class_metrics
    )
