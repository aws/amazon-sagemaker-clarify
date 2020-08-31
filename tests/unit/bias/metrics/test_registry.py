import pytest
from famly.bias.metrics import registry, ProblemType


def test_valid_registration():
    @registry.pretraining(problem_type=ProblemType.BINARY)
    def pretraining_metric():
        """
        Metric Description
        :return: None
        """

    @registry.posttraining(problem_type=ProblemType.MULTICLASS)
    def posttraining_metric():
        """
        Metric Description
        :return: None
        """

    assert pretraining_metric in registry.PRETRAINING_BINARY_METRIC_FUNCTIONS
    assert posttraining_metric in registry.POSTTRAINING_MULTI_CLASS_METRIC_FUNCTIONS
    assert pretraining_metric.__name__ in registry.all_metrics()
    assert posttraining_metric.__name__ in registry.all_metrics()


def test_invalid_registration():
    # non-function
    with pytest.raises(TypeError) as e:

        @registry.pretraining(problem_type=ProblemType.BINARY)
        class TestClass(object):
            pass

    with pytest.raises(TypeError) as e:

        @registry.posttraining(problem_type=ProblemType.BINARY)
        class TestClass(object):
            pass

    # duplicate pretraining metrics
    with pytest.raises(AssertionError) as e:

        @registry.pretraining(problem_type=ProblemType.BINARY)
        @registry.pretraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

    with pytest.raises(AssertionError) as e:

        @registry.pretraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.pretraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

    # duplicate posttraining metrics
    with pytest.raises(AssertionError) as e:

        @registry.posttraining(problem_type=ProblemType.BINARY)
        @registry.posttraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

    with pytest.raises(AssertionError) as e:

        @registry.posttraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.posttraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

    # duplicate metrics
    with pytest.raises(AssertionError) as e:

        @registry.pretraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.posttraining(problem_type=ProblemType.BINARY)
        def metric_function():
            """
            Metric Description
            :return: None
            """
