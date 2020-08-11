import pytest
from famly.bias.metrics import registry


def test_valid_registration():
    @registry.pretraining
    def pretraining_metric():
        """
        Metric Description
        :return: None
        """

    @registry.posttraining
    def posttraining_metric():
        """
        Metric Description
        :return: None
        """

    assert pretraining_metric in registry.PRETRAINING_METRIC_FUNCTIONS
    assert posttraining_metric in registry.POSTTRAINING_METRIC_FUNCTIONS
    assert pretraining_metric.__name__ in registry.all_metrics()
    assert posttraining_metric.__name__ in registry.all_metrics()


def test_invalid_registration():
    # non-function
    with pytest.raises(TypeError) as e:

        @registry.pretraining
        class TestClass(object):
            pass

    with pytest.raises(TypeError) as e:

        @registry.posttraining
        class TestClass(object):
            pass

    # duplicate pretraining metrics
    with pytest.raises(AssertionError) as e:

        @registry.pretraining
        @registry.pretraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

    with pytest.raises(AssertionError) as e:

        @registry.pretraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.pretraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

    # duplicate posttraining metrics
    with pytest.raises(AssertionError) as e:

        @registry.posttraining
        @registry.posttraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

    with pytest.raises(AssertionError) as e:

        @registry.posttraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.posttraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

    # duplicate metrics
    with pytest.raises(AssertionError) as e:

        @registry.pretraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

        @registry.posttraining
        def metric_function():
            """
            Metric Description
            :return: None
            """

    # Missing DocStrings
    with pytest.raises(ValueError) as e:

        @registry.pretraining
        def pretraining_metric_function():
            pass

        @registry.posttraining
        def posttraining_metric_function():
            pass
