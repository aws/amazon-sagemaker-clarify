import pytest
from famly.bias.metrics import registry


def testValidRegistration():
    @registry.pretraining
    def pretraining_metric():
        pass

    @registry.posttraining
    def posttraining_metric():
        pass

    assert pretraining_metric in registry.PRETRAINING_METRIC_FUNCTIONS
    assert posttraining_metric in registry.POSTTRAINING_METRIC_FUNCTIONS
    assert pretraining_metric.__name__ in registry.all_metrics()
    assert posttraining_metric.__name__ in registry.all_metrics()


def testInvalidRegistration():
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
            pass

    with pytest.raises(AssertionError) as e:

        @registry.pretraining
        def metric_function():
            pass

        @registry.pretraining
        def metric_function():
            pass

    # duplicate posttraining metrics
    with pytest.raises(AssertionError) as e:

        @registry.posttraining
        @registry.posttraining
        def metric_function():
            pass

    with pytest.raises(AssertionError) as e:

        @registry.posttraining
        def metric_function():
            pass

        @registry.posttraining
        def metric_function():
            pass

    # duplicate metrics
    with pytest.raises(AssertionError) as e:

        @registry.pretraining
        def metric_function():
            pass

        @registry.posttraining
        def metric_function():
            pass
