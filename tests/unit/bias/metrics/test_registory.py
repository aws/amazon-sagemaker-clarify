import pytest
from famly.bias.metrics import registory


def testValidRegistrition():
    @registory.pretraining
    def pretraining_metric():
        pass

    @registory.posttraining
    def posttraining_metric():
        pass

    assert pretraining_metric in registory.PRETRAINING_METRIC_FUNCTIONS
    assert posttraining_metric in registory.POSTTRAINING_METRIC_FUNCTIONS
    assert pretraining_metric.__name__ in registory.all_metrics()
    assert posttraining_metric.__name__ in registory.all_metrics()


def testInalidRegistrition():
    # non-function
    with pytest.raises(TypeError) as e:

        @registory.pretraining
        class TestClass(object):
            pass

    with pytest.raises(TypeError) as e:

        @registory.posttraining
        class TestClass(object):
            pass

    # duplicate pretraining metrics
    with pytest.raises(AssertionError) as e:

        @registory.pretraining
        @registory.pretraining
        def metric_function():
            pass

    with pytest.raises(AssertionError) as e:

        @registory.pretraining
        def metric_function():
            pass

        @registory.pretraining
        def metric_function():
            pass

    # duplicate posttraining metrics
    with pytest.raises(AssertionError) as e:

        @registory.posttraining
        @registory.posttraining
        def metric_function():
            pass

    with pytest.raises(AssertionError) as e:

        @registory.posttraining
        def metric_function():
            pass

        @registory.posttraining
        def metric_function():
            pass

    # duplicate metrics
    with pytest.raises(AssertionError) as e:

        @registory.pretraining
        def metric_function():
            pass

        @registory.posttraining
        def metric_function():
            pass
