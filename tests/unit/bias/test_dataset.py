import pandas as pd
from famly.bias.dataset import _metricname_fmt, class_imbalance_series


def dataframe():
    data = [["a"], ["a"], ["b"], ["c"]]
    df = pd.DataFrame(data)
    return df


df = dataframe()


def test_class_imbalance():
    assert class_imbalance_series(df[0]) == {"a": 0.0, "b": 0.5, "c": 0.5}
    protected = ["b", "c"]
    assert class_imbalance_series(df[0], protected) == {_metricname_fmt(protected): 0.0}
