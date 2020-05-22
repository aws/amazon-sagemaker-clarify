import pandas as pd
from famly.bias.metrics import class_imbalance, class_imbalance_one_vs_all, diff_positive_labels
from pytest import approx
import pytest


def dataframe():
    data = [["a"], ["a"], ["b"]]
    df = pd.DataFrame(data)
    return df


df = dataframe()


def test_ci():
    """test class imbalance"""
    # reverse bias
    assert class_imbalance(df[0], df[0] == "a") == approx(-1 / 3)
    # bias on b
    assert class_imbalance(df[0], df[0] == "b") == approx(1 / 3)
    with pytest.raises(ValueError) as e:
        class_imbalance(df[0], df[0] == "c")
    assert e.type == ValueError
    assert "class_imbalance: facet set is empty" in str(e.value)
    with pytest.raises(ValueError) as e:
        class_imbalance(df[0], df[0] != "c")
    assert e.type == ValueError
    assert "class_imbalance: negated facet set is empty" in str(e.value)


def test_ci_one_vs_all():
    ci = class_imbalance_one_vs_all(df[0])
    assert ci["a"] == approx(-1 / 3)
    assert ci["b"] == approx(1 / 3)


def test_dppt():
    dfl = df.copy()
    dfl["label"] = pd.Series([0, 1, 0])
    assert diff_positive_labels(dfl[0], dfl[0] == "a", dfl["label"] == 1) == -1.0
    dfl["label"] = pd.Series([0, 0, 1])
    assert diff_positive_labels(dfl[0], dfl[0] == "a", dfl["label"] == 1) == 1.0
    dfl["label"] = pd.Series([0, 1, 1])
    assert diff_positive_labels(dfl[0], dfl[0] == "a", dfl["label"] == 1) == approx(1 / 3)
