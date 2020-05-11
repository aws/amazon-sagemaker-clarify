import pandas as pd
from famly.bias.metrics import class_imbalance, class_imbalance_one_vs_all, diff_positive_labels
from pytest import approx


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


def test_ci_one_vs_all():
    ci = class_imbalance_one_vs_all(df[0])
    assert ci["a"] == approx(-1 / 3)
    assert ci["b"] == approx(1 / 3)


def test_dppt():
    df_dppt = df.copy()
    df_dppt["label"] = pd.Series([0, 1, 0])
    assert diff_positive_labels(df_dppt[0], df_dppt["label"], df_dppt[0] == "a", 1) == -1.0
    df_dppt["label"] = pd.Series([0, 0, 1])
    assert diff_positive_labels(df_dppt[0], df_dppt["label"], df_dppt[0] == "a", 1) == 1.0
    df_dppt["label"] = pd.Series([0, 1, 1])
    assert diff_positive_labels(df_dppt[0], df_dppt["label"], df_dppt[0] == "a", 1) == approx(1 / 3)
