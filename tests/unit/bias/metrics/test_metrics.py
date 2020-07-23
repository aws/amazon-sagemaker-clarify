from famly.bias.metrics import AD, CDD, CI, DCO, DI, DPL, DPPL, DLR, FT, JS, KL, KS, LP, RD, TE
from famly.bias.metrics import metric_one_vs_all
from famly.bias.metrics.constants import INFINITY
from pytest import approx
import pandas as pd
from pandas import Series
import numpy as np
import pytest


def dfBinary():
    """
    :return: dataframe with one column which contains Binary categorical data (length 12)
    """
    data = [["M"], ["F"], ["F"], ["M"], ["F"], ["M"], ["F"], ["F"], ["M"], ["M"], ["F"], ["F"]]

    df = pd.DataFrame(data)
    return df


def dfMulticategory():
    """
    :return: dataframe with one column which contains multicategorical data (length 24)
    """
    data = [
        ["M"],
        ["O"],
        ["M"],
        ["M"],
        ["F"],
        ["O"],
        ["O"],
        ["F"],
        ["M"],
        ["M"],
        ["F"],
        ["F"],
        ["O"],
        ["F"],
        ["M"],
        ["F"],
        ["O"],
        ["F"],
        ["M"],
        ["M"],
        ["F"],
        ["F"],
        ["O"],
        ["O"],
    ]
    # [1,     0,     0,      1,     0,     1,     0,     1,      1,    0,    1,       0,     1,     0,     1,    0,     0,     0,      1,    1,     0,     0,     1,     1]
    df = pd.DataFrame(data)
    return df


def dfContinuous():
    """
    :return: dataframe with one column which contains continuous data (length 12)
    """
    data = pd.Series(
        [
            1.55255404,
            1.87128923,
            1.82640675,
            0.48706083,
            0.21833644,
            0.45007763,
            0.47457823,
            1.5346789,
            1.61042132,
            1.87130261,
            1.97768247,
            1.05499183,
        ]
    )

    df = pd.DataFrame(data)
    return df


def datasetFT():
    X = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ]
    )

    return pd.DataFrame(X)


def datasetFTMult():
    X = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [2, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ]
    )
    return pd.DataFrame(X)


dfB = dfBinary()
dfM = dfMulticategory()
dfC = dfContinuous()
dfFT = datasetFT()


def test_CI():
    """test class imbalance"""
    facet = dfB[0] == "F"
    assert CI(dfB[0], facet) == approx(-1 / 6)

    facet = dfB[0] == "M"
    assert CI(dfB[0], facet) == approx(1 / 6)

    # Continuous Facet, Binary Label
    facet = dfC[0] > 1.0
    assert CI(dfC[0], facet) == approx(-1 / 3)

    facet = dfC[0] < 1.0
    assert CI(dfC[0], facet) == approx(1 / 3)

    # Multicategory Facet, Binary Label

    response = metric_one_vs_all(CI, dfM[0])
    assert response["M"] == approx(1 / 3)
    assert response["F"] == approx(1 / 4)
    assert response["O"] == approx(5 / 12)


def test_DPL():
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": [1, 1, 0, 1]})
    res = metric_one_vs_all(DPL, df["x"], df["y"], 1)
    assert res["a"] == -0.5
    assert res["b"] == 0.5
    return


def test_KL():
    res = KL(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([True, False, False, False, False, False]))
    assert res == approx(1.3219280948873624)

    res = KL(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([True, False, False, False, True, False]))
    assert res == 0.0

    # No facet selection
    res = KL(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([False, False, False, False, False, False]))
    assert res is np.nan


def test_JS():
    res = JS(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([True, False, False, False, False, False]))
    assert res == approx(0.3019448800171307)
    print(res)

    res = JS(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([True, False, False, False, True, False]))
    assert res == 0.0

    # No facet selection
    res = JS(pd.Series([1, 1, 1, 2, 2, 2]), pd.Series([False, False, False, False, False, False]))
    assert res is np.nan
    return


def test_LPnorm():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert LP(dfB[0], facet, positive_label_index) == approx(0.24243661)

    facet = dfB[0] == "M"
    assert LP(dfB[0], facet, positive_label_index) == approx(0.24243661)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    positive_label_index = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    response = metric_one_vs_all(LP, dfM[0], positive_label_index=positive_label_index)
    assert response["M"] < 1.0 and response["M"] > -1.0
    assert response["F"] < 1.0 and response["F"] > -1.0
    assert response["O"] < 1.0 and response["O"] > -1.0

    # Multicategory Facet, Multicategory Label
    labels = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    response = metric_one_vs_all(LP, dfM[0], labels=labels)
    for cat in facet.unique():
        assert response[cat][0] < 1.0 and response[cat][0] > -1.0
        assert response[cat][1] < 1.0 and response[cat][1] > -1.0
        assert response[cat][2] < 1.0 and response[cat][2] > -1.0


def test_KS():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert metric_one_vs_all(KS, dfB[0], positive_label_index=positive_label_index)["F"] == approx(0.171428571)

    facet = dfB[0] == "M"
    assert metric_one_vs_all(KS, dfB[0], positive_label_index=positive_label_index)["M"] == approx(0.171428571)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    positive_label_index = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    response = metric_one_vs_all(KS, dfM[0], positive_label_index=positive_label_index)
    assert response["M"] < 1.0 and response["M"] > -1.0
    assert response["F"] < 1.0 and response["F"] > -1.0
    assert response["O"] < 1.0 and response["O"] > -1.0

    # Multicategory Facet, Multicategory Label
    labels = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    response = metric_one_vs_all(KS, dfM[0], labels=labels)
    for cat in facet.unique():
        assert response[cat][0] < 1.0 and response[cat][0] > -1.0
        assert response[cat][1] < 1.0 and response[cat][1] > -1.0
        assert response[cat][2] < 1.0 and response[cat][2] > -1.0


def test_CDD():
    x = pd.Series(
        [
            "M",
            "M",
            "M",
            "F",
            "F",
            "F",
            "F",
            "M",
            "M",
            "M",
            "M",
            "F",
            "M",
            "M",
            "F",
            "M",
            "F",
            "F",
            "M",
            "M",
            "F",
            "M",
            "M",
            "F",
        ]
    )
    positive_label_index = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0])
    group_variable = pd.Series([1, 0, 2, 2, 1, 1, 2, 1, 1, 2, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 0, 0, 1, 1])

    response = metric_one_vs_all(CDD, x, positive_label_index=positive_label_index, group_variable=group_variable)
    assert response["F"] == approx(0.3982142857)
    assert response["M"] == approx(-0.3982142857)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    positive_label_index = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0])
    group_variable = pd.Series([1, 0, 2, 2, 1, 1, 2, 1, 1, 2, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 0, 0, 1, 1])

    response = metric_one_vs_all(KS, dfM[0], positive_label_index=positive_label_index, group_variable=group_variable)
    assert response["M"] < 1.0 and response["M"] > -1.0
    assert response["F"] < 1.0 and response["F"] > -1.0
    assert response["O"] < 1.0 and response["O"] > -1.0

    # Multicategory Facet, Multicategory Label
    group_variable = pd.Series([1, 0, 2, 2, 1, 1, 2, 1, 1, 2, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 0, 0, 1, 1])
    labels = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    response = metric_one_vs_all(KS, dfM[0], labels=labels, group_variable=group_variable)

    for cat in facet.unique():
        assert response[cat][0] < 1.0 and response[cat][0] > -1.0
        assert response[cat][1] < 1.0 and response[cat][1] > -1.0
        assert response[cat][2] < 1.0 and response[cat][2] > -1.0


def test_DPPL():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DPPL(dfB[0], facet, labels, predicted) == approx(-0.1714285714)

    facet = dfB[0] == "M"
    assert DPPL(dfB[0], facet, labels, predicted) == approx(0.17142857142)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DPPL, dfM[0], predicted_labels=predicted, labels=labels)
    assert response["M"] < 1.0 and response["M"] > -1.0
    assert response["F"] < 1.0 and response["F"] > -1.0
    assert response["O"] < 1.0 and response["O"] > -1.0

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])
    response = metric_one_vs_all(DPPL, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert response[cat][0] < 1.0 and response[cat][0] > -1.0
        assert response[cat][1] < 1.0 and response[cat][1] > -1.0
        assert response[cat][2] < 1.0 and response[cat][2] > -1.0


def test_DI():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DI(dfB[0], facet, labels, predicted) == approx(1.4285714285)

    facet = dfB[0] == "M"
    assert DI(dfB[0], facet, labels, predicted) == approx(0.700000000)

    pred_labels_zero_for_M = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    assert DI(dfB[0], dfB[0] == "F", labels, pred_labels_zero_for_M) == INFINITY
    # Check empty facet selection
    with pytest.raises(ValueError) as e:
        DI(dfB[0], dfB[0] == None, labels, predicted)
    assert str(e.value) == "DI: Facet set is empty"

    # Check empty facet selection
    with pytest.raises(ValueError) as e:
        x = Series(["A", "A"])
        labels = Series([0, 1])
        pred = Series([0, 1])
        DI(x, x == "A", labels, pred)
    assert str(e.value) == "DI: Negated facet set is empty"

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DI, dfM[0], predicted_labels=predicted, labels=labels)
    assert response["M"] != 1e10
    assert response["F"] != 1e10
    assert response["O"] != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])
    response = metric_one_vs_all(DI, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert response[cat][0] != 1e10
        assert response[cat][1] != 1e10
        assert response[cat][2] != 1e10


def test_DCA():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DCO(dfB[0], facet, labels, predicted)[0] == approx(1 / 4)

    facet = dfB[0] == "M"
    assert DCO(dfB[0], facet, labels, predicted)[0] == approx(-1 / 4)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DCO, dfM[0], predicted_labels=predicted, labels=labels)
    assert abs(response["M"][0]) != 1e10
    assert abs(response["F"][0]) != 1e10
    assert abs(response["O"][0]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])
    response = metric_one_vs_all(DCO, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert abs(response[cat][0][0]) != 1e10
        assert abs(response[cat][1][0]) != 1e10
        assert abs(response[cat][2][0]) != 1e10


def test_DCR():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DCO(dfB[0], facet, labels, predicted)[1] == approx(-1 / 3)

    facet = dfB[0] == "M"
    assert DCO(dfB[0], facet, labels, predicted)[1] == approx(1 / 3)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DCO, dfM[0], predicted_labels=predicted, labels=labels)
    assert abs(response["M"][1]) != 1e10
    assert abs(response["F"][1]) != 1e10
    assert abs(response["O"][1]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(DCO, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert abs(response[cat][0][1]) != 1e10
        assert abs(response[cat][1][1]) != 1e10
        assert abs(response[cat][2][1]) != 1e10


def test_RD():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert RD(dfB[0], facet, labels, predicted) == approx(-2 / 3)

    facet = dfB[0] == "M"
    assert RD(dfB[0], facet, labels, predicted) == approx(2 / 3)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(RD, dfM[0], predicted_labels=predicted, labels=labels)
    assert abs(response["M"]) != 1e10
    assert abs(response["F"]) != 1e10
    assert abs(response["O"]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(RD, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert abs(response[cat][0]) != 1e10
        assert abs(response[cat][1]) != 1e10
        assert abs(response[cat][2]) != 1e10


def test_DRR():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DLR(dfB[0], facet, labels, predicted)[1] == approx(-1 / 3)

    facet = dfB[0] == "M"
    assert DLR(dfB[0], facet, labels, predicted)[1] == approx(1 / 3)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DLR, dfM[0], predicted_labels=predicted, labels=labels)
    assert abs(response["M"][1]) != 1e10
    assert abs(response["F"][1]) != 1e10
    assert abs(response["O"][1]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(DLR, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert abs(response[cat][0][1]) != 1e10
        assert abs(response[cat][1][1]) != 1e10
        assert abs(response[cat][2][1]) != 1e10


def test_AD():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert AD(dfB[0], facet, labels, 1, predicted, 1) == approx(-0.3714285714)

    facet = dfB[0] == "M"
    assert AD(dfB[0], facet, labels, 1, predicted, 1) == approx(0.3714285714)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(AD, dfM[0], labels, 1, predicted, 1)
    assert abs(response["M"]) != 1e10
    assert abs(response["F"]) != 1e10
    assert abs(response["O"]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(AD, dfM[0], labels, 1, predicted, 1)
    for cat in facet.unique():
        assert abs(response[cat]) != 1e10
        assert abs(response[cat]) != 1e10
        assert abs(response[cat]) != 1e10


def test_PD():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert DLR(dfB[0], facet, labels, predicted)[0] == approx(-1 / 2)

    facet = dfB[0] == "M"
    assert DLR(dfB[0], facet, labels, predicted)[0] == approx(1 / 2)

    # Multicategory Facet, Binary Label
    facet = dfM[0]
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    response = metric_one_vs_all(DLR, dfM[0], predicted_labels=predicted, labels=labels)
    assert abs(response["M"][0]) != 1e10
    assert abs(response["F"][0]) != 1e10
    assert abs(response["O"][0]) != 1e10

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(DLR, dfM[0], predicted_labels=predicted, labels=labels)
    for cat in facet.unique():
        assert abs(response[cat][0][0]) != 1e10
        assert abs(response[cat][1][0]) != 1e10
        assert abs(response[cat][2][0]) != 1e10


def test_TE():
    # Binary Facet, Binary Label
    facet = dfB[0] == "F"
    predicted = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    labels = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    assert TE(dfB[0], facet, labels, predicted) == approx(-1 / 2)

    facet = dfB[0] == "M"
    assert TE(dfB[0], facet, labels, predicted) == approx(1 / 2)

    # Multicategory Facet, Binary Label
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])

    # For Reference: M   F   O
    # FP_a =       | 3 | 3 | 2
    # FN_a =       | 4 | 3 | 3
    # FP_d =       | 1 | 1 | 2
    # FN_d =       | 1 | 2 | 2

    response = metric_one_vs_all(TE, dfM[0], predicted_labels=predicted, labels=labels)
    assert response["M"] == approx(-1 / 3)
    assert response["F"] == approx(1)
    assert response["O"] == approx(-1 / 2)

    # Multicategory Facet, Multicategory Label
    predicted = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1])
    labels = pd.Series([2, 0, 1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2])

    response = metric_one_vs_all(TE, dfM[0], predicted_labels=predicted, labels=labels)


def test_FT():
    dfFT = datasetFT()
    facet = dfFT[0]

    predicted = pd.Series([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
    labels = pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0])

    assert FT(dfFT, facet, labels, predicted) == approx(-0.5384615384615384)

    # Multicategory Facet, Binary Label
    multDF = datasetFTMult()
    predicted = pd.Series([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0])

    response = metric_one_vs_all(FT, multDF[0], predicted_labels=predicted, labels=labels, dataset=multDF)

    assert response[0] == approx(5 / 8)
    assert response[1] == approx(-3 / 5)
    assert response[2] == approx(1 / 2)
