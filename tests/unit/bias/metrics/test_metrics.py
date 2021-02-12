# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

from smclarify.bias.metrics import AD, CDDL, CI, DAR, DCA, DCR, DI, DPL, DRR, FT, JS, KL, LP, RD, TE, KS
from smclarify.bias.metrics import metric_one_vs_all
from smclarify.bias.metrics.constants import INFINITY
from pytest import approx
import pandas as pd
from pandas import Series
import math
import numpy as np
import pytest

DATASET_PDF = pd.DataFrame(
    np.array(
        [
            ["a", 0, False, True],
            ["b", 0, False, False],
            ["b", 1, True, False],
            ["c", 1, True, True],
            ["a", 2, True, True],
            ["a", 1, True, True],
            ["b", 0, False, False],
            ["c", 1, True, True],
            ["b", 2, True, False],
            ["c", 2, True, True],
            ["b", 0, False, False],
            ["b", 2, True, False],
        ]
    ),
    columns=["x", "label", "positive_label_index", "sensitive_facet_index"],
)


def dfBinary():
    """
    :return: a tuple of below objects
        dataframe with one column which contains Binary categorical data (length 12)
        label
        positive label index
        predicted label
        positive predicted label index
    """
    data = [["M"], ["F"], ["F"], ["M"], ["F"], ["M"], ["F"], ["F"], ["M"], ["M"], ["F"], ["F"]]
    df = pd.DataFrame(data)
    label = pd.Series([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    positive_label_index = label == 1
    predicted_label = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    positive_predicted_label_index = predicted_label == 1
    return (df, label, positive_label_index, predicted_label, positive_predicted_label_index)


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
            [0, 0, 0, 0, True, 1, 1],
            [1, 0, 0, 0, True, 0, 1],
            [1, 0, 1, 0, True, 0, 1],
            [0, 0, 0, 0, False, 1, 1],
            [1, 0, 0, 1, True, 0, 1],
            [0, 0, 1, 0, True, 1, 1],
            [1, 0, 0, 0, True, 1, 1],
            [1, 1, 0, 0, True, 1, 1],
            [0, 0, 1, 0, True, 1, 1],
            [1, 0, 1, 1, True, 1, 0],
            [1, 0, 0, 0, True, 1, 0],
            [1, 0, 1, 0, True, 1, 1],
            [1, 0, 0, 0, False, 1, 1],
            [0, 0, 0, 0, True, 1, 1],
            [0, 0, 1, 0, True, 0, 1],
            [0, 0, 1, 0, True, 1, 1],
            [0, 1, 0, 1, False, 0, 1],
            [1, 0, 0, 0, False, 1, 1],
            [0, 0, 1, 0, False, 1, 1],
            [1, 0, 0, 0, True, 1, 0],
            [1, 0, 1, 0, False, 1, 1],
            [0, 1, 0, 0, False, 1, 1],
            [1, 0, 1, 0, True, 0, 1],
            [0, 0, 0, 1, True, 1, 0],
        ]
    )

    return pd.DataFrame(X)


def datasetFT_small_samples():
    """
    With facet as column 0, the FT selects only 3 rows by ~facet
    """
    X = np.array(
        [
            [0, 0, 0, 0, True, 1, 1],
            [1, 0, 0, 0, True, 0, 1],
            [1, 0, 1, 0, True, 0, 1],
            [1, 0, 0, 1, True, 0, 1],
            [1, 0, 0, 0, True, 1, 1],
            [1, 1, 0, 0, True, 1, 1],
            [1, 0, 1, 1, True, 1, 0],
            [1, 0, 0, 0, True, 1, 0],
            [1, 0, 1, 0, True, 1, 1],
            [1, 0, 0, 0, False, 1, 1],
            [1, 0, 0, 0, False, 1, 1],
            [0, 0, 1, 0, False, 1, 1],
            [1, 0, 0, 0, True, 1, 0],
            [1, 0, 1, 0, False, 1, 1],
            [1, 0, 1, 0, True, 0, 1],
            [0, 0, 0, 1, True, 1, 0],
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


(dfB, dfB_label, dfB_pos_label_idx, dfB_pred_label, dfB_pos_pred_label_idx) = dfBinary()
dfM = dfMulticategory()
dfC = dfContinuous()
dfFT = datasetFT()


def test_CI():
    sensitive_facet_index = dfB[0] == "F"
    assert CI(dfB[0], sensitive_facet_index) == approx(-1 / 6)

    sensitive_facet_index = dfB[0] == "M"
    assert CI(dfB[0], sensitive_facet_index) == approx(1 / 6)

    # Continuous Facet, Binary Label
    sensitive_facet_index = dfC[0] > 1.0
    assert CI(dfC[0], sensitive_facet_index) == approx(-1 / 3)

    sensitive_facet_index = dfC[0] < 1.0
    assert CI(dfC[0], sensitive_facet_index) == approx(1 / 3)

    # Multicategory Facet, Binary Label

    response = metric_one_vs_all(CI, dfM[0])
    assert response["M"] == approx(1 / 3)
    assert response["F"] == approx(1 / 4)
    assert response["O"] == approx(5 / 12)


def test_DPL():
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": [1, 1, 0, 1]})
    res = metric_one_vs_all(DPL, df["x"], label=df["y"], positive_label_index=(df["y"] == 1))
    assert res["a"] == -0.5
    assert res["b"] == 0.5
    return


def test_KL():
    res = KL(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, False, False]))
    assert res == approx(-0.366516)

    res = KL(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, True, False]))
    assert res == 0.0

    with pytest.raises(ValueError) as e:
        KL(pd.Series([True, True, True, False, False, False]), pd.Series([False, False, False, False, False, False]))
    assert str(e.value) == "No instance of common facet found, dataset may be too small"

    # multi-facet, multi-category case
    sensitive_facet_index: pd.Series = DATASET_PDF["x"] == "a"
    sensitive_facet_index += DATASET_PDF["x"] == "c"

    positive_label_index: pd.Series = DATASET_PDF["label"] == "1"
    positive_label_index += DATASET_PDF["label"] == "2"
    res = KL(positive_label_index, sensitive_facet_index)
    assert res == approx(0.2938933)


def test_KS():
    df = pd.DataFrame([["1", "a"], ["0", "a"], ["0", "b"], ["1", "b"], ["1", "b"]], columns=["label", "x"])
    result = KS(df["label"], df["x"] == "b")
    assert result == approx(0.16666666)

    result = KS(DATASET_PDF["label"], DATASET_PDF["x"] != "b")
    assert result == approx(0.33333333)

    result = KS(DATASET_PDF["positive_label_index"], DATASET_PDF["x"] != "b")
    assert result == approx(0.33333333)


def test_JS():
    res = JS(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, False, False]))
    assert res == approx(0.06641431438228168)

    res = JS(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, True, False]))
    assert res == 0.0

    with pytest.raises(ValueError) as e:
        JS(pd.Series([True, True, True, False, False, False]), pd.Series([False, False, False, False, False, False]))
    assert str(e.value) == "No instance of common facet found, dataset may be too small"

    # multi-facet, multi-category case
    sensitive_facet_index: pd.Series = DATASET_PDF["x"] == "a"
    sensitive_facet_index += DATASET_PDF["x"] == "c"

    positive_label_index: pd.Series = DATASET_PDF["label"] == "1"
    positive_label_index += DATASET_PDF["label"] == "2"
    res = JS(positive_label_index, sensitive_facet_index)
    assert res == approx(0.06465997)

    # Calculate JS manually.
    res = JS(pd.Series([True, True, True, True, False, False]), pd.Series([True, False, False, False, True, False]))
    Pa = np.array([0.5, 0.5])
    Pd = np.array([0.25, 0.75])
    P = np.array([0.375, 0.625])
    expected_result = 0.5 * (
        (Pa[0] * math.log(Pa[0] / P[0]))
        + (Pa[1] * math.log(Pa[1] / P[1]))
        + (Pd[0] * math.log(Pd[0] / P[0]))
        + (Pd[1] * math.log(Pd[1] / P[1]))
    )
    assert res == approx(expected_result)


def test_LP():
    res = LP(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, False, False]))
    assert res == approx(0.6)

    res = LP(pd.Series([True, True, True, False, False, False]), pd.Series([True, False, False, False, True, False]))
    assert res == 0.0

    # No facet selection
    with pytest.raises(ValueError) as e:
        LP(pd.Series([True, True, True, False, False, False]), pd.Series([False, False, False, False, False, False]))
    assert str(e.value) == "No instance of common facet found, dataset may be too small"

    # multi-facet, multi-category case
    sensitive_facet_index: pd.Series = DATASET_PDF["x"] == "a"
    sensitive_facet_index += DATASET_PDF["x"] == "c"

    positive_label_index: pd.Series = DATASET_PDF["label"] == "1"
    positive_label_index += DATASET_PDF["label"] == "2"
    res = LP(positive_label_index, sensitive_facet_index)
    assert res == approx(0.471404520)

    return


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

    response = metric_one_vs_all(CDDL, x, positive_label_index=positive_label_index == 1, group_variable=group_variable)
    assert response["F"] == approx(0.3982142857)
    assert response["M"] == approx(-0.3982142857)


def test_DI():
    # Binary Facet, Binary Label
    sensitive_facet_index_f = dfB[0] == "F"
    assert DI(dfB[0], sensitive_facet_index_f, dfB_pos_pred_label_idx) == approx(10 / 7)

    sensitive_facet_index_m = dfB[0] == "M"
    assert DI(dfB[0], sensitive_facet_index_m, dfB_pos_pred_label_idx) == approx(7 / 10)

    predicted_labels_zero_for_M = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
    positive_predicted_labels_index_zero_for_M = predicted_labels_zero_for_M == 1
    assert DI(dfB[0], sensitive_facet_index_f, positive_predicted_labels_index_zero_for_M) == INFINITY
    # Check empty facet selection
    with pytest.raises(ValueError) as e:
        DI(dfB[0], dfB[0] == None, positive_predicted_labels_index_zero_for_M)
    assert str(e.value) == "Facet set is empty"

    # Check empty facet selection
    with pytest.raises(ValueError) as e:
        x = Series(["A", "A"])
        pred = Series([0, 1])
        DI(x, x == "A", pred == 1)
    assert str(e.value) == "Negated facet set is empty"


def test_DCA():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert DCA(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(1 / 4)

    sensitive_facet_index = dfB[0] == "M"
    assert DCA(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-1 / 4)


def test_DCR():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert DCR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(1 / 3)

    sensitive_facet_index = dfB[0] == "M"
    assert DCR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-1 / 3)


def test_RD():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert RD(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-2 / 3)

    sensitive_facet_index = dfB[0] == "M"
    assert RD(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(2 / 3)


def test_DRR():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert DRR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(1 / 3)

    sensitive_facet_index = dfB[0] == "M"
    assert DRR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-1 / 3)


def test_DRR_zero():
    # Binary Facet, Binary Label
    # All M have 1 prediction
    predicted_label = pd.Series([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    positive_predicted_label_index = predicted_label == 1

    sensitive_facet_index = dfB[0] == "F"
    assert DRR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, positive_predicted_label_index) == approx(0.5)

    sensitive_facet_index = dfB[0] == "M"
    assert DRR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, positive_predicted_label_index) == approx(-0.5)


def test_AD():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert AD(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-13 / 35)

    sensitive_facet_index = dfB[0] == "M"
    assert AD(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(13 / 35)


def test_DAR():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert DAR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-1 / 2)

    sensitive_facet_index = dfB[0] == "M"
    assert DAR(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(1 / 2)


def test_TE():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    assert TE(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(-1 / 2)

    sensitive_facet_index = dfB[0] == "M"
    assert TE(dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx) == approx(1 / 2)


def test_FT():
    dfFT = datasetFT()
    sensitive_facet_index = dfFT[0]
    predicted = pd.Series([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    assert FT(dfFT, sensitive_facet_index == 1, predicted == 1) == approx(-0.23076923076923078)

    dfFT[3] = dfFT[3].apply(lambda x: "a")
    with pytest.raises(ValueError) as e:
        FT(dfFT, sensitive_facet_index == 1, predicted == 1)
    assert str(e.value) == "FlipTest does not support non-numeric columns"


def test_FT_small_samples():
    dfFT = datasetFT_small_samples()
    sensitive_facet_index = dfFT[0]
    predicted = pd.Series([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    assert FT(dfFT, sensitive_facet_index == 1, predicted == 1) == approx(-0.15384615384615385)
