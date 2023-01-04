from smclarify.bias.metrics import basic_stats

from .test_metrics import dfBinary

from pytest import approx
import pandas as pd

(dfB, dfB_label, dfB_pos_label_idx, dfB_pred_label, dfB_pos_pred_label_idx) = dfBinary()


def test_proportion():
    sensitive_facet_index = dfB[0] == "F"
    assert basic_stats.proportion(sensitive_facet_index) == approx(7 / 12)

    sensitive_facet_index = dfB[0] == "M"
    assert basic_stats.proportion(sensitive_facet_index) == approx(5 / 12)


def test_confusion_matrix():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    TP = approx(2 / 7.0)
    TN = approx(2 / 7.0)
    FP = approx(2 / 7.0)
    FN = approx(1 / 7.0)
    assert [TP, FP, FN, TN] == basic_stats.confusion_matrix(
        dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx
    )

    sensitive_facet_index = dfB[0] == "M"
    TP = 0
    TN = approx(1 / 5.0)
    FP = approx(2 / 5.0)
    FN = approx(2 / 5.0)
    assert [TP, FP, FN, TN] == basic_stats.confusion_matrix(
        dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx
    )


def test_observed_label_distribution():
    sensitive_facet_index = dfB[0] == "F"
    label_dist = [approx(3 / 7.0), approx(4 / 7.0)]
    assert label_dist == basic_stats.observed_label_distribution(dfB[0], sensitive_facet_index, dfB_pos_label_idx)

    sensitive_facet_index = dfB[0] == "M"
    label_dist = [approx(2 / 5.0), approx(3 / 5.0)]
    assert label_dist == basic_stats.observed_label_distribution(dfB[0], sensitive_facet_index, dfB_pos_label_idx)


def test_performance_metrics():
    TP = 2
    TN = 2
    FP = 1
    FN = 0

    result = [
        basic_stats.accuracy(TP, FP, TN, FN),
        basic_stats.PPL(TP, FP, TN, FN),
        basic_stats.PNL(TP, FP, TN, FN),
        basic_stats.recall(TP, FN),
        basic_stats.specificity(TN, FP),
        basic_stats.precision(TP, FP),
        basic_stats.rejection_rate(TN, FN),
        basic_stats.conditional_acceptance(TP, FP, FN),
        basic_stats.conditional_rejection(FP, TN, FN),
        basic_stats.f1_score(TP, FP, FN),
    ]
    expected = [
        approx(4 / 5.0),
        approx(3 / 5.0),
        approx(2 / 5.0),
        approx(2 / 2.0),
        approx(2 / 3.0),
        approx(2 / 3.0),
        approx(2 / 2.0),
        approx(2 / 3.0),
        approx(3 / 2.0),
        4 / 5.0,
    ]

    assert expected == result


def test_multicategorical_confusion_matrix():
    expected_value = {"0": {"0": 3, "1": 4}, "1": {"0": 3, "1": 2}}
    assert basic_stats.multicategory_confusion_matrix(dfB_label, dfB_pred_label) == expected_value

    # Confusion matrix will be of shape (len(unique_label_values), len(unique_label_values))
    label_series = pd.Series([1])
    predicted_label_series = pd.Series([2])
    expected_value = {"1": {"1": 0.0}}
    assert basic_stats.multicategory_confusion_matrix(label_series, predicted_label_series) == expected_value

    # Strings
    df = pd.DataFrame(
        [
            ("a", "white", 1, "red"),
            ("b", "white", 1, "blue"),
            ("b", "blue", 1, "blue"),
            ("b", "blue", 0, "red"),
            ("a", "green", 1, "white"),
            ("b", "white", 1, "white"),
            ("b", "white", 1, "green"),
            ("b", "white", 0, "white"),
        ]
    )
    df.columns = ["x", "y", "z", "yhat"]

    expected_value = {
        "blue": {"blue": 1.0, "green": 0.0, "white": 0.0},
        "green": {"blue": 0.0, "green": 0.0, "white": 1.0},
        "white": {"blue": 1.0, "green": 1.0, "white": 2.0},
    }

    assert basic_stats.multicategory_confusion_matrix(df["y"], df["yhat"]) == expected_value

    df = pd.DataFrame(
        [
            ("a", 1, 1, 1),
            ("b", 1, 1, 0),
            ("b", 0, 1, 0),
            ("b", 0, 0, 1),
            ("a", 2, 1, 3),
            ("b", 3, 1, 3),
            ("b", 3, 1, 2),
            ("b", 1, 0, 3),
        ]
    )

    df.columns = ["x", "y", "z", "yhat"]

    expected_value = {
        "0": {"0": 1.0, "1": 1.0, "2": 0.0, "3": 0.0},
        "1": {"0": 1.0, "1": 1.0, "2": 0.0, "3": 1.0},
        "2": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0},
        "3": {"0": 0.0, "1": 0.0, "2": 1.0, "3": 1.0},
    }

    basic_stats.multicategory_confusion_matrix(df["y"], df["yhat"]) == expected_value
