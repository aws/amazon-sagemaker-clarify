import pandas as pd
from famly.bias.report import (
    ProblemType,
    problem_type,
    bias_report,
    FacetColumn,
    LabelColumn,
    fetch_metrics_to_run,
    StageType,
)
from famly.bias.metrics import PRETRAINING_METRICS, POSTTRAINING_METRICS, CI, DPL, KL, KS, DPPL, DI, DCO, RD
from typing import List, Any


def dataframe(data: List[List[Any]]):
    df = pd.DataFrame(data, columns=["x", "y", "z", "yhat"])
    return df


df_cat = dataframe([["a", 1, 1, 1], ["b", 1, 1, 0], ["b", 0, 1, 0], ["c", 0, 0, 1]])
df_cont = dataframe([[1, 1, 1, 1], [2, 1, 1, 0], [3, 0, 0, 0], [2, 0, 1, 1]])


def test_report_category_data():
    # test the bias_report function on the category data
    #
    report = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn(df_cat["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cat["yhat"], [1]),
        group_variable=df_cat["z"],
    )
    assert isinstance(report, list)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report[0].items():
        if isinstance(v, dict):
            assert len(v["value"]) == 3

    result = {
        "CDDL": {
            "description": "Conditional Demographic Disparity in Labels (CDDL)",
            "value": {"a": -0.375, "b": 0.375, "c": 0.25},
        },
        "CI": {"description": "Class Imbalance (CI)", "value": {"a": 0.5, "b": 0.0, "c": 0.5}},
        "DPL": {
            "description": "Difference in Positive Proportions in Labels (DPL)",
            "value": {"a": -0.6666666666666667, "b": 0.0, "c": 0.6666666666666666},
        },
        "JS": {
            "description": "Jensen-Shannon Divergence (JS)",
            "value": {"a": 0.2789960722619452, "b": 0.0, "c": 0.2789960722619452},
        },
        "KL": {
            "description": "Kullback-Liebler Divergence (KL)",
            "value": {"a": 1.584962500721156, "b": 0.0, "c": 1.584962500721156},
        },
        "KS": {
            "description": "Kolmogorov-Smirnov Distance (KS)",
            "value": {"a": 0.6666666666666667, "b": 0.0, "c": 0.6666666666666667},
        },
        "LP": {"description": "L-p Norm (LP)", "value": {"a": 0.6666666666666667, "b": 0.0, "c": 0.6666666666666667}},
        "TVD": {
            "description": "Total Variation Distance (TVD)",
            "value": {"a": 0.33333333333333337, "b": 0.0, "c": 0.33333333333333337},
        },
        "label_value": "(0, 1]",
    }
    assert report[0] == result


def test_report_continuous_data():
    #   test the bias_report function on the category data
    #
    report = bias_report(
        df_cont,
        FacetColumn("x"),
        LabelColumn("y", df_cont["y"]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cont["yhat"], [1]),
        group_variable=df_cont["z"],
    )
    assert isinstance(report, list)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report[0].items():
        if isinstance(v, dict):
            assert len(v["value"]) == 1
    result = {
        "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": {"(2, 3]": 0.25}},
        "CI": {"description": "Class Imbalance (CI)", "value": {"(2, 3]": 0.5}},
        "DPL": {
            "description": "Difference in Positive Proportions in Labels (DPL)",
            "value": {"(2, 3]": 0.6666666666666666},
        },
        "JS": {"description": "Jensen-Shannon Divergence (JS)", "value": {"(2, 3]": 0.2789960722619452}},
        "KL": {"description": "Kullback-Liebler Divergence (KL)", "value": {"(2, 3]": 1.584962500721156},},
        "KS": {"description": "Kolmogorov-Smirnov Distance (KS)", "value": {"(2, 3]": 0.6666666666666667}},
        "LP": {"description": "L-p Norm (LP)", "value": {"(2, 3]": 0.6666666666666667}},
        "TVD": {"description": "Total Variation Distance (TVD)", "value": {"(2, 3]": 0.33333333333333337}},
        "label_value": "(0, 1]",
    }
    assert report[0] == result


def test_label_values():
    df = dataframe([["a", "p", 1, 1], ["b", "q", 1, 0], ["b", "r", 1, 0], ["c", "p", 0, 1], ["c", "q", 0, 1]])
    # when no explicit label values are given for categorical data
    report = bias_report(
        df_cont,
        FacetColumn("x"),
        LabelColumn(df["y"]),
        StageType.PRE_TRAINING,
        LabelColumn(df_cont["yhat"], [1]),
        metrics=["DPL", "CDDL"],
        group_variable=(df_cont["z"]),
    )

    assert isinstance(report[0], dict)
    assert len(report) > 0
    expected_result = [
        {
            "DPL": {
                "description": "Difference in Positive Proportions in Labels (DPL)",
                "value": {"(2, 3]": 0.6666666666666666},
            },
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": {"(2, 3]": 0.25}},
            "label_value": "p",
        },
        {
            "DPL": {
                "description": "Difference in Positive Proportions in Labels (DPL)",
                "value": {"(2, 3]": 0.3333333333333333},
            },
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": {"(2, 3]": 0.25}},
            "label_value": "q",
        },
        {
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": {"(2, 3]": -1.0}},
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": {"(2, 3]": -0.25}},
            "label_value": "r",
        },
    ]
    assert report == expected_result

    # when  explicit label values are given for categorical data
    report_1 = bias_report(
        df_cont,
        FacetColumn("x"),
        LabelColumn(df["y"], ["p", "q"]),
        StageType.PRE_TRAINING,
        LabelColumn(df_cont["yhat"], [1]),
        metrics=["DPL", "CDDL"],
        group_variable=(df_cont["z"]),
    )

    assert isinstance(report_1[0], dict)
    expected_result_1 = [
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": {"(2, 3]": 0.25}},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": {"(2, 3]": 1.0}},
            "label_value": "p,q",
        }
    ]
    assert report_1 == expected_result_1


def test_fetch_metrics_to_run():
    """
    test the list of callable metric functions to be run
    """

    input_metrics_1 = ["CI", "DPL", "KL", "KS"]
    metrics_to_run = fetch_metrics_to_run(PRETRAINING_METRICS, input_metrics_1)
    print(metrics_to_run, PRETRAINING_METRICS)
    assert metrics_to_run == [CI, DPL, KL, KS]

    input_metrics_2 = ["DPPL", "DI", "DCO", "RD"]
    metrics_to_run = fetch_metrics_to_run(POSTTRAINING_METRICS, input_metrics_2)
    assert metrics_to_run == [DPPL, DI, DCO, RD]


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
