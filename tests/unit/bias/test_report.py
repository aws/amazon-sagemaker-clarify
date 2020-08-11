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
        LabelColumn(df_cat["y"], [1]),
        StageType.PRE_TRAINING,
        LabelColumn(df_cat["yhat"], [1]),
        group_variable=(df_cat["z"]),
    )
    assert isinstance(report, dict)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report.items():
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
    }
    assert report == result


def test_report_continuous_data():
    #   test the bias_report function on the category data
    #
    report = bias_report(
        df_cont,
        FacetColumn("x"),
        LabelColumn(df_cont["y"]),
        StageType.PRE_TRAINING,
        LabelColumn(df_cont["yhat"], [1]),
        group_variable=(df_cont["z"]),
    )
    assert isinstance(report, dict)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report.items():
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
    }
    assert report == result


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
