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


df_cat = dataframe([["a", 1, 1, 1], ["b", 1, 1, 0], ["b", 0, 1, 0], ["b", 0, 0, 1]])
df_cont = dataframe([[1, 1, 1, 1], [2, 1, 1, 0], [3, 0, 0, 0], [2, 0, 1, 1], [0, 0, 1, 1]])


def test_report_category_data():
    # test the bias_report function on the category data
    #
    # pre training bias metrics
    pretraining_report = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cat["yhat"]),
        group_variable=df_cat["z"],
    )
    assert isinstance(pretraining_report, list)
    assert len(pretraining_report) > 0

    result = [
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": -0.375},
            "CI": {"description": "Class Imbalance (CI)", "value": 0.5},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": -0.6666666666666667},
            "JS": {"description": "Jensen-Shannon Divergence (JS)", "value": 0.2789960722619452},
            "KL": {"description": "Kullback-Liebler Divergence (KL)", "value": 1.584962500721156},
            "KS": {"description": "Kolmogorov-Smirnov Distance (KS)", "value": 0.6666666666666667},
            "LP": {"description": "L-p Norm (LP)", "value": 0.6666666666666667},
            "TVD": {"description": "Total Variation Distance (TVD)", "value": 0.33333333333333337},
            "value_or_threshold": "a",
        },
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": 0.625},
            "CI": {"description": "Class Imbalance (CI)", "value": -0.5},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": 0.6666666666666667},
            "JS": {"description": "Jensen-Shannon Divergence (JS)", "value": 0.2789960722619452},
            "KL": {"description": "Kullback-Liebler Divergence (KL)", "value": -0.5283208335737187},
            "KS": {"description": "Kolmogorov-Smirnov Distance (KS)", "value": 0.6666666666666667},
            "LP": {"description": "L-p Norm (LP)", "value": 0.6666666666666667},
            "TVD": {"description": "Total Variation Distance (TVD)", "value": 0.33333333333333337},
            "value_or_threshold": "b",
        },
    ]
    assert pretraining_report == result

    # post training bias metrics
    posttraining_report = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cat["yhat"]),
        metrics=["AD", "DI", "DPPL", "RD"],
        group_variable=df_cat["z"],
    )
    assert isinstance(posttraining_report, list)
    assert len(posttraining_report) > 0
    expected_result_1 = [
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": -0.6666666666666667},
            "DI": {"description": "Disparate Impact (DI)", "value": 3.0},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": -0.6666666666666667,
            },
            "RD": {"description": "Recall Difference (RD)", "value": -1.0},
            "value_or_threshold": "a",
        },
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": 0.6666666666666667},
            "DI": {"description": "Disparate Impact (DI)", "value": 0.3333333333333333},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": 0.6666666666666667,
            },
            "RD": {"description": "Recall Difference (RD)", "value": 1.0},
            "value_or_threshold": "b",
        },
    ]
    assert posttraining_report == expected_result_1


def test_report_continuous_data():
    #   test the bias_report function on the category data
    #
    # pre training bias metrics
    pretraining_report = bias_report(
        df_cont,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cont["yhat"]),
        group_variable=df_cont["z"],
    )
    assert isinstance(pretraining_report, list)
    assert len(pretraining_report) > 0
    result = [
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": 0.2},
            "CI": {"description": "Class Imbalance (CI)", "value": 0.6},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": 0.5},
            "JS": {"description": "Jensen-Shannon Divergence (JS)", "value": 0.20983242268450672},
            "KL": {"description": "Kullback-Liebler Divergence (KL)", "value": 1.0},
            "KS": {"description": "Kolmogorov-Smirnov Distance (KS)", "value": 0.5},
            "LP": {"description": "L-p Norm (LP)", "value": 0.5},
            "TVD": {"description": "Total Variation Distance (TVD)", "value": 0.25},
            "value_or_threshold": "(2, 3]",
        }
    ]
    assert pretraining_report == result

    posttraining_report = bias_report(
        df_cont,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cont["yhat"]),
        group_variable=df_cont["z"],
    )
    assert isinstance(posttraining_report, list)
    assert len(posttraining_report) > 0
    expected_result_1 = [
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": -0.75},
            "DCO": {"description": "Difference in Conditional Outcomes (DCO)", "value": (float("-inf"), -1.0)},
            "DI": {"description": "Disparate Impact (DI)", "value": 0.0},
            "DLR": {"description": "Difference in Label Rates (DLR)", "value": (float("-inf"), 1.0)},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": 0.75,
            },
            "FT": {"description": "Flip Test (FT)", "value": 0.0},
            "RD": {"description": "Recall Difference (RD)", "value": float("-inf")},
            "TE": {"description": "Treatment Equality (TE)", "value": float("inf")},
            "value_or_threshold": "(2, 3]",
        }
    ]
    assert posttraining_report == expected_result_1


def test_label_values():
    """
    Test bias metrics for multiple label values
    """
    df = dataframe([["a", "p", 1, "p"], ["b", "q", 1, "p"], ["b", "r", 1, "q"], ["c", "p", 0, "p"], ["c", "q", 0, "p"]])
    # when  explicit label values are given for categorical data
    # Pre training bias metrics
    pretraining_report = bias_report(
        df,
        FacetColumn("x"),
        LabelColumn("y", df["y"], ["p", "q"]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["DPL", "CDDL"],
        group_variable=df["z"],
    )

    assert isinstance(pretraining_report[0], dict)
    expected_result_1 = [
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": -0.3},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": -0.25},
            "value_or_threshold": "a",
        },
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": 0.3},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": 0.5},
            "value_or_threshold": "b",
        },
        {
            "CDDL": {"description": "Conditional Demographic Disparity in Labels (CDDL)", "value": -0.4},
            "DPL": {"description": "Difference in Positive Proportions in Labels (DPL)", "value": -0.33333333333333337},
            "value_or_threshold": "c",
        },
    ]
    assert pretraining_report == expected_result_1

    # post training bias metrics
    posttraining_report = bias_report(
        df,
        FacetColumn("x"),
        LabelColumn("y", df["y"], ["p", "q"]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["AD", "DI", "DPPL", "RD", "DLR"],
        group_variable=df["z"],
    )

    assert isinstance(posttraining_report[0], dict)
    expected_result_2 = [
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": -0.25},
            "DI": {"description": "Disparate Impact (DI)", "value": 1.0},
            "DLR": {"description": "Difference in Label Rates (DLR)", "value": (-0.25, 0)},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": 0.0,
            },
            "RD": {"description": "Recall Difference (RD)", "value": 0.0},
            "value_or_threshold": "a",
        },
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": 0.5},
            "DI": {"description": "Disparate Impact (DI)", "value": 1.0},
            "DLR": {"description": "Difference in Label Rates (DLR)", "value": (0.5, 0)},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": 0.0,
            },
            "RD": {"description": "Recall Difference (RD)", "value": 0.0},
            "value_or_threshold": "b",
        },
        {
            "AD": {"description": "Accuracy Difference (AD)", "value": -0.33333333333333337},
            "DI": {"description": "Disparate Impact (DI)", "value": 1.0},
            "DLR": {"description": "Difference in Label Rates (DLR)", "value": (-0.33333333333333337, 0)},
            "DPPL": {
                "description": '"Difference in Positive Proportions in Predicted ' 'Labels (DPPL)")',
                "value": 0.0,
            },
            "RD": {"description": "Recall Difference (RD)", "value": 0.0},
            "value_or_threshold": "c",
        },
    ]
    assert posttraining_report == expected_result_2


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
