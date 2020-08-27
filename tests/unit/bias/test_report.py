import pandas as pd
import pytest
from famly.bias.report import (
    ProblemType,
    problem_type,
    bias_report,
    FacetColumn,
    LabelColumn,
    fetch_metrics_to_run,
    StageType,
)
from famly.bias.metrics import PRETRAINING_METRICS, POSTTRAINING_METRICS, CI, DPL, KL, KS, DPPL, DI, DCA, DCR, RD
from famly.bias.metrics import common
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
            "metrics": [
                {"description": "Class Imbalance (CI)", "name": "CI", "value": 0.5},
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": -0.6666666666666667,
                },
                {"description": "Kullback-Liebler Divergence (KL)", "name": "KL", "value": 1.584962500721156},
                {"description": "Jensen-Shannon Divergence (JS)", "name": "JS", "value": 0.2789960722619452},
                {"description": "L-p Norm (LP)", "name": "LP", "value": 0.6666666666666667},
                {"description": "Total Variation Distance (TVD)", "name": "TVD", "value": 0.33333333333333337},
                {"description": "Kolmogorov-Smirnov Distance (KS)", "name": "KS", "value": 0.6666666666666667},
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": -0.375,
                },
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {"description": "Class Imbalance (CI)", "name": "CI", "value": -0.5},
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": 0.6666666666666667,
                },
                {"description": "Kullback-Liebler Divergence (KL)", "name": "KL", "value": -0.5283208335737187},
                {"description": "Jensen-Shannon Divergence (JS)", "name": "JS", "value": 0.2789960722619452},
                {"description": "L-p Norm (LP)", "name": "LP", "value": 0.6666666666666667},
                {"description": "Total Variation Distance (TVD)", "name": "TVD", "value": 0.33333333333333337},
                {"description": "Kolmogorov-Smirnov Distance (KS)", "name": "KS", "value": 0.6666666666666667},
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": 0.625,
                },
            ],
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
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": -0.6666666666666667,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 3.0},
                {"description": "Recall Difference (RD)", "name": "RD", "value": -1.0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": -0.6666666666666667},
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.6666666666666667,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 0.3333333333333333},
                {"description": "Recall Difference (RD)", "name": "RD", "value": 1.0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": 0.6666666666666667},
            ],
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
            "metrics": [
                {"description": "Class Imbalance (CI)", "name": "CI", "value": 0.6},
                {"description": "Difference in Positive Proportions in Labels " "(DPL)", "name": "DPL", "value": 0.5},
                {"description": "Kullback-Liebler Divergence (KL)", "name": "KL", "value": 1.0},
                {"description": "Jensen-Shannon Divergence (JS)", "name": "JS", "value": 0.20983242268450672},
                {"description": "L-p Norm (LP)", "name": "LP", "value": 0.5},
                {"description": "Total Variation Distance (TVD)", "name": "TVD", "value": 0.25},
                {"description": "Kolmogorov-Smirnov Distance (KS)", "name": "KS", "value": 0.5},
                {"description": "Conditional Demographic Disparity in Labels " "(CDDL)", "name": "CDDL", "value": 0.2},
            ],
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
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.75,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 0.0},
                {"description": "Difference in Conditional Acceptance (DCA)", "name": "DCA", "value": float("-inf")},
                {"description": "Difference in Conditional Rejection (DCR)", "name": "DCR", "value": -1.0},
                {"description": "Recall Difference (RD)", "name": "RD", "value": float("-inf")},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": float("-inf")},
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": 1.0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": -0.75},
                {
                    "description": "Conditional Demographic Disparity in Predicted " "Labels (CDDPL)",
                    "name": "CDDPL",
                    "value": 0.2,
                },
                {"description": "Treatment Equality (TE)", "name": "TE", "value": float("inf")},
                {"description": "Flip Test (FT)", "name": "FT", "value": 0.0},
            ],
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
            "metrics": [
                {"description": "Difference in Positive Proportions in Labels " "(DPL)", "name": "DPL", "value": -0.25},
                {"description": "Conditional Demographic Disparity in Labels " "(CDDL)", "name": "CDDL", "value": -0.3},
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {"description": "Difference in Positive Proportions in Labels " "(DPL)", "name": "DPL", "value": 0.5},
                {"description": "Conditional Demographic Disparity in Labels " "(CDDL)", "name": "CDDL", "value": 0.3},
            ],
            "value_or_threshold": "b",
        },
        {
            "metrics": [
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": -0.33333333333333337,
                },
                {"description": "Conditional Demographic Disparity in Labels " "(CDDL)", "name": "CDDL", "value": -0.4},
            ],
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
        metrics=["AD", "DI", "DPPL", "RD", "DAR", "DRR"],
        group_variable=df["z"],
    )

    assert isinstance(posttraining_report[0], dict)
    expected_result_2 = [
        {
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.0,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 1.0},
                {"description": "Recall Difference (RD)", "name": "RD", "value": 0.0},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": -0.25},
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": 0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": -0.25},
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.0,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 1.0},
                {"description": "Recall Difference (RD)", "name": "RD", "value": 0.0},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": 0.5},
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": 0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": 0.5},
            ],
            "value_or_threshold": "b",
        },
        {
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.0,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 1.0},
                {"description": "Recall Difference (RD)", "name": "RD", "value": 0.0},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": -0.33333333333333337},
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": 0},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": -0.33333333333333337},
            ],
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

    input_metrics_2 = ["DPPL", "DI", "DCA", "DCR", "RD"]
    metrics_to_run = fetch_metrics_to_run(POSTTRAINING_METRICS, input_metrics_2)
    assert metrics_to_run == [DPPL, DI, DCA, DCR, RD]


def test_partial_bias_report():
    """
    Test that bias report is generated in for partial metrics when errors occur to compute some metrics
    """
    df = dataframe([[1, 1, 1, 1], [2, 1, 1, 0], [3, 0, 0, 0], [2, 0, 1, 1], [0, 0, 1, 1]])
    # pre training bias metrics
    pretraining_report = bias_report(
        df,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cont["yhat"]),
        metrics=["CI", "CDDL", "DPL", "KL"],
    )
    assert isinstance(pretraining_report, list)
    expected_result_1 = [
        {
            "metrics": [
                {"description": "Class Imbalance (CI)", "name": "CI", "value": 0.6},
                {"description": "Difference in Positive Proportions in Labels " "(DPL)", "name": "DPL", "value": 0.5},
                {"description": "Kullback-Liebler Divergence (KL)", "name": "KL", "value": 1.0},
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "error": "Group variable is empty or not provided",
                    "name": "CDDL",
                    "value": None,
                },
            ],
            "value_or_threshold": "(2, 3]",
        }
    ]
    assert pretraining_report == expected_result_1

    # post training bias metrics
    posttraining_report = bias_report(
        df,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cont["yhat"]),
        metrics=["AD", "CDDPL", "DCA", "DI", "DPPL", "FT"],
    )
    assert isinstance(posttraining_report, list)
    expected_result_2 = [
        {
            "metrics": [
                {
                    "description": '"Difference in Positive Proportions in ' 'Predicted Labels (DPPL)")',
                    "name": "DPPL",
                    "value": 0.75,
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": 0.0},
                {"description": "Difference in Conditional Acceptance (DCA)", "name": "DCA", "value": float("-inf")},
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": -0.75},
                {
                    "description": "Conditional Demographic Disparity in Predicted " "Labels (CDDPL)",
                    "error": "Group variable is empty or not provided",
                    "name": "CDDPL",
                    "value": None,
                },
                {"description": "Flip Test (FT)", "name": "FT", "value": 0.0},
            ],
            "value_or_threshold": "(2, 3]",
        }
    ]
    assert posttraining_report == expected_result_2


def test_metric_descriptions():
    """
    Test the list of callable metrics have descriptions present
    """
    pretraining_metrics = PRETRAINING_METRICS
    postraining_metrics = POSTTRAINING_METRICS

    pretraining_metric_descriptions = {}
    for metric in pretraining_metrics:
        description = common.metric_description(metric)
        pretraining_metric_descriptions.update({metric.__name__: description})
    expected_result_1 = {
        "CDDL": "Conditional Demographic Disparity in Labels (CDDL)",
        "CI": "Class Imbalance (CI)",
        "DPL": "Difference in Positive Proportions in Labels (DPL)",
        "JS": "Jensen-Shannon Divergence (JS)",
        "KL": "Kullback-Liebler Divergence (KL)",
        "KS": "Kolmogorov-Smirnov Distance (KS)",
        "LP": "L-p Norm (LP)",
        "TVD": "Total Variation Distance (TVD)",
    }
    assert pretraining_metric_descriptions == expected_result_1

    # post training metrics
    posttraining_metric_descriptions = {}
    for metric in postraining_metrics:
        description = common.metric_description(metric)
        posttraining_metric_descriptions.update({metric.__name__: description})
    expected_result_2 = {
        "AD": "Accuracy Difference (AD)",
        "CDDPL": "Conditional Demographic Disparity in Predicted Labels (CDDPL)",
        "DAR": "Difference in Acceptance Rates (DAR)",
        "DCA": "Difference in Conditional Acceptance (DCA)",
        "DCR": "Difference in Conditional Rejection (DCR)",
        "DI": "Disparate Impact (DI)",
        "DPPL": '"Difference in Positive Proportions in Predicted Labels (DPPL)")',
        "DRR": "Difference in Rejection Rates (DRR)",
        "FT": "Flip Test (FT)",
        "RD": "Recall Difference (RD)",
        "TE": "Treatment Equality (TE)",
    }
    assert posttraining_metric_descriptions == expected_result_2


def test_predicted_label_values():
    """
    Tests whether exception is raised when predicted label values are differnt from positive label values
    """
    df = dataframe([["a", "p", 1, "p"], ["b", "q", 1, "p"], ["b", "r", 1, "q"], ["c", "p", 0, "p"], ["c", "q", 0, "p"]])
    # when  explicit label values are given for categorical data
    # Pre training bias metrics
    with pytest.raises(
        ValueError,
        match="Positive predicted label values or threshold should" " be empty or same as label values or thresholds",
    ):
        pretraining_report = bias_report(
            df,
            FacetColumn("x"),
            LabelColumn("y", df["y"], ["p", "q"]),
            StageType.PRE_TRAINING,
            LabelColumn("yhat", df["yhat"], ["q"]),
            metrics=["DPL", "CDDL"],
            group_variable=df["z"],
        )


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
