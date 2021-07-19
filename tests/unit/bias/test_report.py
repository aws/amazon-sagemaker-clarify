# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

import pandas as pd
import pytest
from smclarify.bias.report import (
    ProblemType,
    problem_type,
    bias_basic_stats,
    bias_report,
    FacetColumn,
    LabelColumn,
    fetch_metrics_to_run,
    StageType,
    label_value_or_threshold,
)
from smclarify.bias.metrics import PRETRAINING_METRICS, POSTTRAINING_METRICS, CI, DPL, KL, KS, DPPL, DI, DCA, DCR, RD
from smclarify.bias.metrics import common


def test_invalid_input():
    df_cat = pd.DataFrame(
        [["a", 0, 0, "n"], ["b", 0, 1, "y"], ["c", 1, 0, "n"]],
        columns=["x", "y", "label", "predicted_label"],
    )
    for staging_type in StageType:
        # facet not in dataset
        with pytest.raises(ValueError):
            bias_report(
                df_cat,
                FacetColumn("z"),
                LabelColumn("Label", df_cat["label"]),
                staging_type,
            )
        # no positive label value
        with pytest.raises(ValueError):
            bias_report(
                df_cat,
                FacetColumn("x"),
                LabelColumn("Label", df_cat["label"]),
                staging_type,
            )
    # incorrect stage type
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            "pre_training",
        )
    # post-training but no predicted label column
    with pytest.raises(ValueError):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
        )
    # positive label value of label and predicted label not the same
    with pytest.raises(ValueError):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
            LabelColumn("Prediction", df_cat["predicted_label"], [1]),
        )
    # label and positive label have different data types.
    with pytest.raises(ValueError):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
            LabelColumn("Prediction", df_cat["predicted_label"], [1]),
        )
    # TODO: add more test cases.


def test_report_category_data():
    # test the bias_report function on the category data
    #
    # pre training bias metrics
    df_cat = pd.DataFrame(
        [["a", 1, 1, 1, "1"], ["b", 1, 1, 0, "0"], ["b", 0, 1, 0, "0"], ["b", 0, 0, 1, "1"]],
        columns=["x", "y", "z", "yhat", "yhat_cat"],
    )
    pretraining_report = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cat["yhat"]),
        group_variable=df_cat["z"],
    )

    pretraining_report_cat = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cat["yhat_cat"]),
        group_variable=df_cat["z"],
    )

    assert isinstance(pretraining_report, list)
    assert len(pretraining_report) > 0
    assert pretraining_report == pretraining_report_cat

    result = [
        {
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(-0.375),
                },
                {"description": "Class Imbalance (CI)", "name": "CI", "value": pytest.approx(0.5)},
                {
                    "description": "Difference in Positive Proportions in Labels (DPL)",
                    "name": "DPL",
                    "value": pytest.approx(-0.6666666666666667),
                },
                {
                    "description": "Jensen-Shannon Divergence (JS)",
                    "name": "JS",
                    "value": pytest.approx(0.08720802396075798),
                },
                {
                    "description": "Kullback-Liebler Divergence (KL)",
                    "name": "KL",
                    "value": pytest.approx(-0.3662040962227032),
                },
                {
                    "description": "Kolmogorov-Smirnov Distance (KS)",
                    "name": "KS",
                    "value": pytest.approx(0.6666666666666667),
                },
                {"description": "L-p Norm (LP)", "name": "LP", "value": pytest.approx(0.6666666666666667)},
                {
                    "description": "Total Variation Distance (TVD)",
                    "name": "TVD",
                    "value": pytest.approx(0.33333333333333337),
                },
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(0.625),
                },
                {"description": "Class Imbalance (CI)", "name": "CI", "value": pytest.approx(-0.5)},
                {
                    "description": "Difference in Positive Proportions in Labels (DPL)",
                    "name": "DPL",
                    "value": pytest.approx(0.6666666666666667),
                },
                {
                    "description": "Jensen-Shannon Divergence (JS)",
                    "name": "JS",
                    "value": pytest.approx(0.08720802396075798),
                },
                {
                    "description": "Kullback-Liebler Divergence (KL)",
                    "name": "KL",
                    "value": pytest.approx(1.0986122886681098),
                },
                {
                    "description": "Kolmogorov-Smirnov Distance (KS)",
                    "name": "KS",
                    "value": pytest.approx(0.6666666666666667),
                },
                {"description": "L-p Norm (LP)", "name": "LP", "value": pytest.approx(0.6666666666666667)},
                {
                    "description": "Total Variation Distance (TVD)",
                    "name": "TVD",
                    "value": pytest.approx(0.33333333333333337),
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

    posttraining_report_cat = bias_report(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cat["yhat_cat"]),
        metrics=["AD", "DI", "DPPL", "RD"],
        group_variable=df_cat["z"],
    )

    assert isinstance(posttraining_report, list)
    assert len(posttraining_report) > 0
    assert posttraining_report == posttraining_report_cat

    expected_result_1 = [
        {
            "metrics": [
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(-0.6666666666666667)},
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(3.0)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(-0.6666666666666667),
                },
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(-1.0)},
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(0.6666666666666667)},
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(0.3333333333333333)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(0.6666666666666667),
                },
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(1.0)},
            ],
            "value_or_threshold": "b",
        },
    ]
    assert posttraining_report == expected_result_1


def test_report_continuous_data():
    #   test the bias_report function on the category data
    #
    # pre training bias metrics
    df_cont = pd.DataFrame(
        [
            [0, 0, 0, 0, True, 1, 1, 1],
            [3, 0, 0, 0, True, 0, 1, 1],
            [3, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 0, 0, False, 1, 1, 0],
            [4, 0, 0, 1, True, 0, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 0, 0, True, 1, 1, 1],
            [3, 1, 0, 0, True, 1, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 1, 1, True, 1, 0, 1],
            [4, 0, 0, 0, True, 1, 0, 1],
            [3, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 0, 0, False, 1, 1, 0],
            [0, 0, 0, 0, True, 1, 1, 1],
            [0, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [0, 1, 0, 1, False, 0, 1, 0],
            [3, 0, 0, 0, False, 1, 1, 0],
            [0, 0, 1, 0, False, 1, 1, 1],
            [3, 0, 0, 0, True, 1, 0, 1],
            [3, 0, 1, 0, False, 1, 1, 0],
            [0, 1, 0, 0, False, 1, 1, 0],
            [3, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 0, 1, True, 1, 0, 1],
        ],
        columns=["x", "y", "z", "a", "b", "c", "d", "yhat"],
    )

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
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(0.3851010101010101),
                },
                {"description": "Class Imbalance (CI)", "name": "CI", "value": pytest.approx(-0.08333333333333333)},
                {
                    "description": "Difference in Positive Proportions in Labels (DPL)",
                    "name": "DPL",
                    "value": pytest.approx(0.1048951048951049),
                },
                {
                    "description": "Jensen-Shannon Divergence (JS)",
                    "name": "JS",
                    "value": pytest.approx(0.01252420207928287),
                },
                {
                    "description": "Kullback-Liebler Divergence (KL)",
                    "name": "KL",
                    "value": pytest.approx(0.057704603668062765),
                },
                {
                    "description": "Kolmogorov-Smirnov Distance (KS)",
                    "name": "KS",
                    "value": pytest.approx(0.1048951048951049),
                },
                {"description": "L-p Norm (LP)", "name": "LP", "value": pytest.approx(0.14834407996920576)},
                {
                    "description": "Total Variation Distance (TVD)",
                    "name": "TVD",
                    "value": pytest.approx(0.1048951048951049),
                },
            ],
            "value_or_threshold": "(2, 4]",
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
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(-0.2167832167832168)},
                {
                    "description": "Conditional Demographic Disparity in Predicted " "Labels (CDDPL)",
                    "name": "CDDPL",
                    "value": pytest.approx(0.07592592592592595),
                },
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": pytest.approx(-0.1)},
                {
                    "description": "Difference in Conditional Acceptance (DCA)",
                    "name": "DCA",
                    "value": pytest.approx(0.15),
                },
                {
                    "description": "Difference in Conditional Rejection (DCR)",
                    "name": "DCR",
                    "value": pytest.approx(1.0),
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(1.0576923076923077)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(-0.04195804195804198),
                },
                {
                    "description": "Difference in Rejection Rates (DRR)",
                    "name": "DRR",
                    "value": pytest.approx(0.6666666666666667),
                },
                {"description": "Flip Test (FT)", "name": "FT", "value": pytest.approx(-0.23076923076923078)},
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(-1.0)},
                {"description": "Treatment Equality (TE)", "name": "TE", "value": pytest.approx(-0.25)},
            ],
            "value_or_threshold": "(2, 4]",
        }
    ]
    assert posttraining_report == expected_result_1


def test_report_continuous_data_regression():
    #   test that we correctly apply thresholds for regression tasks.
    #
    df_cont_old = pd.DataFrame(
        [
            [0, 0, 0, 0, True, 1, 1, 1],
            [3, 0, 0, 0, True, 0, 1, 1],
            [3, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 0, 0, False, 1, 1, 0],
            [4, 0, 0, 1, True, 0, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 0, 0, True, 1, 1, 1],
            [3, 1, 0, 0, True, 1, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 1, 1, True, 1, 0, 1],
            [4, 0, 0, 0, True, 1, 0, 1],
            [3, 0, 1, 0, True, 1, 1, 1],
            [3, 0, 0, 0, False, 1, 1, 0],
            [0, 0, 0, 0, True, 1, 1, 1],
            [0, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 1, 0, True, 1, 1, 1],
            [0, 1, 0, 1, False, 0, 1, 0],
            [3, 0, 0, 0, False, 1, 1, 0],
            [0, 0, 1, 0, False, 1, 1, 1],
            [3, 0, 0, 0, True, 1, 0, 1],
            [3, 0, 1, 0, False, 1, 1, 0],
            [0, 1, 0, 0, False, 1, 1, 0],
            [3, 0, 1, 0, True, 0, 1, 1],
            [0, 0, 0, 1, True, 1, 0, 1],
        ],
        columns=["x", "y", "z", "a", "b", "c", "d", "yhat"],
    )

    df_cont = pd.DataFrame(
        [
            [0, 0.0, 0, 0, True, 1, 1, 11],  # 11 is the highest among y and yhat
            [3, 0.5, 0, 0, True, 0, 1, 6],
            [3, 2, 1, 0, True, 0, 1, 6.6],
            [0, 3, 0, 0, False, 1, 1, 0.3],
            [4, 2.2, 0, 1, True, 0, 1, 6],
            [0, 0.1, 1, 0, True, 1, 1, 6],
            [3, 0, 0, 0, True, 1, 1, 6],
            [3, 6, 0, 0, True, 1, 1, 6],
            [0, 0, 1, 0, True, 1, 1, 6],
            [3, 0, 1, 1, True, 1, 0, 6],
            [4, 0, 0, 0, True, 1, 0, 6],
            [3, 0, 1, 0, True, 1, 1, 6],
            [3, 0, 0, 0, False, 1, 1, 0],
            [0, 0, 0, 0, True, 1, 1, 6.2],
            [0, 0, 1, 0, True, 0, 1, 6.6],
            [0, 0, 1, 0, True, 1, 1, 6.6],
            [0, 7, 0, 1, False, 0, 1, 0.1],
            [3, 0, 0, 0, False, 1, 1, 2],
            [0, 0, 1, 0, False, 1, 1, 8],
            [3, 0, 0, 0, True, 1, 0, 9],
            [3, 0, 1, 0, False, 1, 1, 0.1],
            [0, 8, 0, 0, False, 1, 1, 2.2],
            [3, 0, 1, 0, True, 0, 1, 10],
            [0, 0, 0, 1, True, 1, 0, 9],
        ],
        columns=["x", "y", "z", "a", "b", "c", "d", "yhat"],
    )
    # Old and new df should yield the same results if we use threshold 5 for the latter.

    threshold_old = 0.5
    threshold_new = 5
    assert ((df_cont_old[["y"]] > threshold_old) == (df_cont[["y"]] > threshold_new)).all
    assert ((df_cont_old[["yhat"]] > threshold_old) == (df_cont[["yhat"]] > threshold_new)).all

    posttraining_report = bias_report(
        df_cont,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont["y"], positive_label_values=[threshold_new]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cont["yhat"], positive_label_values=[threshold_new]),
        group_variable=df_cont["z"],
    )
    posttraining_report_old = bias_report(
        df_cont_old,
        FacetColumn("x", [2]),
        LabelColumn("y", df_cont_old["y"], positive_label_values=[threshold_old]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cont_old["yhat"], positive_label_values=[threshold_old]),
        group_variable=df_cont["z"],
    )
    assert posttraining_report == posttraining_report_old


def test_label_values():
    """
    Test bias metrics for multiple label values
    """
    df = pd.DataFrame(
        [["a", "p", 1, "p"], ["b", "q", 1, "p"], ["b", "r", 1, "q"], ["c", "p", 0, "p"], ["c", "q", 0, "p"]],
        columns=["x", "y", "z", "yhat"],
    )
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
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(-0.3),
                },
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": pytest.approx(-0.25),
                },
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(0.3),
                },
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": pytest.approx(0.5),
                },
            ],
            "value_or_threshold": "b",
        },
        {
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(-0.4),
                },
                {
                    "description": "Difference in Positive Proportions in Labels (DPL)",
                    "name": "DPL",
                    "value": pytest.approx(-0.33333333333333337),
                },
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
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(-0.25)},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": pytest.approx(-0.25)},
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(1.0)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(0.0),
                },
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": pytest.approx(0)},
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(0.0)},
            ],
            "value_or_threshold": "a",
        },
        {
            "metrics": [
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(0.5)},
                {"description": "Difference in Acceptance Rates (DAR)", "name": "DAR", "value": pytest.approx(0.5)},
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(1.0)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(0.0),
                },
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": pytest.approx(0)},
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(0.0)},
            ],
            "value_or_threshold": "b",
        },
        {
            "metrics": [
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(-0.33333333333333337)},
                {
                    "description": "Difference in Acceptance Rates (DAR)",
                    "name": "DAR",
                    "value": pytest.approx(-0.33333333333333337),
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(1.0)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(0.0),
                },
                {"description": "Difference in Rejection Rates (DRR)", "name": "DRR", "value": pytest.approx(0)},
                {"description": "Recall Difference (RD)", "name": "RD", "value": pytest.approx(0.0)},
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
    assert metrics_to_run == [CI, DPL, KL, KS]

    input_metrics_2 = ["DPPL", "DI", "DCA", "DCR", "RD"]
    metrics_to_run = fetch_metrics_to_run(POSTTRAINING_METRICS, input_metrics_2)
    assert metrics_to_run == [DPPL, DI, DCA, DCR, RD]


def test_partial_bias_report():
    """
    Test that bias report is generated in for partial metrics when errors occur to compute some metrics
    """
    df = pd.DataFrame(
        [[1, 1, 1, 1], [2, 1, 1, 0], [3, 0, 0, 0], [2, 0, 1, 1], [0, 0, 1, 1]], columns=["x", "y", "z", "yhat"]
    )
    # pre training bias metrics
    pretraining_report = bias_report(
        df,
        FacetColumn("x", [2]),
        LabelColumn("y", df["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["CI", "CDDL", "DPL", "KL"],
    )
    assert isinstance(pretraining_report, list)
    expected_result_1 = [
        {
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels (CDDL)",
                    "error": "Group variable is empty or not provided",
                    "name": "CDDL",
                    "value": None,
                },
                {"description": "Class Imbalance (CI)", "name": "CI", "value": pytest.approx(0.6)},
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": pytest.approx(0.5),
                },
                {
                    "description": "Kullback-Liebler Divergence (KL)",
                    "name": "KL",
                    "value": pytest.approx(-0.34657359027997264),
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
        LabelColumn("y", df["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["AD", "CDDPL", "DCA", "DI", "DPPL", "FT"],
    )
    assert isinstance(posttraining_report, list)
    expected_result_2 = [
        {
            "metrics": [
                {"description": "Accuracy Difference (AD)", "name": "AD", "value": pytest.approx(-0.75)},
                {
                    "description": "Conditional Demographic Disparity in Predicted " "Labels (CDDPL)",
                    "error": "Group variable is empty or not provided",
                    "name": "CDDPL",
                    "value": None,
                },
                {
                    "description": "Difference in Conditional Acceptance (DCA)",
                    "name": "DCA",
                    "value": pytest.approx(0.6666666666666666),
                },
                {"description": "Disparate Impact (DI)", "name": "DI", "value": pytest.approx(0.0)},
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(0.75),
                },
                {"description": "Flip Test (FT)", "name": "FT", "value": pytest.approx(-1.0)},
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
        "DPPL": "Difference in Positive Proportions in Predicted Labels (DPPL)",
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
    df = pd.DataFrame(
        [["a", "p", 1, "p"], ["b", "q", 1, "p"], ["b", "r", 1, "q"], ["c", "p", 0, "p"], ["c", "q", 0, "p"]],
        columns=["x", "y", "z", "yhat"],
    )
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


def test_bias_basic_stats():
    df_cat = pd.DataFrame(
        [["a", 1, 1, 1, "1"], ["b", 1, 1, 0, "0"], ["b", 0, 1, 0, "0"], ["b", 0, 0, 1, "1"]],
        columns=["x", "y", "z", "yhat", "yhat_cat"],
    )

    # Proportion
    results = bias_basic_stats(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df_cat["yhat"]),
    )
    expected_results = [
        {
            "value_or_threshold": "a",
            "metrics": [
                {
                    "name": "proportion",
                    "description": "Proportion of examples in sensitive facet.",
                    "value": pytest.approx(0.25),
                }
            ],
        },
        {
            "value_or_threshold": "b",
            "metrics": [
                {
                    "name": "proportion",
                    "description": "Proportion of examples in sensitive facet.",
                    "value": pytest.approx(0.75),
                }
            ],
        },
    ]
    assert expected_results == results

    # Confusion matrix
    results = bias_basic_stats(
        df_cat,
        FacetColumn("x"),
        LabelColumn("y", df_cat["y"], [0]),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df_cat["yhat"]),
    )

    expected_results = [
        {
            "value_or_threshold": "a",
            "metrics": [
                {
                    "name": "proportion",
                    "description": "Proportion of examples in sensitive facet.",
                    "value": pytest.approx(0.25),
                },
                {
                    "name": "confusion_matrix",
                    "description": "Fractions of TP, FP, FN, TN.",
                    "value": [pytest.approx(1.0), pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0)],
                },
            ],
        },
        {
            "value_or_threshold": "b",
            "metrics": [
                {
                    "name": "proportion",
                    "description": "Proportion of examples in sensitive facet.",
                    "value": pytest.approx(0.75),
                },
                {
                    "name": "confusion_matrix",
                    "description": "Fractions of TP, FP, FN, TN.",
                    "value": [
                        pytest.approx(0.0),
                        pytest.approx(1 / 3.0),
                        pytest.approx(1 / 3.0),
                        pytest.approx(1 / 3.0),
                    ],
                },
            ],
        },
    ]
    assert expected_results == results


def test_thresholds_small_data():
    """Test that for trivial dataset where labels don't have as many element as label_values thresholds are calculated correctly"""
    data = pd.Series([0])
    positive_values = [1]
    res = label_value_or_threshold(data, positive_values)
    assert res == "(0, 1]"
