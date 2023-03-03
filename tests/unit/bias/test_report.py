# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
from typing import NamedTuple, List, Any, Union, Dict

import pandas as pd
import pytest
from smclarify.bias.report import (
    ProblemType,
    problem_type,
    bias_basic_stats,
    model_performance_report,
    bias_report,
    FacetColumn,
    LabelColumn,
    fetch_metrics_to_run,
    StageType,
    label_value_or_threshold,
)
from smclarify.bias.metrics import PRETRAINING_METRICS, POSTTRAINING_METRICS, CI, DPL, KL, KS, DPPL, DI, DCA, DCR, RD
from smclarify.bias.metrics import common


class LabelValueInput(NamedTuple):
    df: pd.DataFrame
    positive_label_values: List[Union[str, float, int, bool]]


class LabelValueOutput(NamedTuple):
    value_or_threshold: str
    metrics: Dict[str, float]


def test_invalid_input():
    df_cat = pd.DataFrame(
        [["a", 0, 0, "n"], ["b", 0, 1, "y"], ["c", 1, 0, "n"]],
        columns=["x", "y", "label", "predicted_label"],
    )
    for staging_type in StageType:
        # facet not in dataset
        with pytest.raises(ValueError, match="Facet column z is not present in the dataset"):
            bias_report(
                df_cat,
                FacetColumn("z"),
                LabelColumn("Label", df_cat["label"]),
                staging_type,
            )
        # no positive label value
        with pytest.raises(ValueError, match="Positive label values or thresholds are empty for Label column"):
            bias_report(
                df_cat,
                FacetColumn("x"),
                LabelColumn("Label", df_cat["label"]),
                staging_type,
            )
    # incorrect stage type
    with pytest.raises(ValueError, match="stage_type should be a Enum value of StageType"):
        # noinspection PyTypeChecker
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            "pre_training",
        )
    # post-training but no predicted label column
    with pytest.raises(ValueError, match="predicted_label_column has to be provided for Post training metrics"):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
        )
    # positive label value of label and predicted label not the same
    match_message = "Positive predicted label values or threshold should be empty or same as label values or thresholds"
    with pytest.raises(ValueError, match=match_message):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
            LabelColumn("Prediction", df_cat["predicted_label"], [0]),
        )

    # label and positive label have different data types.
    match_message = "Predicted Label Column series datatype is not the same as Label Column series"
    with pytest.raises(ValueError, match=match_message):
        bias_report(
            df_cat,
            FacetColumn("x"),
            LabelColumn("Label", df_cat["label"], [1]),
            StageType.POST_TRAINING,
            LabelColumn("Prediction", df_cat["predicted_label"], [1]),
        )

    # threshold not provided for continuous facet
    df = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        columns=["Label", "Facet", "Feature", "PredictedLabel"],
    )
    with pytest.raises(ValueError, match="Threshold values must be provided for continuous features"):
        bias_report(
            df=df,
            facet_column=FacetColumn("Facet"),
            label_column=LabelColumn("Label", df["Label"], [2.0]),
            stage_type=StageType.POST_TRAINING,
            predicted_label_column=LabelColumn("PredictedLabel", df["PredictedLabel"], [2.0]),
        )

    with pytest.raises(
        ValueError, match="Facet/label value provided must be a single numeric threshold for continuous data"
    ):
        bias_report(
            df=df,
            facet_column=FacetColumn("Facet", [3.0]),
            label_column=LabelColumn("Label", df["Label"], ["string_threshold"]),
            stage_type=StageType.PRE_TRAINING,
        )


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
                {
                    "description": "Generalized Entropy (GE)",
                    "name": "GE",
                    "value": 0.07593688362919139,
                },
                {"description": "Recall Difference (RD)", "name": "RD", "value": -1.0},
                {"description": "Specificity Difference (SD)", "name": "SD", "value": 0.1388888888888889},
                {"description": "Treatment Equality (TE)", "name": "TE", "value": pytest.approx(-0.25)},
            ],
            "value_or_threshold": "(2, 4]",
        }
    ]
    print(posttraining_report)
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


def test_report_string_data_determined_as_continuous():
    # Although the data columns look like categorical, they are determined as continuous
    # because the data can be casted to numbers, and the data uniqueness is high.
    # The test case means to check if the report method can handle the case correctly.
    df = pd.DataFrame(
        data=[
            ["1", "1", "1", "1"],
            ["2", "2", "2", "2"],
            ["3", "3", "3", "3"],
            ["4", "4", "4", "4"],
        ],
        columns=["Label", "Facet", "Feature", "PredictedLabel"],
    )
    pretraining_report = bias_report(
        df=df,
        facet_column=FacetColumn("Facet", [2]),
        label_column=LabelColumn("Label", df["Label"], [2]),
        stage_type=StageType.POST_TRAINING,
        predicted_label_column=LabelColumn("PredictedLabel", df["PredictedLabel"], [2]),
        metrics=["DPPL"],
    )
    # Actually the validation below is not really needed. If there was problem then the report method
    # should have failed with error like "TypeError: bad operand type for abs(): 'str'" when it tried to
    # manipulate string as number.
    assert pretraining_report == [
        {
            "value_or_threshold": "(2, 4]",  # <== range, so the facet is indeed determined as continuous
            "metrics": [
                {
                    "name": "DPPL",
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "value": pytest.approx(-1.0),
                }
            ],
        }
    ]


def test_report_integer_data_determined_as_categorical():
    # Although the data columns look like continuous, they are determined as categorical because
    # the facet values or label values are categorical. Note that the label column and the predicted
    # label column have different categories (1,3,4 and 2,3,4 respectively). They can not be cast to
    # the type of each other, but no problem to get the positive label index.
    df = pd.DataFrame(
        data=[
            [1, 1, 1, 2],
            [1, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
        ],
        columns=["Label", "Facet", "Feature", "PredictedLabel"],
    )
    pretraining_report = bias_report(
        df=df,
        facet_column=FacetColumn("Facet", [1, 2]),
        label_column=LabelColumn("Label", df["Label"], [1, 2]),
        stage_type=StageType.POST_TRAINING,
        predicted_label_column=LabelColumn("PredictedLabel", df["PredictedLabel"], [1, 2]),
        metrics=["DPPL"],
    )
    assert pretraining_report == [
        {
            "value_or_threshold": "1,2",  # <== range, so the facet is indeed determined as categorical
            "metrics": [
                {
                    "name": "DPPL",
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "value": pytest.approx(-1.0),
                }
            ],
        }
    ]


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


def label_values_test_cases() -> List[List[Union[LabelValueInput, List[LabelValueOutput]]]]:
    """
    Setting the `y` and `yhat` series .astype('category'),
    as this conversion feature is supposed to work only on categorical data.
    """
    test_cases = []
    output = [
        LabelValueOutput("a", {"CDDL": -0.3, "DPL": -0.25}),
        LabelValueOutput("b", {"CDDL": 0.3, "DPL": 0.5}),
        LabelValueOutput("c", {"CDDL": -0.4, "DPL": -0.33333333333333337}),
    ]

    df = pd.DataFrame(
        [["a", None, 1, None], ["b", None, 1, None], ["b", None, 1, None], ["c", None, 0, None], ["c", None, 0, None]],
        columns=["x", "y", "z", "yhat"],
    )

    # series - int, label values - int
    df["y"] = pd.Series([1, 2, 0, 1, 2]).astype("category")
    df["yhat"] = pd.Series([1, 1, 0, 1, 1]).astype("category")
    function_input = LabelValueInput(df=df.copy(), positive_label_values=[1, 2])
    test_cases.append([function_input, output[:]])

    # series - str, label values - int
    df["y"] = pd.Series(["1", "2", "0", "1", "2"]).astype("category")
    df["yhat"] = pd.Series(["1", "1", "0", "1", "1"]).astype("category")
    function_input = LabelValueInput(df=df.copy(), positive_label_values=[1, 2])
    test_cases.append([function_input, output[:]])

    # series - int, label values - str
    df["y"] = pd.Series([1, 2, 0, 1, 2]).astype("category")
    df["yhat"] = pd.Series([1, 1, 0, 1, 1]).astype("category")
    function_input = LabelValueInput(df=df.copy(), positive_label_values=["1", "2"])
    test_cases.append([function_input, output[:]])

    # series - str, label values - str
    df["y"] = pd.Series(["1", "2", "0", "1", "2"]).astype("category")
    df["yhat"] = pd.Series(["1", "1", "0", "1", "1"]).astype("category")
    function_input = LabelValueInput(df=df.copy(), positive_label_values=["1", "2"])
    test_cases.append([function_input, output[:]])

    return test_cases


@pytest.mark.parametrize("function_input,function_output", label_values_test_cases())
def test_label_values_with_different_types_for_pre_training(
    function_input: LabelValueInput, function_output: List[LabelValueOutput]
):
    df = function_input.df
    pretraining_report = bias_report(
        df,
        FacetColumn("x"),
        LabelColumn("y", df["y"], function_input.positive_label_values),
        StageType.PRE_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["DPL", "CDDL"],
        group_variable=df["z"],
    )
    expected_result_1 = [
        {
            "value_or_threshold": output.value_or_threshold,
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Labels " "(CDDL)",
                    "name": "CDDL",
                    "value": pytest.approx(output.metrics["CDDL"]),
                },
                {
                    "description": "Difference in Positive Proportions in Labels " "(DPL)",
                    "name": "DPL",
                    "value": pytest.approx(output.metrics["DPL"]),
                },
            ],
        }
        for output in function_output
    ]
    assert pretraining_report == expected_result_1


@pytest.mark.parametrize("function_input,function_output", label_values_test_cases())
def test_label_values_with_different_types_for_post_training(
    function_input: LabelValueInput, function_output: List[LabelValueOutput]
):
    df = function_input.df
    pretraining_report = bias_report(
        df,
        FacetColumn("x"),
        LabelColumn("y", df["y"], function_input.positive_label_values),
        StageType.POST_TRAINING,
        LabelColumn("yhat", df["yhat"]),
        metrics=["DPPL", "CDDPL"],
        group_variable=df["z"],
    )
    expected_result_1 = [
        {
            "value_or_threshold": output.value_or_threshold,
            "metrics": [
                {
                    "description": "Conditional Demographic Disparity in Predicted " "Labels (CDDPL)",
                    "name": "CDDPL",
                    "value": pytest.approx(output.metrics["CDDL"]),
                },
                {
                    "description": "Difference in Positive Proportions in Predicted " "Labels (DPPL)",
                    "name": "DPPL",
                    "value": pytest.approx(output.metrics["DPL"]),
                },
            ],
        }
        for output in function_output
    ]
    assert pretraining_report == expected_result_1


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
        metrics=["AD", "CDDPL", "DCA", "DI", "DPPL", "FT", "GE", "SD"],
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
                {"description": "Generalized Entropy (GE)", "name": "GE", "value": 0.19444444444444456},
                {"description": "Specificity Difference (SD)", "name": "SD", "value": 1.0},
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
        "GE": "Generalized Entropy (GE)",
        "RD": "Recall Difference (RD)",
        "SD": "Specificity Difference (SD)",
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
                },
                {
                    "name": "observed_label_distribution",
                    "description": "Distribution of observed label outcomes for sensitive facet",
                    "value": [pytest.approx(1.0), pytest.approx(0.0)],
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
                    "name": "observed_label_distribution",
                    "description": "Distribution of observed label outcomes for sensitive facet",
                    "value": [pytest.approx(1 / 3.0), pytest.approx(2 / 3.0)],
                },
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
                    "name": "observed_label_distribution",
                    "description": "Distribution of observed label outcomes for sensitive facet",
                    "value": [pytest.approx(1.0), pytest.approx(0.0)],
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
                    "name": "observed_label_distribution",
                    "description": "Distribution of observed label outcomes for sensitive facet",
                    "value": [pytest.approx(1 / 3.0), pytest.approx(2 / 3.0)],
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


def test_model_performance_categorical():
    df_cat = pd.DataFrame(
        [["a", "p", 1, 1, "q"], ["b", "p", 1, 0, "r"], ["b", "r", 1, 0, "q"], ["b", "q", 0, 1, "p"]],
        columns=["x", "y_cat", "z", "yhat", "yhat_cat"],
    )
    result = model_performance_report(
        df=df_cat,
        label_column=LabelColumn("y_cat", df_cat["y_cat"], ["p"]),
        predicted_label_column=LabelColumn("yhat_cat", df_cat["yhat_cat"], ["p"]),
    )

    expected_result = {
        "label": "y_cat",
        "model_performance_metrics": [
            {
                "name": "Accuracy",
                "description": "Proportion of inputs assigned the correct predicted label by the model.",
                "value": pytest.approx(1 / 4.0),
            },
            {
                "name": "Proportion of Positive Predictions in Labels",
                "description": "Proportion of input assigned in positive predicted label.",
                "value": pytest.approx(1 / 4.0),
            },
            {
                "name": "Proportion of Negative Predictions in Labels",
                "description": "Proportion of input assigned the negative predicted label.",
                "value": pytest.approx(3 / 4.0),
            },
            {
                "name": "True Positive Rate / Recall",
                "description": "Proportion of inputs with positive observed label correctly assigned the positive predicted label.",
                "value": pytest.approx(0.0),
            },
            {
                "name": "True Negative Rate / Specificity",
                "description": "Proportion of inputs with negative observed label correctly assigned the negative predicted label.",
                "value": pytest.approx(1 / 2.0),
            },
            {
                "name": "Acceptance Rate / Precision",
                "description": "Proportion of inputs with positive predicted label that actually have a positive observed label.",
                "value": pytest.approx(0.0),
            },
            {
                "name": "Rejection Rate",
                "description": "Proportion of inputs with negative predicted label that actually have a negative observed label.",
                "value": pytest.approx(1 / 3.0),
            },
            {
                "name": "Conditional Acceptance",
                "description": "Ratio between the positive observed labels and positive predicted labels.",
                "value": pytest.approx(2.0),
            },
            {
                "name": "Conditional Rejection",
                "description": "Ratio between the negative observed labels and negative predicted labels.",
                "value": pytest.approx(2 / 3.0),
            },
            {"name": "F1 Score", "description": "Harmonic mean of precision and recall.", "value": pytest.approx(0.0)},
        ],
        "binary_confusion_matrix": [pytest.approx(0.0), pytest.approx(0.25), pytest.approx(0.5), pytest.approx(0.25)],
        "confusion_matrix": {
            "p": {"p": pytest.approx(0.0), "q": pytest.approx(1.0), "r": pytest.approx(1.0)},
            "q": {"p": pytest.approx(1.0), "q": pytest.approx(0.0), "r": pytest.approx(0.0)},
            "r": {"p": pytest.approx(0.0), "q": pytest.approx(1.0), "r": pytest.approx(0.0)},
        },
    }
    assert expected_result == result


def test_model_performance_continuous():
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

    result = model_performance_report(
        df=df_cont,
        label_column=LabelColumn("y", df_cont["y"], [5]),
        predicted_label_column=LabelColumn("yhat", df_cont["yhat"], [5]),
    )
    # No multicategory confusion matrix

    expected_result = {
        "label": "y",
        "model_performance_metrics": [
            {
                "name": "Accuracy",
                "description": "Proportion of inputs assigned the correct predicted label by the model.",
                "value": pytest.approx(5 / 24),
            },
            {
                "name": "Proportion of Positive Predictions in Labels",
                "description": "Proportion of input assigned in positive predicted label.",
                "value": pytest.approx(0.75),
            },
            {
                "name": "Proportion of Negative Predictions in Labels",
                "description": "Proportion of input assigned the negative predicted label.",
                "value": pytest.approx(0.25),
            },
            {
                "name": "True Positive Rate / Recall",
                "description": "Proportion of inputs with positive observed label correctly assigned the positive predicted label.",
                "value": pytest.approx(1 / 3),
            },
            {
                "name": "True Negative Rate / Specificity",
                "description": "Proportion of inputs with negative observed label correctly assigned the negative predicted label.",
                "value": pytest.approx(4 / 21),
            },
            {
                "name": "Acceptance Rate / Precision",
                "description": "Proportion of inputs with positive predicted label that actually have a positive observed label.",
                "value": pytest.approx(1 / 18),
            },
            {
                "name": "Rejection Rate",
                "description": "Proportion of inputs with negative predicted label that actually have a negative observed label.",
                "value": pytest.approx(2 / 3.0),
            },
            {
                "name": "Conditional Acceptance",
                "description": "Ratio between the positive observed labels and positive predicted labels.",
                "value": pytest.approx(1 / 6),
            },
            {
                "name": "Conditional Rejection",
                "description": "Ratio between the negative observed labels and negative predicted labels.",
                "value": pytest.approx(3.5),
            },
            {
                "name": "F1 Score",
                "description": "Harmonic mean of precision and recall.",
                "value": pytest.approx(2 / 21),
            },
        ],
        "binary_confusion_matrix": [
            pytest.approx(1 / 24),
            pytest.approx(17 / 24),
            pytest.approx(1 / 12),
            pytest.approx(1 / 6),
        ],
    }
    assert expected_result == result


class LabelValueOrThresholdFunctionInput(NamedTuple):
    data: pd.Series
    values: List[Any]


class LabelValueOrThresholdFunctionOutput(NamedTuple):
    result: str


def label_value_or_threshold_test_cases():
    test_cases = []

    # categorical data series
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([1, 2, 3]).astype("category"), values=[2])
    function_output = LabelValueOrThresholdFunctionOutput(result="2")  # instead of "(2, 3]"
    test_cases.append([function_input, function_output])

    # categorical values
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([1, 2, 3]), values=[1, 2])
    function_output = LabelValueOrThresholdFunctionOutput(result="1,2")
    test_cases.append([function_input, function_output])

    # continuous data series
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([1.0, 2.0, 3.0]), values=[2.0])
    function_output = LabelValueOrThresholdFunctionOutput(result="(2.0, 3.0]")
    test_cases.append([function_input, function_output])

    # continuous data series, positive value less than all data
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([1.0, 2.0, 3.0]), values=[0.0])
    function_output = LabelValueOrThresholdFunctionOutput(result="(0.0, 3.0]")
    test_cases.append([function_input, function_output])

    # continuous data series, positive value greater than all data
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([1.0, 2.0, 3.0]), values=[5.0])
    function_output = LabelValueOrThresholdFunctionOutput(result="(3.0, 5.0]")
    test_cases.append([function_input, function_output])

    # object data series, can NOT be converted to numeric
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series(["yes", "no", "yes"]), values=["yes"])
    function_output = LabelValueOrThresholdFunctionOutput(result="yes")
    test_cases.append([function_input, function_output])

    # object data series, can be converted to numeric, and uniqueness is high
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series(["1", "2", "3"]), values=[2])
    function_output = LabelValueOrThresholdFunctionOutput(result="(2, 3]")
    test_cases.append([function_input, function_output])

    # boolean data series
    function_input = LabelValueOrThresholdFunctionInput(data=pd.Series([True, False, True]), values=[True])
    function_output = LabelValueOrThresholdFunctionOutput(result="True")
    test_cases.append([function_input, function_output])

    # Test that for trivial dataset where labels don't have as many element as label_values
    data = pd.Series([0])
    positive_values = [1]
    function_input = LabelValueOrThresholdFunctionInput(data=data, values=positive_values)
    function_output = LabelValueOrThresholdFunctionOutput(result="(0, 1]")
    test_cases.append([function_input, function_output])

    return test_cases


@pytest.mark.parametrize("function_input,function_output", label_value_or_threshold_test_cases())
def test_label_value_or_threshold(function_input, function_output):
    result = label_value_or_threshold(*function_input)
    assert result == function_output.result
