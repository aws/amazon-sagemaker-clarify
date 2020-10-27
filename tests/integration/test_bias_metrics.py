#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging

from typing import Dict, Optional

import pandas as pd
from famly.bias.report import FacetColumn, LabelColumn, bias_report, StageType
from famly.util.dataset import Datasets

logger = logging.getLogger(__name__)
RESOURCES_DIR = os.path.join(os.getcwd(), "tests", "resources")


def fetch_input_data() -> pd.DataFrame:
    dataset = Datasets()
    s3_input_obj = dataset("german_csv")
    df = s3_input_obj.read_csv_data()
    return df


def get_predicted_labels() -> pd.DataFrame:
    dataset = Datasets()
    s3_pred_label_obj = dataset("german_predicted_labels")
    predicted_labels = s3_pred_label_obj.read_csv_data(index_col=0)
    return predicted_labels.squeeze()


def get_pretraining_bias_metrics(
    dataframe: pd.DataFrame, facet_column: FacetColumn, label_column: LabelColumn, group_variable: Optional[pd.Series]
) -> Dict:
    # Measure pre-training bias for the ForeignWorker attribute
    return bias_report(
        dataframe,
        facet_column,
        label_column,
        stage_type=StageType.PRE_TRAINING,
        metrics=["all"],
        group_variable=group_variable,
    )


def get_posttraining_bias_metrics(
    dataframe: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    pred_label_column: LabelColumn,
    group_variable: Optional[pd.Series],
) -> Dict:
    # Measure the post-training bias for the ForeignWorker attribute
    report = bias_report(
        dataframe,
        facet_column,
        label_column,
        stage_type=StageType.POST_TRAINING,
        predicted_label_column=pred_label_column,
        metrics=["all"],
        group_variable=group_variable,
    )
    return report


def test_bias_metrics():
    dataframe = fetch_input_data()
    label_data = dataframe.pop("Class1Good2Bad")
    label_column = LabelColumn("Class1Good2Bad", label_data, [1])
    facet_column = FacetColumn("ForeignWorker", [1])
    group_variable = dataframe["A151"]

    # pre_training_bias metrics
    pre_training_metrics = get_pretraining_bias_metrics(dataframe, facet_column, label_column, group_variable)

    # post training bias metrics
    predicted_labels = get_predicted_labels()
    pred_label_column = LabelColumn("_predicted_labels", predicted_labels, [1])

    post_training_metrics = get_posttraining_bias_metrics(
        dataframe, facet_column, label_column, pred_label_column, group_variable
    )

    pre_training_expected_result = [
        {
            "value_or_threshold": "1",
            "metrics": [
                {
                    "name": "CDDL",
                    "description": "Conditional Demographic Disparity in Labels (CDDL)",
                    "value": 0.029771892530848814,
                },
                {"name": "CI", "description": "Class Imbalance (CI)", "value": -0.9288888888888889},
                {
                    "name": "DPL",
                    "description": "Difference in Positive Proportions in Labels (DPL)",
                    "value": 0.17453917050691248,
                },
                {"name": "JS", "description": "Jensen-Shannon Divergence (JS)", "value": 0.04021236938805562},
                {"name": "KL", "description": "Kullback-Liebler Divergence (KL)", "value": 0.08543332780657628},
                {"name": "KS", "description": "Kolmogorov-Smirnov Distance (KS)", "value": 0.3490783410138249},
                {"name": "LP", "description": "L-p Norm (LP)", "value": 0.2468356620962257},
                {"name": "TVD", "description": "Total Variation Distance (TVD)", "value": 0.17453917050691245},
            ],
        }
    ]

    post_training_expected_result = [
        {
            "value_or_threshold": "1",
            "metrics": [
                {"name": "AD", "description": "Accuracy Difference (AD)", "value": 0.03312211981566815},
                {
                    "name": "CDDPL",
                    "description": "Conditional Demographic Disparity in Predicted Labels (CDDPL)",
                    "value": 0.032647137172999274,
                },
                {"name": "DAR", "description": "Difference in Acceptance Rates (DAR)", "value": 0.017096617181796114},
                {
                    "name": "DCA",
                    "description": "Difference in Conditional Acceptance (DCA)",
                    "value": -0.035775127768313375,
                },
                {
                    "name": "DCR",
                    "description": "Difference in Conditional Rejection (DCR)",
                    "value": -0.07473309608540923,
                },
                {"name": "DI", "description": "Disparate Impact (DI)", "value": 0.7728768926925609},
                {
                    "name": "DPPL",
                    "description": "Difference in Positive Proportions in Predicted Labels (DPPL)",
                    "value": 0.19873271889400923,
                },
                {"name": "DRR", "description": "Difference in Rejection Rates (DRR)", "value": 0.06494661921708189},
                {"name": "FT", "description": "Flip Test (FT)", "value": -0.32373271889400923},
                {"name": "RD", "description": "Recall Difference (RD)", "value": 0.049812030075187974},
                {"name": "TE", "description": "Treatment Equality (TE)", "value": 0.6774193548387097},
            ],
        }
    ]

    assert pre_training_metrics == pre_training_expected_result
    assert post_training_metrics == post_training_expected_result
