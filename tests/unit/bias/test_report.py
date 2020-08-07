import pandas as pd
from famly.bias.report import ProblemType, problem_type, bias_report, FacetColumn, LabelColumn
from typing import List, Any


def dataframe(data: List[List[Any]]):
    df = pd.DataFrame(data, columns=["x", "y", "yhat"])
    return df


df_cat = dataframe([["a", 1, 1], ["b", 1, 0], ["b", 0, 0], ["c", 0, 1]])
df_cont = dataframe([[1, 1, 1], [2, 1, 0], [3, 0, 0], [2, 0, 1]])


def test_report_category_data():
    # once the report is refactored, fill the test
    #
    report = bias_report(df_cat, FacetColumn("x"), LabelColumn("y", 1), LabelColumn("yhat", 1))
    assert isinstance(report, dict)
    assert len(report) > 0
    print(report)
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report.items():
        assert len(v) == 3
    cat_result = {
        "CI": {"a": 0.5, "b": 0.0, "c": 0.5},
        "DPL": {"a": 0.0, "b": 0.0, "c": 0.0},
        "KL": {"a": 1.584962500721156, "b": 0.0, "c": 1.584962500721156},
        "JS": {"a": 0.2789960722619452, "b": 0.0, "c": 0.2789960722619452},
        "LP": {"a": 0.6666666666666667, "b": 0.0, "c": 0.6666666666666667},
        "TVD": {"a": 0.33333333333333337, "b": 0.0, "c": 0.33333333333333337},
        "KS": {"a": 0.6666666666666667, "b": 0.0, "c": 0.6666666666666667},
    }
    assert report == report


def test_report_continuous_data():
    # Todo: check if there is better way to
    #
    report = bias_report(df_cont, FacetColumn("x"), LabelColumn("y", 1), LabelColumn("yhat", 1))
    assert isinstance(report, dict)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report.items():
        assert len(v) == 1
    result = {
        "CI": {"(2, 3]": 0.5},
        "DPL": {"(2, 3]": 0.0},
        "KL": {"(2, 3]": 1.584962500721156},
        "JS": {"(2, 3]": 0.2789960722619452},
        "LP": {"(2, 3]": 0.6666666666666667},
        "TVD": {"(2, 3]": 0.33333333333333337},
        "KS": {"(2, 3]": 0.6666666666666667},
    }
    assert report == result


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
