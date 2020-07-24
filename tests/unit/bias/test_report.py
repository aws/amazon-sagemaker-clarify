import pandas as pd
from famly.bias.report import ProblemType, problem_type, bias_report, FacetCategoricalColumn, LabelColumn


def dataframe():
    data = [["a", 1, 1], ["a", 1, 0], ["b", 0, 0], ["c", 0, 1]]
    df = pd.DataFrame(data, columns=["x", "y", "yhat"])
    return df


df = dataframe()


def test_report():
    # once the report is refactored, fill the test
    #
    report = bias_report(df, FacetCategoricalColumn("x"), LabelColumn("y", 1), LabelColumn("yhat", 1))
    assert isinstance(report, dict)
    assert len(report) > 0
    # Check that we have metric for each of the 3 classes vs the rest
    for k, v in report.items():
        assert len(v) == 3


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
