import pandas as pd
from famly.bias.report import ProblemType, problem_type, bias_report, FacetCategoricalColumn, LabelColumn


def dataframe():
    data = [["a", 1], ["a", 1], ["b", 0], ["c", 0]]
    df = pd.DataFrame(data)
    return df


df = dataframe()
df.columns = ["data", "label"]


def test_report():
    # once the report is refactored, fill the test
    #
    bias_report(df, FacetCategoricalColumn("data"), LabelColumn("label", 1))


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
