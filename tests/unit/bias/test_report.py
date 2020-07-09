import pandas as pd
from famly.bias.report import ProblemType, problem_type


def dataframe():
    data = [["a", 1], ["a", 1], ["b", 0], ["c", 0]]
    df = pd.DataFrame(data)
    return df


df = dataframe()
df.columns = ["x", "y"]


def test_report():
    pass


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
