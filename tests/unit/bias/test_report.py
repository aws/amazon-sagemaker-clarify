import pandas as pd
from famly.bias.report import ProblemType, class_imbalance_values, column_list_to_str, problem_type


def dataframe():
    data = [["a"], ["a"], ["b"], ["c"]]
    df = pd.DataFrame(data)
    return df


df = dataframe()


def test_class_imbalance():
    assert class_imbalance_values(df[0]) == {"a": 0.0, "b": 0.5, "c": 0.5}
    protected = ["b", "c"]
    assert class_imbalance_values(df[0], protected) == {column_list_to_str(protected): 0.0}


def test_problem_type():
    series = pd.Series([1, 2, 1, 2])
    assert problem_type(series) == ProblemType.BINARY
