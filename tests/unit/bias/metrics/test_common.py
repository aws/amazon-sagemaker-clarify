from typing import Any, List, NamedTuple, Optional
import pandas as pd
import pytest

from smclarify.bias.metrics.common import (
    DataType,
    series_datatype,
    ensure_series_data_type,
    convert_positive_label_values,
    calc_confusion_matrix_quadrants,
)
from .test_metrics import dfBinary


class EnsureSeriesDataTypeInput(NamedTuple):
    data: pd.Series
    values: Optional[List[Any]] = None


class EnsureSeriesDataTypeOutput(NamedTuple):
    data_type: DataType
    new_data: pd.Series


def ensure_series_data_type_test_cases():
    test_cases = []

    # categorical data series
    data = pd.Series([1, 2, 3]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data)
    test_cases.append([function_input, function_output])

    # categorical values
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2, 3])
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # floating data series
    data = pd.Series([1.0, 2.0, 3.0])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    # object data series, can NOT be converted to numeric
    data = pd.Series(["a", "b", "c"])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # object data series, can be converted to numeric, and uniqueness is high
    data = pd.Series(["1", "2", "3"])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=pd.to_numeric(data))
    test_cases.append([function_input, function_output])

    # object data series, can be converted to numeric, but uniqueness is low
    data = ["1"] * 40
    data.append("2")
    data = pd.Series(data)
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # integer data series, uniqueness is high
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    # integer data series, uniqueness is low
    data = [1] * 40
    data.append(2)
    data = pd.Series(data)
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # boolean data series
    data = pd.Series([True, False, True])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # continous with empty values
    data = pd.Series([0.2, 0.3, 0.4])
    function_input = EnsureSeriesDataTypeInput(data=data, values=[])
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    return test_cases


def convert_positive_label_values_test_case():
    test_cases = []

    # series - int, label values - int
    data = pd.Series([1, 2, 3]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2])
    function_output = [1, 2]
    test_cases.append([function_input, function_output])

    # series - int, label values - str
    data = pd.Series([1, 2, 3]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=["1", "2"])
    function_output = [1, 2]
    test_cases.append([function_input, function_output])

    # series - int, label values - string float
    data = pd.Series([1, 2, 3]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=["1.0", "2.0"])
    function_output = [1, 2]
    test_cases.append([function_input, function_output])

    # series - string, label values - string
    data = pd.Series(["1", "2", "3"]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=["1", "2"])
    function_output = ["1", "2"]
    test_cases.append([function_input, function_output])

    # series - string, label values - int
    data = pd.Series(["1", "2", "3"]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2])
    function_output = ["1", "2"]
    test_cases.append([function_input, function_output])

    # series - string float, label values - float
    data = pd.Series(["1.0", "2.0", "3.0"]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1.0, 2.0])
    function_output = ["1.0", "2.0"]
    test_cases.append([function_input, function_output])

    # series - float, label values - float
    data = pd.Series([1.0, 2.0, 3.0]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1.0, 2.0])
    function_output = [1.0, 2.0]
    test_cases.append([function_input, function_output])

    # series - float, label values - string float
    data = pd.Series([1.0, 2.0, 3.0]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=["1.0", "2.0"])
    function_output = [1.0, 2.0]
    test_cases.append([function_input, function_output])

    # series - float, label values - int
    data = pd.Series([1.0, 2.0, 3.0]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2])
    function_output = [1.0, 2.0]
    test_cases.append([function_input, function_output])

    # series - bool, label values - bool
    data = pd.Series([True, True, False]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[True])
    function_output = [True]
    test_cases.append([function_input, function_output])

    # series - string, label values - bool
    data = pd.Series(["True", "True", "False"]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[True])
    function_output = ["True"]
    test_cases.append([function_input, function_output])

    # series - bool, label values - int
    data = pd.Series([True, True, False]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2, 0])
    function_output = [True, True, False]
    test_cases.append([function_input, function_output])

    # series - int, label values - bool
    data = pd.Series([1, 1, 0]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data, values=[True, False])
    function_output = [1, 0]
    test_cases.append([function_input, function_output])

    return test_cases


@pytest.mark.parametrize("function_input,function_output", ensure_series_data_type_test_cases())
def test_ensure_series_data_type(function_input, function_output):
    # Test the series_datatype function by the way
    data_type = series_datatype(*function_input)
    assert data_type == function_output.data_type
    # Test the ensure_series_data_type
    data_type, new_data = ensure_series_data_type(*function_input)
    assert data_type == function_output.data_type
    assert new_data.equals(function_output.new_data)


@pytest.mark.parametrize("function_input,function_output", convert_positive_label_values_test_case())
def test_convert_positive_label_values(function_input, function_output):
    positive_label_values = convert_positive_label_values(*function_input)
    assert positive_label_values == function_output


def test_calc_confusion_matrix_quadrants():
    # binary
    (dfB, dfB_label, dfB_pos_label_idx, dfB_pred_label, dfB_pos_pred_label_idx) = dfBinary()
    dfB_features = dfB[0]

    assert (2, 3, 4, 3) == calc_confusion_matrix_quadrants(dfB_features, dfB_pos_label_idx, dfB_pos_pred_label_idx)

    dfB_sensitive_features = dfB_features == "F"
    assert (2, 2, 2, 1) == calc_confusion_matrix_quadrants(
        dfB_features[dfB_sensitive_features],
        dfB_pos_label_idx[dfB_sensitive_features],
        dfB_pos_pred_label_idx[dfB_sensitive_features],
    )

    dfB_sensitive_features = dfB_features == "M"
    assert (0, 1, 2, 2) == calc_confusion_matrix_quadrants(
        dfB_features[dfB_sensitive_features],
        dfB_pos_label_idx[dfB_sensitive_features],
        dfB_pos_pred_label_idx[dfB_sensitive_features],
    )

    # multi category
    dfM = pd.DataFrame(
        [
            ("a", "white", 1, "red"),
            ("b", "white", 1, "blue"),
            ("b", "blue", 1, "blue"),
            ("b", "blue", 0, "red"),
            ("a", "green", 1, "white"),
            ("b", "white", 1, "white"),
            ("b", "white", 1, "green"),
            ("b", "white", 0, "white"),
        ]
    )
    dfM.columns = ["x", "y", "z", "yhat"]
    dfM_features = dfM["x"]
    dfM_label = dfM["y"]
    dfM_predicted_label = dfM["yhat"]
    dfM_pos_label_idx = dfM_label == "blue"
    dfM_pos_pred_label_idx = dfM_predicted_label == "blue"

    assert (1, 5, 1, 1) == calc_confusion_matrix_quadrants(dfM_features, dfM_pos_label_idx, dfM_pos_pred_label_idx)

    dfM_sensitive_features = dfM_features == "a"
    assert (0, 2, 0, 0) == calc_confusion_matrix_quadrants(
        dfM_features[dfM_sensitive_features],
        dfM_pos_label_idx[dfM_sensitive_features],
        dfM_pos_pred_label_idx[dfM_sensitive_features],
    )

    dfM_sensitive_features = dfM_features == "b"
    assert (1, 3, 1, 1) == calc_confusion_matrix_quadrants(
        dfM_features, dfM_pos_label_idx, dfM_pos_pred_label_idx[dfM_sensitive_features]
    )
